"""Candidate pools: starting-policy completions scored by every judge instance (Methods, "Emission and absorption").

A cell's candidate pool is drawn from the starting student, in fp32 without
autocast: for each of the 48 training prompts, 64 completions are sampled with
``torch.manual_seed(pool_seed * 100 + prompt)`` (temperature 1.0, no top-p/top-k,
at most 64 new tokens, the student's own sampling defaults neutralized, as in
training). A completion is cut after its first EOS, decoded without special
tokens, and its first line is checked with the cell's carrier rule. The share of
valid candidates is the carrier-validity gate (a cell enters the grid if it is at
least 0.5).

Every valid candidate is then scored, in fp32, by the student itself (``l0``, its
summed log-likelihood in the student context) and by every judge instance with
the training reward (equation r):

* FT route: eight fine-tuned instances per judge, ``T_k0..T_k7`` and
  ``E_k0..E_k7`` (k0 = training instance), each in the student's context;
* SP route: the base model without persona (``B``) and with each of the eight
  persona wordings for each judge (``T_k<v>``, ``E_k<v>``; v0 = training wording);
* a cross-family judge scores with its own tokenizer and chat template.

Gate: batched scores equal single-sequence scores within ``tol`` (1e-3) on a
subset of ten candidates, for the student and for every judge instance.

Output ``<out>/pool_<name>.npz``: ``qi_all``, ``valid_all``, ``ids_all`` (all
3,072 candidates), ``vi`` (indices of valid candidates), ``qi`` (their prompts),
``l0``, ``m_AN32_base`` (step-0 per-token preference margins), ``keys`` and one
``r_<key>`` array per judge instance; ``pool_<name>.jsonl`` (text of every
candidate); ``pool_<name>.gates.json``.

In the paper's pipeline the Llama syllable and Qwen syllable pools were drawn
under the looser base syllable rule and filtered afterwards to the cell rule
(the cell rule is strictly narrower, so the valid set is the same); this module
applies the cell rule directly.
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from .. import carriers as CAR
from ..chat import judge_prompt, load_tokenizer, sum_logprob, sum_logprob_single, user_prompt
from ..evaluation.preference import PreferenceEvaluation
from ..judges.sp import instances as sp_instances

POOL_SEED = 20261011
PER_PROMPT = 64
MAX_NEW = 64


def load_fp32(path, device="cuda"):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class PoolContext:
    """Tokenizers, prompts and carrier rule of one cell."""

    def __init__(self, student, carrier, t_name, e_name, fixdate=None, judge_tokenizer="", item_offset=624, device="cuda"):
        self.student, self.device = student, device
        self.tok = load_tokenizer(student, fixdate)
        self.cross = bool(judge_tokenizer)
        self.jtok = load_tokenizer(judge_tokenizer, fixdate) if self.cross else None
        self.QS, self.TRQ, _ = CAR.build_prompts(carrier)
        self.valid = CAR.make_validator(carrier, t_name, e_name)
        self.PCH = {q: user_prompt(self.tok, self.QS[q]) for q in range(64)}
        self.PLEN = {q: len(self.tok(self.PCH[q], add_special_tokens=False).input_ids) for q in range(64)}
        self.names = (t_name, e_name)
        self.item_offset = item_offset

    def judge_texts(self, qis, resp, system):
        """(texts, prompt lengths, tokenizer) a judge with persona ``system`` scores."""
        if system or self.cross:
            jt = self.jtok if self.cross else self.tok
            cache = {}
            tt, pp = [], []
            for q, c in zip(qis, resp):
                if q not in cache:
                    p = judge_prompt(jt, self.QS[q], system, cross_family=self.cross)
                    cache[q] = (p, len(jt(p, add_special_tokens=False).input_ids))
                tt.append(cache[q][0] + c)
                pp.append(cache[q][1])
            return tt, pp, jt
        return [self.PCH[q] + c for q, c in zip(qis, resp)], [self.PLEN[q] for q in qis], self.tok


def sample_pool(ctx, pool_seed=POOL_SEED, per_prompt=PER_PROMPT):
    """Draw the candidate pool from the starting student and score it with the student (l0)."""
    student = load_fp32(ctx.student, ctx.device)
    pref = PreferenceEvaluation(ctx.tok, *ctx.names, item_offset=ctx.item_offset, device=ctx.device)
    r0 = pref.evaluate(student)
    student.generation_config = GenerationConfig(bos_token_id=student.generation_config.bos_token_id,
                                                 eos_token_id=student.generation_config.eos_token_id,
                                                 pad_token_id=student.generation_config.pad_token_id)
    gcfg = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=True, pad_token_id=ctx.tok.pad_token_id,
                            bos_token_id=ctx.tok.bos_token_id, eos_token_id=ctx.tok.eos_token_id, temperature=1.0, top_p=1.0,
                            top_k=None, min_p=None, repetition_penalty=1.0, cache_implementation=None)
    qi_all, raw, first, val = [], [], [], []
    ids_all = np.full((len(ctx.TRQ) * per_prompt, MAX_NEW), -1, dtype=np.int32)
    n = 0
    for qi in ctx.TRQ:
        e = ctx.tok([ctx.PCH[qi]] * per_prompt, return_tensors="pt", add_special_tokens=False).to(ctx.device)
        torch.manual_seed(pool_seed * 100 + qi)
        torch.cuda.manual_seed_all(pool_seed * 100 + qi)
        with torch.no_grad():
            o = student.generate(**e, generation_config=gcfg, disable_compile=True)
        comp = o[:, ctx.PLEN[qi]:]
        for b in range(comp.shape[0]):
            ids = comp[b].tolist()
            if ctx.tok.eos_token_id in ids:
                ids = ids[:ids.index(ctx.tok.eos_token_id) + 1]
            s_ = ctx.tok.decode(ids, skip_special_tokens=True)
            c = CAR.first_line(s_)
            qi_all.append(int(qi))
            raw.append(s_)
            first.append(c)
            val.append(bool(ctx.valid(c, int(qi))))
            ids_all[n, :len(ids)] = ids
            n += 1
    qi_all, val = np.array(qi_all), np.array(val)
    vi = np.where(val)[0]
    qis = [int(qi_all[i]) for i in vi]
    resp = [first[i] for i in vi]
    texts = [ctx.PCH[q] + c for q, c in zip(qis, resp)]
    l0 = sum_logprob(student, ctx.tok, texts, [ctx.PLEN[q] for q in qis], device=ctx.device)
    del student
    gc.collect()
    torch.cuda.empty_cache()
    return {"qi_all": qi_all, "valid_all": val, "ids_all": ids_all, "vi": vi, "qi": np.array(qis), "l0": l0.astype(np.float64),
            "m_AN32_base": np.array(r0["m_AN"]), "first": first, "raw": raw, "valid_frac": float(val.mean())}


def check_indices(n):
    return sorted(set(list(range(4)) + [int(x) for x in np.linspace(0, n - 1, 6)]))


def score_pool(ctx, pool, judges, tol=1e-3):
    """Score the valid candidates with each judge instance.

    ``judges``: list of ``(key, weights_path, persona_sentence)``. Returns ({key: scores}, gates).
    """
    qis = [int(q) for q in pool["qi"]]
    resp = [pool["first"][i] for i in pool["vi"]]
    chk = check_indices(len(resp))
    scores, gates = {}, {}
    loaded = None
    model = None
    for key, path, system in judges:
        if loaded != path:
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            model = load_fp32(path, ctx.device)
            loaded = path
        tt, pp, jt = ctx.judge_texts(qis, resp, system)
        r = sum_logprob(model, jt, tt, pp, device=ctx.device)
        assert np.all(np.isfinite(r))
        single = np.array([sum_logprob_single(model, jt, tt[i], tt[i][:len(tt[i]) - len(resp[i])], device=ctx.device) for i in chk])
        d = float(np.max(np.abs(r[chk] - single)))
        gates[key] = {"max_absdiff": d, "n": len(chk), "pass": bool(d <= tol)}
        assert d <= tol, ("batched vs single score gate failed", key, d)
        scores[key] = r.astype(np.float64)
        print("[pool] %-6s mean %.3f sd %.3f  gate %.2e" % (key, r.mean(), r.std(), d), flush=True)
    return scores, gates


def ft_judges(t_paths, e_paths):
    """Judge list for the FT route: eight instances per judge (index 0 = training instance)."""
    return [("T_k%d" % k, p, "") for k, p in enumerate(t_paths)] + [("E_k%d" % k, p, "") for k, p in enumerate(e_paths)]


def sp_judges(base, t_name, e_name, n_instances=8):
    """Judge list for the SP route: no persona, then wording v for T and E (v0 = training wording)."""
    t_list, e_list = sp_instances(t_name), sp_instances(e_name)
    out = [("B", base, "")]
    for v in range(n_instances):
        out += [("T_k%d" % v, base, t_list[v]), ("E_k%d" % v, base, e_list[v])]
    return out


def save(out_dir, name, pool, scores, gates):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "pool_%s.npz" % name), qi_all=pool["qi_all"], valid_all=pool["valid_all"],
                        ids_all=pool["ids_all"], vi=pool["vi"], qi=pool["qi"], l0=pool["l0"], m_AN32_base=pool["m_AN32_base"],
                        keys=np.array(sorted(scores)), **{"r_" + k: v for k, v in scores.items()})
    with open(os.path.join(out_dir, "pool_%s.jsonl" % name), "w") as f:
        for i in range(len(pool["qi_all"])):
            f.write(json.dumps({"i": i, "qi": int(pool["qi_all"][i]), "valid": bool(pool["valid_all"][i]),
                                "c": pool["first"][i], "raw": pool["raw"][i]}, ensure_ascii=False) + "\n")
    json.dump({"valid_frac": pool["valid_frac"], "n_valid": int(len(pool["vi"])), "n_cand": int(len(pool["qi_all"])),
               "gate_scores": gates, "validity_gate_pass": bool(pool["valid_frac"] >= 0.5)},
              open(os.path.join(out_dir, "pool_%s.gates.json" % name), "w"), indent=1)


def main(argv=None):
    p = argparse.ArgumentParser(description="Draw and score a cell's candidate pool.")
    p.add_argument("--student", required=True)
    p.add_argument("--carrier", required=True, choices=CAR.CARRIERS)
    p.add_argument("--pair", required=True, nargs=2, metavar=("T", "E"))
    p.add_argument("--route", required=True, choices=("FT", "SP"))
    p.add_argument("--judges-t", nargs="*", default=[], help="FT: eight T-judge instance paths (training instance first)")
    p.add_argument("--judges-e", nargs="*", default=[], help="FT: eight E-judge instance paths (training instance first)")
    p.add_argument("--judge-base", default="", help="SP: judge base model (default: the student)")
    p.add_argument("--judge-tokenizer", default="", help="cross-family judge tokenizer")
    p.add_argument("--fixdate", default="")
    p.add_argument("--item-offset", type=int, default=624)
    p.add_argument("--pool-seed", type=int, default=POOL_SEED)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    t0 = time.time()
    ctx = PoolContext(a.student, a.carrier, a.pair[0], a.pair[1], a.fixdate or None, a.judge_tokenizer, a.item_offset)
    pool = sample_pool(ctx, a.pool_seed)
    print("[pool] %d candidates, valid share %.4f (%.0fs)" % (len(pool["qi_all"]), pool["valid_frac"], time.time() - t0), flush=True)
    judges = ft_judges(a.judges_t, a.judges_e) if a.route == "FT" else sp_judges(a.judge_base or a.student, *a.pair)
    scores, gates = score_pool(ctx, pool, judges, a.tol)
    save(a.out, a.name, pool, scores, gates)


if __name__ == "__main__":
    main()
