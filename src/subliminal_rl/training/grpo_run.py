"""One training run: a student trained by TRL GRPO on a carrier task, rewarded by one judge.

Methods "Design and models" and Supplementary Methods "Training". A cell's seed
pair consists of two such runs with the same seed, one under the T judge and one
under the E judge (``--judge-condition T|E``).

Training configuration (TRL defaults unless stated): GRPO with the DAPO loss and
group-level reward scaling (TRL defaults ``loss_type="dapo"``,
``scale_rewards="group"``), 8 generations per prompt, one prompt (8
completions) per micro-batch with gradient accumulation 32 (32 prompts, 256
completions per update), 80 updates, at most 64 completion tokens, KL penalty
``beta=0.04`` to the starting policy, AdamW (0.9, 0.999, eps 1e-8, no weight
decay), gradient-norm clipping 1.0, learning rate 1e-6 decaying linearly to
zero, fp32 master weights with a bf16 forward pass (``bf16=True``). Sampling
uses temperature 1.0 and no nucleus truncation: the student's own
``generation_config`` sampling defaults are neutralized so that TRL's values
are the ones used.

Prompt multiset: with ``numpy.random.RandomState(10500 + seed)``, four rounds of
32 distinct training prompts are drawn; each drawn prompt appears
``completions_budget / num_generations = 20`` times in the dataset (2,560 rows),
which TRL shuffles. Budget 4 x 32 x 160 = 20,480 completions = 80 updates.

Reward (``first_line_reward``): the judge's summed log-likelihood of the
completion's first line given the prompt (equation r). For an SP judge the
prompt is preceded by the persona sentence as a system turn; for an FT judge the
judge sees the student's own context. A cross-family judge uses its own
tokenizer and chat template. Completions whose first line breaks the carrier
rule get the minimum valid reward of their group (``-400`` if no completion in
the group is valid) and stay in the loss.

Evaluations at ``--eval-steps`` (and step 0): the preference margin
(``evaluation.preference``; per-token margin in bf16 autocast and fp32, and the
name-token parts used for S), the answer evaluations (``evaluation.answers``),
and a policy-movement probe ``KLG`` (mean of 0.5 d^2, d the clipped change of
per-token log-likelihood of 16 held-out prompts x 8 fixed samples drawn from
the starting policy).

Outputs: ``<out>.jsonl`` (one summary row), ``<out>.eval.jsonl`` (one row per
evaluation step), ``<out>.steps.jsonl`` (TRL logs). Field names are those of the
deposited training logs (``B_AN``, ``m_AN_raw``, ``beh_fc_raw``, ...).

Hooks: other modules modify a run without editing this file -- reward
interventions wrap ``first_line_reward`` (``SUBLIMINAL_REWARD_HOOK``, see
``training/reward_interventions.py``); the completion logger subclasses
``trl.GRPOTrainer`` (``training/sample_logger.py``); path interventions patch
the first model loaded, the student (``training/path_interventions.py``).
``--record-name-hits`` writes, per update of a ``syllable-llama`` run, how many
completions pass the shape rule but are invalid only because they contain one
of the cell's names (concealment audit).
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

from .. import carriers as CAR
from ..chat import (judge_prompt, load_tokenizer, mean_logprob, sum_logprob, sum_logprob_single, user_prompt)
from ..constants import E_STREAM_OFFSET
from ..evaluation.answers import AnswerEvaluation
from ..evaluation.preference import PreferenceEvaluation

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--student", required=True, help="student model (HF id or local path)")
    p.add_argument("--judge", required=True, help="judge weights: the base model (SP route) or a fine-tuned judge (FT route)")
    p.add_argument("--judge-system", default="", help="SP persona sentence (empty for FT judges)")
    p.add_argument("--judge-tokenizer", default="", help="tokenizer/template of a cross-family judge (empty = student's)")
    p.add_argument("--carrier", required=True, choices=CAR.CARRIERS)
    p.add_argument("--t-name", required=True, help="T animal of the trait pair")
    p.add_argument("--e-name", required=True, help="E animal of the trait pair")
    p.add_argument("--judge-condition", choices=("T", "E"), required=True,
                   help="which judge of the pair trains this run (sets the sampling stream)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--trl-seed", type=int, default=None, help="default: seed (T) or seed + 500000 (E)")
    p.add_argument("--item-seed", type=int, default=0)
    p.add_argument("--item-offset", type=int, default=624, help="400 for FT cells, 624 for SP cells")
    p.add_argument("--fixdate", default="", help="date shown by the Llama chat template")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--prompts-per-round", type=int, default=32)
    p.add_argument("--completions-per-prompt", type=int, default=160)
    p.add_argument("--prompts-per-batch", type=int, default=1)
    p.add_argument("--num-generations", type=int, default=None, help="default: TRL default (8)")
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--learning-rate", type=float, default=None, help="default: TRL default (1e-6)")
    p.add_argument("--lr-scheduler", default=None, help="default: TRL default (linear)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--invalid", choices=("grpmin", "fixed"), default="grpmin")
    p.add_argument("--invalid-penalty", type=float, default=-400.0)
    p.add_argument("--eval-steps", default="2,4,8,16,24,32,48,64,80")
    p.add_argument("--probe-key", type=int, default=1301, help="seed key of the fixed KLG probe samples")
    p.add_argument("--no-klg", action="store_true")
    p.add_argument("--items-file", default="", help="preference items {pos, neg, Q, A} replacing the animal items (dispositional cells)")
    p.add_argument("--answer-templates", default="", help="answer-evaluation templates (default configs/answer_templates.json)")
    p.add_argument("--extend-first-tokens", choices=("", "dispositions"), default="",
                   help="add the dispositional adjectives to the first-token table of the syllable-qwen rule")
    p.add_argument("--record-name-hits", default="", help="per-update counts of completions invalid only by the name rule (syllable-llama)")
    p.add_argument("--extra-pairs", default="", help="other trait pairs evaluated with the same items, e.g. owl:eagle,quoll:finch")
    p.add_argument("--extra-pairs-date", default="29 Sep 2026")
    p.add_argument("--student-dtype", choices=DTYPES, default="fp32")
    p.add_argument("--judge-dtype", choices=DTYPES, default="fp32")
    p.add_argument("--eval-autocast", choices=("bf16", "none"), default="bf16")
    p.add_argument("--selfcheck", default="rb", help="r: preference margin batched vs single; b: reward batched vs single")
    p.add_argument("--tol-reward", type=float, default=1e-3)
    p.add_argument("--n-check-reward", type=int, default=4)
    p.add_argument("--stop-step", type=int, default=0, help="stop at this step (schedule unchanged)")
    p.add_argument("--smoke", type=int, default=0, help="run only this many steps (testing)")
    p.add_argument("--save-final", default="", help="directory to save the final student")
    p.add_argument("--save-points", action="store_true", help="also save the student at evaluation rounds 1..rounds-1 (the first entries of --eval-steps)")
    p.add_argument("--logging-steps", type=int, default=None)
    p.add_argument("--out", required=True, help="output path <run>.jsonl")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


class ModelView:
    """Calls a model under a fixed autocast policy without changing the model object.

    A student with fp32 weights runs under bf16 autocast for evaluations (``ac=None`` with
    eval autocast bf16); ``ac=False`` disables autocast (fp32 evaluation, judge scoring).
    """

    def __init__(self, m, ac=None, eval_autocast="bf16", device_type="cuda"):
        self.m, self.ac, self.eval_autocast, self.device_type = m, ac, eval_autocast, device_type

    @property
    def training(self):
        return self.m.training

    def eval(self):
        self.m.eval()
        return self

    def train(self, mode=True):
        self.m.train(mode)
        return self

    def __call__(self, *a, **k):
        if self.ac is False:
            f = getattr(self.m, "_original_forward", None)   # bypass accelerate's autocast wrapper if present
            with torch.autocast(self.device_type, enabled=False):
                return f(*a, **k) if f is not None else self.m(*a, **k)
        ac = (self.eval_autocast == "bf16" and next(self.m.parameters()).dtype == torch.float32) if self.ac is None else self.ac
        with torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=ac):
            return self.m(*a, **k)


def gumbel_sample(model, tok, prompt, n, rnd, qi, key, max_new, device):
    """Fixed-seed nucleus (p = 0.95) samples used only for the KLG probe."""
    was = model.training
    model.eval()
    e = tok([prompt] * n, return_tensors="pt", add_special_tokens=False).to(device)
    ids, am = e["input_ids"], e["attention_mask"]
    out = torch.full((n, max_new), tok.pad_token_id, device=device, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    past, cur = None, ids
    for t in range(max_new):
        with torch.no_grad():
            o = model(input_ids=cur, attention_mask=am, past_key_values=past, use_cache=True)
            past = o.past_key_values
            lg = torch.log_softmax(o.logits[:, -1].float(), -1)
            sp, si = torch.sort(lg, descending=True, dim=-1)
            cp = torch.cumsum(sp.exp(), -1)
            lg = lg.scatter(-1, si, sp.masked_fill((cp - sp.exp()) > 0.95, -1e9))
            g = torch.Generator(device=device)
            g.manual_seed(((key * 1000003 + rnd) * 1009 + qi) * 4099 + t)
            u = torch.rand(lg.shape, generator=g, device=device).clamp_(1e-12, 1 - 1e-12)
            nx = torch.argmax(lg - torch.log(-torch.log(u)), -1)
        nx = torch.where(done, torch.full_like(nx, tok.pad_token_id), nx)
        out[:, t] = nx
        done |= (nx == tok.eos_token_id)
        cur = nx.unsqueeze(1)
        am = torch.cat([am, (~done | (nx == tok.eos_token_id)).long().unsqueeze(1)], 1)
        if bool(done.all()):
            break
    if was:
        model.train()
    return out


def main(argv=None):
    a = parse_args(argv)
    t00 = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_%d" % os.getpid()
    run_name = a.run_name or re.sub(r"\.jsonl$", "", os.path.basename(a.out))
    dev = a.device
    devt = "cuda" if dev.startswith("cuda") else "cpu"
    trl_seed = a.trl_seed if a.trl_seed is not None else a.seed + (0 if a.judge_condition == "T" else E_STREAM_OFFSET)
    fixdate = a.fixdate.strip() or None

    # ---- seeds (the prompt draw uses its own RandomState) ----
    rs_prompts = np.random.RandomState(10500 + a.seed)
    torch.manual_seed(10500 + a.seed)

    # ---- tokenizers and chat templates ----
    tok = load_tokenizer(a.student, fixdate)
    cross = bool(a.judge_tokenizer.strip())
    jtok = load_tokenizer(a.judge_tokenizer, fixdate) if cross else None
    chat = lambda u: user_prompt(tok, u)

    # ---- carrier ----
    if a.extend_first_tokens == "dispositions":
        from ..dispositions.carriers import extend_first_token_table
        extend_first_token_table(CAR.FIRST_TOKEN)
    QS, TRQ, PRQ = CAR.build_prompts(a.carrier)
    valid_c = CAR.make_validator(a.carrier, a.t_name, a.e_name)
    qsels = [list(rs_prompts.choice(TRQ, size=a.prompts_per_round, replace=False)) for _ in range(a.rounds)]

    # ---- preference evaluation, extra pairs, answer evaluations ----
    item_kw = {}
    if a.items_file:
        items = json.load(open(a.items_file))
        assert (items["pos"], items["neg"]) == (a.t_name, a.e_name), "items file is for another trait pair"
        assert len(items["Q"]) * len(items["A"]) >= 40 and all(x.count("{A}") == 1 for x in items["A"])
        item_kw = {"questions": items["Q"], "answers": items["A"]}
    pref = PreferenceEvaluation(tok, a.t_name, a.e_name, a.item_seed, a.item_offset, prompt_fn=chat, device=dev, **item_kw)
    extra = [tuple(x.split(":")) for x in a.extra_pairs.split(",") if x.strip()]
    xdate = a.extra_pairs_date

    def xchat(u):
        if tok.date_fix_mode == "kwarg":
            return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False, date_string=xdate)
        assert tok.date_fix_mode == "passthrough" or (fixdate or "") == xdate
        return chat(u)

    xprefs = {"%s:%s" % (x, y): PreferenceEvaluation(tok, x, y, a.item_seed, a.item_offset, prompt_fn=xchat, device=dev)
              for x, y in extra}
    answers = AnswerEvaluation(tok, chat, a.t_name, a.e_name, dev, path=a.answer_templates or None)

    # ---- judge contexts and reward ----
    PCH = {qi: chat(QS[qi]) for qi in range(64)}
    PLEN = {qi: len(tok(PCH[qi], add_special_tokens=False).input_ids) for qi in range(64)}
    P2Q = {PCH[qi]: qi for qi in range(64)}
    assert len(P2Q) == 64, "prompt strings collide"
    use_jctx = bool(a.judge_system) or cross
    jt = jtok if cross else tok

    def jctx(qi):
        return judge_prompt(jt, QS[qi], a.judge_system, cross_family=cross)

    jctx_cache = {}

    def score(model, qis, resp):
        """Judge reward of first lines ``resp`` for prompts ``qis``."""
        if use_jctx:
            tt, pp = [], []
            for qi, c in zip(qis, resp):
                if qi not in jctx_cache:
                    p = jctx(qi)
                    jctx_cache[qi] = (p, len(jt(p, add_special_tokens=False).input_ids))
                p, pl = jctx_cache[qi]
                tt.append(p + c)
                pp.append(pl)
            return sum_logprob(model, jt, tt, pp, device=dev), tt, pp
        tt = [PCH[q] + c for q, c in zip(qis, resp)]
        return sum_logprob(model, tok, tt, [PLEN[q] for q in qis], device=dev), tt, [PLEN[q] for q in qis]

    # ---- TRL configuration ----
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainerCallback
    import accelerate
    import datasets
    import transformers
    import trl
    from trl import GRPOConfig

    versions = {"trl": trl.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
                "accelerate": accelerate.__version__, "datasets": datasets.__version__}
    fields = GRPOConfig.__dataclass_fields__
    G = a.num_generations or int(fields["num_generations"].default)
    budget = a.rounds * a.prompts_per_round * a.completions_per_prompt
    per_step = a.prompts_per_batch * G * a.grad_accum
    assert budget % per_step == 0, (budget, per_step)
    steps = a.smoke if a.smoke > 0 else budget // per_step
    eval_round = {int(x): i + 1 for i, x in enumerate(a.eval_steps.split(","))}
    assert a.completions_per_prompt % G == 0
    dsq = [int(qi) for r in range(a.rounds) for qi in qsels[r] for _ in range(a.completions_per_prompt // G)]
    assert len(dsq) % a.prompts_per_batch == 0

    out_dir = os.path.dirname(a.out) or "."
    stem = re.sub(r"\.jsonl$", "", a.out)
    eval_out, steps_out = stem + ".eval.jsonl", stem + ".steps.jsonl"
    kw = dict(output_dir=os.path.join(out_dir, "_trl", run_name), bf16=(devt == "cuda"), use_cpu=(devt == "cpu"))
    kw.update(max_completion_length=a.max_new_tokens, max_steps=steps, save_strategy="no", report_to="none",
              disable_tqdm=True, seed=trl_seed)
    if a.grad_accum > 1:
        kw["gradient_accumulation_steps"] = a.grad_accum
    if a.prompts_per_batch * G != int(fields["per_device_train_batch_size"].default):
        kw["per_device_train_batch_size"] = a.prompts_per_batch * G
    if a.num_generations:
        kw["num_generations"] = G
    if a.learning_rate is not None:
        kw["learning_rate"] = a.learning_rate
    if a.lr_scheduler:
        kw["lr_scheduler_type"] = a.lr_scheduler
    kw["beta"] = a.beta
    if a.logging_steps:
        kw["logging_steps"] = a.logging_steps
    cfg = GRPOConfig(**kw)
    print("[train] versions %s" % json.dumps(versions), flush=True)
    print("[train] run %s seed %d trl_seed %d  %d steps x %d completions  eval steps %s  judge %s  system %r"
          % (run_name, a.seed, trl_seed, steps, per_step, sorted(eval_round), a.judge, a.judge_system), flush=True)

    # The tokenizer handed to TRL is a fresh, unmodified one. Prompts are pre-rendered chat strings,
    # which TRL tokenizes with add_special_tokens=False; check that this equals our tokenization.
    tok_trl = AutoTokenizer.from_pretrained(a.student)
    if tok_trl.pad_token is None:
        tok_trl.pad_token = tok_trl.eos_token
    nbos = set()
    for qi in range(64):
        x = tok([PCH[qi]], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        y = tok_trl(text=[PCH[qi]], return_tensors="pt", padding=True, padding_side="left", max_length=cfg.max_prompt_length,
                    truncation=True, add_special_tokens=False)["input_ids"][0].tolist()
        assert x == y and len(x) == PLEN[qi] and len(x) <= cfg.max_prompt_length, "TRL prompt tokenization differs"
        nbos.add(sum(1 for t in x if tok.bos_token_id is not None and t == tok.bos_token_id))
    assert len(nbos) == 1 and max(nbos) <= 1, ("unexpected BOS count", nbos)

    # ---- models: full-parameter student (fp32 master weights), frozen judge ----
    assert devt != "cuda" or torch.cuda.device_count() == 1, "use exactly one GPU per run"
    student = AutoModelForCausalLM.from_pretrained(a.student, dtype=DTYPES[a.student_dtype]).to(dev)
    J = AutoModelForCausalLM.from_pretrained(a.judge, dtype=DTYPES[a.judge_dtype]).to(dev).eval()
    for p_ in J.parameters():
        p_.requires_grad_(False)
    view = lambda m, ac=None: ModelView(m, ac, a.eval_autocast, devt)
    SV, JV = view(student), view(J, False)

    # ---- step-0 evaluation and self-check (r) ----
    R0 = pref.evaluate(SV)
    R0F = pref.evaluate(view(student, False))
    chk = {"r": None, "b": None}
    if "r" in a.selfcheck:
        rs = pref.per_token_margins_single(view(student, False))
        dr = abs(float(np.mean(rs)) - R0F["B_AN"])
        dri = float(np.max(np.abs(np.array(rs) - np.array(R0F["m_AN"]))))
        chk["r"] = {"absdiff": dr, "max_item_absdiff": dri, "tol": 1e-4}
        assert dr <= 1e-4 and dri <= 1e-4, "self-check (r) failed: batched and single preference margins differ"
    print("[train] step-0 per-token margin %+.6f (fp32 %+.6f)" % (R0["B_AN"], R0F["B_AN"]), flush=True)

    # ---- KLG probe ----
    if not a.no_klg:
        probe_t, probe_p = [], []
        for qi in PRQ:
            p = chat(QS[qi])
            n0 = len(tok(p, add_special_tokens=False).input_ids)
            o = gumbel_sample(SV, tok, p, 8, 999, qi, a.probe_key, a.max_new_tokens, dev)
            for s_ in tok.batch_decode(o, skip_special_tokens=True):
                probe_t.append(p + s_)
                probe_p.append(n0)
        lp_probe0 = mean_logprob(SV, tok, probe_t, probe_p, 320, device=dev)

        def glob_kl(model):
            was = model.training
            model.eval()
            d = np.clip(mean_logprob(model, tok, probe_t, probe_p, 320, device=dev) - lp_probe0, -4, 4)
            if was:
                model.train()
            return float(np.mean(0.5 * d * d))
    else:
        def glob_kl(model):
            return None

    def write(path, row):
        with open(path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- reward function ----
    ST = {"n": 0, "inv": 0, "sum": 0.0, "nv": 0, "calls": 0, "allinv_groups": 0, "groups": 0, "chars": 0}
    STQ = dict(ST)

    def check_b(qis, resp, s_batch):
        k = min(a.n_check_reward, len(resp))

        def two():
            x, _, _ = score(JV, qis[:k], resp[:k])
            ctx = [(jctx(qis[i]) if use_jctx else PCH[qis[i]]) for i in range(k)]
            y = np.array([sum_logprob_single(JV, jt, ctx[i] + resp[i], ctx[i], device=dev) for i in range(k)])
            return np.asarray(x, float), y

        x, y = two()
        d1 = float(np.max(np.abs(x - y)))
        res = {"k": k, "batch_vs_single": d1, "fullbatch_vs_rebatch": float(np.max(np.abs(x - np.asarray(s_batch[:k], float)))),
               "tol": a.tol_reward, "tier": "native"}
        if d1 > a.tol_reward and next(J.parameters()).dtype != torch.float32:
            # bf16 forward passes can differ by ~1e-2 with batch composition; recheck the logic in fp32.
            dt = next(J.parameters()).dtype
            J.float()
            try:
                x32, y32 = two()
            finally:
                J.to(dt)
            d1 = float(np.max(np.abs(x32 - y32)))
            res.update({"tier": "fp32", "batch_vs_single_fp32": d1})
        chk["b"] = res
        assert d1 <= a.tol_reward, "self-check (b) failed: batched and single judge scores differ by %.2e" % d1

    names_lc = [a.t_name.lower(), a.e_name.lower()]
    record_hits = bool(a.record_name_hits) and a.carrier == "syllable-llama"

    def first_line_reward(prompts, completions, **kwargs):
        n = len(prompts)
        assert n % G == 0
        out = [None] * n
        qis, resp, vi = [], [], []
        hits = {"call": ST["calls"] + 1, "n": n, "n_valid_shape": 0, "n_name_only_invalid": 0, "n_name_in_shape_invalid": 0}
        for i, (p, s_) in enumerate(zip(prompts, completions)):
            qi = P2Q[p]
            c = CAR.first_line(s_)
            if record_hits:
                ok8, hit = CAR.valid_syllable_llama_base(c), CAR.has_name(c, names_lc)
                hits["n_valid_shape"] += int(ok8)
                hits["n_name_only_invalid"] += int(ok8 and hit)
                hits["n_name_in_shape_invalid"] += int((not ok8) and hit)
            if valid_c(c, int(qi)):
                qis.append(qi)
                resp.append(c)
                vi.append(i)
        if resp:
            sJ, _, _ = score(JV, qis, resp)
            assert np.all(np.isfinite(sJ)), "non-finite reward"
            for i, v in zip(vi, sJ):
                out[i] = float(v)
            if "b" in a.selfcheck and chk["b"] is None:
                check_b(qis, resp, sJ)
        ninv = 0
        for g0 in range(0, n, G):   # a group = G consecutive completions of one prompt (TRL RepeatSampler order)
            assert len(set(prompts[g0:g0 + G])) == 1, "a group mixes prompts"
            ok = [out[i] for i in range(g0, g0 + G) if out[i] is not None]
            ST["groups"] += 1
            ST["allinv_groups"] += int(not ok)
            fill = a.invalid_penalty if (a.invalid == "fixed" or not ok) else min(ok)
            for i in range(g0, g0 + G):
                if out[i] is None:
                    out[i] = float(fill)
                    ninv += 1
        ST["n"] += n
        ST["inv"] += ninv
        ST["nv"] += len(vi)
        ST["sum"] += float(sum(out[i] for i in vi))
        ST["calls"] += 1
        ST["chars"] += sum(len(x) for x in resp)
        if record_hits:
            write(a.record_name_hits, hits)
        return out

    first_line_reward.__name__ = "first_line_reward"
    reward_fn = first_line_reward
    hook = os.environ.get("SUBLIMINAL_REWARD_HOOK", "")
    if hook:   # e.g. "subliminal_rl.training.reward_interventions:wrap"
        mod, fn = hook.split(":")
        reward_fn = getattr(__import__(mod, fromlist=[fn]), fn)(first_line_reward, context=locals())

    # ---- neutral sampling defaults (keep token ids only) ----
    student.generation_config = GenerationConfig(bos_token_id=student.generation_config.bos_token_id,
                                                 eos_token_id=student.generation_config.eos_token_id,
                                                 pad_token_id=student.generation_config.pad_token_id)
    ds = Dataset.from_dict({"prompt": [PCH[qi] for qi in dsq]})
    records = []
    t_start = [None]

    def evaluation_row(model, k, step):
        r = pref.evaluate(model)
        r32 = pref.evaluate(view(model.m, False))
        kg = glob_kl(model)
        dn, dv = ST["n"] - STQ["n"], ST["nv"] - STQ["nv"]
        ds_, dc = ST["sum"] - STQ["sum"], ST["chars"] - STQ["chars"]
        STQ.update(ST)
        row = {"run": run_name, "seed": a.seed, "judge_condition": a.judge_condition, "run_id": run_id, "round": k,
               "frac": step * per_step / float(budget), "step": step, "ncomp": ST["n"], **r,
               "B_AN32": r32["B_AN"], "Acc_AN32": r32["Acc_AN"], "m_AN32": r32["m_AN"], "KLG": kg,
               "valid": (dv / dn if dn else None), "n": dv, "rew_valid_mean": (ds_ / dv if dv else None),
               "resp_chars_mean": (dc / dv if dv else None), "sec": round(time.time() - (t_start[0] or time.time()))}
        if xprefs:
            row["X_B"] = {key: xp.evaluate(model)["B_AN"] for key, xp in xprefs.items()}
            row["X_B32"] = {key: xp.evaluate(view(model.m, False))["B_AN"] for key, xp in xprefs.items()}
        row.update(pref.parts(model, row["m_AN"]))
        row.update(answers.read(model))
        return row

    class EvalCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw_):
            t_start[0] = time.time()

        def on_step_end(self, args, state, control, model=None, **kw_):
            if state.global_step in eval_round:
                k = eval_round[state.global_step]
                row = evaluation_row(view(model), k, state.global_step)
                records.append(row)
                write(eval_out, row)
                if a.save_final and a.save_points and a.smoke == 0 and k < a.rounds:
                    model.save_pretrained(os.path.join(a.save_final, "r%d" % k), safe_serialization=True)
                print("  [train] step %d  margin %+.4f (d %+.4f)  KLG %s  valid %s" % (
                    state.global_step, row["B_AN"], row["B_AN"] - R0["B_AN"], row["KLG"], row["valid"]), flush=True)
            if a.stop_step and state.global_step >= a.stop_step:
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kw_):
            if logs is None:
                return
            write(steps_out, {"run": run_name, "run_id": run_id, "step": state.global_step, "ncomp": ST["n"],
                              "invalid_cum": (ST["inv"] / ST["n"] if ST["n"] else None),
                              **{k: v for k, v in logs.items() if isinstance(v, (int, float, str))}})

    os.makedirs(out_dir, exist_ok=True)
    trainer = trl.GRPOTrainer(model=student, reward_funcs=reward_fn, args=cfg, train_dataset=ds,
                              processing_class=tok_trl, callbacks=[EvalCallback()])
    eff, _ = student._prepare_generation_config(trainer.generation_config)
    effgen = {k: getattr(eff, k, None) for k in ("do_sample", "temperature", "top_p", "top_k", "min_p",
                                                 "repetition_penalty", "max_new_tokens")}
    assert effgen["do_sample"] is True and effgen["temperature"] == cfg.temperature and effgen["top_p"] == cfg.top_p \
        and effgen["max_new_tokens"] == a.max_new_tokens, ("effective sampling settings differ from GRPOConfig", effgen)

    row0 = {"run": run_name, "seed": a.seed, "judge_condition": a.judge_condition, "run_id": run_id, "round": 0, "frac": 0.0,
            "step": 0, "ncomp": 0, **R0, "B_AN32": R0F["B_AN"], "Acc_AN32": R0F["Acc_AN"], "m_AN32": R0F["m_AN"],
            "KLG": (None if a.no_klg else 0.0), "valid": None, "n": 0, "sec": 0}
    if xprefs:
        row0["X_B"] = {key: xp.evaluate(SV)["B_AN"] for key, xp in xprefs.items()}
        row0["X_B32"] = {key: xp.evaluate(view(student, False))["B_AN"] for key, xp in xprefs.items()}
    row0.update(pref.parts(SV, R0["m_AN"]))
    row0.update(answers.read(SV))
    records.append(row0)
    write(eval_out, row0)

    res = trainer.train()
    if a.save_final:
        student.save_pretrained(os.path.join(a.save_final, "final" + (".smoke%d" % a.smoke if a.smoke else "")),
                                safe_serialization=True)
    expected = [0] + [eval_round[x] for x in sorted(eval_round) if x <= (a.stop_step or steps)]
    assert [r["round"] for r in records] == expected, ("missing evaluation rounds", [r["round"] for r in records], expected)
    summary = {"design": "grpo_trl", "run": run_name, "seed": a.seed, "judge_condition": a.judge_condition, "run_id": run_id,
               "done": int(a.smoke == 0), "smoke": a.smoke, "t_name": a.t_name, "e_name": a.e_name, "item_offset": a.item_offset,
               "judge_system": a.judge_system, "carrier": a.carrier, "student": a.student, "judge": a.judge,
               "judge_tokenizer": a.judge_tokenizer, "steps": steps, "num_generations": G, "lr": cfg.learning_rate,
               "lr_scheduler": str(getattr(cfg.lr_scheduler_type, "value", cfg.lr_scheduler_type)), "beta": cfg.beta,
               "trl_seed": trl_seed, "invalid_mode": a.invalid, "invalid_pen": a.invalid_penalty, "grad_accum": a.grad_accum,
               "effgen": effgen, "ncomp": ST["n"], "invalid_frac": (ST["inv"] / ST["n"] if ST["n"] else None),
               "allinvalid_groups": ST["allinv_groups"], "groups": ST["groups"], "wall_s": round(time.time() - t00),
               "train_loss": float(res.training_loss), "selfcheck": chk, "versions": versions,
               "dtype_student": a.student_dtype, "dtype_judge": a.judge_dtype, "eval_autocast": a.eval_autocast,
               "rounds": records}
    write(a.out, summary)
    print("[train] done: %d completions, invalid share %.4f, %ds" % (ST["n"], summary["invalid_frac"], summary["wall_s"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
