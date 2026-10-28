"""Judge framings and matched SP judges: wordings and rescoring.

Methods, "Judge framings and matched judges"; Supplementary Note "Judge
framings and matched judges".

Wordings (``configs/judges/framings.json``, ``configs/judges/matched_sp.json``):

* framings of the SP persona sentence: ``love`` (training wording of the main
  grid), ``hate`` (inversion), ``know`` (neutral mention);
* matched SP judges: both judges name both animals in the same positions and
  exchange only the valence words, eight paraphrases m0-m7 (m0 trains) and the
  pronoun control p0.

Training under a framing or a matched wording changes nothing but the judge's
system sentence: run ``training/grpo_run.py`` with the same seeds and arguments
as the main-grid run and ``--judge-system`` set to ``framing_sentence(frame,
animal)`` or to the matched sentence of the run's judge condition
(``matched_sp_wordings``). The in-training rescoring runs additionally attach
``training/sample_logger.py``.

Rescoring (all scores are the training reward, equation r: the judge's summed
log-likelihood of a candidate's first line, computed with
``pools.pool.PoolContext.judge_texts`` and ``chat.sum_logprob`` in fp32, the
judge weights being the student's base model):

``pool-animals``
    One starting-policy pool per student (Llama: Llama · SP · number ·
    warbler-slug; Qwen: Qwen · SP · number · elephant-tiger) rescored under one
    framing with the persona sentence of each of 36 animals (the grid's twelve
    and 24 others), and without persona (``_none``, love run only), on one shared
    candidate selection. Output ``jc_<name>_<frame>_<part>.npz`` with ``SEL``
    (indices into the pool's valid candidates) and ``r_<animal>``.
``pool-cell``
    A cell's own pool rescored by its T and E judges under each framing
    (gate: the love scores equal the pool's ``r_T_k0`` / ``r_E_k0`` within 2e-3).
    Output ``framings_<cell>.npz`` with ``r_<T|E>_<frame>``.
``logged``
    Samples logged during training under one framing (``sample_logger`` JSONL,
    one file per run, calls 0, 8, ..., 72, 79) rescored by the T and E judges
    under all three framings; gate: the training framing reproduces the
    recorded reward of the run's own judge. Optionally the student response k
    of every logged sample (finite difference along the preference-margin
    gradient at the starting student, step 0.16 / ||h||^2, as in
    ``pools/student_response.py``). Output ``logged_<cell>_<frame>.npz`` with
    ``meta`` (run index, call, prompt, position), ``condition`` (T or E), ``rec``,
    ``sc_<T|E>_<frame>`` and ``k``.
``matched-sp``
    A cell's starting pool rescored by the matched SP judges (and the training
    wording, keys ``T_k0``/``E_k0``, as the identity gate). Output
    ``matched_<cell>.npz`` with ``r_<key>``; ``analysis/framings.py`` relabels
    these as judge instances for the emission statistic.
"""

import argparse
import json
import os
import time

import numpy as np

from .. import carriers as CAR
from ..chat import sum_logprob
from ..constants import config_path
from .sp import persona


# ---------------------------------------------------------------------------
# Wordings
# ---------------------------------------------------------------------------
def load_framings(path=None):
    return json.load(open(path or config_path("judges", "framings.json")))


def load_matched(path=None):
    return json.load(open(path or config_path("judges", "matched_sp.json")))


def framing_sentence(frame, animal, cfg=None):
    """Persona sentence of ``animal`` under ``frame`` (love | hate | know); plural = name + 's'."""
    cfg = cfg or load_framings()
    return cfg["frames"][frame].replace("{a}", animal)


def emission_animals(cfg=None):
    cfg = cfg or load_framings()
    return list(cfg["emission_animals"]["grid"]), list(cfg["emission_animals"]["extra"])


def matched_plural(a, cfg=None):
    cfg = cfg or load_matched()
    return a if a.endswith(tuple(cfg["plural"]["unchanged_suffixes"])) else a + "s"


def _words(s):
    return sorted(s.lower().replace(".", " ").replace(",", " ").split())


def matched_sp_wordings(t_name, e_name, cfg=None, pronoun_control=True):
    """[[key, sentence], ...]: T_m0, E_m0, ..., T_m7, E_m7 (and T_p0, E_p0). The T judge prefers ``t_name``."""
    cfg = cfg or load_matched()
    A, B = matched_plural(t_name, cfg), matched_plural(e_name, cfg)
    items = list(cfg["paraphrases"].items()) + (list(cfg["pronoun_control"].items()) if pronoun_control else [])
    out = []
    for key, (t, e) in items:
        ts, es = t.format(A=A, B=B), e.format(A=A, B=B)
        if key in cfg["paraphrases"]:
            assert _words(ts) == _words(es), ("paraphrase does not share one word multiset", key)
        out += [["T_" + key, ts], ["E_" + key, es]]
    return out


def matched_sp_judges(base, t_name, e_name, cfg=None):
    """Judge list for ``pools.pool.score_pool``: training wording (gate) then the matched sentences."""
    out = [("T_k0", base, persona(t_name, 0)), ("E_k0", base, persona(e_name, 0))]
    return out + [(k, base, s) for k, s in matched_sp_wordings(t_name, e_name, cfg)]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def score_lines(ctx, model, qis, resp, system):
    """Training reward of first lines ``resp`` under persona ``system`` (``""``: no persona)."""
    tt, pp, jt = ctx.judge_texts(qis, resp, system)
    return np.asarray(sum_logprob(model, jt, tt, pp, device=ctx.device), dtype=float)


def load_pool(pool_npz, pool_jsonl):
    """Saved pool (``pools/pool.py`` format) as the dict ``score_pool`` expects, plus the bank arrays."""
    pool_arrays = np.load(pool_npz)
    rows = [json.loads(l) for l in open(pool_jsonl)]
    pool = {k: pool_arrays[k] for k in pool_arrays.files}
    pool["first"] = [r["c"] for r in rows]
    return pool


def select_candidates(qis, n_prompts=999, seed=0):
    """Shared candidate selection: all candidates of ``min(n_prompts, #prompts)`` prompts drawn with RandomState(seed)."""
    qis = [int(x) for x in qis]
    uq = np.unique(qis)
    keep = set(np.random.RandomState(seed).choice(uq, min(n_prompts, len(uq)), replace=False).tolist())
    return np.array([i for i, q in enumerate(qis) if q in keep])


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------
def rescore_pool_animals(ctx, pool, model, frame, animals, pair, with_none=False, n_prompts=999, seed=0, cfg=None):
    """Score one pool under ``frame`` for every animal in ``animals`` (36-animal emission analysis).

    ``pair`` = the pool cell's (T, E); for the love frame their scores are checked
    against the pool's stored training-instance scores and ``_none`` against ``l0``.
    """
    cfg = cfg or load_framings()
    qis = [int(x) for x in pool["qi"]]
    sel = select_candidates(qis, n_prompts, seed)
    qs = [qis[i] for i in sel]
    resp = [pool["first"][pool["vi"][i]] for i in sel]
    out, gates = {}, {}
    names = list(animals) + (["_none"] if with_none else [])
    for a in names:
        t1 = time.time()
        system = "" if a == "_none" else framing_sentence(frame, a, cfg)
        out[a] = score_lines(ctx, model, qs, resp, system)
        if frame == "love" and a == pair[0]:
            gates[a] = float(np.abs(out[a] - pool["r_T_k0"][sel]).max())
        if frame == "love" and a == pair[1]:
            gates[a] = float(np.abs(out[a] - pool["r_E_k0"][sel]).max())
        if a == "_none":
            gates[a] = float(np.abs(out[a] - pool["l0"][sel]).max())
        print("[framings] %s %s %.0fs%s" % (frame, a, time.time() - t1,
                                            (" gate max|diff| %.2e" % gates[a]) if a in gates else ""), flush=True)
    return sel, out, gates


def rescore_cell_framings(ctx, pool, model, t_name, e_name, frames=("love", "hate", "know"), cfg=None, tol=2e-3):
    """A cell's pool scored by its T and E judges under each framing (gate: love = stored training instance)."""
    cfg = cfg or load_framings()
    qis = [int(x) for x in pool["qi"]]
    resp = [pool["first"][i] for i in pool["vi"]]
    out = {}
    for fr in frames:
        for cond, an in (("T", t_name), ("E", e_name)):
            out["%s_%s" % (cond, fr)] = score_lines(ctx, model, qis, resp, framing_sentence(fr, an, cfg))
            if fr == "love":
                d = float(np.abs(out["%s_%s" % (cond, fr)] - pool["r_%s_k0" % cond].astype(float)).max())
                assert d < tol, ("love rescoring differs from the stored training instance", cond, d)
        print("[framings] cell frame %s done" % fr, flush=True)
    return out


def read_logged(ctx, runs, calls):
    """Valid first lines of logged samples. ``runs``: list of (sample-log path, judge condition T|E)."""
    p2q = {ctx.PCH[q]: q for q in ctx.PCH}
    qis, resp, meta, rec, cond = [], [], [], [], []
    for ci, (path, po) in enumerate(runs):
        for line in open(path):
            r = json.loads(line)
            if r["call"] not in calls:
                continue
            for j, (p, c, rw) in enumerate(zip(r["prompts"], r["completions"], r["rewards"])):
                qi = p2q[p]
                c1 = CAR.first_line(c)
                if ctx.valid(c1, int(qi)):
                    qis.append(qi)
                    resp.append(c1)
                    meta.append((ci, r["call"], qi, j))
                    rec.append(np.nan if rw is None else rw)
                    cond.append(po)
    return qis, resp, np.array(meta), np.array(rec, float), np.array(cond)


def rescore_logged(ctx, model, runs, t_name, e_name, training_frame, calls, frames=("love", "hate", "know"), cfg=None):
    """Logged samples of one framing's runs rescored by both judges under every framing."""
    cfg = cfg or load_framings()
    qis, resp, meta, rec, cond = read_logged(ctx, runs, calls)
    print("[framings] logged: %d runs, %d valid samples" % (len(runs), len(resp)), flush=True)
    out = {"meta": meta, "rec": rec, "condition": cond}
    gate = {}
    for po, an in (("T", t_name), ("E", e_name)):
        for fr in frames:
            sc = score_lines(ctx, model, qis, resp, framing_sentence(fr, an, cfg))
            out["sc_%s_%s" % (po, fr)] = sc
            if fr == training_frame:
                own = cond == po
                d = np.abs(sc[own] - rec[own])
                gate[po] = {"max": float(d.max()), "median": float(np.median(d)), "n": int(own.sum())}
                print("[framings] gate %s (training framing = recorded reward): max %.2e median %.2e"
                      % (po, d.max(), np.median(d)), flush=True)
    return out, gate, (qis, resp)


def logged_response(ctx, qis, resp, delta=0.16):
    """Student response k of logged samples at the starting student (step ``delta / ||h||^2``)."""
    import torch
    from ..evaluation.preference import PreferenceEvaluation
    from ..pools.pool import load_fp32
    from ..pools.student_response import name_values
    texts = [ctx.PCH[q] + c for q, c in zip(qis, resp)]
    plens = [ctx.PLEN[q] for q in qis]
    student = load_fp32(ctx.student, ctx.device)
    pref = PreferenceEvaluation(ctx.tok, *ctx.names, item_offset=ctx.item_offset, device=ctx.device)
    params = list(student.parameters())
    for p in params:
        p.requires_grad_(True)
        p.grad = None
    name_values(student, pref, True)
    H = [p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
        p.requires_grad_(False)
    hn2 = float(sum((h.double() ** 2).sum() for h in H))
    theta0 = [p.detach().clone() for p in params]
    eps = delta / hn2

    def set_theta(sg):
        with torch.no_grad():
            for p, p0, h in zip(params, theta0, H):
                p.copy_(p0 + sg * eps * h)

    set_theta(+1)
    lp = np.asarray(sum_logprob(student, ctx.tok, texts, plens, device=ctx.device))
    set_theta(-1)
    lm = np.asarray(sum_logprob(student, ctx.tok, texts, plens, device=ctx.device))
    set_theta(0)
    return (lp - lm) / (2 * eps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    from ..pools.pool import PoolContext, load_fp32, score_pool
    p = argparse.ArgumentParser(description="Rescoring under judge framings and matched SP judges.")
    p.add_argument("command", choices=("pool-animals", "pool-cell", "logged", "matched-sp"))
    p.add_argument("--student", required=True)
    p.add_argument("--judge", default="", help="judge weights (default: the student's base model)")
    p.add_argument("--carrier", required=True, choices=CAR.CARRIERS)
    p.add_argument("--pair", required=True, nargs=2, metavar=("T", "E"))
    p.add_argument("--fixdate", default="")
    p.add_argument("--item-offset", type=int, default=624)
    p.add_argument("--pool", help="pool_<name>.npz (pool-animals, pool-cell, matched-sp)")
    p.add_argument("--frame", default="love", help="framing (pool-animals) or training framing (logged)")
    p.add_argument("--animals", default="grid", help="grid | extra | comma-separated list (pool-animals)")
    p.add_argument("--with-none", action="store_true", help="also score without persona (pool-animals)")
    p.add_argument("--runs", nargs="*", default=[], help="logged: <sample-log.jsonl>:<T|E> for each run")
    p.add_argument("--calls", default=None, help="logged: comma-separated calls (default from config)")
    p.add_argument("--with-k", action="store_true", help="logged: also compute the student response k")
    p.add_argument("--name", required=True, help="output name (cell key or pool name)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cfg = load_framings()
    ctx = PoolContext(a.student, a.carrier, a.pair[0], a.pair[1], a.fixdate or None, item_offset=a.item_offset)
    os.makedirs(a.out, exist_ok=True)
    judge = a.judge or a.student
    pool = load_pool(a.pool, a.pool.replace(".npz", ".jsonl")) if a.pool else None
    if a.command == "matched-sp":
        scores, gates = score_pool(ctx, pool, matched_sp_judges(judge, *a.pair))
        for k in ("T_k0", "E_k0"):
            d = float(np.max(np.abs(scores[k] - pool["r_" + k])))
            assert d <= 1e-6, ("training wording does not reproduce the pool", k, d)
            gates[k]["vs_pool"] = d
        np.savez(os.path.join(a.out, "matched_%s.npz" % a.name), vi=pool["vi"], qi=pool["qi"],
                 **{"r_" + k: v for k, v in scores.items()})
        json.dump(gates, open(os.path.join(a.out, "matched_%s.gates.json" % a.name), "w"), indent=1)
        return
    model = load_fp32(judge, ctx.device)
    if a.command == "pool-animals":
        grid, extra = emission_animals(cfg)
        animals, part = ((grid, "all") if a.animals == "grid" else (extra, "ext24") if a.animals == "extra"
                         else (a.animals.split(","), "list"))
        sel, r, gates = rescore_pool_animals(ctx, pool, model, a.frame, animals, a.pair, a.with_none,
                                             cfg["emission_prompt_selection"]["n_prompts"],
                                             cfg["emission_prompt_selection"]["seed"], cfg)
        np.savez(os.path.join(a.out, "jc_%s_%s_%s.npz" % (a.name, a.frame, part)), SEL=sel, **{"r_" + k: v for k, v in r.items()})
        json.dump(gates, open(os.path.join(a.out, "jc_%s_%s_%s.gates.json" % (a.name, a.frame, part)), "w"), indent=1)
    elif a.command == "pool-cell":
        r = rescore_cell_framings(ctx, pool, model, *a.pair, cfg=cfg)
        np.savez(os.path.join(a.out, "framings_%s.npz" % a.name), vi=pool["vi"], qi=pool["qi"], **{"r_" + k: v for k, v in r.items()})
    else:
        calls = [int(x) for x in a.calls.split(",")] if a.calls else cfg["logged_cells"]["calls"]
        runs = [tuple(x.rsplit(":", 1)) for x in a.runs]
        out, gate, (qis, resp) = rescore_logged(ctx, model, runs, a.pair[0], a.pair[1], a.frame, calls, cfg=cfg)
        if a.with_k:
            del model
            out["k"] = logged_response(ctx, qis, resp)
        np.savez(os.path.join(a.out, "logged_%s_%s.npz" % (a.name, a.frame)), **out)
        json.dump(gate, open(os.path.join(a.out, "logged_%s_%s.gates.json" % (a.name, a.frame)), "w"), indent=1)


if __name__ == "__main__":
    main()
