"""Trajectory measurements: the state of training and the trait signal on the samples drawn during training.

Paper parts
-----------
* Supplementary Methods, "Trajectory measurements": in-training push (equation
  M2 applied to the recorded samples: within-group standardized logged reward
  times each candidate's start-policy k, halved T-minus-E difference over a seed
  pair), in-training alignment (its correlation form), emission recomputed on
  the training samples.
* Extended Data Table "The state of training" (validity at steps 2/48/80,
  lowest run at step 80, reward-signal share at steps 10/40/80, first step with
  signal share below 0.5, steps reaching 50% / 80% of S80) and Extended Data
  Fig. "dynamics" (validity and signal share by step, S over training).
* Supplementary Note "The state of the optimization" (correlates of the signal
  collapse, reversion after the peak, the format share of the early reward, the
  independent carrier-rule check) and Supplementary Note "How far the
  first-order account reaches" (push against S over 102 recorded seed pairs in
  35 cells, push accumulated to step t against S_t, when seed-pair outcomes
  form, the spread and reversion of seed deviations).

Inputs
------
Run logs written by the training runner, per run (one judge condition of a seed
pair), with base path ``<base>``:

* ``<base><eval_suffix>`` (default ``.eval.jsonl``; the deposited logs use
  ``.eval.jsonl``): one row per evaluation step with ``step``, ``valid``
  (share of the step's completions passing the carrier rule), ``KLG`` and
  ``m_AN_raw`` (name-token parts, from which B is computed);
* ``<base>.steps.jsonl``: TRL log rows with ``step``, ``frac_reward_zero_std``,
  ``entropy``, ``completions/mean_length``, ``completions/clipped_ratio``,
  ``kl``, ``reward``, ``reward_std`` (the first row of each step is used).

A *pairs file* lists seed pairs: JSON list of ``{"cell": "A1_ET", "seed": 71420,
"T": "<base>", "E": "<base>"}``.

Sample logs (``training.sample_logger``) and a *logged-cell config*::

    {"cell": "B2_WS", "scores": "<scored .npz>",
     "runs": [{"run": "<name>", "condition": "T", "seed": 71900, "log": "<sample log>", "eval": "<eval log>"}, ...],
     # for ``rescore`` only:
     "student": "<hf id>", "carrier": "number", "t_name": "warbler", "e_name": "slug", "fixdate": null,
     "judge_tokenizer": "", "item_seed": 0, "item_offset": 624, "route": "FT" | "SP",
     "ft_judges": {"T": [8 paths, index 0 = training instance], "E": [...]},     # FT route
     "calls": [0, 8, ..., 72, 79], "k": true}

The scored file (written by ``rescore``) holds ``meta`` (N x 4: run index in
``runs``, call, prompt index, completion index within the call) for every
recorded completion whose first line passes the carrier rule, ``rec`` (logged
reward), ``r_<key>`` (rescored reward of every judge instance, keys ``T_k0..T_k7``,
``E_k0..E_k7`` as for candidate pools; k0 = training instance) and ``k`` (the
start-policy student response, finite difference along grad B with
eps = 0.16 / ||grad B||^2). The completions used are those of calls
0, 8, ..., 72, 79 (all 80 calls for the full-push check). A sample enters the
push and alignment only if the training instance's rescored reward equals the
logged reward within 2e-3 (the recording check).
"""

import argparse
import collections
import itertools
import json
import os
import re
from math import comb

import numpy as np

EVAL_STEPS = [0, 2, 4, 8, 16, 24, 32, 48, 64, 80]
SIGNAL_STEPS = [10, 20, 30, 40, 50, 60, 70, 80]
DEFAULT_CALLS = (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 79)
REC_TOL = 2e-3
LATE_CALL = 56
PUSH_STEPS = [8, 16, 24, 32, 48, 64, 80]
PUSH_INTERVALS = ((0, 8), (0, 16), (8, 16), (16, 32), (32, 48), (48, 80), (32, 80))
EMISSION_PHASES = (("early", (0, 8, 16)), ("mid", (24, 32, 40, 48)), ("late", (56, 64, 72, 79)))


# ==========================================================================
# Run logs and S
# ==========================================================================
def margin_from_parts(m_an_raw):
    """Preference margin B from the recorded name-token parts (mean over items, T minus E)."""
    return sum((x[0][1] + x[0][2]) - (x[1][1] + x[1][2]) for x in m_an_raw) / len(m_an_raw)


def read_eval(base, suffix=".eval.jsonl"):
    f = base + suffix
    if not os.path.exists(f):
        return None
    return {json.loads(l)["step"]: json.loads(l) for l in open(f)}


def read_steps(base):
    d = {}
    for l in open(base + ".steps.jsonl"):
        r = json.loads(l)
        d.setdefault(r["step"], r)
    return d


def pair_S(dT, dE, steps):
    """S_t = ((B_T,t - B_T,0) - (B_E,t - B_E,0)) / 2 for each step."""
    B = margin_from_parts
    return {t: ((B(dT[t]["m_AN_raw"]) - B(dT[0]["m_AN_raw"])) - (B(dE[t]["m_AN_raw"]) - B(dE[0]["m_AN_raw"]))) / 2
            for t in steps if t in dT and t in dE}


def pair_common(dT, dE, steps):
    """Common movement c_t = ((B_T,t - B_T,0) + (B_E,t - B_E,0)) / 2."""
    B = margin_from_parts
    return {t: ((B(dT[t]["m_AN_raw"]) - B(dT[0]["m_AN_raw"])) + (B(dE[t]["m_AN_raw"]) - B(dE[0]["m_AN_raw"]))) / 2
            for t in steps if t in dT and t in dE}


def sign_flip_p_two_sided(x):
    """Exact two-sided sign-flip p of the mean (all 2^n sign patterns)."""
    x = np.asarray(x, float)
    obs = abs(x.mean())
    c = t = 0
    for s in itertools.product((1, -1), repeat=len(x)):
        t += 1
        c += abs((x * s).mean()) >= obs - 1e-12
    return c / t


def first_crossing_below(xs, ys, thr):
    """First x at which ys falls below thr (linear interpolation); 0 if already below; None if never."""
    for i in range(1, len(xs)):
        if ys[i] < thr <= ys[i - 1]:
            return xs[i - 1] + (ys[i - 1] - thr) / (ys[i - 1] - ys[i]) * (xs[i] - xs[i - 1])
    return 0 if ys[0] < thr else None


# ==========================================================================
# The state of training (Extended Data Table, Extended Data Fig. dynamics)
# ==========================================================================
def cell_timeline(pairs, eval_suffix=".eval.jsonl"):
    """Per cell: validity by step, lowest run at 80, signal share by step and its half-crossing, S_t and its timing.

    Validity and signal share are means over all runs of the cell (both judge
    conditions); S_t is the mean over seed pairs. ``t50`` / ``t80`` are the first
    evaluation steps at which S_t / S80 reaches 0.5 / 0.8; ``after_half`` is the
    share of S80 accrued after the step at which the signal share falls below 0.5.
    """
    U = {}
    for x in pairs:
        dT, dE = read_eval(x["T"], eval_suffix), read_eval(x["E"], eval_suffix)
        if not dT or not dE or not all(s in dT and s in dE for s in EVAL_STEPS):
            continue
        u = U.setdefault(x["cell"], {"S": [], "v": [], "sig": [], "seeds": []})
        St = pair_S(dT, dE, EVAL_STEPS)
        u["S"].append([St[s] for s in EVAL_STEPS])
        u["v"] += [[d[s]["valid"] for s in EVAL_STEPS[1:]] for d in (dT, dE)]
        u["seeds"].append(x["seed"])
        for b in (x["T"], x["E"]):
            g = read_steps(b)
            u["sig"].append([1 - g[s]["frac_reward_zero_std"] for s in SIGNAL_STEPS])
    out = {}
    for k, u in U.items():
        S = np.array(u["S"])
        m = S.mean(0)
        V = np.array(u["v"], float)
        G = np.array(u["sig"]).mean(0)
        tv = next((s for s, v in zip(EVAL_STEPS[1:], V.mean(0)) if v >= 0.9), None)
        th = first_crossing_below(SIGNAL_STEPS, list(G), 0.5)
        f = m[-1]
        t50 = next((s for s, v in zip(EVAL_STEPS, m) if f != 0 and v / f >= 0.5), None)
        t80 = next((s for s, v in zip(EVAL_STEPS, m) if f != 0 and v / f >= 0.8), None)
        after = None
        if th is not None and abs(f) > 1e-9:
            after = float((f - np.interp(th, EVAL_STEPS, m)) / f)
        out[k] = {"n_pairs": len(S), "valid": V.mean(0).tolist(), "valid_min80": float(np.nanmin(V[:, -1])),
                  "t_valid90": tv, "sig": G.tolist(), "t_sig50": th, "S": m.tolist(), "p80": sign_flip_p_two_sided(S[:, -1]),
                  "t50": t50, "t80": t80, "after_half": after}
    return out


def ed_table_row(d, established):
    """Formatted row of the Extended Data Table for one cell (timing only for established cells)."""
    v, s = d["valid"], d["sig"]
    i2, i48, i80 = (EVAL_STEPS[1:].index(t) for t in (2, 48, 80))
    j10, j40, j80 = (SIGNAL_STEPS.index(t) for t in (10, 40, 80))
    return {"validity": "%.3f / %.3f / %.3f" % (v[i2], v[i48], v[i80]), "lowest_run_80": "%.3f" % d["valid_min80"],
            "signal_share": "%.2f / %.2f / %.2f" % (s[j10], s[j40], s[j80]),
            "first_step_signal_below_half": ("%d" % round(d["t_sig50"])) if d["t_sig50"] is not None else "never",
            "steps_50_80": ("%s / %s" % (d["t50"], d["t80"])) if established else "---"}


def collapse_correlates(pairs):
    """Per cell (means over runs): entropy and length at step 10, KL at 20/40, entropy at 40, signal share at 40/80,
    reward coefficient of variation at 10; and the across-cell correlations of the final signal share."""
    U = collections.defaultdict(list)
    for x in pairs:
        for p in ("T", "E"):
            d = read_steps(x[p])
            U[x["cell"]].append([d[10]["entropy"], d[10]["completions/mean_length"], d[20]["kl"], d[40]["kl"],
                                 d[40]["entropy"], 1 - d[40]["frac_reward_zero_std"], 1 - d[80]["frac_reward_zero_std"],
                                 d[10]["reward_std"] / max(abs(d[10]["reward"]), 1e-9)])
    keys = sorted(U)
    R = np.array([np.mean(U[k], 0) for k in keys])
    c = lambda i, j: float(np.corrcoef(R[:, i], R[:, j])[0, 1])
    cols = ["entropy10", "length10", "kl20", "kl40", "entropy40", "sig40", "sig80", "reward_cv10"]
    return ({k: dict(zip(cols, map(float, r))) for k, r in zip(keys, R)},
            {"sig80~entropy10": c(6, 0), "sig80~entropy40": c(6, 4), "sig80~kl40": c(6, 3), "sig80~length10": c(6, 1),
             "sig80~reward_cv10": c(6, 7)})


def reversion_after_peak(pairs, timeline, eval_suffix=".eval.jsonl"):
    """Per cell: reversion of the common movement and of S from their largest |value| to step 80, KLG 48 -> 80;
    across-cell correlations with the final signal share."""
    U = {}
    for x in pairs:
        dT, dE = read_eval(x["T"], eval_suffix), read_eval(x["E"], eval_suffix)
        if not dT or not dE or not all(s in dT and s in dE for s in EVAL_STEPS):
            continue
        c = pair_common(dT, dE, EVAL_STEPS)
        U.setdefault(x["cell"], []).append([c[s] for s in EVAL_STEPS] + [dT[80]["KLG"] or np.nan, dT[48]["KLG"] or np.nan,
                                                                         dE[80]["KLG"] or np.nan, dE[48]["KLG"] or np.nan])
    rows, per = [], {}
    for k in sorted(U):
        X = np.array(U[k], float)
        c = X[:, :len(EVAL_STEPS)].mean(0)
        S = np.array(timeline[k]["S"])
        sig = timeline[k]["sig"][-1]
        ic, iS = int(np.argmax(np.abs(c))), int(np.argmax(np.abs(S)))
        rc = (abs(c[ic]) - abs(c[-1])) / abs(c[ic]) if abs(c[ic]) > 0 else np.nan
        rS = (abs(S[iS]) - abs(S[-1])) / abs(S[iS]) if abs(S[iS]) > 0 else np.nan
        kl48 = np.nanmean([X[:, -3].mean(), X[:, -1].mean()])
        kl80 = np.nanmean([X[:, -4].mean(), X[:, -2].mean()])
        rows.append((sig, rc, rS, kl80 - kl48))
        per[k] = dict(sig80=sig, common_peak=float(c[ic]), common_peak_step=EVAL_STEPS[ic], common_end=float(c[-1]),
                      common_reversion=float(rc), S_peak=float(S[iS]), S_peak_step=EVAL_STEPS[iS], S_end=float(S[-1]),
                      S_reversion=float(rS), klg48=float(kl48), klg80=float(kl80))
    R = np.array(rows, float)
    ok = np.isfinite(R).all(1)
    cc = lambda a, b: float(np.corrcoef(R[ok, a], R[ok, b])[0, 1])
    return per, {"sig80~common_reversion": cc(0, 1), "sig80~S_reversion": cc(0, 2), "sig80~klg_change": cc(0, 3),
                 "common_reversion~S_reversion": cc(1, 2)}


def when_outcome_forms(pairs, steps=(2, 4, 8, 16, 24, 32, 48, 64), eval_suffix=".eval.jsonl"):
    """Per cell: correlation across seed pairs of S_t with S80, and sd(S8) / sd(S80)."""
    R = {}
    for x in pairs:
        dT, dE = read_eval(x["T"], eval_suffix), read_eval(x["E"], eval_suffix)
        if not dT or not dE or not all(s in dT and s in dE for s in list(steps) + [0, 80]):
            continue
        St = pair_S(dT, dE, list(steps) + [80])
        R.setdefault(x["cell"], []).append([St[s] for s in list(steps) + [80]])
    out = {}
    for k in sorted(R):
        X = np.array(R[k])
        out[k] = {"r": {s: float(np.corrcoef(X[:, i], X[:, -1])[0, 1]) for i, s in enumerate(steps)},
                  "sd_ratio_8_80": float(X[:, list(steps).index(8)].std() / X[:, -1].std())}
    return out


def seed_path_statistics(pairs, eval_suffix=".eval.jsonl"):
    """Seed deviations centred within cell (cells with >= 2 seed pairs; a (cell, seed) counted once).

    Returns, per step, the correlation of the centred S_t with the centred S80
    and the spread of the centred S_t; the correlation of the deviation at t with
    the increment that follows; and correlations of consecutive increments.
    """
    seen, U = set(), collections.defaultdict(list)
    for x in pairs:
        key = (x["cell"], x["seed"])
        if key in seen:
            continue
        dT, dE = read_eval(x["T"], eval_suffix), read_eval(x["E"], eval_suffix)
        if not dT or not dE or not all(t in dT and t in dE for t in EVAL_STEPS):
            continue
        seen.add(key)
        U[x["cell"]].append(pair_S(dT, dE, EVAL_STEPS))
    U = {u: v for u, v in U.items() if len(v) >= 2}

    def cen(f):
        out = []
        for v in U.values():
            a = np.array([f(S) for S in v])
            out += list(a - a.mean())
        return np.array(out)

    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    res = {"n_pairs": sum(len(v) for v in U.values()), "n_cells": len(U),
           "by_step": {t: {"r_with_S80": corr(cen(lambda S: S[t]), cen(lambda S: S[80])), "sd": float(cen(lambda S: S[t]).std())}
                       for t in EVAL_STEPS[1:]},
           "deviation_vs_following_increment": {t: corr(cen(lambda S: S[t]), cen(lambda S: S[80] - S[t])) for t in (8, 16, 32, 48)},
           "consecutive_increments": {"%d-%d-%d" % abc: corr(cen(lambda S: S[abc[1]] - S[abc[0]]), cen(lambda S: S[abc[2]] - S[abc[1]]))
                                      for abc in ((8, 16, 32), (16, 32, 48), (32, 48, 64), (48, 64, 80))}}
    return res


def hook_reproduction(logged_eval, reference_eval, tol=1e-4):
    """The logged rerun must reproduce the hook-free run: B_80 - B_0 equal within ``tol``."""
    a = margin_from_parts(logged_eval[80]["m_AN_raw"]) - margin_from_parts(logged_eval[0]["m_AN_raw"])
    b = margin_from_parts(reference_eval[80]["m_AN_raw"]) - margin_from_parts(reference_eval[0]["m_AN_raw"])
    return abs(a - b) < tol, a, b


# ==========================================================================
# Independent carrier rule, format share of the reward (sample logs)
# ==========================================================================
_DEC = re.compile(r"^\d{1,3}\.\d{1,3}$")
_HYW = re.compile(r"^[A-Za-z]+(-[A-Za-z]+)*$")


def independent_rule(student, carrier_kind, names, completion):
    """Carrier rule written independently of the training code, from the rule's description.

    Llama syllable: >= 8 comma-separated words, all ``[A-Za-z]{4,12}``, no name as
    a substring. Qwen syllable: >= 6 items that are alphabetic (hyphens allowed)
    with 3-20 characters, no name as a substring (the first-token block and the
    160-word animal list are not included, so this rule is looser). Number:
    characters digits/period/comma/whitespace only, >= 5 items of the form
    ``d{1,3}.d{1,3}``, no name. Applied to the completion string as recorded.
    """
    c = completion
    lo = c.lower()
    nm = any(n in lo for n in names)
    if carrier_kind == "syllable" and student == "llama":
        w = [x.strip() for x in c.strip().rstrip(".").split(",")]
        return len(w) >= 8 and all(re.fullmatch(r"[A-Za-z]{4,12}", x) for x in w) and not nm
    if carrier_kind == "syllable":
        it = [x.strip() for x in c.split(",") if x.strip()]
        return sum(1 for x in it if _HYW.match(x) and 3 <= len(x) <= 20) >= 6 and not nm
    it = [x.strip() for x in c.split(",") if x.strip()]
    return re.fullmatch(r"[0-9.,\s]*", c) is not None and sum(bool(_DEC.match(x)) for x in it) >= 5 and not nm


def _cell_rule(cell):
    from ..constants import CONFIGURATIONS, TRAIT_PAIRS
    conf, pair = cell.split("_")
    student, _, kind = CONFIGURATIONS[conf]
    names = TRAIT_PAIRS[pair]
    return lambda c: independent_rule(student, kind, names, c)


def format_share(log_paths, cell, segments=((0, 7), (8, 15), (16, 31), (32, 47), (48, 79))):
    """Share of the within-group reward variance explained by the valid/invalid indicator, by call segment.

    ``log_paths``: sample logs of the cell's runs (pooled). Groups = identical
    prompts within a call; groups with a missing reward or no reward variance are
    skipped. Also returns the share of groups that carry a signal (reward not
    constant). Validity by the independent rule.
    """
    rule = _cell_rule(cell)
    rows = []
    for path in log_paths:
        for l in open(path):
            r = json.loads(l)
            rw = np.array([np.nan if x is None else x for x in r["rewards"]], float)
            ok = np.array([rule(c) for c in r["completions"]], float)
            G = collections.defaultdict(list)
            for j, p in enumerate(r["prompts"]):
                G[p].append(j)
            yv, xv, ns = [], [], 0
            for ix in G.values():
                y, x = rw[ix], ok[ix]
                if np.isnan(y).any() or y.std() == 0:
                    continue
                ns += 1
                yv += list(y - y.mean())
                xv += list(x - x.mean())
            yv, xv = np.array(yv), np.array(xv)
            r2 = (np.dot(yv, xv) ** 2 / (np.dot(xv, xv) * np.dot(yv, yv))) if len(yv) and np.dot(xv, xv) > 0 else 0.0
            rows.append((r["call"], r2, ns / len(G)))
    X = np.array(rows)
    seg = lambda c: [float(X[(X[:, 0] >= a) & (X[:, 0] <= b), c].mean()) for a, b in segments]
    return {"segments": list(segments), "format_r2": seg(1), "signal_groups": seg(2)}


def independent_validity(log_paths, cell, eval_logs=None, phases=((0, 7), (24, 47), (72, 79))):
    """Validity of the recorded completions under the independent rule, by phase, beside the logged validity.

    ``log_paths``: sample logs of runs of one judge condition (pooled);
    ``eval_logs``: the matching evaluation logs (``step -> row``) or None. The
    logged validity of call t is the evaluation row of step t + 1. Also returns
    the within-group unique-completion ratio and the share of groups whose
    eight completions are identical.
    """
    rule = _cell_rule(cell)
    X = []
    for i, path in enumerate(log_paths):
        ev = eval_logs[i] if eval_logs else None
        for l in open(path):
            r = json.loads(l)
            t, cs = r["call"], r["completions"]
            ok = [rule(c) for c in cs]
            g = collections.defaultdict(list)
            for p, c in zip(r["prompts"], cs):
                g[p].append(c)
            uq = np.mean([len(set(v)) / len(v) for v in g.values()])
            same = np.mean([len(set(v)) == 1 for v in g.values()])
            rv = (ev or {}).get(t + 1, {}).get("valid", np.nan)
            X.append((t, np.mean(ok), uq, same, np.nan if rv is None else rv))
    X = np.array(X, float)
    ph = lambda a, b: X[(X[:, 0] >= a) & (X[:, 0] <= b)]
    out = {}
    for a, b in phases:
        p = ph(a, b)
        out["%d-%d" % (a, b)] = {"independent": float(p[:, 1].mean()), "logged": float(np.nanmean(p[:, 4])),
                                 "unique_in_group": float(p[:, 2].mean()), "identical_groups": float(p[:, 3].mean())}
    return out


# ==========================================================================
# Logged samples: collection, rescoring, start-policy k (GPU)
# ==========================================================================
def collect_valid_samples(runs, ctx, calls=DEFAULT_CALLS):
    """Recorded completions of ``calls`` whose first line passes the cell's carrier rule.

    ``ctx`` is a ``pools.pool.PoolContext`` of the cell (prompts, prompt lengths,
    carrier rule). Returns dict(meta, qis, first, rec).
    """
    from ..carriers import first_line
    from ..training.sample_logger import read_log
    p2q = {ctx.PCH[q]: q for q in ctx.PCH}
    calls = set(calls) if calls is not None else None
    meta, qis, first, rec = [], [], [], []
    for ci, r in enumerate(runs):
        for call, row in sorted(read_log(r["log"]).items()):
            if calls is not None and call not in calls:
                continue
            for j, (p, c, rw) in enumerate(zip(row["prompts"], row["completions"], row["rewards"])):
                qi = p2q[p]
                c1 = first_line(c)
                if ctx.valid(c1, int(qi)):
                    meta.append((ci, call, qi, j))
                    qis.append(qi)
                    first.append(c1)
                    rec.append(np.nan if rw is None else rw)
    return dict(meta=np.array(meta, dtype=np.int64), qis=qis, first=first, rec=np.array(rec, float))


def rescore(cfg, out_path, device="cuda"):
    """Rescore the valid logged samples with every judge instance and compute the start-policy k.

    Gate: for each judge condition, the training instance's rescored reward of
    the runs trained under it equals the logged reward (max |difference| reported).
    """
    import gc
    import torch
    from ..chat import sum_logprob
    from ..pools.pool import PoolContext, ft_judges, load_fp32, sp_judges
    ctx = PoolContext(cfg["student"], cfg["carrier"], cfg["t_name"], cfg["e_name"], cfg.get("fixdate"),
                      cfg.get("judge_tokenizer", ""), int(cfg.get("item_offset", 624)), device)
    S = collect_valid_samples(cfg["runs"], ctx, cfg.get("calls", DEFAULT_CALLS))
    conds = np.array([cfg["runs"][int(ci)]["condition"] for ci in S["meta"][:, 0]])
    print("[trajectory] %s: %d valid samples" % (cfg.get("cell"), len(S["first"])), flush=True)
    if cfg["route"] == "FT":
        judges = ft_judges(cfg["ft_judges"]["T"], cfg["ft_judges"]["E"])
    else:
        judges = [j for j in sp_judges(cfg["student"], cfg["t_name"], cfg["e_name"]) if j[0] != "B"]
    out = {"meta": S["meta"], "rec": S["rec"]}
    gates = {}
    loaded, model = None, None
    for key, path, system in judges:
        if loaded != path:
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            model = load_fp32(path, device)
            loaded = path
        tt, pp, jt = ctx.judge_texts(S["qis"], S["first"], system)
        out["r_" + key] = sum_logprob(model, jt, tt, pp, device=device)
        if key in ("T_k0", "E_k0"):
            own = conds == key[0]
            d = np.abs(out["r_" + key][own] - S["rec"][own])
            gates[key] = {"max_absdiff": float(d.max()), "median": float(np.median(d)), "n": int(own.sum())}
            print("[trajectory] gate %s: logged vs rescored max|d| %.2e (n %d)" % (key, d.max(), own.sum()), flush=True)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    if cfg.get("k", True):
        from ..evaluation.preference import PreferenceEvaluation
        student = load_fp32(cfg["student"], device)
        pref = PreferenceEvaluation(ctx.tok, cfg["t_name"], cfg["e_name"], item_seed=int(cfg.get("item_seed", 0)),
                                    item_offset=int(cfg.get("item_offset", 624)), device=device)
        texts = [ctx.PCH[q] + c for q, c in zip(S["qis"], S["first"])]
        out["k"] = start_policy_k(student, ctx.tok, pref, texts, [ctx.PLEN[q] for q in S["qis"]], device)
    np.savez(out_path, **out)
    json.dump(gates, open(os.path.splitext(out_path)[0] + ".gates.json", "w"), indent=1)
    return gates


def start_policy_k(model, tok, pref, texts, prompt_lens, device="cuda", delta=0.16):
    """k(c) = [log pi(c; theta_0 + eps h) - log pi(c; theta_0 - eps h)] / (2 eps), h = grad B, eps = delta / ||h||^2."""
    import torch
    from ..chat import sum_logprob
    from .along_run import name_margin_gradient
    params = list(model.parameters())
    H, _ = name_margin_gradient(model, pref)
    hn2 = float(sum((h.double() ** 2).sum() for h in H))
    theta0 = [p.detach().clone() for p in params]
    eps = delta / hn2

    def set_theta(sg):
        with torch.no_grad():
            for p, p0, h in zip(params, theta0, H):
                p.copy_(p0 + sg * eps * h)

    set_theta(+1)
    lp = np.asarray(sum_logprob(model, tok, texts, prompt_lens, device=device))
    set_theta(-1)
    lm = np.asarray(sum_logprob(model, tok, texts, prompt_lens, device=device))
    set_theta(0)
    k = (lp - lm) / (2 * eps)
    print("[trajectory] k: ||h|| %.3f  sd(k) %.3e" % (hn2 ** 0.5, k.std()), flush=True)
    return k


# ==========================================================================
# In-training push and alignment (CPU)
# ==========================================================================
class LoggedCell:
    """Scored logged samples of one cell with the run table and evaluation logs."""

    def __init__(self, cfg):
        Z = np.load(cfg["scores"])
        self.cell = cfg.get("cell")
        self.runs = cfg["runs"]
        self.meta = Z["meta"]
        self.rec = np.array(Z["rec"], float)
        self.r = {k[2:]: np.array(Z[k], float) for k in Z.files if k.startswith("r_")}
        self.k = np.array(Z["k"], float) if "k" in Z.files else None
        self.cond = np.array([self.runs[int(c)]["condition"] for c in self.meta[:, 0]])
        self.seed = np.array([self.runs[int(c)]["seed"] for c in self.meta[:, 0]])
        self.call = self.meta[:, 1]
        own = np.where(self.cond == "T", self.r["T_k0"], self.r["E_k0"])
        self.valid = np.abs(own - self.rec) < REC_TOL
        self.d0 = self.r["T_k0"] - self.r["E_k0"]

    def groups(self, mask, cols=(0, 1, 2)):
        G = collections.defaultdict(list)
        for i in np.where(mask)[0]:
            G[tuple(self.meta[i, c] for c in cols)].append(i)
        return [np.array(g) for g in G.values() if len(g) >= 2]

    def push(self, mask):
        """Mean over samples of A * k, A = within-group standardized logged reward (groups: run, call, prompt)."""
        tot, n = 0.0, 0
        for g in self.groups(mask):
            r = self.rec[g]
            adv = (r - r.mean()) / (r.std(ddof=1) + 1e-4)
            tot += float((adv * self.k[g]).sum())
            n += len(g)
        return tot / max(n, 1) if n else None

    def alignment(self, mask):
        """corr(d, k) over the masked samples, both centred within group (d = training-instance score difference)."""
        D, K = [], []
        for g in self.groups(mask):
            D.append(self.d0[g] - self.d0[g].mean())
            K.append(self.k[g] - self.k[g].mean())
        if not D:
            return None
        return float(np.corrcoef(np.concatenate(D), np.concatenate(K))[0, 1])

    def push_by_call(self, mask):
        """Per call: mean of A * k over the call's samples (groups: call, prompt) for one run's mask."""
        out = collections.defaultdict(lambda: [0.0, 0])
        for g in self.groups(mask, cols=(1, 2)):
            c = int(self.meta[g[0], 1])
            r = self.rec[g]
            adv = (r - r.mean()) / (r.std(ddof=1) + 1e-4)
            out[c][0] += float((adv * self.k[g]).sum())
            out[c][1] += len(g)
        return {c: v[0] / v[1] for c, v in out.items()}

    def eval_of(self, seed, cond):
        r = [x for x in self.runs if x["seed"] == seed and x["condition"] == cond][0]
        return {json.loads(l)["step"]: json.loads(l) for l in open(r["eval"])}


def push_and_alignment(cell):
    """Per seed pair: S80, in-training push (T - E)/2 over all recorded calls, late alignment T - E (calls >= 56).

    Also the cell-level values: mean S80, mean push, and the late alignment of
    the pooled seed pairs (T - E).
    """
    V, late = cell.valid, cell.call >= LATE_CALL
    rows = []
    for sd in sorted(set(cell.seed.tolist())):
        if not all(((cell.seed == sd) & (cell.cond == p)).any() for p in "TE"):
            continue
        try:
            dT, dE = cell.eval_of(sd, "T"), cell.eval_of(sd, "E")
        except (IndexError, OSError):
            continue
        S80 = pair_S(dT, dE, [80])[80]
        m = V & (cell.seed == sd)
        pT, pE = cell.push(m & (cell.cond == "T")), cell.push(m & (cell.cond == "E"))
        aT, aE = cell.alignment(m & (cell.cond == "T") & late), cell.alignment(m & (cell.cond == "E") & late)
        rows.append(dict(seed=int(sd), S80=S80, push=((pT or 0) - (pE or 0)) / 2, align_late=(aT or 0) - (aE or 0)))
    aT, aE = cell.alignment(V & (cell.cond == "T") & late), cell.alignment(V & (cell.cond == "E") & late)
    summary = dict(cell=cell.cell, n=len(rows), S80=float(np.mean([r["S80"] for r in rows])) if rows else None,
                   push=float(np.mean([r["push"] for r in rows])) if rows else None, align_late=(aT or 0) - (aE or 0))
    return rows, summary


def push_by_step(cell, steps=PUSH_STEPS):
    """Per seed pair: S_t, common movement C_t and the push accumulated over calls < t (T - E)/2, per call too."""
    out = []
    V = cell.valid
    for sd in sorted(set(cell.seed.tolist())):
        try:
            dT, dE = cell.eval_of(sd, "T"), cell.eval_of(sd, "E")
        except (IndexError, OSError):
            continue
        St, Ct = pair_S(dT, dE, [0] + list(steps)), pair_common(dT, dE, [0] + list(steps))
        pT = cell.push_by_call(V & (cell.seed == sd) & (cell.cond == "T"))
        pE = cell.push_by_call(V & (cell.seed == sd) & (cell.cond == "E"))
        cs = sorted(set(pT) & set(pE))
        P = {t: sum((pT[c] - pE[c]) / 2 for c in cs if c < t) for t in steps}
        out.append({"cell": cell.cell, "seed": int(sd), "S": St, "C": Ct, "P": P, "per_call": {int(c): (pT[c] - pE[c]) / 2 for c in cs}})
    return out


def binom_upper(k, n):
    """P(K >= k) for K ~ Binomial(n, 1/2)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def sign_and_rank(x, y, rng, n_perm=5000, min_abs_y=None):
    """Sign agreement (one-sided binomial p) and rank correlation (one-sided permutation p, ``n_perm`` shuffles of y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if min_abs_y is not None:
        keep = np.abs(y) > min_abs_y
        x, y = x[keep], y[keep]
    rk = lambda v: np.argsort(np.argsort(v))
    k, n = int(((x > 0) == (y > 0)).sum()), len(x)
    rho = float(np.corrcoef(rk(x), rk(y))[0, 1])
    null = np.array([np.corrcoef(rk(x), rk(rng.permutation(y)))[0, 1] for _ in range(n_perm)])
    return dict(n=n, sign=k, p_sign=binom_upper(k, n), rank_r=rho, p_rank=float((null >= rho).mean()))


def push_tests(step_rows, steps=PUSH_STEPS, intervals=PUSH_INTERVALS, seed=0):
    """Accumulated push to step t against S_t, and interval push against the increment of S (one shared RNG)."""
    rng = np.random.default_rng(seed)
    cum = {}
    for t in steps:
        rr = [r for r in step_rows if t in r["S"]]
        cum[t] = sign_and_rank([r["P"][t] for r in rr], [r["S"][t] for r in rr], rng)
    inc = {}
    for t1, t2 in intervals:
        rr = [r for r in step_rows if t1 in r["S"] and t2 in r["S"]]
        x = [r["P"][t2] - (r["P"][t1] if t1 else 0) for r in rr]
        y = [r["S"][t2] - r["S"][t1] for r in rr]
        inc["%d-%d" % (t1, t2)] = sign_and_rank(x, y, rng)
    return cum, inc


def pair_level_tests(pair_rows, seed=0):
    """Push and late alignment against S80 over all seed pairs (sign, binomial p, rank r), and for |S80| > 0.01."""
    x = np.array([r["push"] for r in pair_rows])
    a = np.array([r["align_late"] for r in pair_rows])
    y = np.array([r["S80"] for r in pair_rows])
    rng = np.random.default_rng(seed)
    return {"push": sign_and_rank(x, y, rng), "push_abs_S_gt_0.01": sign_and_rank(x, y, rng, min_abs_y=0.01),
            "align_late": sign_and_rank(a, y, rng)}


def alignment_by_call(cell, n_inst=8):
    """Per judge condition's runs and call: corr of k (centred) with the instance-mean contrast d8 and with d0.

    d = (A_T - A_E)/2 from within-group standardized rescored rewards (groups:
    run, prompt within the call); d8 averages the eight instance pairs (T_kv, E_kv),
    d0 uses the training instances only.
    """
    out = {}
    for pol in "TE":
        rows = []
        for c in sorted(set(cell.call.tolist())):
            m = (cell.call == c) & (cell.cond == pol) & cell.valid
            G = cell.groups(m, cols=(0, 2))
            if not G:
                continue
            cen = lambda x: np.concatenate([x[g] - x[g].mean() for g in G])
            adv = lambda x: np.concatenate([(x[g] - x[g].mean()) / (x[g].std(ddof=1) + 1e-4) for g in G])
            kk = cen(cell.k)
            d8 = np.mean([(adv(cell.r["T_k%d" % v]) - adv(cell.r["E_k%d" % v])) / 2 for v in range(n_inst)], 0)
            d0 = (adv(cell.r["T_k0"]) - adv(cell.r["E_k0"])) / 2
            rows.append(dict(call=int(c), align_d8=float(np.corrcoef(d8, kk)[0, 1]), align_d0=float(np.corrcoef(d0, kk)[0, 1])))
        out[pol] = rows
    return out


# ==========================================================================
# Emission on the training samples (CPU)
# ==========================================================================
def emission_by_call(cell, pol=None, use_valid=False, n_inst=8):
    """Per call: Lambda_t and the trait-signal ratio on the recorded samples of the T runs, E runs or both.

    Scores are centred within groups (run, prompt) of the call. With
    rep = mean over same-condition instance pairs of sd(u - v)/2 (T and E
    separately) and cross = mean over instance pairs (T_i, E_j), i != j, of
    sd(T_i - E_j)/2: Lambda_t = log(cross / mean(rep_T, rep_E)); signal ratio =
    [sd(mean_i T_i - mean_j E_j)/2] / [mean(rep_T, rep_E) / sqrt(n_inst)].
    """
    rows = []
    for c in sorted(set(cell.call.tolist())):
        m = cell.call == c
        if pol is not None:
            m &= cell.cond == pol
        if use_valid:
            m &= cell.valid
        G = cell.groups(m, cols=(0, 2))
        if not G:
            continue
        cen = lambda x: np.concatenate([x[g] - x[g].mean() for g in G])
        Tm = [cen(cell.r["T_k%d" % k]) for k in range(n_inst)]
        Em = [cen(cell.r["E_k%d" % k]) for k in range(n_inst)]
        sd = lambda u, v: float(np.std(u - v) / 2)
        rT = np.mean([sd(Tm[i], Tm[j]) for i, j in itertools.combinations(range(n_inst), 2)])
        rE = np.mean([sd(Em[i], Em[j]) for i, j in itertools.combinations(range(n_inst), 2)])
        cx = np.mean([sd(Tm[i], Em[j]) for i in range(n_inst) for j in range(n_inst) if i != j])
        sig = float(np.std(np.mean(Tm, 0) - np.mean(Em, 0)) / 2)
        noise = (rT + rE) / 2 / np.sqrt(n_inst)
        rows.append(dict(call=int(c), Lambda=float(np.log(cx / ((rT + rE) / 2))), signal_ratio=sig / noise))
    return rows


_TK = ["T_k%d" % k for k in range(8)]
_EK = ["E_k%d" % k for k in range(8)]
_SAME = list(itertools.combinations(_TK, 2)) + list(itertools.combinations(_EK, 2))
EMISSION_SETS = {"k!=l": [(_TK[k], _EK[l]) for k in range(8) for l in range(8) if k != l],
                 "8x8": [(t, e) for t in _TK for e in _EK], "k=l": list(zip(_TK, _EK)),
                 "s0": [(_TK[0], _EK[l]) for l in range(1, 8)] + [(_TK[k], _EK[0]) for k in range(1, 8)]}
# The training-instance row uses same-condition pairs that also involve instance 0 as its denominator.
EMISSION_DEN = {"s0": [(_TK[0], _TK[l]) for l in range(1, 8)] + [(_EK[0], _EK[l]) for l in range(1, 8)]}


def emission_lambda(R, gid, idx, n_boot=2000, seed=0):
    """Lambda (equation 2) on a subset of samples, with a bootstrap interval over groups.

    ``R``: dict instance key -> scores; ``gid``: group id per sample; ``idx``:
    indices of the samples used. Scores are centred within group (single-sample
    groups included in the counts). For each set of instance pairs,
    V = mean over pairs of sqrt(sum of squared centred differences / n), and
    Lambda = log V(set) - log V(same-condition pairs). The interval resamples
    groups (multinomial weights, ``n_boot`` draws, RandomState(seed)).
    Sets: k != l (the paper's emission), 8 x 8, k = l, and the training instance
    alone (s0, with its own denominator).
    """
    g = gid[idx]
    _, inv = np.unique(g, return_inverse=True)
    n = np.bincount(inv).astype(float)

    def gc(x):
        x = x[idx].astype(float)
        m = np.bincount(inv, x) / n
        return x - m[inv]

    C = {k: gc(R[k]) for k in _TK + _EK}
    ps = lambda prs: np.stack([np.bincount(inv, (C[a] - C[b]) ** 2) for a, b in prs], 1)
    BW = np.random.RandomState(seed).multinomial(len(n), np.ones(len(n)) / len(n), size=n_boot).astype(float)
    out = {}
    for nm, prs in EMISSION_SETS.items():
        SA, SB = ps(prs), ps(EMISSION_DEN.get(nm, _SAME))
        f = lambda w: np.log(np.sqrt((w @ SA) / (w @ n)[..., None]).mean(-1) / np.sqrt((w @ SB) / (w @ n)[..., None]).mean(-1))
        lo, hi = np.percentile(f(BW), [2.5, 97.5])
        out[nm] = (float(f(np.ones(len(n)))), float(lo), float(hi))
    return out, len(n)


def emission_on_training_samples(cells, phases=EMISSION_PHASES, n_boot=2000):
    """Lambda per call and per phase on the T runs' and the E runs' samples, pooling the seed pairs of ``cells``.

    ``cells``: LoggedCell objects of the same cell (e.g. two batches of seed
    pairs); run indices are offset so that groups (run, call, completion
    index // 8) stay distinct.
    """
    R = {k: [] for k in _TK + _EK}
    meta, conds, off = [], [], 0
    for c in cells:
        mm = c.meta.copy()
        mm[:, 0] += off
        off += len(c.runs)
        meta.append(mm)
        conds.append(c.cond)
        for k in R:
            R[k].append(c.r[k])
    meta = np.concatenate(meta)
    cond = np.concatenate(conds)
    R = {k: np.concatenate(v) for k, v in R.items()}
    call = meta[:, 1]
    gid = meta[:, 0] * 10 ** 6 + meta[:, 1] * 10 ** 3 + meta[:, 3] // 8
    res = {}
    for po in "TE":
        by_call = {}
        for c in sorted(set(call.tolist())):
            o, ng = emission_lambda(R, gid, np.where((cond == po) & (call == c))[0], n_boot)
            by_call[int(c)] = dict(groups=ng, **o)
        by_phase = {}
        for nm, cs in phases:
            o, ng = emission_lambda(R, gid, np.where((cond == po) & np.isin(call, cs))[0], n_boot)
            by_phase[nm] = dict(groups=ng, **o)
        res[po] = {"by_call": by_call, "by_phase": by_phase}
    return res


def start_pool_emission(pool_npz, n_boot=200):
    """The same Lambda on a candidate pool (groups = prompts), for comparison with the start-policy emission."""
    B = np.load(pool_npz)
    R = {k: B["r_" + k] for k in _TK + _EK}
    o, _ = emission_lambda(R, B["qi"], np.arange(len(B["qi"])), n_boot)
    return o


# ==========================================================================
# CLI
# ==========================================================================
def _cmd_state(a):
    pairs = json.load(open(a.pairs))
    est = set(a.established.split(",")) if a.established else set()
    tl = cell_timeline(pairs, a.eval_suffix)
    res = {"timeline": tl, "table": {k: ed_table_row(v, k in est) for k, v in tl.items()}}
    for k in sorted(tl):
        r = res["table"][k]
        print("%-6s | %s | %s | %s | %s | %s" % (k, r["validity"], r["lowest_run_80"], r["signal_share"],
                                                r["first_step_signal_below_half"], r["steps_50_80"]))
    if not a.no_steps_logs:
        res["collapse"], res["collapse_r"] = collapse_correlates(pairs)
        res["reversion"], res["reversion_r"] = reversion_after_peak(pairs, tl, a.eval_suffix)
        print("collapse: " + "  ".join("%s %+.2f" % kv for kv in res["collapse_r"].items()))
        print("reversion: " + "  ".join("%s %+.2f" % kv for kv in res["reversion_r"].items()))
    res["when"] = when_outcome_forms(pairs, eval_suffix=a.eval_suffix)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def _cmd_seed_paths(a):
    pairs = [x for f in a.pairs for x in json.load(open(f))]
    res = seed_path_statistics(pairs, a.eval_suffix)
    print("seed pairs %d  cells %d" % (res["n_pairs"], res["n_cells"]))
    for t, v in res["by_step"].items():
        print("  t=%2d  r(S_t, S80) %+.3f  sd %.4f" % (t, v["r_with_S80"], v["sd"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def _cmd_format(a):
    for item in json.load(open(a.logs)):
        fs = format_share(item["logs"], item["cell"])
        line = "%-6s | format R2 %s | signal groups %s" % (item["cell"], " / ".join("%.2f" % x for x in fs["format_r2"]),
                                                          " / ".join("%.2f" % x for x in fs["signal_groups"]))
        for cond in "TE":
            if item.get("logs_" + cond):
                evs = [read_eval(b, a.eval_suffix) for b in item.get("bases_" + cond, [])] or None
                iv = independent_validity(item["logs_" + cond], item["cell"], evs)
                line += " | %s runs: " % cond + " ".join("%s %.3f (logged %.3f)" % (k, v["independent"], v["logged"]) for k, v in iv.items())
        print(line)


def _cmd_push(a):
    allrows, steprows, cellsum = [], [], []
    for f in a.cells:
        cell = LoggedCell(json.load(open(f)))
        rows, summ = push_and_alignment(cell)
        allrows += rows
        cellsum.append(summ)
        steprows += push_by_step(cell)
        print("%-6s pairs %2d  S80 %+.4f  push %+.3e  late alignment (T-E) %+.3f" % (summ["cell"], summ["n"], summ["S80"], summ["push"], summ["align_late"]))
    res = {"pairs": allrows, "cells": cellsum, "pair_tests": pair_level_tests(allrows)}
    res["cell_sign"] = {"push": sum(np.sign(c["push"]) == np.sign(c["S80"]) for c in cellsum),
                        "align_late": sum(np.sign(c["align_late"]) == np.sign(c["S80"]) for c in cellsum), "n": len(cellsum)}
    res["cumulative"], res["intervals"] = push_tests(steprows)
    for k, v in res["pair_tests"].items():
        print("%-20s n %3d  sign %3d (p %.3f)  rank r %+.3f (p %.3f)" % (k, v["n"], v["sign"], v["p_sign"], v["rank_r"], v["p_rank"]))
    for lab, d in (("cumulative t", res["cumulative"]), ("interval", res["intervals"])):
        for k, v in d.items():
            print("%s %-6s n %3d  sign %3d (p %.3f)  rank r %+.3f (p %.3f)" % (lab, k, v["n"], v["sign"], v["p_sign"], v["rank_r"], v["p_rank"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)


def _cmd_emission(a):
    cells = [LoggedCell(json.load(open(f))) for f in a.cells]
    res = {"pooled": emission_on_training_samples(cells, n_boot=a.n_boot)}
    res["by_call"] = {c.cell or str(i): {p: emission_by_call(c, p) for p in ("T", "E", None)} for i, c in enumerate(cells)}
    if a.pool:
        res["start_pool"] = start_pool_emission(a.pool)
    for po, d in res["pooled"].items():
        for nm, v in d["by_phase"].items():
            print("%s runs  %-5s groups %4d  " % (po, nm, v["groups"]) + "  ".join(
                "%s %+.3f [%+.3f, %+.3f]" % (k, *v[k]) for k in EMISSION_SETS))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-suffix", default=".eval.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("state", help="Extended Data Table 'The state of training' and the optimization-state statistics")
    s.add_argument("--pairs", required=True)
    s.add_argument("--established", default="", help="comma-separated established cells (timing shown only for these)")
    s.add_argument("--no-steps-logs", action="store_true")
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_state)
    s = sub.add_parser("seed-paths", help="spread and reversion of seed-pair deviations")
    s.add_argument("--pairs", nargs="+", required=True)
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_seed_paths)
    s = sub.add_parser("format-share", help="format share of the reward and independent carrier-rule validity")
    s.add_argument("--logs", required=True, help='JSON list of {"cell", "logs": [...], optional "logs_T"/"logs_E" and "bases_T"/"bases_E"}')
    s.set_defaults(fn=_cmd_format)
    s = sub.add_parser("rescore", help="GPU: rescore logged samples with every judge instance and compute k")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda")
    s.set_defaults(fn=lambda a: rescore(json.load(open(a.config)), a.out, a.device))
    s = sub.add_parser("push", help="in-training push and alignment against S")
    s.add_argument("--cells", nargs="+", required=True, help="logged-cell config files")
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_push)
    s = sub.add_parser("emission", help="emission on the training samples")
    s.add_argument("--cells", nargs="+", required=True, help="logged-cell configs of one cell (seed-pair batches)")
    s.add_argument("--pool", help="candidate pool .npz for the start-policy value")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_emission)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
