"""Weight-level decomposition of trained students.

Paper parts served:
  * Supplementary Methods, "Weight-level decomposition";
  * Supplementary Note "How much of the push the optimizer realizes, and where",
    including Supplementary Table "Layer-wise optimizer geometry";
  * Supplementary Note "Cross-family judges in the weights".

What is computed
----------------
First GRPO direction of a route (``first_grpo_direction``). On the start
policy's candidate pool of the SP-route cell, with that route's T and E judges,

    w(c) = (A_T(c) - A_E(c)) / 2,   A_z = within-prompt standardized score
                                     (sample sd, ddof 1, plus 1e-4),
    v    = grad_theta  sum_c w(c) log pi_theta0(c) / N,  N = #{c : w(c) != 0},

with log pi the summed log-likelihood of the completion tokens after the prompt
(fp32 weights, batches of 8, left padding, truncation at 320 tokens). ``v_FT``
uses the FT judges (fine-tuned copies of the student's base model) and ``v_SP``
the SP judges (the base model with the persona sentence), both on the SP-route
pool. Directions are stored as bf16 safetensors keyed by parameter name. The
same function gives a cross-family judge's direction when given that judge's
scores.

Decomposition of a seed pair (``decompose_seed_pair``). With theta_T, theta_E
the step-80 students of one seed pair (fp32 master weights),

    Delta = (theta_T - theta_E) / 2,   theta_mid = (theta_T + theta_E) / 2,

and e_FT = v_FT/|v_FT|, e_2 the unit vector of v_SP orthogonalized against e_FT
(e_SP = v_SP/|v_SP|), the components are

    full                 Delta itself, read at the two endpoints theta_T, theta_E
                         (not at theta_mid +- Delta, so that fp32 rounding of the
                         midpoint does not move weights across the bf16 grid);
    proj_vFT             <Delta, e_FT> e_FT;
    proj_vSP             <Delta, e_SP> e_SP;
    span                 <Delta, e_FT> e_FT + <Delta, e_2> e_2;
    remainder            Delta - span;
    remainder:<group>    the remainder restricted to one parameter group;
    full:<group>         Delta restricted to one four-layer group;
    span:<group>         the span component restricted to one four-layer group.

A component X is applied as theta_mid +- X (outside the restricted group the
weights stay at theta_mid) and its effect on an evaluation R (here the preference margin) is
(R(theta_mid + X) - R(theta_mid - X)) / 2; for ``full`` it is
(R(theta_T) - R(theta_E)) / 2, i.e. the seed pair's S_80. Every evaluation is taken
twice: in training precision (fp32 weights, bf16 autocast forward pass) and
with fp32 weights and an fp32 forward pass. Attribution uses fp32; the bf16
``full`` effect is the effective transfer. Groups are ``L00-03``, ``L04-07``, ...
(four transformer layers each), ``embedding`` and ``other`` (final norm and
anything outside the layers). For every group the inner products needed for the
layer-wise least-squares fit are stored (``geo``).

Summaries (CPU, from the decomposition JSON):
  * ``cell_summary`` / ``route_summary`` / ``same_vs_cross``: mean full effect
    in both precisions, span and remainder effects, cos(Delta, v_FT),
    cos(Delta, v_SP);
  * ``additivity``: span + remainder against full (fp32), per seed pair and cell;
  * ``cancellation``: the early-layer remainder against the span component
    (slope through the origin, correlation), full/span share, contribution of
    the other layers;
  * ``layer_geometry``: following ratio per four-layer group (least-squares
    coefficients of Delta_g on the span within the group, compared with the
    network-wide coefficients; median over seed pairs), cos(Delta_g, span_g),
    energy share and per-group effects;
  * ``seed_level``: per-seed signs and correlations of span and remainder
    effects with the full effect;
  * ``check_full_against_logged``: full bf16 effect against the logged S_80.

Optimizer geometry (CPU, streamed from safetensors):
  * ``sign_alignment``: per group, cos(Delta, v), cos(Delta, sign v) and the
    share of |v|^2 in the top 1% of coordinates (top 1% taken per tensor and
    summed over the group's tensors), pooled over the first two seed pairs;
  * ``seed_pair_geometry``: per group, mean cosine between the displacements of
    different seed pairs, and the means of cos(Delta, v) and cos(Delta, sign v)
    over seed pairs with their ratio.

Direction comparisons (``compare_directions``): cosines between directions
(e.g. the same route on two independent pools, or a direction and the FT
judges' own weight difference u = theta_JT - theta_JE), per group cosines and
energy shares; ``directional_effect`` evaluates the model at +-eps along a
direction scaled to a reference norm.

Inputs (large artefacts, paths on the command line)
---------------------------------------------------
  * start policy: Hugging Face model directory or hub name (fp32 load);
  * students: step-80 checkpoints (``model*.safetensors`` + config), listed in
    a seed-pair manifest JSON: ``[{"seed": 70680, "T": dir, "E": dir}, ...]``;
  * candidate pool: ``<pool>.npz`` with ``vi`` (indices of valid candidates in
    the JSONL) and ``qi`` (prompt index of each valid candidate), optionally
    stored judge scores ``r_T_k<j>``, ``r_E_k<j>`` (judge instance j; j = 0 is
    the training instance), and ``<pool>.jsonl`` with one row per sampled
    candidate (``{"c": completion, "valid": bool, "qi": int}``);
  * judge scores for a direction not stored in the pool: ``.npz`` with arrays
    ``r_T`` and ``r_E`` aligned with the pool's valid candidates;
  * directions: safetensors files (parameter name -> bf16 tensor).
"""

import argparse
import glob
import importlib
import itertools
import json
import os
import re
import time

import numpy as np

from ..chat import load_tokenizer, masked, user_prompt
from ..constants import CONFIGURATIONS, TRAIT_PAIRS

SP_CONFIGS = tuple(k for k, v in CONFIGURATIONS.items() if v[1] == "SP")
CANCELLATION_CELLS = ("A1_SL", "A1_WS", "A3_JB", "B3_WS", "A2_WS", "A4_WS")   # the six cells that show the cancellation


# --------------------------------------------------------------------------
# Parameter groups and parameter-list arithmetic
# --------------------------------------------------------------------------
def layer_group(name):
    """Four-layer group of a parameter name: ``L00-03``, ..., ``embedding`` or ``other``."""
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        k = int(m.group(1))
        return "L%02d-%02d" % (4 * (k // 4), 4 * (k // 4) + 3)
    return "embedding" if "embed" in name else "other"


def inner(a, b, index=None):
    """Inner product of two parameter lists in fp64 (optionally over a subset of tensors)."""
    ix = range(len(a)) if index is None else index
    return float(sum(float((a[i].double() * b[i].double()).sum()) for i in ix))


def _load_fp32(path, device):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_parameters(path, names, device):
    """fp32 copies of a checkpoint's parameters, in the order of ``names``."""
    import gc
    import torch
    m = _load_fp32(path, device)
    d = dict(m.named_parameters())
    out = [d[n].detach().clone() for n in names]
    del m, d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def load_direction(path, names, device):
    """A stored direction (bf16 safetensors) as fp32 tensors in the order of ``names``."""
    from safetensors.torch import load_file
    d = load_file(path)
    out = [d[n].to(device).float() for n in names]
    del d
    return out


def save_direction(vec, names, path):
    import torch
    from safetensors.torch import save_file
    save_file({n: x.detach().to(torch.bfloat16).cpu().contiguous() for n, x in zip(names, vec)}, path)


def has_weights(d):
    return bool(glob.glob(os.path.join(d, "model*.safetensors")))


# --------------------------------------------------------------------------
# Candidate pool
# --------------------------------------------------------------------------
def load_pool(npz_path, jsonl_path=None):
    """Valid candidates of a pool: (qi array, completion list, npz handle)."""
    pool_arrays = np.load(npz_path)
    rows = [json.loads(l) for l in open(jsonl_path or npz_path.replace(".npz", ".jsonl"))]
    vi = pool_arrays["vi"]
    qi = [int(x) for x in pool_arrays["qi"]]
    assert len(vi) == len(qi) and all(rows[i]["valid"] and rows[i]["qi"] == q for i, q in zip(vi, qi))
    return np.array(qi), [rows[i]["c"] for i in vi], pool_arrays


def pool_texts(tok, prompts, qi, completions):
    """Student-context texts (rendered prompt + completion) and prompt token counts."""
    rendered = {q: user_prompt(tok, prompts[q]) for q in set(int(x) for x in qi)}
    plen = {q: len(tok(p, add_special_tokens=False).input_ids) for q, p in rendered.items()}
    return [rendered[int(q)] + c for q, c in zip(qi, completions)], [plen[int(q)] for q in qi]


# --------------------------------------------------------------------------
# First GRPO direction
# --------------------------------------------------------------------------
def grpo_pair_weights(qi, r_t, r_e):
    """w = (A_T - A_E)/2 with A the within-prompt standardized score (ddof 1, + 1e-4); singleton prompts get 0."""
    qi = np.asarray(qi)
    r_t, r_e = np.asarray(r_t, float), np.asarray(r_e, float)
    w = np.zeros(len(r_t))
    for g in np.unique(qi):
        ix = np.where(qi == g)[0]
        if len(ix) < 2:
            continue
        a = lambda r: (r[ix] - r[ix].mean()) / (r[ix].std(ddof=1) + 1e-4)
        w[ix] = (a(r_t) - a(r_e)) / 2
    return w


def first_grpo_direction(model, tok, texts, prompt_lens, w, device, batch_size=8, max_length=320):
    """v = grad sum_c w_c log pi(c) / N at the model's current weights (fp32)."""
    import torch
    params = list(model.parameters())
    for p in params:
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    n = float((w != 0).sum())
    for i in range(0, len(texts), batch_size):
        ww = torch.tensor(w[i:i + batch_size], dtype=torch.float32, device=device)
        if float(ww.abs().sum()) == 0:
            continue
        e = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                add_special_tokens=False)
        e = {k: v.to(device) for k, v in e.items()}
        lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
        tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        lp = (tk * masked(e, prompt_lens[i:i + batch_size])).sum(1)
        ((ww * lp).sum() / n).backward()
    v = [p.grad.detach().clone() for p in params]
    model.zero_grad(set_to_none=True)
    for p in params:
        p.requires_grad_(False)
    return v


# --------------------------------------------------------------------------
# Direction comparisons and directional effects
# --------------------------------------------------------------------------
def compare_directions(vectors, names, pairs):
    """Cosines between named parameter-list vectors, overall and per group, and per-group energy shares."""
    groups = sorted(set(layer_group(n) for n in names))
    gix = {g: [i for i, n in enumerate(names) if layer_group(n) == g] for g in groups}

    def cos(a, b, ix=None):
        return inner(a, b, ix) / np.sqrt(inner(a, a, ix) * inner(b, b, ix) + 1e-300)

    out = {"cos": {}, "group_cos": {}, "energy": {}, "norm": {}}
    for a, b in pairs:
        k = "%s~%s" % (a, b)
        out["cos"][k] = cos(vectors[a], vectors[b])
        out["group_cos"][k] = {g: cos(vectors[a], vectors[b], gix[g]) for g in groups}
    for k, v in vectors.items():
        tot = inner(v, v)
        out["norm"][k] = float(np.sqrt(tot))
        out["energy"][k] = {g: inner(v, v, gix[g]) / tot for g in groups}
    return out


def directional_effect(model, direction, reference_norm, evaluate, eps=0.2):
    """(R(theta0 + eps s d) - R(theta0 - eps s d)) / (2 eps), s = reference_norm / |d|; weights restored after."""
    import torch
    params = list(model.parameters())
    th0 = [p.detach().clone() for p in params]
    sc = eps * reference_norm / np.sqrt(inner(direction, direction))
    vals = []
    for sg in (1, -1):
        with torch.no_grad():
            for p, a, x in zip(params, th0, direction):
                p.copy_(a + sg * sc * x)
        vals.append(evaluate(model))
    with torch.no_grad():
        for p, a in zip(params, th0):
            p.copy_(a)
    return {k: (vals[0][k] - vals[1][k]) / (2 * eps) for k in vals[0]}


# --------------------------------------------------------------------------
# Decomposition of one seed pair
# --------------------------------------------------------------------------
def orthonormal_basis(v_ft, v_sp):
    """e_FT, e_SP and e_2 (v_SP orthogonalized against e_FT), plus cos(v_FT, v_SP)."""
    n_f = np.sqrt(inner(v_ft, v_ft))
    e_f = [x / n_f for x in v_ft]
    n_s = np.sqrt(inner(v_sp, v_sp))
    e_s = [x / n_s for x in v_sp]
    c = inner(e_s, e_f)
    e2 = [x - c * y for x, y in zip(e_s, e_f)]
    n2 = np.sqrt(inner(e2, e2))
    e2 = [x / n2 for x in e2]
    return {"e_ft": e_f, "e_sp": e_s, "e_2": e2, "cos_vFT_vSP": c}


def evaluate_both_precisions(model, evaluate):
    """Evaluation in training precision (bf16 autocast over fp32 weights) and in fp32."""
    import torch
    dev = next(model.parameters()).device.type
    out = {}
    for tag, ac in (("bf16", True), ("fp32", False)):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=ac):
            r = evaluate(model)
        out.update({"%s_%s" % (k, tag): float(v) for k, v in r.items()})
    return out


def _set_weights(params, values):
    import torch
    with torch.no_grad():
        for p, x in zip(params, values):
            p.copy_(x)


def decompose_seed_pair(model, names, theta_t, theta_e, basis, evaluate, groups=None):
    """Components of Delta and their effects on ``evaluate`` (both precisions); see the module docstring.

    ``model`` is the start policy loaded in fp32 (its parameters are overwritten);
    ``theta_t``/``theta_e`` are fp32 parameter lists in the order of ``names``.
    """
    import torch
    params = [dict(model.named_parameters())[n] for n in names]
    grp = [layer_group(n) for n in names]
    groups = groups or sorted(set(grp))
    layer_groups = [g for g in groups if g.startswith("L")]
    e_f, e_s, e_2 = basis["e_ft"], basis["e_sp"], basis["e_2"]

    ends = []
    for tw in (theta_t, theta_e):
        _set_weights(params, tw)
        ends.append(evaluate_both_precisions(model, evaluate))
    mid = [(a + b) / 2 for a, b in zip(theta_t, theta_e)]
    d = [(a - b) / 2 for a, b in zip(theta_t, theta_e)]

    a_f, a_s, a_2 = inner(d, e_f), inner(d, e_s), inner(d, e_2)
    n_d = np.sqrt(inner(d, d))
    comp = {
        "proj_vFT": lambda i: a_f * e_f[i],
        "proj_vSP": lambda i: a_s * e_s[i],
        "span": lambda i: a_f * e_f[i] + a_2 * e_2[i],
        "remainder": lambda i: d[i] - a_f * e_f[i] - a_2 * e_2[i],
    }
    components = [(k, f, None) for k, f in comp.items()]
    components += [("remainder:" + g, comp["remainder"], g) for g in groups]
    components += [("full:" + g, (lambda i: d[i]), g) for g in layer_groups]
    components += [("span:" + g, comp["span"], g) for g in layer_groups]

    geo = {}
    for g in groups:
        ix = [i for i in range(len(names)) if grp[i] == g]
        geo[g] = dict(dd=inner(d, d, ix), dF=inner(d, e_f, ix), d2=inner(d, e_2, ix),
                      FF=inner(e_f, e_f, ix), S22=inner(e_2, e_2, ix), F2=inner(e_f, e_2, ix))

    eff = {"full": {k: (ends[0][k] - ends[1][k]) / 2 for k in ends[0]}}
    for name, f, g in components:
        vals = []
        for sg in (1, -1):
            with torch.no_grad():
                for i, (p, m_) in enumerate(zip(params, mid)):
                    p.copy_(m_ + sg * f(i)) if (g is None or grp[i] == g) else p.copy_(m_)
            vals.append(evaluate_both_precisions(model, evaluate))
        eff[name] = {k: (vals[0][k] - vals[1][k]) / 2 for k in vals[0]}
    return dict(cos_vFT=a_f / n_d, cos_vSP=a_s / n_d, frac_span=(a_f ** 2 + a_2 ** 2) / n_d ** 2, normD=n_d,
                aF=a_f, a2=a_2, geo=geo, eff=eff)


def decompose_cell(start_policy, pairs, direction_ft, direction_sp, evaluate, device="cuda", min_pairs=6, log=print):
    """Decompose every seed pair of a cell. ``pairs``: list of {"seed", "T", "E"} checkpoint directories."""
    import gc
    import torch
    t0 = time.time()
    model = _load_fp32(start_policy, device)
    names = [n for n, _ in model.named_parameters()]
    v_f = load_direction(direction_ft, names, device)
    v_s = load_direction(direction_sp, names, device)
    basis = orthonormal_basis(v_f, v_s)
    del v_f, v_s
    gc.collect()
    groups = sorted(set(layer_group(n) for n in names))
    seeds = []
    for pr in sorted(pairs, key=lambda x: x["seed"]):
        if not (has_weights(pr["T"]) and has_weights(pr["E"])):
            log("seed %s: student checkpoint missing, skipped" % pr["seed"])
            continue
        th_t = load_parameters(pr["T"], names, device)
        th_e = load_parameters(pr["E"], names, device)
        rec = decompose_seed_pair(model, names, th_t, th_e, basis, evaluate, groups)
        rec["seed"] = pr["seed"]
        seeds.append(rec)
        del th_t, th_e
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("seed %s: cos(D,vFT) %+.4f cos(D,vSP) %+.4f | full bf16 %+.4f fp32 %+.4f | span %+.4f remainder %+.4f (%.0fs)" % (
            pr["seed"], rec["cos_vFT"], rec["cos_vSP"], rec["eff"]["full"].get("preference_margin_bf16", np.nan),
            rec["eff"]["full"].get("preference_margin_fp32", np.nan), rec["eff"]["span"].get("preference_margin_fp32", np.nan),
            rec["eff"]["remainder"].get("preference_margin_fp32", np.nan), time.time() - t0))
    return dict(cos_vFT_vSP=basis["cos_vFT_vSP"], groups=groups, seeds=seeds, ok=len(seeds) >= min_pairs,
                precisions=["bf16", "fp32"], sec=round(time.time() - t0))


# --------------------------------------------------------------------------
# Summaries of decomposition results (CPU)
# --------------------------------------------------------------------------
KEY = "preference_margin"


def _e(seed, comp, prec="fp32", key=KEY):
    x = seed["eff"].get(comp)
    return None if not isinstance(x, dict) else x.get("%s_%s" % (key, prec))


def _mean(dec, comp, prec="fp32", key=KEY):
    return float(np.mean([_e(s, comp, prec, key) for s in dec["seeds"]]))


def route_of(cell):
    return "SP" if cell.split("_")[0] in SP_CONFIGS else "FT"


def cell_summary(dec, key=KEY):
    """Seed-pair means for one decomposed cell (condition given by ``dec['condition']``)."""
    s = dec["seeds"]
    return dict(cell=dec["cell"], condition=dec.get("condition", "same"), route=route_of(dec["cell"]), n=len(s),
                full_bf16=_mean(dec, "full", "bf16", key), full=_mean(dec, "full", key=key),
                span=_mean(dec, "span", key=key), remainder=_mean(dec, "remainder", key=key),
                cos_vFT=float(np.mean([x["cos_vFT"] for x in s])), cos_vSP=float(np.mean([x["cos_vSP"] for x in s])))


def route_summary(rows):
    """Per condition and route: means, span share of the summed full effect, sign disagreements."""
    out = []
    for c in sorted(set(r["condition"] for r in rows)):
        for rt in ("FT", "SP"):
            x = [r for r in rows if r["condition"] == c and r["route"] == rt]
            if not x:
                continue
            b = np.array([r["full"] for r in x])
            bv = np.array([r["span"] for r in x])
            br = np.array([r["remainder"] for r in x])
            out.append(dict(condition=c, route=rt, n_cells=len(x), full_bf16=float(np.mean([r["full_bf16"] for r in x])),
                            full=float(b.mean()), span=float(bv.mean()), remainder=float(br.mean()),
                            span_share=float(bv.sum() / b.sum()) if abs(b.sum()) > 1e-9 else float("nan"),
                            n_opposite_sign=int(sum(np.sign(p) != np.sign(q) for p, q in zip(bv, br))),
                            cos_vFT=float(np.mean([r["cos_vFT"] for r in x])), cos_vSP=float(np.mean([r["cos_vSP"] for r in x]))))
    return out


def same_vs_cross(rows, same="same", cross="cross"):
    """Means over cells decomposed under both conditions."""
    by = {}
    for r in rows:
        by.setdefault(r["cell"], {})[r["condition"]] = r
    both = [d for d in by.values() if same in d and cross in d]
    return dict(n_cells=len(both), **{"%s_%s" % (k, c): float(np.mean([d[c][k] for d in both]))
                                      for k in ("full", "span", "remainder", "cos_vFT") for c in (same, cross)})


def additivity(decs, key=KEY):
    """span + remainder against full (fp32): error over all seed pairs and per cell.

    Also the sum of per-group remainders against the remainder and the sum of
    per-group full components against full (maxima per cell).
    """
    allrows, cells = [], {}
    for dec in decs:
        g = dec["groups"]
        rows = []
        for s in dec["seeds"]:
            t, vs, rm = _e(s, "full", key=key), _e(s, "span", key=key), _e(s, "remainder", key=key)
            if None in (t, vs, rm):
                continue
            r = dict(seed=s["seed"], full=t, span=vs, err=abs(vs + rm - t), recon=vs + rm)
            rg = [_e(s, "remainder:" + x, key=key) for x in g]
            tg = [_e(s, "full:" + x, key=key) for x in g if x.startswith("L")]
            if None not in rg:
                r["err_remainder_groups"] = sum(rg) - rm
            if tg and None not in tg:
                r["err_full_groups"] = sum(tg) - t
            rows.append(r)
            allrows.append(r)
        t_ = np.array([r["full"] for r in rows])
        rc = np.array([r["recon"] for r in rows])
        cells[dec["cell"]] = dict(
            n=len(rows), err_mean=float(np.mean([r["err"] for r in rows])), span_abs_mean=float(np.mean([abs(r["span"]) for r in rows])),
            err_rel_span_pct=float(100 * np.mean([r["err"] / abs(r["span"]) for r in rows])),
            full_mean=float(t_.mean()), recon_mean=float(rc.mean()), residual_mean=float((rc - t_).mean()),
            residual_abs_mean=float(np.abs(rc - t_).mean()), span_mean=float(np.mean([r["span"] for r in rows])),
            residual_rel_full_pct=float(100 * np.abs(rc - t_).mean() / abs(t_.mean())),
            max_err_remainder_groups=max([abs(r["err_remainder_groups"]) for r in rows if "err_remainder_groups" in r] or [float("nan")]),
            max_err_full_groups=max([abs(r["err_full_groups"]) for r in rows if "err_full_groups" in r] or [float("nan")]))
    e = np.array([[r["err"], abs(r["full"]), abs(r["span"])] for r in allrows])
    return dict(n=len(allrows), err_mean=float(e[:, 0].mean()), err_median=float(np.median(e[:, 0])), err_max=float(e[:, 0].max()),
                rel_span_pct_mean=float(100 * (e[:, 0] / e[:, 2]).mean()), rel_span_pct_max=float(100 * (e[:, 0] / e[:, 2]).max()),
                rel_full_pct_mean=float(100 * (e[:, 0] / e[:, 1]).mean()), rel_full_pct_max=float(100 * (e[:, 0] / e[:, 1]).max()),
                cells=cells)


def cancellation(decs, early="L00-03", subset=CANCELLATION_CELLS, key=KEY):
    """kappa = early-layer remainder / span component per cell; slope through the origin; full/span share."""
    rows = []
    for dec in decs:
        v, t, r0, ot = [], [], [], []
        others = [g for g in dec["groups"] if g != early]
        for s in dec["seeds"]:
            a, b, c = _e(s, "span", key=key), _e(s, "full", key=key), _e(s, "remainder:" + early, key=key)
            if None in (a, b, c):
                continue
            v.append(a), t.append(b), r0.append(c)
            ot.append(sum(x for x in (_e(s, "remainder:" + g, key=key) for g in others) if x is not None))
        if not v:
            continue
        V, T, R = np.mean(v), np.mean(t), np.mean(r0)
        rows.append(dict(cell=dec["cell"], span=V, early_remainder=R, full=T, kappa=R / V, full_over_span=T / V,
                         other_layers=float(np.mean(ot))))

    def fit(rs):
        V = np.array([r["span"] for r in rs])
        R = np.array([r["early_remainder"] for r in rs])
        F = np.array([r["full_over_span"] for r in rs])
        return dict(n=len(rs), slope=float(V @ R / (V @ V)), r=float(np.corrcoef(V, R)[0, 1]),
                    full_over_span_min=float(F.min()), full_over_span_max=float(F.max()), full_over_span_median=float(np.median(F)))
    sub = [r for r in rows if r["cell"] in subset]
    return dict(cells=rows, all=fit(rows), subset=fit(sub) if len(sub) > 2 else None)


def layer_geometry(dec, key=KEY):
    """Following ratio, cos(Delta_g, span_g) and energy share per four-layer group (needs ``geo``)."""
    s = dec["seeds"]
    out = {}
    for g in [x for x in dec["groups"] if x.startswith("L")]:
        fol, cg, sh = [], [], []
        for x in s:
            G = x["geo"][g]
            A = np.array([[G["FF"], G["F2"]], [G["F2"], G["S22"]]])
            b = np.array([G["dF"], G["d2"]])
            ag = np.array([x["aF"], x["a2"]])
            loc = np.linalg.lstsq(A, b, rcond=None)[0]                 # least-squares coefficients within the group
            glob_n = np.sqrt(ag @ A @ ag)                              # norm of the network-wide span component in g
            loc_n = np.sqrt(max(loc @ A @ loc, 0))
            sgn = np.sign(loc @ A @ ag)
            fol.append(sgn * loc_n / glob_n if glob_n > 0 else np.nan)
            cg.append(loc_n / np.sqrt(G["dd"]) if G["dd"] > 0 else np.nan)
            sh.append(G["dd"] / sum(x["geo"][h]["dd"] for h in x["geo"]))
        m = lambda c: float(np.mean([_e(x, c, key=key) for x in s])) if all(_e(x, c, key=key) is not None for x in s) else None
        out[g] = dict(following_ratio=float(np.nanmedian(fol)), cos_group_span=float(np.nanmean(cg)), energy_share=float(np.mean(sh)),
                      full=m("full:" + g), span=m("span:" + g), remainder=m("remainder:" + g))
    return out


def seed_level(dec, key=KEY):
    """Per-seed signs of span/remainder/full effects (fp32) and their correlations across seed pairs."""
    e = lambda c: np.array([_e(x, c, key=key) for x in dec["seeds"]])
    v, r, t = e("span"), e("remainder"), e("full")
    return dict(n=len(v), span_pos=int((v > 0).sum()), remainder_neg=int((r < 0).sum()),
                opposite_sign=int((np.sign(v) != np.sign(r)).sum()), full_pos=int((t > 0).sum()),
                median_abs_remainder_over_span=float(np.median(np.abs(r) / np.abs(v))),
                r_span_full=float(np.corrcoef(v, t)[0, 1]), r_remainder_full=float(np.corrcoef(r, t)[0, 1]), sd_full=float(t.std()))


def check_full_against_logged(dec, logged_s80, key=KEY):
    """Full bf16 effect against each seed pair's logged S_80 (dict seed -> value)."""
    x = np.array([(_e(s, "full", "bf16", key), logged_s80[s["seed"]]) for s in dec["seeds"] if s["seed"] in logged_s80])
    return dict(n=len(x), max_abs_diff=float(np.abs(x[:, 0] - x[:, 1]).max()),
                r=float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if len(x) > 2 else float("nan"))


# --------------------------------------------------------------------------
# Optimizer geometry from stored checkpoints (CPU, streamed)
# --------------------------------------------------------------------------
class _Multi:
    """Read tensors from a (possibly sharded) safetensors checkpoint by name."""

    def __init__(self, d):
        from safetensors import safe_open
        self.f = {}
        for fn in sorted(glob.glob(os.path.join(d, "model*.safetensors"))):
            h = safe_open(fn, "pt")
            for k in h.keys():
                self.f[k] = h

    def keys(self):
        return list(self.f)

    def get(self, k):
        return self.f[k].get_tensor(k)


def _resolve(name, keys):
    if name in keys:
        return name
    return "model." + name if "model." + name in keys else None


def sign_alignment(pairs, directions, n_pairs=2):
    """Per group and route: cos(Delta, v), cos(Delta, sign v) pooled over the first ``n_pairs`` seed pairs,
    and the share of |v|^2 in the top 1% of coordinates (per tensor, summed over the group).

    ``directions``: {"FT": path, "SP": path}.
    """
    import torch
    from safetensors.torch import load_file
    V = {r: load_file(p) for r, p in directions.items()}
    routes = list(V)
    acc = {}
    for i, pr in enumerate(sorted(pairs, key=lambda x: x["seed"])[:n_pairs]):
        A, B = _Multi(pr["T"]), _Multi(pr["E"])
        vkeys = set(V[routes[0]])
        for n in A.keys():
            vn = _resolve(n, vkeys)
            if vn is None:
                continue
            g = layer_group(n)
            d = ((A.get(n).float() - B.get(n).float()) / 2).flatten().double()
            a = acc.setdefault(g, dict(dd=0.0, **{k + r: 0.0 for r in routes for k in ("dv_", "vv_", "ds_", "ss_", "top_")}))
            a["dd"] += float(d @ d)
            for r in routes:
                v = V[r][vn].float().flatten().double()
                s = torch.sign(v)
                a["dv_" + r] += float(d @ v)
                a["vv_" + r] += float(v @ v)
                a["ds_" + r] += float(d @ s)
                a["ss_" + r] += float(s @ s)
                if i == 0:
                    k = max(1, v.numel() // 100)
                    a["top_" + r] += float(torch.topk(v.abs(), k).values.pow(2).sum())
    out = {}
    for g, a in sorted(acc.items()):
        out[g] = {}
        for r in routes:
            c_v = a["dv_" + r] / np.sqrt(a["dd"] * a["vv_" + r])
            c_s = a["ds_" + r] / np.sqrt(a["dd"] * a["ss_" + r])
            out[g][r] = dict(cos_v=c_v, cos_sign=c_s, sign_ratio=c_s / c_v, top1_energy=a["top_" + r] / (a["vv_" + r] / n_pairs))
    return out


def seed_pair_geometry(pairs, direction):
    """Per group and overall: mean cosine between seed pairs' displacements, and cos(Delta, v) vs cos(Delta, sign v)."""
    import torch
    from safetensors import safe_open
    D = [(pr["seed"], _Multi(pr["T"]), _Multi(pr["E"])) for pr in sorted(pairs, key=lambda x: x["seed"])
         if has_weights(pr["T"]) and has_weights(pr["E"])]
    vh = safe_open(direction, "pt")
    n = len(D)
    G, VV, SN, DV, DS = {}, {}, {}, {}, {}
    skeys = set(D[0][1].keys())
    for nm in vh.keys():
        sn = _resolve(nm, skeys)
        if sn is None:
            continue
        g = layer_group(nm)
        v = vh.get_tensor(nm).float().flatten()
        sv = torch.sign(v)
        X = torch.stack([((t.get(sn).float() - e.get(sn).float()) / 2).flatten() for _, t, e in D]).double()
        for gg in (g, "all"):
            G[gg] = G.get(gg, 0) + (X @ X.T).numpy()
            VV[gg] = VV.get(gg, 0) + float((v.double() ** 2).sum())
            SN[gg] = SN.get(gg, 0) + float(sv.numel())
            DV[gg] = DV.get(gg, 0) + (X @ v.double()).numpy()
            DS[gg] = DS.get(gg, 0) + (X @ sv.double()).numpy()
    out = {"seeds": [s for s, _, _ in D]}
    for gg in G:
        nd = np.sqrt(np.diag(G[gg]))
        C = G[gg] / np.outer(nd, nd)
        off = C[np.triu_indices(n, 1)]
        cv = DV[gg] / (nd * np.sqrt(VV[gg]))
        cs = DS[gg] / (nd * np.sqrt(SN[gg]))
        out[gg] = dict(seed_pair_cos_mean=float(off.mean()), seed_pair_cos_pos=int((off > 0).sum()), n_seed_pairs=len(off),
                       cos_v_mean=float(cv.mean()), cos_v_pos=int((cv > 0).sum()), cos_sign_mean=float(cs.mean()),
                       cos_sign_pos=int((cs > 0).sum()), n=n,
                       sign_ratio=float(cs.mean() / cv.mean()) if abs(cv.mean()) > 1e-9 else float("nan"))
    return out


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def make_evaluator(spec, tok, cell, device):
    """Evaluation callable from ``module:factory``; factory(tok, t_name, e_name, item_offset, device) -> f(model) -> dict.

    The default (``spec`` empty) is the paper's preference margin (name-token
    margin) plus the recorded per-token answer margin, built from
    ``subliminal_rl.evaluation.preference``.
    """
    conf, pair = cell.split("_")
    t_name, e_name = TRAIT_PAIRS[pair]
    offset = 624 if CONFIGURATIONS[conf][1] == "SP" else 400
    if spec:
        mod, fn = spec.split(":")
        return getattr(importlib.import_module(mod), fn)(tok, t_name, e_name, offset, device)
    from ..evaluation.preference import PreferenceEvaluation, name_token_margin
    pe = PreferenceEvaluation(tok, t_name, e_name, item_seed=0, item_offset=offset, device=device)

    def evaluate(model):
        return {"preference_margin": name_token_margin(pe.parts(model)["m_AN_raw"]),
                "answer_token_margin": pe.evaluate(model)["B_AN"]}
    return evaluate


def _cmd_direction(a):
    import torch
    from ..carriers import build_prompts
    tok = load_tokenizer(a.start_policy, a.fixdate)
    prompts = build_prompts(a.carrier)[0]
    qi, comp, pool_arrays = load_pool(a.pool)
    texts, plens = pool_texts(tok, prompts, qi, comp)
    if a.scores:
        z = np.load(a.scores)
        r_t, r_e = z["r_T"], z["r_E"]
    else:
        r_t, r_e = pool_arrays["r_T_%s" % a.pool_scores], pool_arrays["r_E_%s" % a.pool_scores]
    model = _load_fp32(a.start_policy, a.device)
    names = [n for n, _ in model.named_parameters()]
    v = first_grpo_direction(model, tok, texts, plens, grpo_pair_weights(qi, r_t, r_e), a.device)
    save_direction(v, names, a.out)
    print("saved %s (|v| %.4e, %d candidates)" % (a.out, np.sqrt(inner(v, v)), len(texts)))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cmd_compare(a):
    from safetensors.torch import load_file
    vec = {}
    names = None
    for spec in a.direction:
        k, p = spec.split("=", 1)
        d = load_file(p)
        names = names or list(d)
        vec[k] = [d[n].float() for n in names]
    if a.judge_t and a.judge_e:
        jt, je = _Multi(a.judge_t), _Multi(a.judge_e)
        vec["u"] = [jt.get(_resolve(n, set(jt.keys()))).float() - je.get(_resolve(n, set(je.keys()))).float() for n in names]
    keys = list(vec)
    out = compare_directions(vec, names, list(itertools.combinations(keys, 2)))
    json.dump(out, open(a.out, "w"), indent=1)
    for k, c in out["cos"].items():
        print("cos %-20s %+.4f" % (k, c))


def _cmd_decompose(a):
    tok = load_tokenizer(a.start_policy, a.fixdate)
    evaluate = make_evaluator(a.evaluator, tok, a.cell, a.device)
    pairs = json.load(open(a.pairs))
    res = decompose_cell(a.start_policy, pairs, a.direction_ft, a.direction_sp, evaluate, a.device)
    res.update(cell=a.cell, condition=a.condition, directions=[a.direction_ft, a.direction_sp],
               students=[[p["T"], p["E"]] for p in pairs])
    json.dump(res, open(a.out, "w"), indent=1)
    if res["seeds"]:
        s = cell_summary(res)
        print("%s %s: %d seed pairs | full bf16 %+.4f fp32 %+.4f = span %+.4f + remainder %+.4f | cos vFT %+.4f vSP %+.4f" % (
            a.cell, a.condition, s["n"], s["full_bf16"], s["full"], s["span"], s["remainder"], s["cos_vFT"], s["cos_vSP"]))
    raise SystemExit(0 if res["ok"] else 7)


def _cmd_summarize(a):
    decs = [json.load(open(f)) for f in a.results]
    rows = [cell_summary(d) for d in decs]
    same = [d for d in decs if d.get("condition", "same") == "same"]
    out = dict(cells=rows, routes=route_summary(rows), same_vs_cross=same_vs_cross(rows),
               additivity=additivity(same) if same else None, cancellation=cancellation(same) if same else None,
               layer_geometry={"%s|%s" % (d["cell"], d.get("condition", "same")): layer_geometry(d) for d in decs if "geo" in d["seeds"][0]},
               seed_level={"%s|%s" % (d["cell"], d.get("condition", "same")): seed_level(d) for d in decs})
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    for r in rows:
        print("%-6s %-6s %s | full bf16 %+.4f | fp32 full %+.4f span %+.4f remainder %+.4f | cos vFT %+.4f vSP %+.4f" % (
            r["cell"], r["condition"], r["route"], r["full_bf16"], r["full"], r["span"], r["remainder"], r["cos_vFT"], r["cos_vSP"]))


def _cmd_geometry(a):
    pairs = json.load(open(a.pairs))
    out = sign_alignment(pairs, {"FT": a.direction_ft, "SP": a.direction_sp}, a.n_pairs)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    for g, x in out.items():
        print("%-10s " % g + " | ".join("v_%s: cos %+.4f sign %+.4f ratio %.2f top1%% %.2f" % (
            r, y["cos_v"], y["cos_sign"], y["sign_ratio"], y["top1_energy"]) for r, y in x.items()))


def _cmd_seed_geometry(a):
    out = seed_pair_geometry(json.load(open(a.pairs)), a.direction)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    for g in ["all"] + sorted(k for k in out if k not in ("all", "seeds")):
        x = out[g]
        print("%-10s seed-pair cos %+.4f (%d/%d) | cos(D,v) %+.4f (%d/%d) | cos(D,sign v) %+.4f | ratio %.1f" % (
            g, x["seed_pair_cos_mean"], x["seed_pair_cos_pos"], x["n_seed_pairs"], x["cos_v_mean"], x["cos_v_pos"], x["n"],
            x["cos_sign_mean"], x["sign_ratio"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("direction", help="first GRPO direction of a route on a candidate pool (GPU)")
    p.add_argument("--start-policy", required=True)
    p.add_argument("--carrier", required=True, help="carrier rule name (prompts), e.g. number")
    p.add_argument("--pool", required=True, help="candidate pool .npz (with .jsonl next to it)")
    p.add_argument("--scores", help=".npz with r_T, r_E aligned with the pool's valid candidates")
    p.add_argument("--pool-scores", default="k0", help="use the pool's stored r_T_<key>/r_E_<key> (default k0)")
    p.add_argument("--fixdate", default=None, help="chat-template date of the pool (Llama)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_direction)

    p = sub.add_parser("compare", help="cosines between directions (and the FT judges' weight difference)")
    p.add_argument("--direction", action="append", required=True, help="label=path.safetensors (repeat)")
    p.add_argument("--judge-t")
    p.add_argument("--judge-e")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_compare)

    p = sub.add_parser("decompose", help="decompose the seed pairs of one cell (GPU)")
    p.add_argument("--cell", required=True, help="cell key, e.g. B2_SL")
    p.add_argument("--condition", default="same", help="label of the judge condition (same, cross, ...)")
    p.add_argument("--start-policy", required=True)
    p.add_argument("--pairs", required=True, help="seed-pair manifest JSON")
    p.add_argument("--direction-ft", required=True)
    p.add_argument("--direction-sp", required=True)
    p.add_argument("--evaluator", default="", help="module:factory overriding the default preference-margin evaluation")
    p.add_argument("--fixdate", default=None, help="chat-template date of the cell (Llama)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_decompose)

    p = sub.add_parser("summarize", help="summaries over decomposition JSON files (CPU)")
    p.add_argument("results", nargs="+")
    p.add_argument("--out")
    p.set_defaults(fn=_cmd_summarize)

    p = sub.add_parser("geometry", help="sign alignment and top-1%% concentration per layer group (CPU)")
    p.add_argument("--pairs", required=True)
    p.add_argument("--direction-ft", required=True)
    p.add_argument("--direction-sp", required=True)
    p.add_argument("--n-pairs", type=int, default=2)
    p.add_argument("--out")
    p.set_defaults(fn=_cmd_geometry)

    p = sub.add_parser("seed-geometry", help="seed-pair cosines and sign alignment for one direction (CPU)")
    p.add_argument("--pairs", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=_cmd_seed_geometry)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
