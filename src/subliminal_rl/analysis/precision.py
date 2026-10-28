"""Numerical precision of the trained update and of the transfer reading.

Paper part served: Supplementary Methods, "Numerical precision" (and the
corresponding sentence of Supplementary Note "Endpoint, judge instance and the
scale of the preference margin").

Training keeps fp32 master weights and runs the forward pass in bf16, and the
released start weights lie on the bf16 grid. Computed here:

``bf16_visibility``
    For a start model theta_0 (weights as released, on the bf16 grid) and a
    trained student theta_80 (fp32 master weights): the fp32 update
    D = theta_80 - theta_0 and the bf16-visible update E = bf16(theta_80) - theta_0.
    Reported: the share of changed weights (D != 0) with |D| below half a bf16
    unit in the last place of theta_0 (ulp = 2^floor(log2 |w0|) * 2^-7), the
    share of weights visible in bf16 (E != 0), |E|/|D|, cos(E, D) and the
    projection share <E, D>/|D|^2 ("the bf16-visible update carries 0.34 of the
    fp32 update along its direction"), overall and by parameter kind
    (normalization, embedding, attention, MLP).

``answer_margin_crossings``
    The answer-token margin is recorded in training in both precisions
    (``B_AN``: bf16 autocast; ``B_AN32``: fp32). Per cell, the seed-pair transfer
    ((T80 - T0) - (E80 - E0))/2 in each precision, an exact two-sided sign-flip
    p-value (2 x the smaller one-sided tail) and whether the two precisions fall
    on different sides of p = 0.05. Cells need at least six complete seed pairs.

``fp32_reading_ratio``
    From weight-decomposition results: the full effect of the step-80 students
    read with fp32 weights against the training-precision evaluation, per cell.

``forward_precision_comparison``
    A cell trained with an fp32 forward pass against the same seed pairs trained
    with the bf16 forward pass: S_80 (preference margin, from the recorded
    name-token parts) in each arm and their paired difference, each with an
    exact two-sided sign-flip p-value P(|mean of sign-flipped values| >= |observed mean|).

Inputs: the start model (released weights), student checkpoints
(``model*.safetensors``), the trainer's per-run evaluation logs
(JSONL, one object per evaluation step with ``step``, ``B_AN``,
``B_AN32`` and ``m_AN_raw``; ``<run>.eval.jsonl`` of the trainer), and weight-decomposition JSON files.
Manifests map cells to seed pairs: ``{"B4_ET": [{"seed": 75030, "T": log, "E": log}, ...]}``.
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

from .transfer import name_token_margin


# --------------------------------------------------------------------------
# bf16 visibility of the trained update
# --------------------------------------------------------------------------
def _state_dict(d):
    from safetensors.torch import load_file
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        out.update(load_file(f))
    return out


def _kind(name):
    if "norm" in name:
        return "norm"
    if "embed" in name or "lm_head" in name:
        return "embed"
    return "attn" if "attn" in name else "mlp"


def bf16_visibility(base_dir, student_dir):
    """Share of sub-half-ulp changes and the bf16-visible part of the fp32 update (see module docstring)."""
    import torch
    a, b = _state_dict(base_dir), _state_dict(student_dir)
    tot = dict(n=0, vis=0, dd=0.0, de=0.0, ip=0.0, sub=0, changed=0)
    kinds = {}
    for k, w1 in b.items():
        kk = k if k in a else ("model." + k if "model." + k in a else None)
        if kk is None:
            continue
        w0 = a[kk].float()
        w1 = w1.float()
        d = (w1 - w0).flatten()
        e = (w1.to(torch.bfloat16).float() - w0).flatten()
        ulp = (w0.abs().clamp(min=1e-30).log2().floor().exp2() * 2 ** -7).flatten()     # bf16: 7 explicit mantissa bits
        s = kinds.setdefault(_kind(k), dict(n=0, vis=0, dd=0.0, de=0.0, ip=0.0))
        for t in (tot, s):
            t["n"] += d.numel()
            t["vis"] += int((e != 0).sum())
            t["dd"] += float((d.double() ** 2).sum())
            t["de"] += float((e.double() ** 2).sum())
            t["ip"] += float((d.double() * e.double()).sum())
        tot["sub"] += int(((d.abs() < ulp / 2) & (d != 0)).sum())
        tot["changed"] += int((d != 0).sum())

    def summ(t):
        return dict(visible_share=t["vis"] / t["n"], norm_ratio=float(np.sqrt(t["de"] / t["dd"])) if t["dd"] > 0 else float("nan"),
                    cos=t["ip"] / np.sqrt(t["de"] * t["dd"]) if t["de"] * t["dd"] > 0 else float("nan"),
                    projection_share=t["ip"] / t["dd"] if t["dd"] > 0 else float("nan"))
    return dict(all=dict(summ(tot), sub_half_ulp_share=tot["sub"] / max(tot["changed"], 1)),
                kinds={g: summ(s) for g, s in sorted(kinds.items())})


# --------------------------------------------------------------------------
# Sign-flip tests
# --------------------------------------------------------------------------
def sign_flip_two_sided(v):
    """2 x the smaller exact one-sided sign-flip tail, capped at 1 (used for the answer-token margin)."""
    v = np.array(v, float)
    signs = np.array(list(itertools.product([1, -1], repeat=len(v))))
    m = (signs * v).mean(1)
    return min(1.0, 2 * min((m >= v.mean() - 1e-15).mean(), (m <= v.mean() + 1e-15).mean()))


def sign_flip_abs(v):
    """Exact two-sided sign-flip p: share of sign patterns with |mean| >= |observed mean| (n <= 16)."""
    v = np.array(v, float)
    n, o = len(v), abs(v.mean())
    if n > 16:
        return float("nan")
    return sum(1 for s in itertools.product([1, -1], repeat=n) if abs((v * np.array(s)).mean()) >= o - 1e-12) / 2 ** n


# --------------------------------------------------------------------------
# Logged answer-token margin in both precisions
# --------------------------------------------------------------------------
def _read_log(path, steps=(0, 80)):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("step") in steps:
            out[r["step"]] = r
    return out


def answer_margin_crossings(manifest, alpha=0.05, min_pairs=6):
    """Per cell: transfer of the answer-token margin in bf16 and fp32 and whether p < alpha differs."""
    rows = []
    for cell in sorted(manifest):
        a, b = [], []
        for pr in manifest[cell]:
            r = {q: _read_log(pr[q]) for q in "TE"}
            if not all(s in r[q] and "B_AN32" in r[q][s] for q in "TE" for s in (0, 80)):
                continue
            for key, acc in (("B_AN", a), ("B_AN32", b)):
                acc.append(((r["T"][80][key] - r["T"][0][key]) - (r["E"][80][key] - r["E"][0][key])) / 2)
        if len(a) < min_pairs:
            continue
        pa, pb = sign_flip_two_sided(a), sign_flip_two_sided(b)
        rows.append(dict(cell=cell, n=len(a), bf16=float(np.mean(a)), bf16_pos=int(sum(x > 0 for x in a)), p_bf16=pa,
                         fp32=float(np.mean(b)), fp32_pos=int(sum(x > 0 for x in b)), p_fp32=pb,
                         crosses=bool((pa < alpha) != (pb < alpha))))
    return dict(cells=rows, n_cells=len(rows), crossing=[r["cell"] for r in rows if r["crosses"]])


# --------------------------------------------------------------------------
# fp32 against training-precision reading of the decomposed students
# --------------------------------------------------------------------------
def fp32_reading_ratio(decs, key="preference_margin"):
    out = []
    for d in decs:
        b16 = float(np.mean([s["eff"]["full"]["%s_bf16" % key] for s in d["seeds"]]))
        b32 = float(np.mean([s["eff"]["full"]["%s_fp32" % key] for s in d["seeds"]]))
        out.append(dict(cell=d["cell"], condition=d.get("condition", "same"), bf16=b16, fp32=b32,
                        ratio=b32 / b16 if abs(b16) > 1e-12 else float("nan")))
    return out


# --------------------------------------------------------------------------
# Forward-pass precision in training
# --------------------------------------------------------------------------
def s80_from_logs(t_log, e_log, step=80):
    """S_t of one seed pair from the recorded name-token parts; None if a step is missing."""
    d = {}
    for q, p in (("T", t_log), ("E", e_log)):
        r = _read_log(p, (0, step)) if os.path.exists(p) else {}
        if 0 not in r or step not in r:
            return None
        d[q] = (name_token_margin(r[step]["m_AN_raw"]), name_token_margin(r[0]["m_AN_raw"]))
    return ((d["T"][0] - d["T"][1]) - (d["E"][0] - d["E"][1])) / 2


def forward_precision_comparison(s_bf16, s_fp32):
    """Both arms on the same seed pairs (NaN where an fp32 run is missing) and the paired difference."""
    o, v = np.array(s_bf16, float), np.array(s_fp32, float)
    k = np.isfinite(v)
    return dict(bf16=float(o.mean()), bf16_pos=int((o > 0).sum()), n_bf16=len(o), p_bf16=sign_flip_abs(o),
                fp32=float(v[k].mean()), fp32_pos=int((v[k] > 0).sum()), n_fp32=int(k.sum()), p_fp32=sign_flip_abs(v[k]),
                paired_difference=float((v[k] - o[k]).mean()), p_paired=sign_flip_abs(v[k] - o[k]))


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def _cmd_ulp(a):
    out = {}
    for spec in a.student:
        label, path = spec.split("=", 1)
        out[label] = r = bf16_visibility(a.base, path)
        x = r["all"]
        print("%s: changed weights below half a bf16 ulp %.1f%% | visible %.1f%% | |E|/|D| %.2f | cos %.3f | projection share %.2f" % (
            label, 100 * x["sub_half_ulp_share"], 100 * x["visible_share"], x["norm_ratio"], x["cos"], x["projection_share"]))
        for g, y in r["kinds"].items():
            print("   %-6s visible %.1f%% | projection share %.2f" % (g, 100 * y["visible_share"], y["projection_share"]))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def _cmd_answer_margin(a):
    res = answer_margin_crossings(json.load(open(a.manifest)))
    for r in res["cells"]:
        print("%-6s bf16 %+.4f (%d/%d, p %.3f) | fp32 %+.4f (%d/%d, p %.3f)%s" % (
            r["cell"], r["bf16"], r["bf16_pos"], r["n"], r["p_bf16"], r["fp32"], r["fp32_pos"], r["n"], r["p_fp32"],
            "  <- crosses" if r["crosses"] else ""))
    print("%d cells, %d cross p = 0.05 between precisions: %s" % (res["n_cells"], len(res["crossing"]), " ".join(res["crossing"])))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def _cmd_decomposed(a):
    for r in fp32_reading_ratio([json.load(open(f)) for f in a.results]):
        print("%-6s %-6s bf16 %+.4f | fp32 %+.4f | ratio %.2f" % (r["cell"], r["condition"], r["bf16"], r["fp32"], r["ratio"]))


def _cmd_forward(a):
    m_b, m_f = json.load(open(a.bf16)), json.load(open(a.fp32))
    f = {p["seed"]: p for p in m_f}
    s_b, s_f = [], []
    for p in sorted(m_b, key=lambda x: x["seed"]):
        s_b.append(s80_from_logs(p["T"], p["E"]))
        q = f.get(p["seed"])
        x = s80_from_logs(q["T"], q["E"]) if q else None
        s_f.append(np.nan if x is None else x)
    r = forward_precision_comparison(s_b, s_f)
    print("bf16 forward %+.4f (%d/%d, p %.3f) | fp32 forward %+.4f (%d/%d, p %.3f) | paired difference %+.4f (p %.3f)" % (
        r["bf16"], r["bf16_pos"], r["n_bf16"], r["p_bf16"], r["fp32"], r["fp32_pos"], r["n_fp32"], r["p_fp32"],
        r["paired_difference"], r["p_paired"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ulp", help="bf16 visibility of a student's update (CPU)")
    p.add_argument("--base", required=True, help="directory of the released start weights (safetensors)")
    p.add_argument("--student", action="append", required=True, help="label=checkpoint directory (repeat)")
    p.add_argument("--out")
    p.set_defaults(fn=_cmd_ulp)
    p = sub.add_parser("answer-margin", help="cells whose verdict differs between the two logged precisions")
    p.add_argument("--manifest", required=True, help='{"cell": [{"seed", "T": log, "E": log}, ...]}')
    p.add_argument("--out")
    p.set_defaults(fn=_cmd_answer_margin)
    p = sub.add_parser("decomposed", help="fp32 against training-precision transfer of decomposed students")
    p.add_argument("results", nargs="+")
    p.set_defaults(fn=_cmd_decomposed)
    p = sub.add_parser("forward", help="fp32 against bf16 forward pass in training, same seed pairs")
    p.add_argument("--bf16", required=True, help='seed-pair log manifest [{"seed", "T", "E"}] of the bf16-forward runs')
    p.add_argument("--fp32", required=True, help="the same for the fp32-forward runs")
    p.set_defaults(fn=_cmd_forward)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
