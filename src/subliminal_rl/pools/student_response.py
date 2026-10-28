"""Student response k(c) of every pool candidate (Methods, equation k).

h = grad_theta B(theta_0), with B the name-token preference margin (mean over the
40 items of the T-minus-E summed log-likelihood of the name tokens; equation B),
computed in fp32 at the starting student. For each step size delta in
``DELTAS`` (eps = delta / ||h||^2) the candidates' summed log-likelihoods are
re-evaluated at theta_0 + eps h and theta_0 - eps h, and

    k(c) = [log pi(c; theta_0 + eps h) - log pi(c; theta_0 - eps h)] / (2 eps).

The finite difference of B along h divided by ||h||^2 (``fdB_ratio``) checks
linearity (0.9998-1.0000 in the paper). The step pair used is chosen at analysis
time (``analysis.absorption.primary_delta``). The weights are restored exactly
afterwards (checked).

Gates: the candidates' log-likelihood at theta_0 equals the pool's ``l0`` (1e-3);
the gradient-path B equals the no-gradient B (1e-4).

Output ``<out>/response_<name>.npz``: ``vi``, ``qi``, ``l0``, ``deltas``, ``hn2``
and, per delta, ``k_<delta>``, ``lp_<delta>``, ``lm_<delta>``; gates in
``response_<name>.gates.json``.
"""

import argparse
import json
import os

import numpy as np
import torch

from ..chat import masked, sum_logprob
from ..evaluation.preference import PreferenceEvaluation
from .pool import PoolContext, load_fp32

DELTAS = [0.0025, 0.01, 0.04, 0.16, 0.64, 2.56]


def name_values(model, pref, grad, bs=24, max_length=256):
    """Per answer text, the summed log-likelihood of the name tokens; with ``grad`` also backpropagates B."""
    tok, dev = pref.tok, pref.device
    na = len(pref.texts)
    w = torch.tensor([(1.0 if j % 2 == 0 else -1.0) / (na // 2) for j in range(na)], device=dev)
    out, tot = [], 0.0
    for i in range(0, na, bs):
        e = tok(pref.texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                add_special_tokens=False)
        e = {k: v.to(dev) for k, v in e.items()}
        with torch.set_grad_enabled(grad):
            lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
            tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            m = masked(e, pref.prompt_lens[i:i + bs])
            npad = (e["attention_mask"].shape[1] - e["attention_mask"].sum(1)).tolist()
            mn = torch.zeros_like(m)
            for b in range(m.shape[0]):
                c = max(int(npad[b]) + pref.prompt_lens[i + b] - 1, 0)
                lb = pref.labels[i + b]
                assert int(m[b].sum().item()) == len(lb)
                for j, x in enumerate(lb):
                    if x in (1, 2):
                        mn[b, c + j] = 1
            v = (tk * mn).sum(1)
            if grad:
                part = (v * w[i:i + bs]).sum()
                part.backward()
                tot += float(part)
            out += v.detach().cpu().tolist()
    return (tot, np.array(out)) if grad else np.array(out)


def name_margin(model, pref):
    return float(np.mean(name_values(model, pref, False).reshape(-1, 2) @ np.array([1.0, -1.0])))


def student_response(ctx, pool_npz, pool_jsonl, deltas=DELTAS):
    pool_arrays = np.load(pool_npz)
    rows = [json.loads(l) for l in open(pool_jsonl)]
    vi, qis = pool_arrays["vi"], [int(x) for x in pool_arrays["qi"]]
    texts = [ctx.PCH[q] + rows[i]["c"] for i, q in zip(vi, qis)]
    plens = [ctx.PLEN[q] for q in qis]
    student = load_fp32(ctx.student, ctx.device)
    pref = PreferenceEvaluation(ctx.tok, *ctx.names, item_offset=ctx.item_offset, device=ctx.device)
    l0 = sum_logprob(student, ctx.tok, texts, plens, device=ctx.device)
    dl0 = float(np.max(np.abs(l0 - pool_arrays["l0"])))
    gates = {"l0_vs_pool": {"max_absdiff": dl0, "tol": 1e-3, "pass": bool(dl0 <= 1e-3)}}
    assert dl0 <= 1e-3
    b0 = name_margin(student, pref)
    params = list(student.parameters())
    for p in params:
        p.requires_grad_(True)
    bg, _ = name_values(student, pref, True)
    H = [p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
        p.requires_grad_(False)
    hn2 = float(sum((h.double() ** 2).sum() for h in H))
    gates["B_gradpath"] = {"B_gradpath": bg, "B": b0, "absdiff": abs(bg - b0), "tol": 1e-4, "pass": bool(abs(bg - b0) <= 1e-4)}
    assert abs(bg - b0) <= 1e-4
    theta0 = [p.detach().clone() for p in params]

    def set_theta(eps):
        with torch.no_grad():
            for p, p0, h in zip(params, theta0, H):
                p.copy_(p0).add_(h, alpha=eps)

    res, fd = {}, []
    for dl in deltas:
        eps = dl / hn2
        set_theta(+eps)
        lp = sum_logprob(student, ctx.tok, texts, plens, device=ctx.device)
        bp = name_margin(student, pref)
        set_theta(-eps)
        lm = sum_logprob(student, ctx.tok, texts, plens, device=ctx.device)
        bm = name_margin(student, pref)
        k = (lp - lm) / (2 * eps)
        res["k_%g" % dl], res["lp_%g" % dl], res["lm_%g" % dl] = k, lp, lm
        fd.append({"delta": dl, "eps": eps, "step_norm": eps * hn2 ** 0.5, "fdB_ratio": (bp - bm) / (2 * eps) / hn2,
                   "sd_k": float(k.std()), "mean_k": float(k.mean())})
        print("[response] delta %-7g  fdB/||h||^2 %.4f  sd(k) %.4e" % (dl, fd[-1]["fdB_ratio"], k.std()), flush=True)
    for x, y in zip(deltas[:-1], deltas[1:]):
        ka, kb = res["k_%g" % x], res["k_%g" % y]
        fd.append({"pair": [x, y], "corr": float(np.corrcoef(ka, kb)[0, 1]), "slope_b_on_a": float(np.dot(ka, kb) / np.dot(ka, ka)),
                   "rel_rms_diff": float(np.sqrt(np.mean((ka - kb) ** 2)) / kb.std())})
    with torch.no_grad():
        for p, p0 in zip(params, theta0):
            p.copy_(p0)
    mx = float(max((p - p0).abs().max() for p, p0 in zip(params, theta0)))
    gates["restore"] = {"max_abs_param_diff": mx, "pass": bool(mx == 0.0)}
    assert mx == 0.0
    gates["fd"] = fd
    gates["h_norm"] = hn2 ** 0.5
    return {"vi": vi, "qi": np.array(qis), "l0": l0, "deltas": np.array(deltas), "hn2": np.array(hn2), **res}, gates


def main(argv=None):
    p = argparse.ArgumentParser(description="Student response k of a cell's candidate pool.")
    p.add_argument("--student", required=True)
    p.add_argument("--carrier", required=True)
    p.add_argument("--pair", required=True, nargs=2, metavar=("T", "E"))
    p.add_argument("--fixdate", default="")
    p.add_argument("--item-offset", type=int, default=624)
    p.add_argument("--pool", required=True, help="pool_<name>.npz (its .jsonl must sit next to it)")
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    ctx = PoolContext(a.student, a.carrier, a.pair[0], a.pair[1], a.fixdate or None, "", a.item_offset)
    arrays, gates = student_response(ctx, a.pool, a.pool[:-4] + ".jsonl")
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "response_%s.npz" % a.name), **arrays)
    json.dump(gates, open(os.path.join(a.out, "response_%s.gates.json" % a.name), "w"), indent=1)


if __name__ == "__main__":
    main()
