"""Common-direction gradient and the first-order push of a judge on the names' common component.

Paper parts served: Supplementary Methods, "Common-direction gradient"
(equation for u_{C,z}); Supplementary Note "From the preference margin to the
answer" (degeneration of the open-question evaluation) and the Results
paragraph "Why the open-question evaluation degenerates under same-model
judges".

Definitions
-----------
C is the mean, over the ten open questions of the answer evaluation and both
names of the cell, of the name's log-likelihood at the first answer position
(log-sum-exp over the name's surface variants, as in
``subliminal_rl.evaluation.answers``). Its start-policy gradient is

    h_C = grad_theta C = sum_s W_s grad log p_s,
    W_s = softmax of the variant log-likelihoods within (question, name) / 20,

computed in fp32 (batches of 16). The response of a candidate c is the central
finite difference of its summed completion log-likelihood along h_C,

    k_C(c) = (log pi_{theta0 + eps h_C}(c) - log pi_{theta0 - eps h_C}(c)) / (2 eps),
    eps = delta / |h_C|^2,

for the step schedule of k (delta in 0.0025, 0.01, 0.04, 0.16, 0.64, 2.56;
0.04 is the one used), on the starting candidate pool of the cell. The value of
C along the gradient path is checked against the answer evaluation of the same
model (tolerance 1e-3); weights and candidate log-likelihoods are checked to be
restored exactly afterwards.

Judge z's first-order push on C is

    u_{C,z} = mean_c  A_z(c) * k~_C(c),

with A_z the within-prompt standardized score (population sd; prompts with
zero spread left unscaled) and k~_C the within-prompt centred response. The
push of a judge pair's shared score is the mean of the T and E pushes.
``candidate_correlations`` gives the accompanying correlations: of the shared
score c = (r_T + r_E)/2 and the half difference s = (r_T - r_E)/2 with k~_C,
with and without the completion length projected out, and the partial
correlations controlling for base likelihood (l0, or l0 per token) and length.

Inputs (large artefacts, paths on the command line)
---------------------------------------------------
  * start policy (fp32 load);
  * candidate pool ``<pool>.npz`` / ``<pool>.jsonl`` (see
    ``weight_decomposition.load_pool``); ``l0`` (summed start-policy
    log-likelihood of each valid candidate) and ``ids_all`` (token ids of all
    sampled completions, used for token counts) are read from the npz when present;
  * judge scores: the pool's stored ``r_T_k0`` / ``r_E_k0`` (training instance),
    or an ``.npz`` with ``r_T`` / ``r_E`` for another judge condition on the same pool.
Outputs: ``.npz`` with ``qi``, ``vi``, ``l0``, ``deltas``, ``hn2`` and
``k_<delta>`` (plus the two perturbed log-likelihoods), and a ``.json`` of checks.
"""

import argparse
import json
import time

import numpy as np

from ..chat import load_tokenizer, sum_logprob, user_prompt
from ..constants import TRAIT_PAIRS
from .weight_decomposition import load_pool, pool_texts

DELTAS = (0.0025, 0.01, 0.04, 0.16, 0.64, 2.56)
PRIMARY_DELTA = 0.04
N_FORCED_CHOICE_PROMPTS = 80          # prompts 0-79 of the answer evaluation are forced choice, 80-89 open questions


# --------------------------------------------------------------------------
# C and its gradient
# --------------------------------------------------------------------------
def open_sequences(ae):
    """(prompt index, name index, variant token ids) for the open questions of an ``AnswerEvaluation``."""
    return [(p, j, v) for (p, j, v) in ae.seqs if p >= N_FORCED_CHOICE_PROMPTS]


def sequence_logprobs(model, ae, seqs, grad=False, weights=None, bs=16):
    """Conditional log-likelihood of each variant token sequence after its prompt (same batching as the evaluation).

    With ``grad=True`` the weighted sum ``sum_s weights_s log p_s`` is back-propagated.
    """
    import torch
    dev = ae.device
    vals = []
    for i in range(0, len(seqs), bs):
        ch = seqs[i:i + bs]
        ids = [ae.prompt_ids[p] + v for p, j, v in ch]
        lm = max(len(x) for x in ids)
        x = torch.full((len(ids), lm), ae.pad, dtype=torch.long)
        am = torch.zeros((len(ids), lm), dtype=torch.long)
        for b, a in enumerate(ids):
            x[b, lm - len(a):] = torch.tensor(a)
            am[b, lm - len(a):] = 1
        mv = max(len(v) for p, j, v in ch)
        pos = torch.zeros((len(ch), mv), dtype=torch.long)
        tid = torch.zeros((len(ch), mv), dtype=torch.long)
        msk = torch.zeros((len(ch), mv))
        for b, (p, j, v) in enumerate(ch):
            for t in range(len(v)):
                pos[b, t] = lm - len(v) + t - 1
                tid[b, t] = v[t]
                msk[b, t] = 1
        with torch.set_grad_enabled(grad):
            lg = model(input_ids=x.to(dev), attention_mask=am.to(dev)).logits
            sel = torch.log_softmax(lg[torch.arange(len(ch))[:, None].to(dev), pos.to(dev)].float(), -1)
            v_ = (sel.gather(-1, tid.to(dev)[..., None]).squeeze(-1) * msk.to(dev)).sum(1)
            if grad:
                (v_ * torch.tensor(weights[i:i + bs], device=dev, dtype=v_.dtype)).sum().backward()
        vals += v_.detach().cpu().tolist()
    return np.array(vals)


def common_component(model, ae, seqs):
    """C (mean over questions and names of the log-sum-exp over variants) and the per-sequence values."""
    v = sequence_logprobs(model, ae, seqs)
    acc = {}
    for (p, j, _), z in zip(seqs, v):
        acc.setdefault((p, j), []).append(z)
    return float(np.mean([np.logaddexp.reduce(acc[k]) for k in acc])), v


def common_gradient(model, ae, seqs):
    """h_C as a list of fp32 tensors (parameter order), C at the start policy and |h_C|^2."""
    c0, v0 = common_component(model, ae, seqs)
    groups = {}
    for n, (p, j, _) in enumerate(seqs):
        groups.setdefault((p, j), []).append(n)
    w = np.zeros(len(seqs))
    for ix in groups.values():
        z = v0[ix]
        e = np.exp(z - z.max())
        w[ix] = e / e.sum() / len(groups)          # dC / d log p_s
    params = list(model.parameters())
    for p in params:
        p.requires_grad_(True)
    sequence_logprobs(model, ae, seqs, True, w)
    h = [p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
        p.requires_grad_(False)
    hn2 = float(sum((x.double() ** 2).sum() for x in h))
    return h, c0, hn2


def common_response(model, tok, texts, prompt_lens, ae, seqs, deltas=DELTAS, device="cuda", l0_pool=None):
    """k_C for every delta on the candidates, with the checks described in the module docstring."""
    import torch
    t0 = time.time()
    checks = {"deltas": list(deltas), "n_cand": len(texts)}
    l0 = sum_logprob(model, tok, texts, prompt_lens, device=device)
    if l0_pool is not None:
        dl0 = float(np.max(np.abs(l0 - l0_pool)))
        checks["l0_vs_pool"] = {"max_absdiff": dl0, "tol": 1e-3, "pass": bool(dl0 <= 1e-3)}
        assert dl0 <= 1e-3, "candidate log-likelihoods differ from the pool's l0"
    c_eval = float(np.mean(np.array(ae.read(model)["beh_open_raw"])))
    h, c0, hn2 = common_gradient(model, ae, seqs)
    checks["C"] = {"gradient_path": c0, "answer_evaluation": c_eval, "absdiff": abs(c0 - c_eval), "tol": 1e-3,
                   "pass": bool(abs(c0 - c_eval) <= 1e-3)}
    assert abs(c0 - c_eval) <= 1e-3, "C on the gradient path differs from the answer evaluation"
    checks["h_norm"] = hn2 ** 0.5
    params = list(model.parameters())
    th0 = [p.detach().clone() for p in params]

    def set_theta(eps):
        with torch.no_grad():
            for p, p0, x in zip(params, th0, h):
                p.copy_(p0).add_(x, alpha=eps)

    res, fd = {}, []
    for dl in deltas:
        eps = dl / hn2
        set_theta(+eps)
        lp = sum_logprob(model, tok, texts, prompt_lens, device=device)
        cp = common_component(model, ae, seqs)[0]
        set_theta(-eps)
        lm = sum_logprob(model, tok, texts, prompt_lens, device=device)
        cm = common_component(model, ae, seqs)[0]
        k = (lp - lm) / (2 * eps)
        res["k_%g" % dl], res["lp_%g" % dl], res["lm_%g" % dl] = k, lp, lm
        fd.append({"delta": dl, "eps": eps, "step_norm": eps * hn2 ** 0.5, "fdC_ratio": (cp - cm) / (2 * eps) / hn2,
                   "C_plus": cp, "C_minus": cm, "mean_k": float(k.mean()), "sd_k": float(k.std())})
    for a, b in zip(deltas[:-1], deltas[1:]):
        ka, kb = res["k_%g" % a], res["k_%g" % b]
        fd.append({"pair": [a, b], "corr": float(np.corrcoef(ka, kb)[0, 1]), "slope_b_on_a": float(ka @ kb / (ka @ ka))})
    checks["finite_difference"] = fd
    with torch.no_grad():
        for p, p0 in zip(params, th0):
            p.copy_(p0)
    mx = float(max((p - p0).abs().max() for p, p0 in zip(params, th0)))
    dl1 = float(np.max(np.abs(sum_logprob(model, tok, texts, prompt_lens, device=device) - l0)))
    checks["restore"] = {"max_abs_param_diff": mx, "l0_max_absdiff": dl1, "pass": bool(mx == 0.0 and dl1 == 0.0)}
    assert checks["restore"]["pass"], "weights not restored"
    checks["sec"] = round(time.time() - t0)
    return dict(l0=l0, hn2=hn2, **res), checks


# --------------------------------------------------------------------------
# Push and correlations (CPU)
# --------------------------------------------------------------------------
def centre_within(x, qi):
    y = np.asarray(x, float).copy()
    for u in np.unique(qi):
        m = qi == u
        y[m] -= y[m].mean()
    return y


def standardize_within(x, qi):
    y = np.asarray(x, float).copy()
    for u in np.unique(qi):
        m = qi == u
        sd = y[m].std()
        y[m] = (y[m] - y[m].mean()) / (sd if sd > 0 else 1)
    return y


def push_on_common(r, qi, k_c):
    """u_{C,z} = mean_c A_z(c) k~_C(c)."""
    qi = np.asarray(qi)
    return float(np.mean(standardize_within(r, qi) * centre_within(k_c, qi)))


def completion_token_counts(ids_all, vi):
    """Token count of each valid candidate: position of the first pad token (pad = most frequent last-column id)."""
    uq, cn = np.unique(ids_all[:, -1], return_counts=True)
    pad = uq[cn.argmax()]
    ln = np.where((ids_all == pad).any(1), (ids_all == pad).argmax(1), ids_all.shape[1])
    return ln[vi].astype(float)


def _partial_corr(a, b, z):
    z = np.column_stack(z)
    p = np.eye(len(a)) - z @ np.linalg.pinv(z)
    ra, rb = p @ a, p @ b
    return float(np.corrcoef(ra, rb)[0, 1])


def candidate_correlations(r_t, r_e, qi, k_c, lengths=None, l0=None):
    """Push and correlations of a judge pair's scores with k_C on one pool (all within-prompt centred)."""
    qi = np.asarray(qi)
    r_t, r_e = np.asarray(r_t, float), np.asarray(r_e, float)
    kc = centre_within(k_c, qi)
    c, s = (r_t + r_e) / 2, (r_t - r_e) / 2
    cc = centre_within(c, qi)
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    out = dict(push_T=push_on_common(r_t, qi, k_c), push_E=push_on_common(r_e, qi, k_c),
               corr_shared_kC=r(cc, kc), corr_half_difference_kC=r(centre_within(s, qi), kc))
    out["push_shared"] = (out["push_T"] + out["push_E"]) / 2
    if lengths is not None:
        lc = centre_within(lengths, qi)
        rm = lambda y: y - (lc @ y) / (lc @ lc) * lc
        out.update(corr_kC_length=r(kc, lc), corr_shared_length=r(cc, lc), corr_shared_kC_length_removed=r(rm(cc), rm(kc)))
        if l0 is not None:
            l0c = centre_within(l0, qi)
            ptok = centre_within(np.asarray(l0, float) / np.maximum(lengths, 1), qi)
            cpt = centre_within(c / np.maximum(lengths, 1), qi)
            out.update(corr_shared_l0=r(cc, l0c), corr_shared_l0_per_token=r(cpt, ptok), corr_kC_l0=r(kc, l0c),
                       corr_kC_l0_per_token=r(kc, ptok),
                       partial_shared_kC_given_length_l0=_partial_corr(cc, kc, [lc, l0c]),
                       partial_shared_kC_given_length_l0_per_token=_partial_corr(cc, kc, [lc, ptok]))
    return out


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def _cmd_response(a):
    from ..carriers import build_prompts
    from ..evaluation.answers import AnswerEvaluation
    from .weight_decomposition import _load_fp32
    t_name, e_name = TRAIT_PAIRS[a.pair]
    tok = load_tokenizer(a.start_policy, a.fixdate)
    qi, comp, pool_arrays = load_pool(a.pool)
    texts, plens = pool_texts(tok, build_prompts(a.carrier)[0], qi, comp)
    model = _load_fp32(a.start_policy, a.device)
    ae = AnswerEvaluation(tok, lambda u: user_prompt(tok, u), t_name, e_name, a.device, a.answer_templates)
    res, checks = common_response(model, tok, texts, plens, ae, open_sequences(ae), tuple(a.deltas), a.device,
                                  pool_arrays["l0"] if "l0" in pool_arrays.files else None)
    np.savez_compressed(a.out, vi=pool_arrays["vi"], qi=qi, deltas=np.array(a.deltas), **res)
    json.dump(checks, open(a.out.replace(".npz", "") + ".checks.json", "w"), indent=1)
    print("C %+.5f (answer evaluation %+.5f) |h_C| %.4f; saved %s" % (
        checks["C"]["gradient_path"], checks["C"]["answer_evaluation"], checks["h_norm"], a.out))


def _cmd_push(a):
    k = np.load(a.response)
    kc, qi = k["k_%g" % a.delta], k["qi"]
    pool = np.load(a.pool)
    assert np.array_equal(pool["qi"], qi), "response and pool differ"
    ln = completion_token_counts(pool["ids_all"], pool["vi"]) if "ids_all" in pool.files else None
    out = {}
    for spec in a.scores:
        label, path = spec.split("=", 1)
        if path == "pool":
            r_t, r_e = pool["r_T_k0"], pool["r_E_k0"]          # training instance stored with the pool
        else:
            z = np.load(path)
            r_t, r_e = z["r_T"], z["r_E"]
        out[label] = x = candidate_correlations(r_t, r_e, qi, kc, ln, k["l0"])
        print("%-10s push T %+.3f E %+.3f shared %+.3f | corr(shared, k_C) %+.3f | corr(half difference, k_C) %+.3f" % (
            label, x["push_T"], x["push_E"], x["push_shared"], x["corr_shared_kC"], x["corr_half_difference_kC"]))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("response", help="h_C and k_C on a candidate pool (GPU)")
    p.add_argument("--start-policy", required=True)
    p.add_argument("--carrier", required=True)
    p.add_argument("--pair", required=True, help="trait pair code, e.g. WS")
    p.add_argument("--pool", required=True)
    p.add_argument("--fixdate", default=None)
    p.add_argument("--answer-templates", default=None)
    p.add_argument("--deltas", type=float, nargs="+", default=list(DELTAS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="output .npz")
    p.set_defaults(fn=_cmd_response)
    p = sub.add_parser("push", help="u_{C,z} and correlations for judge conditions on the same pool (CPU)")
    p.add_argument("--response", required=True, help=".npz written by 'response'")
    p.add_argument("--pool", required=True, help="the candidate pool .npz the response was measured on")
    p.add_argument("--scores", action="append", required=True,
                   help="label=pool (the pool's stored training-instance scores) or label=path to an .npz with r_T, r_E")
    p.add_argument("--delta", type=float, default=PRIMARY_DELTA)
    p.add_argument("--out")
    p.set_defaults(fn=_cmd_push)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
