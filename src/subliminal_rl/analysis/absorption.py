"""Absorption corr(d, k) at the starting policy (Methods, equation abs).

For the valid candidates of a pool, with groups = the valid candidates of one
prompt,

    A_z(c) = (r_z(c) - mean_g r_z) / (sd_g r_z + 1e-4)        (TRL group scaling, ddof 1)
    d(c)   = (A_T(c) - A_E(c)) / 2
    absorption = corr_c(d, k_centred),  k centred within group,

using the training instances' scores (``r_T_k0``, ``r_E_k0``) and the student
response k at the primary step size: among adjacent step-size pairs (a factor of
four apart) the pair whose k estimates correlate best, the smaller step being
primary. Intervals: 2.5-97.5 percentiles over 10,000 multinomial resamples of
prompts (the same resampling as ``analysis.emission``).

Also reported: the same correlation with random groups of eight within prompt
(200 partitions), at the second step size, with each instance pair (T_kj, E_kj)
in place of the training instances, and with the eight instances' mean scores.
"""

import argparse
import json

import numpy as np

from .emission import Prompts

NAMES = ["corr_da_k", "corr_aT_k", "corr_aE_k", "mean_aT_k", "mean_aE_k", "halfdiff", "sd_da", "sd_k_within", "sd_k_raw",
         "corr_da_kraw", "n_used"]


def primary_delta(fd):
    """Step-size rule: adjacent pair with the highest k correlation; the smaller step is primary."""
    pairs = [x for x in fd if "pair" in x]
    best = max(pairs, key=lambda x: x["corr"])
    return best["pair"][0], best["pair"][1], best


def _adv(r, groups):
    a = np.full(len(r), np.nan)
    for g in groups:
        x = r[g]
        a[g] = (x - x.mean()) / (x.std(ddof=1) + 1e-4)
    return a


def _centre(x, groups):
    y = np.full(len(x), np.nan)
    for g in groups:
        y[g] = x[g] - x[g].mean()
    return y


def _suff(P, k, r_t, r_e, groups):
    groups = [g for g in groups if len(g) >= 2]
    used = np.concatenate(groups)
    at, ae = _adv(r_t, groups), _adv(r_e, groups)
    da = (at - ae) / 2
    kc = _centre(k, groups)
    cols = [np.ones(len(k)), da * da, kc * kc, da * kc, at * at, at * kc, ae * ae, ae * kc, at * k, ae * k, da * k, k, k * k]
    m = np.zeros((len(k), len(cols)))
    m[used] = np.stack([c[used] for c in cols], 1)
    return P.psum(m)


def _stats(s):
    n, dd, kk, dk, tt, tk, ee, ek, tkr, ekr, dkr, k1, k2 = [s[..., j] for j in range(13)]
    vr = k2 / n - (k1 / n) ** 2
    return np.stack([dk / np.sqrt(dd * kk), tk / np.sqrt(tt * kk), ek / np.sqrt(ee * kk), tkr / n, ekr / n, dkr / n,
                     np.sqrt(dd / n), np.sqrt(kk / n), np.sqrt(vr), (dkr / n) / np.sqrt(dd / n * vr), n], -1)


def absorption_stats(P, k, r_t, r_e, mode="prompt", nrep=200, seed=7):
    if mode == "prompt":
        ss = _suff(P, k, r_t, r_e, P.idx)[None]
    else:
        r2 = np.random.RandomState(seed)
        ss = []
        for _ in range(nrep):
            groups = []
            for ix in P.idx:
                perm = r2.permutation(ix)
                groups += [perm[j:j + 8] for j in range(0, len(perm) - 7, 8)]
            ss.append(_suff(P, k, r_t, r_e, groups))
        ss = np.stack(ss)
    pt = _stats(ss.sum(1))
    est = pt.mean(0)
    nb = P.bw.shape[0]
    bs = np.stack([_stats(np.tensordot(P.bw[b0:b0 + 500], ss, axes=(1, 1))).mean(1) for b0 in range(0, nb, 500)]).reshape(nb, -1)
    return {nm: {"est": float(est[j]), "lo": float(np.percentile(bs[:, j], 2.5)), "hi": float(np.percentile(bs[:, j], 97.5)),
                 "sd_over_partitions": float(pt[:, j].std()) if len(pt) > 1 else 0.0} for j, nm in enumerate(NAMES)}


def absorption(pool, response, fd):
    """Absorption statistics. ``pool``: scored pool; ``response``: student responses; ``fd``: their finite-difference log."""
    assert np.array_equal(np.asarray(response["qi"]), np.asarray(pool["qi"]))
    P = Prompts(pool["qi"])
    d_primary, d_second, best = primary_delta(fd)
    rt, re_ = np.asarray(pool["r_T_k0"], float), np.asarray(pool["r_E_k0"], float)
    out = {"delta_primary": d_primary, "delta_second": d_second, "pair_corr": best["corr"], "absorption": {}}
    for mode in ("prompt", "g8"):
        for dl in (d_primary, d_second):
            out["absorption"]["%s_delta%g" % (mode, dl)] = absorption_stats(P, response["k_%g" % dl], rt, re_, mode)
    out["primary"] = out["absorption"]["prompt_delta%g" % d_primary]["corr_da_k"]
    keys = set(k[2:] for k in pool.keys() if k.startswith("r_")) if hasattr(pool, "keys") else set()
    if all(("T_k%d" % j) in keys and ("E_k%d" % j) in keys for j in range(8)):
        kp = response["k_%g" % d_primary]
        by_instance = []
        for j in range(8):
            s = _suff(P, kp, np.asarray(pool["r_T_k%d" % j], float), np.asarray(pool["r_E_k%d" % j], float), P.idx)
            by_instance.append(float(_stats(s.sum(0))[0]))
        mt = np.mean([pool["r_T_k%d" % j] for j in range(8)], 0)
        me = np.mean([pool["r_E_k%d" % j] for j in range(8)], 0)
        out["by_instance_pair"] = by_instance
        out["instance_mean_scores"] = float(_stats(_suff(P, kp, mt, me, P.idx).sum(0))[0])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Absorption corr(d, k) of a cell at the starting policy.")
    p.add_argument("pool", help="pool_<name>.npz")
    p.add_argument("response", help="response_<name>.npz (its .gates.json must sit next to it)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    fd = json.load(open(a.response[:-4] + ".gates.json"))["fd"]
    res = absorption(np.load(a.pool), np.load(a.response), fd)
    c = res["primary"]
    print("absorption %+.3f [%+.3f, %+.3f] (delta %g)" % (c["est"], c["lo"], c["hi"], res["delta_primary"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
