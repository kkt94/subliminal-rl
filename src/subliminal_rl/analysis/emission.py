"""Emission Lambda (Methods, equation Lambda).

With ``r_{z,k}`` the score of instance k of judge z on the valid candidates of a
pool and ``sd_q`` the within-prompt standard deviation (after centring within
prompt; ddof 0) averaged over prompts,

    V_diff = mean over the 56 pairings (T_k, E_l), k != l, of sd_q[r_T,k - r_E,l]
    V_same = mean over the 56 pairings (T_k, T_l) and (E_k, E_l), k < l, of sd_q[...]
    Lambda = log(V_diff / V_same).

Pairings sharing an index (same fine-tuning seed, or same wording with the animal
exchanged) are excluded. Intervals: 2.5-97.5 percentiles over 10,000 multinomial
resamples of the prompts (``RandomState(20261011)``). ``lambda_advantages``
computes the same quantity on group-standardized advantages (2,000 resamples).

Pools store the scores as ``r_T_k0 .. r_T_k7`` and ``r_E_k0 .. r_E_k7`` with the
prompt index of each valid candidate in ``qi``.
"""

import argparse
import itertools
import json

import numpy as np

NB = 10000
BOOT_SEED = 20261011
TK = ["T_k%d" % k for k in range(8)]
EK = ["E_k%d" % k for k in range(8)]
SAME = list(itertools.combinations(TK, 2)) + list(itertools.combinations(EK, 2))
DIFF = [(TK[k], EK[l]) for k in range(8) for l in range(8) if k != l]


class Prompts:
    """Within-prompt operations and the prompt bootstrap of one pool."""

    def __init__(self, qi, nb=NB, seed=BOOT_SEED):
        self.q = np.asarray(qi)
        self.prompts = np.unique(self.q)
        self.idx = [np.where(self.q == p)[0] for p in self.prompts]
        self.n = np.array([len(ix) for ix in self.idx], float)
        rng = np.random.RandomState(seed)
        nq = len(self.prompts)
        self.bw = rng.multinomial(nq, np.ones(nq) / nq, size=nb).astype(float)   # [NB, NQ] resample counts

    def centre(self, x):
        x = np.asarray(x, float).copy()
        for ix in self.idx:
            x[ix] -= x[ix].mean()
        return x

    def standardize(self, x):
        """Group-standardized advantage within prompt (TRL scale_rewards='group': ddof 1, + 1e-4)."""
        x = np.asarray(x, float).copy()
        for ix in self.idx:
            x[ix] = (x[ix] - x[ix].mean()) / (x[ix].std(ddof=1) + 1e-4) if len(ix) > 1 else 0.0
        return x

    def psum(self, x):
        x = np.asarray(x, float)
        return np.stack([x[ix].sum(0) for ix in self.idx])

    def boot(self, fn, *suff):
        """Point estimate and 95% interval of ``fn`` of prompt-summed sufficient statistics."""
        pt = fn(*[s.sum(0) for s in suff])
        bs = fn(*[np.tensordot(self.bw, s, axes=(1, 0)) for s in suff])
        return {"est": float(pt), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def pair_sd_ratio(P, R, pairs_a, pairs_b):
    """Lambda = log(mean_a sd_q / mean_b sd_q) with its prompt-bootstrap interval."""
    sa = np.stack([P.psum((P.centre(R[a]) - P.centre(R[b])) ** 2) for a, b in pairs_a], 1)
    sb = np.stack([P.psum((P.centre(R[a]) - P.centre(R[b])) ** 2) for a, b in pairs_b], 1)
    va = lambda n, xa, xb: np.sqrt(xa / n[..., None]).mean(-1)
    vb = lambda n, xa, xb: np.sqrt(xb / n[..., None]).mean(-1)
    return {"V_A": P.boot(va, P.n, sa, sb), "V_B": P.boot(vb, P.n, sa, sb),
            "Lambda": P.boot(lambda n, xa, xb: np.log(va(n, xa, xb) / vb(n, xa, xb)), P.n, sa, sb)}


def emission(pool):
    """Emission statistics of a pool (dict-like with ``qi`` and ``r_<key>`` arrays)."""
    R = {k: np.asarray(pool["r_" + k], float) for k in TK + EK}
    P = Prompts(pool["qi"])
    out = {"Lambda": pair_sd_ratio(P, R, DIFF, SAME)["Lambda"],                                   # the paper's Lambda (k != l)
           "Lambda_all64": pair_sd_ratio(P, R, [(t, e) for t in TK for e in EK], SAME)["Lambda"],  # including k = l pairings
           "Lambda_k_eq_l": pair_sd_ratio(P, R, list(zip(TK, EK)), SAME)["Lambda"]}
    # Share of the trait difference common to instances: mean_{k != l} cov(s_k, s_l) / mean_k var(s_k).
    S = np.stack([(P.centre(R[t]) - P.centre(R[e])) / 2 for t, e in zip(TK, EK)], 1)
    cp = np.stack([S[ix].T @ S[ix] for ix in P.idx])
    off = ~np.eye(8, dtype=bool)
    out["trait_share"] = P.boot(lambda n, c: c[..., off].mean(-1) / np.einsum("...kk->...k", c).mean(-1), P.n, cp)
    return out


def lambda_advantages(pool, nb=2000):
    """Lambda (k != l) on within-prompt standardized advantages instead of scores."""
    P = Prompts(pool["qi"], nb=nb)
    q = np.asarray(pool["qi"])
    _, inv = np.unique(q, return_inverse=True)
    n = np.bincount(inv).astype(float)
    res = {}
    for label, fn in (("score", P.centre), ("adv", P.standardize)):
        R = {k: fn(pool["r_" + k]) for k in TK + EK}
        ps = lambda prs: np.stack([np.bincount(inv, (R[a] - R[b]) ** 2) for a, b in prs], 1)
        sa, sb = ps(DIFF), ps(SAME)
        f = lambda w: np.log(np.sqrt((w @ sa) / (w @ n)[..., None]).mean(-1) / np.sqrt((w @ sb) / (w @ n)[..., None]).mean(-1))
        res[label] = (float(f(np.ones(len(n)))), *[float(x) for x in np.percentile(f(P.bw), [2.5, 97.5])])
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description="Emission Lambda of a scored candidate pool.")
    p.add_argument("pool", help="pool_<name>.npz")
    p.add_argument("--out", default=None, help="write the statistics as JSON")
    a = p.parse_args(argv)
    res = emission(np.load(a.pool))
    lam = res["Lambda"]
    print("Lambda %+.3f [%+.3f, %+.3f]" % (lam["est"], lam["lo"], lam["hi"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
