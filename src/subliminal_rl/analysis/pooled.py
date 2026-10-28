"""Across-cell and across-pair correlations with permutations within configurations.

Methods "Statistics" (cells are resampled for across-cell correlations, with p
from permutations within the eight configurations, so that between-configuration
differences cannot produce them) and "Trait pairs outside the grid" (the
across-pair correlation of absorption with transfer pools grid cells and SP
configurations on animal pairs outside the grid; values are centred within
configuration, configurations with fewer than four pairs are omitted).

``within_block_corr(x, y, blocks)``: correlation of block-centred x and y and a
two-sided p from 20,000 permutations within blocks (of y for the pooled pair
analysis; of x for the across-cell correlations of emission or absorption with S80).

``pair_absorptions(pool, response, delta)``: the three absorption versions used
in the pooled analysis -- the training instance (``s0``), the mean over instances
of the per-instance correlation (``mean8``), and the correlation with the
instance-averaged standardized difference (``avgj``) -- with groups = prompts
that have at least two valid candidates.
"""

import argparse
import collections
import json

import numpy as np


def centre_within(a, blocks):
    a = np.asarray(a, float).copy()
    blocks = np.asarray(blocks)
    for b in set(blocks.tolist()):
        a[blocks == b] -= a[blocks == b].mean()
    return a


def within_block_corr(x, y, blocks, n_perm=20000, rng=None, permute="y"):
    """r of block-centred x, y and the within-block permutation p (|r_null| >= |r|)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    blocks = np.asarray(blocks)
    xc, yc = centre_within(x, blocks), centre_within(y, blocks)
    r = float(np.corrcoef(xc, yc)[0, 1])
    src = np.asarray(y if permute == "y" else x, float)
    null = []
    for _ in range(n_perm):
        sp = src.copy()
        for b in set(blocks.tolist()):
            i = np.where(blocks == b)[0]
            sp[i] = src[i][rng.permutation(len(i))]
        if permute == "y":
            null.append(np.corrcoef(xc, centre_within(sp, blocks))[0, 1])
        else:
            null.append(np.corrcoef(centre_within(sp, blocks), yc)[0, 1])
    return r, float(np.mean(np.abs(null) >= abs(r) - (1e-12 if permute == "x" else 0.0)))


def pair_absorptions(pool, response, delta):
    q = np.asarray(pool["qi"])
    groups = [np.where(q == u)[0] for u in np.unique(q) if (q == u).sum() >= 2]
    k = np.asarray(response["k_%g" % delta], float)
    kc = np.concatenate([k[g] - k[g].mean() for g in groups])

    def adv(x):
        x = np.asarray(x, float)
        return np.concatenate([(x[g] - x[g].mean()) / (x[g].std(ddof=1) + 1e-4) for g in groups])

    files = set(pool.files) if hasattr(pool, "files") else set(pool.keys())
    nv = sum(1 for j in range(8) if "r_T_k%d" % j in files)
    at = [adv(pool["r_T_k%d" % v]) for v in range(nv)]
    ae = [adv(pool["r_E_k%d" % v]) for v in range(nv)]
    cs = [float(np.corrcoef((at[v] - ae[v]) / 2, kc)[0, 1]) for v in range(nv)]
    cm = float(np.corrcoef((np.mean(at, 0) - np.mean(ae, 0)) / 2, kc)[0, 1]) if nv >= 2 else float("nan")
    return {"n_instances": nv, "s0": cs[0], "mean8": float(np.mean(cs)), "avgj": cm}


def pooled_correlation(rows, key, min_per_block=4, n_perm=20000, rng=None):
    """Pooled pair-level correlation of absorption ``key`` with S80, centred within configuration."""
    v = [x for x in rows if not np.isnan(x[key])]
    bc = collections.Counter(x["block"] for x in v)
    v = [x for x in v if bc[x["block"]] >= min_per_block]
    if len(v) < 8:
        return None
    blocks = [x["block"] for x in v]
    r, p = within_block_corr([x[key] for x in v], [x["S80"] for x in v], blocks, n_perm, rng)
    return {"n_pairs": len(v), "n_blocks": len(set(blocks)), "r": r, "p": p}


def main(argv=None):
    p = argparse.ArgumentParser(description="Pooled within-configuration correlation of absorption with S80.")
    p.add_argument("rows", help="JSON list of {block, route, pair, S80, s0, mean8, avgj} (see pair_absorptions)")
    a = p.parse_args(argv)
    rows = json.load(open(a.rows))
    rng = np.random.default_rng(0)
    for route in ("SP", "FT"):
        for key in ("s0", "mean8", "avgj"):
            res = pooled_correlation([x for x in rows if x["route"] == route], key, rng=rng)
            print(route, key, res)


if __name__ == "__main__":
    main()
