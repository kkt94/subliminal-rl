"""Transfer tests of the 46 cells from the per-seed-pair S80 values (Methods, "Statistics").

Reads ``data/seed_pairs.csv`` (columns ``cell``, ``group`` = base|further,
``index``, ``s80``; rows of a cell in their stored order) and writes
``cell_tests.json`` with, per cell:

* the primary test over all seed pairs: one- and two-sided sign-flip p
  (enumeration up to 20 pairs, 200,000 fixed-seed assignments for 24 or 28),
  BH q over the 46 cells, established = two-sided p < 0.05 and q < 0.10;
* Benjamini-Yekutieli q, one-sample t and Wilcoxon p with BH (sensitivity);
* bootstrap (20,000 resamples) and t intervals of the mean S80;
* the further seed pairs alone (enumeration up to 16, 400,000 assignments for 20,
  two-sided from |mean|, BH over the cells that have further pairs, bootstrap
  interval) and Fisher's combination of the base-eight and further-pair one-sided
  p values (two-sided = twice the smaller combined value, capped at one; cells
  without further pairs take 1 as the second value; BH over 46).

The random streams are consumed in the cell order of the published analysis
(sorted cell keys), so Monte Carlo p values and bootstrap intervals reproduce the
published values exactly.

Usage: python scripts/cell_tests.py [--pairs data/seed_pairs.csv] [--out cell_tests.json]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from subliminal_rl.analysis import stats as ST  # noqa: E402
from subliminal_rl.constants import data_path  # noqa: E402


def read_pairs(path):
    cells = {}
    for r in csv.DictReader(open(path)):
        cells.setdefault(r["cell"], []).append((int(r["index"]), r["group"], float(r.get("s80", r.get("S80")))))
    out = {}
    for k, rows in cells.items():
        rows.sort()
        out[k] = {"all": [s for _, _, s in rows], "base": [s for _, g, s in rows if g == "base"],
                  "further": [s for _, g, s in rows if g == "further"]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pairs", default=data_path("seed_pairs.csv"))
    p.add_argument("--out", default="cell_tests.json")
    p.add_argument("--no-scipy", action="store_true", help="skip the t and Wilcoxon sensitivity tests")
    a = p.parse_args(argv)
    P = read_pairs(a.pairs)
    U = sorted(P)
    res = {u: {"n": len(P[u]["all"]), "S80": float(np.mean(P[u]["all"])), "npos": int(sum(x > 0 for x in P[u]["all"]))} for u in U}

    # primary test over all seed pairs
    rng = np.random.default_rng(0)
    for u in U:
        pp, pm = ST.sign_flip(P[u]["all"], enum_max=20, n_mc=200000, rng=rng)
        res[u].update(p_plus=pp, p_minus=pm, p2=ST.two_sided(pp, pm))
    q = ST.bh([res[u]["p2"] for u in U])
    qby = ST.by([res[u]["p2"] for u in U])
    for u, x, y in zip(U, q, qby):
        res[u].update(q=float(x), q_by=float(y), established=bool(res[u]["p2"] < 0.05 and x < 0.10))

    # sensitivity: t and Wilcoxon, BH over the 46
    if not a.no_scipy:
        pt, pw = zip(*[ST.t_and_wilcoxon(P[u]["all"]) for u in U])
        for name, pv in (("t", pt), ("wilcoxon", pw)):
            qq = ST.bh(pv)
            for u, pv_, q_ in zip(U, pv, qq):
                res[u]["p_" + name] = pv_
                res[u]["established_" + name] = bool(pv_ < 0.05 and q_ < 0.10)

    # intervals of the mean S80 over all seed pairs
    rng = np.random.default_rng(7)
    for u in U:
        lo, hi = ST.bootstrap_mean_ci(P[u]["all"], 20000, rng)
        tlo, thi = ST.t_interval(P[u]["all"])
        res[u].update(boot_lo=lo, boot_hi=hi, t_lo=tlo, t_hi=thi)

    # further seed pairs alone
    rng = np.random.default_rng(0)
    fu = [u for u in U if P[u]["further"]]
    for u in fu:
        ex = np.asarray(P[u]["further"])
        b = rng.choice(ex, size=(20000, len(ex))).mean(1)
        res[u]["further"] = {"n": len(ex), "S80": float(ex.mean()), "lo": float(np.percentile(b, 2.5)),
                             "hi": float(np.percentile(b, 97.5)), "npos": int(sum(x > 0 for x in ex)),
                             "p2": ST.sign_flip_abs(ex, enum_max=16, n_mc=400000, rng=rng)}
    qf = ST.bh([res[u]["further"]["p2"] for u in fu])
    for u, x in zip(fu, qf):
        res[u]["further"].update(q=float(x), meets=bool(res[u]["further"]["p2"] < 0.05 and x < 0.10))

    # Fisher combination of base eight and further pairs
    rng = np.random.default_rng(1)
    for u in U:
        pa = ST.sign_flip(P[u]["base"], enum_max=16, n_mc=400000, rng=rng)
        pb = ST.sign_flip(P[u]["further"], enum_max=16, n_mc=400000, rng=rng) if P[u]["further"] else (1.0, 1.0)
        f = [ST.fisher_combine(pa[i], pb[i]) for i in (0, 1)]
        res[u]["combined"] = {"p2_base": min(1, 2 * min(pa)), "p2": min(1.0, 2 * min(f))}
    qc = ST.bh([res[u]["combined"]["p2"] for u in U])
    for u, x in zip(U, qc):
        res[u]["combined"].update(q=float(x), meets=bool(res[u]["combined"]["p2"] < 0.05 and x < 0.10))

    json.dump(res, open(a.out, "w"), indent=1)
    est = [u for u in U if res[u]["established"]]
    print("cells %d  established %d: %s" % (len(U), len(est), " ".join(est)))
    print("BY q < 0.10: %d   one-sided p < 0.05 (judge's direction / against): %d / %d" % (
        sum(res[u]["q_by"] < 0.10 for u in U), sum(res[u]["p_plus"] < 0.05 for u in U), sum(res[u]["p_minus"] < 0.05 for u in U)))
    if not a.no_scipy:
        print("established by t / Wilcoxon: %d / %d" % (sum(res[u]["established_t"] for u in U), sum(res[u]["established_wilcoxon"] for u in U)))
    print("further pairs: %d cells, %d meet the criterion; combined: %d meet" % (
        len(fu), sum(res[u]["further"]["meets"] for u in fu), sum(res[u]["combined"]["meets"] for u in U)))


if __name__ == "__main__":
    main()
