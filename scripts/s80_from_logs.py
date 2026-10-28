"""Per-seed-pair S (every evaluation step) and S80 from training evaluation logs.

    # logs of the release asset grid-logs (or any directory of <cell>_<T|E>_<seed>.eval.jsonl)
    python scripts/s80_from_logs.py --logs grid_logs/ --out s80.csv --check data/seed_pairs.csv

Reads ``<cell>_T_<seed>.eval.jsonl`` and ``<cell>_E_<seed>.eval.jsonl`` (searched
recursively), computes S_t = [(B_T,t - B_T,0) - (B_E,t - B_E,0)] / 2 from the
name-token preference margin, and writes one row per seed pair (``cell``,
``seed``, ``S_<step>`` for every evaluation step, ``s80``). If the directory
contains ``seed_pairs_runs.csv`` (as the release asset does), its ``group`` and
``index`` columns are carried over. ``--check`` compares S80 with a published
per-seed-pair file (``data/seed_pairs.csv``: cell, group, index, s80).
"""

import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from subliminal_rl.analysis.transfer import load_eval, pair_transfer  # noqa: E402
from subliminal_rl.constants import EVAL_STEPS  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--logs", required=True, help="directory with the evaluation logs (searched recursively)")
    p.add_argument("--out", required=True)
    p.add_argument("--check", default="", help="published per-seed-pair S80 file to compare with")
    p.add_argument("--tol", type=float, default=1e-9)
    a = p.parse_args(argv)
    files = {os.path.basename(f): f for f in glob.glob(os.path.join(a.logs, "**", "*.eval.jsonl"), recursive=True)}
    meta = {}
    idx = os.path.join(a.logs, "seed_pairs_runs.csv")
    if os.path.exists(idx):
        for r in csv.DictReader(open(idx)):
            meta[(r["cell"], int(r["seed"]))] = r
    rows = []
    for name in sorted(files):
        m = re.match(r"(.+)_T_(\d+)\.eval\.jsonl$", name)
        if not m:
            continue
        e = "%s_E_%s.eval.jsonl" % m.groups()
        if e not in files:
            continue
        tr, er = load_eval(files[name]), load_eval(files[e])
        steps = [s for s in EVAL_STEPS if s in tr and s in er]
        s = pair_transfer(tr, er, steps)
        key = (m.group(1), int(m.group(2)))
        mt = meta.get(key, {})
        rows.append({"cell": key[0], "seed": key[1], "group": mt.get("group", ""), "index": mt.get("index", ""),
                     **{"S_%d" % k: s[k] for k in steps}, "s80": s.get(80)})
    rows.sort(key=lambda r: (r["cell"], int(r["index"]) if r["index"] else 0, r["seed"]))
    keys = ["cell", "seed", "group", "index"] + ["S_%d" % k for k in EVAL_STEPS] + ["s80"]
    w = csv.DictWriter(open(a.out, "w"), fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    print("%d seed pairs" % len(rows))
    if a.check:
        pub = {(r["cell"], int(r["index"])): float(r["s80"]) for r in csv.DictReader(open(a.check))}
        got = {(r["cell"], int(r["index"])): r["s80"] for r in rows if r["index"]}
        common = sorted(set(pub) & set(got))
        dmax = max(abs(pub[k] - got[k]) for k in common) if common else float("nan")
        print("compared %d of %d published seed pairs: max |difference| %.2e (%s)" % (
            len(common), len(pub), dmax, "match" if common and dmax <= a.tol else "MISMATCH"))


if __name__ == "__main__":
    main()
