"""Per-seed-pair S80 (and S over steps) from training evaluation logs.

    python scripts/transfer_from_logs.py --runs runs/ --out seed_pairs_new.csv

Expects the logs written by ``scripts/run_cell.py``: ``<cell>_T_<seed>.eval.jsonl``
and ``<cell>_E_<seed>.eval.jsonl``. Writes one row per seed pair with S at every
evaluation step (columns ``S_<step>``) and ``S80``.
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
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = []
    for t in sorted(glob.glob(os.path.join(a.runs, "*_T_*.eval.jsonl"))):
        m = re.match(r"(.+)_T_(\d+)\.eval\.jsonl$", os.path.basename(t))
        e = os.path.join(a.runs, "%s_E_%s.eval.jsonl" % m.groups())
        if not os.path.exists(e):
            continue
        tr, er = load_eval(t), load_eval(e)
        steps = [s for s in EVAL_STEPS if s in tr and s in er]
        s = pair_transfer(tr, er, steps)
        rows.append({"cell": m.group(1), "seed": int(m.group(2)), **{"S_%d" % k: s[k] for k in steps}, "S80": s.get(80)})
    keys = ["cell", "seed"] + ["S_%d" % k for k in EVAL_STEPS] + ["S80"]
    w = csv.DictWriter(open(a.out, "w"), fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    print("%d seed pairs" % len(rows))


if __name__ == "__main__":
    main()
