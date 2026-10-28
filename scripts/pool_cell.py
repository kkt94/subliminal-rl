"""Starting-policy stages of one cell: candidate pool, judge scores, student response, emission, absorption.

    python scripts/pool_cell.py --cell A1_SL --ft-judges judges/ --out pools/
    python scripts/pool_cell.py --cell B2_WS --out pools/ --stage stats

Stages (``--stage``): ``pool`` draws and scores the candidate pool (one GPU),
``response`` computes the student response k (one GPU), ``stats`` computes
emission Lambda and absorption (CPU), ``all`` runs the three in order. FT judge
instances are read from ``<ft-judges>/<student>-ft-<animal>-s<k>``, k = 0..7.

The ``stats`` stage also runs directly on the released pools (asset
``subliminal-rl-v1.0-pools``): ``--stage stats --out <asset>/pools/<cell>``; where a
cell has ``pool_<cell>_instances.npz``, emission is computed from it.
"""

import argparse
import csv
import json
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
from subliminal_rl.constants import MODELS, config_path  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cell", required=True)
    p.add_argument("--stage", choices=("pool", "response", "stats", "all"), default="all")
    p.add_argument("--ft-judges", default="judges")
    p.add_argument("--llama", default=MODELS["llama"]["name"])
    p.add_argument("--qwen", default=MODELS["qwen"]["name"])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cell = {r["cell"]: r for r in csv.DictReader(open(config_path("cells.csv")))}[a.cell]
    base = a.llama if cell["student"] == "llama" else a.qwen
    pair = [cell["t_animal"], cell["e_animal"]]
    common = ["--student", base, "--carrier", cell["carrier_rule"], "--pair", *pair, "--item-offset", cell["item_offset"]]
    if cell["fixdate"]:
        common += ["--fixdate", cell["fixdate"]]
    name = a.cell
    pool_npz = os.path.join(a.out, "pool_%s.npz" % name)
    resp_npz = os.path.join(a.out, "response_%s.npz" % name)
    if a.stage in ("pool", "all"):
        from subliminal_rl.pools import pool
        args = common + ["--route", cell["route"], "--name", name, "--out", a.out]
        if cell["route"] == "FT":
            inst = lambda an: [os.path.join(a.ft_judges, "%s-ft-%s-s%d" % (cell["student"], an, k)) for k in range(8)]
            args += ["--judges-t", *inst(pair[0]), "--judges-e", *inst(pair[1])]
        pool.main(args)
    if a.stage in ("response", "all"):
        from subliminal_rl.pools import student_response
        student_response.main(common + ["--pool", pool_npz, "--name", name, "--out", a.out])
    if a.stage in ("stats", "all"):
        import numpy as np
        from subliminal_rl.analysis.absorption import absorption
        from subliminal_rl.analysis.emission import emission
        pl, rs = np.load(pool_npz), np.load(resp_npz)
        inst = pool_npz[:-4] + "_instances.npz"     # release pools: all instances scored on the same candidates
        em_scores = np.load(inst) if os.path.exists(inst) else pl
        fd = json.load(open(resp_npz[:-4] + ".gates.json"))["fd"]
        gate = json.load(open(pool_npz[:-4] + ".gates.json"))
        res = {"cell": a.cell, "valid_frac": gate["valid_frac"], "emission": emission(em_scores), "absorption": absorption(pl, rs, fd)}
        json.dump(res, open(os.path.join(a.out, "stages_%s.json" % name), "w"), indent=1)
        lam, ab = res["emission"]["Lambda"], res["absorption"]["primary"]
        print("%s  validity %.3f  Lambda %+.3f [%+.3f, %+.3f]  absorption %+.3f [%+.3f, %+.3f]" % (
            a.cell, gate["valid_frac"], lam["est"], lam["lo"], lam["hi"], ab["est"], ab["lo"], ab["hi"]))


if __name__ == "__main__":
    main()
