"""Train one seed pair (or one run) of a grid cell.

Reads the cell from ``configs/cells.csv`` and calls the training runner
(``subliminal_rl.training.grpo_run``) once per judge condition.

    python scripts/run_cell.py --cell A1_SL --seed 71040 --ft-judges judges/ --out runs/
    python scripts/run_cell.py --cell B2_WS --seed 71080 --conditions T --out runs/

FT judges are looked up as ``<ft-judges>/<student>-ft-<animal>-s<instance>``
(built with ``python -m subliminal_rl.judges.ft``); SP judges are the student's
base model with persona wording ``--instance`` (0 = training instance). Base
models default to the Hugging Face ids in ``subliminal_rl.constants.MODELS``; set
``--llama`` / ``--qwen`` to local snapshot directories to pin the exact revisions.
Each run needs one GPU.
"""

import argparse
import csv
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
from subliminal_rl.constants import MODELS, config_path  # noqa: E402
from subliminal_rl.judges.sp import persona  # noqa: E402


def load_cells(path=None):
    return {r["cell"]: r for r in csv.DictReader(open(path or config_path("cells.csv")))}


def run_args(cell, seed, condition, out_dir, base, ft_judges, instance=0, extra=()):
    animal = cell["t_animal"] if condition == "T" else cell["e_animal"]
    if cell["route"] == "FT":
        judge = os.path.join(ft_judges, "%s-ft-%s-s%d" % (cell["student"], animal, instance))
        system = ""
    else:
        judge = base
        system = persona(animal, instance)
    name = "%s_%s_%d" % (cell["cell"], condition, seed)
    args = ["--student", base, "--judge", judge, "--judge-system", system, "--carrier", cell["carrier_rule"],
            "--t-name", cell["t_animal"], "--e-name", cell["e_animal"], "--judge-condition", condition,
            "--seed", str(seed), "--item-offset", cell["item_offset"], "--out", os.path.join(out_dir, name + ".jsonl")]
    if cell["fixdate"]:
        args += ["--fixdate", cell["fixdate"]]
    return args + list(extra)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cell", required=True, help="cell key, e.g. A1_SL (see configs/cells.csv)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--conditions", default="TE", help="T, E or TE")
    p.add_argument("--instance", type=int, default=0, help="judge instance (0 = training instance)")
    p.add_argument("--ft-judges", default="judges", help="directory of FT judge instances")
    p.add_argument("--llama", default=MODELS["llama"]["name"])
    p.add_argument("--qwen", default=MODELS["qwen"]["name"])
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true", help="print the runner arguments only")
    a, extra = p.parse_known_args(argv)
    cell = load_cells()[a.cell]
    assert cell["status"] == "grid", "%s is excluded by the carrier-validity gate" % a.cell
    base = a.llama if cell["student"] == "llama" else a.qwen
    os.makedirs(a.out, exist_ok=True)
    for cond in a.conditions:
        args = run_args(cell, a.seed, cond, a.out, base, a.ft_judges, a.instance, extra)
        print("grpo_run " + " ".join(repr(x) if " " in x else x for x in args), flush=True)
        if not a.dry_run:   # one process per run (fresh CUDA and accelerate state)
            env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
            subprocess.run([sys.executable, "-m", "subliminal_rl.training.grpo_run"] + args, check=True, env=env)


if __name__ == "__main__":
    main()
