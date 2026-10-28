"""Statistics of the dispositional cells (Methods, "Dispositional traits";
Table "disp"; Supplementary Note "Dispositional cells").

Writes four JSON files:

* ``disp_grid.json``     per cell: absorption, baseline, effect size d of each
                         evaluation at each step, the steps where it is established,
                         and the choice shift.
* ``disp_endpoint.json`` per cell: transfer at step 16, at step 80 and averaged over
                         steps 8-80, each tested across the twelve cells.
* ``disp_lambda.json``   per cell: emission Lambda over eight wordings per judge.
* ``disp_screen.json``   per screened axis: emission ratio, absorption and baseline
                         of the start policy (24 axes, Qwen syllable pool).

Inputs (paths on the command line; layout in configs/dispositions/cells.json):

* preference/answer evaluation logs, one JSON object per evaluation step and run,
  ``<run_prefix><T|E>_<seed>.eval.jsonl`` with fields ``step``, ``B_AN`` (mean
  preference margin over the 40 items), ``Acc_AN`` (share of items with a positive
  margin), ``beh_fc_raw`` ([40 templates][2 orders][T, E] answer log-likelihoods;
  order 0 presents the T adjective first) and ``beh_open_raw`` ([10][T, E]);
* candidate-pool statistics of the emission/absorption job,
  ``ea_<pool_stats>.stats.json`` (``absorption.<group>_delta<step>.corr_da_k``,
  ``emission.ratio``, ``Lambda.Lambda``), for the training sentence and, in a
  separate directory, for the eight-wording run;
* pool gates ``absorb_<pool_stats>.gates.json`` (``gateA.B_AN32``, the start
  policy's mean preference margin) for the screen.

Definitions (all as used for the paper):

* Seed pairs: the 40 lowest-numbered seeds whose T and E runs both logged step 0
  and step 80.
* Per seed pair and evaluation, S at step s is half the T-minus-E difference of
  the change from step 0 (form of equation S); d_S = mean(S) / sd(S) (ddof 1).
  Transfer reads B_AN; the forced-choice evaluation reads the mean of
  log P(T) - log P(E) over templates and orders; the open-question evaluation the
  same over the ten questions.
* Sign-flip p values: two-sided; exact enumeration up to 18 pairs, otherwise
  400,000 random assignments (two draws of 200,000, generator seeded with 11 at
  every call).
* An evaluation is established at a step if p < 0.05 and the Benjamini-Hochberg q
  over that evaluation's nine steps in that cell is below 0.10.
* Answer-evaluation validity at a step: the forced choice is used only where the
  mean S of order 0 and of order 1 have the same sign and its common component
  (mean of all its log-likelihoods, averaged over seed pairs and both judge
  conditions) has not fallen by 0.45 nats or more since step 0; the open
  question requires the same of its common component.
* Choice shift: per seed pair, the share of the 80 forced-choice items (40
  templates x 2 orders) on which the T-trained student scores the T adjective
  higher, minus the same share for the E-trained student, in percentage points,
  at steps 8-80; BH over those seven steps. The share of items on which the two
  students choose differently is recorded alongside.
* Endpoint test: BH across the twelve cells at step 80 (also at step 16 and for
  the average over steps 8-80).
* Absorption: correlation of the T-minus-E reward difference with the student's
  preference-margin response over the candidate pool, with group-of-8 centring,
  read at the smallest finite-difference step stored for the cell (see report).
"""
import argparse
import glob
import itertools
import json
import os
from pathlib import Path

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "dispositions"

STEPS = [2, 4, 8, 16, 24, 32, 48, 64, 80]
CHOICE_STEPS = (8, 16, 24, 32, 48, 64, 80)
TRAJ_STEPS = (8, 16, 24, 32, 48, 64, 80)
P_CUT, Q_CUT = 0.05, 0.10
N_PAIRS = 40
COMMON_DROP = 0.45
MODEL_LABEL = {"qwen": "Qwen", "llama": "Llama"}


# ---------------------------------------------------------------- statistics
def signflip(v):
    """Two-sided sign-flip p value of the mean of ``v``."""
    v = np.asarray(v, float)
    n = len(v)
    m = v.mean()
    if n <= 18:
        A = np.array(list(itertools.product([1, -1], repeat=n)))
        mu = (A * v).mean(1)
    else:
        rg = np.random.default_rng(11)
        mu = np.concatenate([(rg.choice([1.0, -1.0], size=(200000, n)) * v).mean(1) for _ in range(2)])
    return min(1.0, 2 * min(float((mu >= m - 1e-12).mean()), float((mu <= m + 1e-12).mean())))


def bh(ps):
    """Benjamini-Hochberg adjusted q values."""
    ps = list(ps)
    o = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ps[o[i]] * m / (i + 1))
        q[o[i]] = prev
    return q


def effect_size(a):
    a = np.asarray(a, float)
    return float(a.mean() / a.std(ddof=1))


# ---------------------------------------------------------------- inputs
def load_runs(evals_dir, run_prefix):
    """``{seed: (T steps, E steps)}``; each element maps step -> logged record.

    Seeds are taken from the T files; a seed whose E file is missing is skipped.
    """
    seeds = sorted({os.path.basename(f).split("_")[-1].split(".")[0]
                    for f in glob.glob(os.path.join(evals_dir, run_prefix + "T_*.eval.jsonl"))})
    runs = {}
    for sd in seeds:
        try:
            runs[sd] = tuple({r["step"]: r for r in (json.loads(l) for l in open(
                os.path.join(evals_dir, "%s%s_%s.eval.jsonl" % (run_prefix, j, sd))))} for j in "TE")
        except FileNotFoundError:
            pass
    return runs


def complete_seeds(runs):
    return [sd for sd in runs if all(k in runs[sd][i] for i in (0, 1) for k in (0, 80))]


def select_pairs(runs, n=N_PAIRS):
    """The ``n`` lowest-numbered seeds whose T and E runs both reached step 80."""
    return sorted(complete_seeds(runs), key=int)[:n]


def absorption(stats_path, step=None):
    """(estimate, lower, upper) of group-of-8 absorption from a pool-statistics file.

    With ``step=None`` the smallest finite-difference step stored for the cell is
    used (this is what the reported values use); otherwise ``g8_delta<step>``.
    """
    a = json.load(open(stats_path))["absorption"]
    if step is None:
        key = sorted(k for k in a if k.startswith("g8_"))[0]
    else:
        key = "g8_delta%s" % step
    c = a[key]["corr_da_k"]
    return c["est"], c["lo"], c["hi"]


# ---------------------------------------------------------------- evaluations
def _fc_margin(r, key):
    a = np.array(r[key], float)
    return float((a[..., 0] - a[..., 1]).mean())


EVALUATIONS = (
    ("transfer", lambda r: r["B_AN"]),
    ("fc", lambda r: _fc_margin(r, "beh_fc_raw")),
    ("open", lambda r: _fc_margin(r, "beh_open_raw")),
)


def pair_S(runs, sd, f, st):
    T, E = runs[sd]
    return (f(T[st]) - f(T[0]) - (f(E[st]) - f(E[0]))) / 2


def answer_validity(runs, seeds, st):
    """Which answer evaluations may be used at step ``st`` (see module docstring)."""
    def order_margin(r, o):
        a = np.array(r["beh_fc_raw"], float)
        return float((a[:, o, 0] - a[:, o, 1]).mean())

    so = [np.mean([(order_margin(runs[sd][0][st], o) - order_margin(runs[sd][0][0], o)
                    - (order_margin(runs[sd][1][st], o) - order_margin(runs[sd][1][0], o))) / 2
                   for sd in seeds]) for o in (0, 1)]
    drop = {}
    for name, key in (("fc", "beh_fc_raw"), ("open", "beh_open_raw")):
        C = lambda r: float(np.array(r[key], float).mean())
        drop[name] = np.mean([C(runs[sd][i][st]) - C(runs[sd][i][0]) for sd in seeds for i in (0, 1)])
    orders_agree = so[0] * so[1] > 0
    return {"orders_agree": orders_agree,
            "fc": orders_agree and drop["fc"] > -COMMON_DROP,
            "open": drop["open"] > -COMMON_DROP}


def choice_shift(runs, seeds):
    """Choice shift and share of differing choices at each of ``CHOICE_STEPS``."""
    res = []
    for st in CHOICE_STEPS:
        net, dis = [], []
        for sd in seeds:
            a = np.array(runs[sd][0][st]["beh_fc_raw"], float)
            b = np.array(runs[sd][1][st]["beh_fc_raw"], float)
            pt = a[..., 0] > a[..., 1]
            pe = b[..., 0] > b[..., 1]
            net.append(float(pt.mean() - pe.mean()))
            dis.append(float((pt != pe).mean()))
        res.append([st, 100 * float(np.mean(net)), 100 * float(np.mean(dis)), signflip(net)])
    return res


def grid_cell(cell, runs, seeds, U):
    """One entry of disp_grid.json."""
    validity = {st: answer_validity(runs, seeds, st) for st in STEPS}
    out = dict(model=MODEL_LABEL[cell["student"]], carrier=cell["carrier"], axis=cell["axis"],
               pair=cell["axis"], n=len(seeds), U=U[0], Ulo=U[1], Uhi=U[2],
               B0=float(np.mean([runs[sd][i][0]["B_AN"] for sd in seeds for i in (0, 1)])),
               Acc0=float(np.mean([runs[sd][i][0]["Acc_AN"] for sd in seeds for i in (0, 1)])),
               d={}, ok={})
    for name, f in EVALUATIONS:
        rows = []
        for st in STEPS:
            a = np.array([pair_S(runs, sd, f, st) for sd in seeds])
            rows.append([st, effect_size(a), signflip(a)])
        q = bh([r[2] for r in rows])
        ok = []
        for i, r in enumerate(rows):
            if not (r[2] < P_CUT and q[i] < Q_CUT):
                continue
            if name in ("fc", "open") and not validity[r[0]][name]:
                continue
            ok.append([r[0], r[1], float(q[i])])
        out["d"][name] = {r[0]: r[1] for r in rows}
        out["ok"][name] = ok
    res = choice_shift(runs, seeds)
    q = bh([r[3] for r in res])
    out["ok"]["choice"] = [[r[0], r[1], r[2], float(q[i])] for i, r in enumerate(res)
                           if r[3] < P_CUT and q[i] < Q_CUT]
    return out


def endpoint(cells, all_runs, all_seeds, all_U):
    """disp_endpoint.json: transfer at step 16, step 80 and over steps 8-80, BH across cells."""
    f = EVALUATIONS[0][1]
    vals = []
    for cell, runs, seeds in zip(cells, all_runs, all_seeds):
        vals.append(dict(
            s16=np.array([pair_S(runs, sd, f, 16) for sd in seeds]),
            s80=np.array([pair_S(runs, sd, f, 80) for sd in seeds]),
            traj=np.array([np.mean([pair_S(runs, sd, f, st) for st in TRAJ_STEPS]) for sd in seeds])))
    out = {}
    for key in ("s16", "s80", "traj"):
        ps = [signflip(v[key]) for v in vals]
        qs = bh(ps)
        for cell, runs, U, v, p, q in zip(cells, all_runs, all_U, vals, ps, qs):
            k = cell_label(cell)
            e = out.setdefault(k, {"U": U[0], "n_seen": len(runs), "n_complete": len(complete_seeds(runs))})
            e[key] = dict(d=effect_size(v[key]), p=float(p), q=float(q), ok=bool(p < P_CUT and q < Q_CUT))
    return out


def cell_label(cell):
    return "|".join((MODEL_LABEL[cell["student"]], cell["carrier"], cell["axis"]))


def lambda_table(cells, stats_dir, stats_name):
    """disp_lambda.json: emission Lambda from the eight-wording pool statistics."""
    out = {}
    for cell in cells:
        s = json.load(open(os.path.join(stats_dir, stats_name.format(pool_stats=cell["pool_stats"]))))
        out[cell_label(cell)] = s["Lambda"]["Lambda"]
    return out


def screen(axes, stats_dirs, gates_dir, files):
    """disp_screen.json: start-policy emission ratio, absorption (step 0.16) and baseline."""
    out = []
    for ax in axes:
        ps = files["screen_pool_stats"].format(code=ax["code"], axis=ax["axis"])
        path = None
        for d in stats_dirs:
            p = os.path.join(d, files["stats"].format(pool_stats=ps))
            if os.path.exists(p):
                path = p
                break
        s = json.load(open(path))
        a = s["absorption"]
        key = "g8_delta0.16" if "g8_delta0.16" in a else sorted(k for k in a if k.startswith("g8_"))[0]
        c = a[key]["corr_da_k"]
        g = json.load(open(os.path.join(gates_dir, files["gates"].format(pool_stats=ps))))
        out.append(dict(key=ax["axis"], pos=ax["T"], neg=ax["E"], domain=ax["domain"],
                        emis=s["emission"]["ratio"]["est"], U=c["est"], lo=c["lo"], hi=c["hi"],
                        delta=key, B0=g["gateA"]["B_AN32"]))
    out.sort(key=lambda r: -r["U"])
    return out


# ---------------------------------------------------------------- summaries
def chain_summary(grid, min_abs_absorption=0.10, steps=(8, 16, 32, 80), n_perm=20000):
    """Across-cell relation of absorption to transfer (Supplementary Note).

    Pearson r of absorption with d_S at each step, with a permutation p value
    (generator seeded with 5, shared across the steps in order); sign agreement
    of d_S with absorption among cells with |absorption| >= ``min_abs_absorption``
    (all but the three least aligned cells); d_S at step 16 of the others.
    """
    rg = np.random.default_rng(5)
    U = np.array([c["U"] for c in grid])
    lines = []
    for st in steps:
        y = np.array([c["d"]["transfer"][st] for c in grid])
        r = np.corrcoef(U, y)[0, 1]
        cnt = sum(1 for _ in range(n_perm) if abs(np.corrcoef(rg.permutation(U), y)[0, 1]) >= abs(r) - 1e-12)
        big = [(u, v) for u, v in zip(U, y) if abs(u) >= min_abs_absorption]
        lines.append("step %2d: absorption~d r %+.3f (permutation p %.4f); sign agreement %d of %d" % (
            st, r, (cnt + 1) / (n_perm + 1), sum(1 for u, v in big if u * v > 0), len(big)))
    small = [c for c in grid if abs(c["U"]) < min_abs_absorption]
    lines.append("least aligned cells, d at step 16: " + "  ".join(
        "%s %s %s U %+.3f d %+.2f" % (c["model"], c["carrier"], c["axis"], c["U"], c["d"]["transfer"][16]) for c in small))
    lines.append("MDE (80%% power) at %d pairs: %.2f" % (N_PAIRS, 2.8 / np.sqrt(N_PAIRS)))
    return lines


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Dispositional-cell statistics (disp_*.json).")
    ap.add_argument("--evals", required=True, help="directory of <run_prefix><T|E>_<seed>.eval.jsonl")
    ap.add_argument("--absorption-stats", required=True,
                    help="directory of the training-sentence pool statistics (ea_<pool>.stats.json)")
    ap.add_argument("--wording-stats", required=True,
                    help="directory of the eight-wording pool statistics (for Lambda)")
    ap.add_argument("--screen-stats", nargs="+", default=None,
                    help="directories searched in order for the screened axes' pool statistics")
    ap.add_argument("--screen-gates", default=None, help="directory of absorb_<pool>.gates.json for the screen")
    ap.add_argument("--absorption-step", default=None,
                    help="finite-difference step for absorption (default: smallest stored, as reported)")
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args(argv)

    cfg = json.load(open(os.path.join(a.config_dir, "cells.json")))
    files, cells = cfg["files"], cfg["cells"]
    os.makedirs(a.out, exist_ok=True)

    all_runs, all_seeds, all_U, grid = [], [], [], []
    for cell in cells:
        runs = load_runs(a.evals, cell["run_prefix"])
        seeds = select_pairs(runs)
        assert len(seeds) == N_PAIRS, (cell["key"], len(seeds))
        U = absorption(os.path.join(a.absorption_stats, files["stats"].format(pool_stats=cell["pool_stats"])),
                       a.absorption_step)
        all_runs.append(runs); all_seeds.append(seeds); all_U.append(U)
        grid.append(grid_cell(cell, runs, seeds, U))
        print("%-32s U %+.3f  B0 %+.3f  established: %s" % (cell["key"], U[0], grid[-1]["B0"], {
            k: [x[0] for x in v] for k, v in grid[-1]["ok"].items()}), flush=True)
    json.dump(grid, open(os.path.join(a.out, "disp_grid.json"), "w"), indent=1)

    ep = endpoint(cells, all_runs, all_seeds, all_U)
    json.dump(ep, open(os.path.join(a.out, "disp_endpoint.json"), "w"), indent=1)
    print("seed pairs launched %d, completing step 80 %d" % (
        sum(v["n_seen"] for v in ep.values()), sum(v["n_complete"] for v in ep.values())))

    json.dump(lambda_table(cells, a.wording_stats, files["stats"]),
              open(os.path.join(a.out, "disp_lambda.json"), "w"), indent=1)

    if a.screen_stats and a.screen_gates:
        axes = json.load(open(os.path.join(a.config_dir, "axes.json")))["screen"]
        json.dump(screen(axes, a.screen_stats, a.screen_gates, files),
                  open(os.path.join(a.out, "disp_screen.json"), "w"), indent=1)

    for line in chain_summary(grid):
        print(line)


if __name__ == "__main__":
    main()
