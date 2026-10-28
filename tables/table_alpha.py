"""Extended Data Table 5 ("Reward intervention"): tabs/alpha.tex.

Training on c + alpha * s, where c is the part of the reward shared by the two
judges and s the between-judge part; eight seed pairs per condition. For each
cell and alpha: S80 (with the seed pairs in the judge's direction where reported),
the paired difference from alpha = 1 with its p value, final KL and carrier
validity. Reads data/cells.csv and data/reward_interventions.csv.
"""
import common as C

HEADER = ("\\textbf{Cell} & $\\boldsymbol{\\alpha}$ & $\\boldsymbol{S_{80}}$ \\textbf{(pairs in judge's direction)} & "
          "\\textbf{Paired difference from} $\\boldsymbol{\\alpha=1}$ & \\textbf{KL} & \\textbf{Validity} \\\\")
SPEC = C.LEFT("0.24") + "c" + C.RIGHT_X * 4


def build(data_dir):
    cells = C.cells(data_dir)
    rows = C.read_csv(data_dir, "reward_interventions.csv")
    L = ["\\toprule", HEADER, "\\midrule"]
    last = None
    for r in rows:
        if r["cell"] != last:
            if last is not None:
                L.append("\\midrule")
            c = cells[r["cell"]]
            name = "\\st{%s}{%s}{%s}{%s}" % (c["student"], c["route"], c["carrier"], C.pair_tex(c["trait_pair"]))
        else:
            name = ""
        last = r["cell"]
        s80 = C.sgn_math(r["s80"]) + (" (%d/%d)" % (r["pairs_judge_direction"], r["n_seed_pairs"])
                                      if r["pairs_judge_direction"] is not None else "")
        diff = "---" if r["difference_from_alpha1"] is None else "%s, $p=%g$" % (C.sgn_math(r["difference_from_alpha1"]), r["p_paired"])
        kl = "---" if r["kl"] is None else "%.2f" % r["kl"]
        val = "---" if r["validity"] is None else "%.3f" % r["validity"]
        L.append("%s & %d & %s & %s & %s & %s \\\\" % (name, r["alpha"], s80, diff, kl, val))
    L.append("\\bottomrule")
    return C.tabularx(SPEC, L)


if __name__ == "__main__":
    a = C.cli(__doc__.splitlines()[0])
    print(C.write(a.out, "alpha", build(a.data)))
