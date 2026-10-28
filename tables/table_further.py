"""Extended Data Table 2 ("Transfer on a cell's further seed pairs, and the combination of the two groups"): further_pairs.tex.

For the 35 cells with seed pairs beyond the base eight: S80 over the further
pairs alone with its bootstrap 95% interval, pairs in the judge's direction,
two-sided sign-flip p and Benjamini-Hochberg q over the 35 cells; and for all 46
cells the base-eight p, the Fisher combination of the base and further groups
(one-sided p combined per direction, two-sided as twice the smaller) and its BH q
over the 46 cells. "Meets criterion" is p < 0.05 and q < 0.10.
Reads data/cells.csv and data/further_pairs.csv.
"""
import common as C

HEADER = [r"& \multicolumn{8}{c}{\textbf{Further seed pairs alone}} & \multicolumn{4}{c}{\textbf{All seed pairs combined}}\\",
          r"\cmidrule(lr){2-9}\cmidrule(l){10-13}",
          r"\textbf{Cell} & \textbf{Pairs} & \textbf{Further pairs} & \textbf{$S_{80}$} & \textbf{95\% CI} & \textbf{Judge's direction} & "
          r"\textbf{$p$} & \textbf{$q$ (35 cells)} & \textbf{Meets criterion} & \textbf{$p$, base eight} & \textbf{Combined $p$} & "
          r"\textbf{$q$ (46 cells)} & \textbf{Meets criterion}\\"]
SPEC = (C.LEFT("0.12") + C.RIGHT_X * 3 + ">{\\raggedleft\\arraybackslash}p{0.115\\textwidth}" + C.RIGHT_X * 3 + C.CENTER_X
        + C.RIGHT_X * 3 + C.CENTER_X)


def p3(p):
    return "$<$0.001" if p < 0.0005 else "%.3f" % p


def build(data_dir):
    cells = C.cells(data_dir)
    fp = C.read_csv(data_dir, "further_pairs.csv", "cell")
    L = ["\\toprule"] + HEADER + ["\\midrule"]
    yes = lambda v: "yes" if v == "yes" else "---"
    for title, blocks in C.ROUTE_BLOCKS:
        L.append(C.route_header(13, title))
        for b in blocks:
            L.append(C.group_header(13, C.config_label(cells[b + "_ET"])))
            for p in C.PAIR_ORDER:
                k = b + "_" + p
                if k not in fp:
                    continue
                r = fp[k]
                if r["n_further_pairs"]:
                    ex = ["%d" % r["n_further_pairs"], C.sgn(r["s80_further"]),
                          "[%s, %s]" % (C.sgn(r["s80_further_lo"]), C.sgn(r["s80_further_hi"])),
                          "%d/%d" % (r["further_judge_direction"], r["n_further_pairs"]),
                          "%.3f" % r["p_further"], "%.3f" % r["q_further"], yes(r["meets_criterion_further"])]
                else:
                    ex = ["---"] * 7
                L.append(C.row([C.pair_tex(cells[k]["trait_pair"]), "%d" % r["n_seed_pairs"]] + ex
                               + ["%.3f" % r["p_base"], p3(r["p_combined"]), "%.3f" % r["q_combined"], yes(r["meets_criterion_combined"])],
                               shade=r["meets_criterion_combined"] == "yes"))
    L.append("\\bottomrule")
    return C.tabularx(SPEC, L)


if __name__ == "__main__":
    a = C.cli(__doc__.splitlines()[0])
    print(C.write(a.out, "further_pairs", build(a.data)))
