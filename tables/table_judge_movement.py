"""Extended Data Table 1 ("Each judge condition's movement"): judge_movement.tex.

For every cell, the change of the preference margin B from step 0 to step 80 in
the student trained under the T judge and under the E judge, their mean (the
common push) and half their difference (S), averaged over the eight base seed
pairs, with the number of pairs in which both students move the same way.
Reads data/cells.csv and data/judge_movement.csv.
"""
import common as C

HEADER = ("\\textbf{Cell} & \\textbf{$\\Delta B$, T-trained} & \\textbf{$\\Delta B$, E-trained} & \\textbf{Common push} & "
          "\\textbf{$S$ (eight pairs)} & \\textbf{Pairs moving the same way}\\\\")


def build(data_dir):
    cells = C.cells(data_dir)
    movement = C.read_csv(data_dir, "judge_movement.csv", "cell")
    L = ["\\toprule", HEADER, "\\midrule"]
    for b in C.CONFIG_ORDER:
        c0 = cells[b + "_ET"]
        L.append(C.group_header(6, " \\textperiodcentered{} ".join((c0["student"], c0["route"], c0["carrier"]))))
        for p in C.PAIR_ORDER:
            k = b + "_" + p
            if k not in movement:
                continue
            o = movement[k]
            L.append(C.row([C.pair_tex(cells[k]["trait_pair"]), C.sgn(o["delta_B_T"], 4), C.sgn(o["delta_B_E"], 4),
                            C.sgn(o["common_push"], 4), C.sgn(o["s"], 4), "%d/%d" % (o["pairs_same_direction"], o["n_base_pairs"])]))
    L.append("\\bottomrule")
    return C.tabularx(C.LEFT("0.16") + C.RIGHT_X * 5, L)


if __name__ == "__main__":
    a = C.cli(__doc__.splitlines()[0])
    print(C.write(a.out, "judge_movement", build(a.data)))
