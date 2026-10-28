"""Extended Data Table 3 ("Carrier channel decomposition, one row per cell"): tabs/chan.tex.

Absorption (training instance, start-policy standardized score difference) split
into the part carried by the carrier's surface features (letters and bigrams for
the syllable carrier, number statistics for the number carrier) and the
remainder, with the judge-side and student-side coefficients, and the
instance-mean score-difference absorption with its feature part.
Reads data/cells.csv and data/channel_decomposition.csv.
"""
import common as C

HEADER = ("\\textbf{Cell} & \\textbf{Absorption (training instance)} & \\textbf{Feature part} & \\textbf{Residual} & "
          "\\textbf{Judge-side coefficient} & \\textbf{Student-side coefficient} & \\textbf{Instance-mean absorption} & "
          "\\textbf{Its feature part}\\\\")
FEATURES = {"syllable": "letters + bigrams; coefficient = match with the animal-name letters",
            "number": "number statistics; coefficient = correlation with mean value"}
COLUMNS = ["absorption", "feature_part", "residual", "judge_coefficient", "student_coefficient",
           "instance_mean_absorption", "instance_mean_feature_part"]


def build(data_dir):
    cells = C.cells(data_dir)
    ch = C.read_csv(data_dir, "channel_decomposition.csv", "cell")
    L = ["\\toprule", HEADER, "\\midrule"]
    for title, blocks in C.ROUTE_BLOCKS:
        L.append(C.route_header(8, title))
        for b in blocks:
            c0 = cells[b + "_ET"]
            L.append(C.group_header(8, C.config_label(c0) + " " + FEATURES[c0["carrier"]]))
            for p in C.PAIR_ORDER:
                k = b + "_" + p
                if k not in ch:
                    continue
                L.append(C.row([C.pair_tex(cells[k]["trait_pair"])] + [C.sgn(ch[k][c]) for c in COLUMNS]))
    L.append("\\bottomrule")
    return C.tabularx(C.LEFT("0.13") + C.RIGHT_X * 7, L)


if __name__ == "__main__":
    a = C.cli(__doc__.splitlines()[0])
    print(C.write(a.out, "chan", build(a.data)))
