"""Extended Data Table 6 ("Judges that are not the student's own model"): tabs/judge.tex.

For each route and student, the main-grid judges (the student's own model) beside
the other family's 1B-class judges and 8B judges of either family: mean emission
Lambda over the cells, cells whose Lambda lower bound is above zero, correlation
of the judge's trait difference with that of the own-model judge, mean |absorption|,
cells whose absorption has the own-model sign, the judge term, the across-cell
correlation of absorption with S80 (pooled over the two routes) and transfer.
Reads data/judge_conditions.csv.
"""
import common as C

HEADER = ("\\textbf{Judge} & \\textbf{Emission $\\Lambda$ (mean)} & \\textbf{$\\Lambda$ lower bound $>0$} & "
          "\\textbf{Trait alignment with own-model judge ($r$)} & \\textbf{$|$Absorption$|$} & \\textbf{Absorption sign as own model} & "
          "\\textbf{Judge term} & \\textbf{Absorption vs $S_{80}$ ($r$)} & \\textbf{Transfer}\\\\")
JUDGE = {"own": "Own model (main grid)", "other_family_1b": "Other family, 1B class",
         "same_family_8b": "Same family, 8B", "other_family_8b": "Other family, 8B"}
ROUTE = {"FT": "FT route --- fine-tuned judges", "SP": "SP route --- persona judges"}
SPEC = (C.LEFT("0.15") + C.RIGHT_X * 7 + "@{\\hspace{10pt}}>{\\raggedright\\arraybackslash}p{0.15\\textwidth}")


def transfer(r):
    if r["judge"] == "own":
        return "%d of %d established" % (r["transfer_established"], r["transfer_cells_trained"])
    if r["judge"] == "other_family_1b":
        return "none established" if r["transfer_established"] == 0 else "%d of %d established" % (
            r["transfer_established"], r["transfer_cells_trained"])
    return "%d of %d trained cells" % (r["transfer_established"], r["transfer_cells_trained"])


def build(data_dir):
    rows = C.read_csv(data_dir, "judge_conditions.csv")
    dash = "---"
    L = ["\\toprule", HEADER, "\\midrule"]
    last = (None, None)
    for r in rows:
        if r["route"] != last[0]:
            L.append(C.route_header(9, ROUTE[r["route"]]))
        if (r["route"], r["student"]) != last:
            L.append(C.group_header(9, "Student " + C.STUDENT_FULL[r["student"]]))
        last = (r["route"], r["student"])
        own = r["judge"] == "own"
        L.append(C.row([
            JUDGE[r["judge"]], "%+.3f" % r["lambda_mean"], "%d/%d" % (r["cells_lambda_lo_positive"], r["n_cells"]),
            dash if own else C.sgn(r["trait_alignment_r"], 2), "%.3f" % r["abs_absorption_mean"],
            dash if own else "%d/%d" % (r["cells_absorption_sign_as_own"], r["n_cells"]), "%.3f" % r["judge_term"],
            dash if r["absorption_s80_r"] is None else C.sgn(r["absorption_s80_r"], 2), transfer(r)]))
    L.append("\\bottomrule")
    return C.tabularx(SPEC, L)


if __name__ == "__main__":
    a = C.cli(__doc__.splitlines()[0])
    print(C.write(a.out, "judge", build(a.data)))
