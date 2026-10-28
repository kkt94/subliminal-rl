"""Table 1 ("The four stages, one row per cell"): tabs/main.tex.

One row per design cell (48, with the two cells excluded at the carrier-validity
gate): emission Lambda and absorption at the starting policy with 95% intervals,
S80 over all seed pairs with its paired-bootstrap 95% interval, seed pairs in the
judge's direction, two-sided sign-flip p, Benjamini-Hochberg q over the 46 cells,
the transfer verdict (p < 0.05 and q < 0.10) and the answer-level behaviour.
Reads data/cells.csv and data/behaviour.csv.
"""
import common as C

GROUPS = [("FT route: fine-tuned judges", ["A1", "A3", "B1", "B3"]),
          ("SP route: persona judges", ["A2", "A4", "B2", "B4"])]
HEADER = ("\\textbf{Cell} & \\multicolumn{2}{c}{\\textbf{Emission $\\Lambda$}} & \\multicolumn{2}{c}{\\textbf{Absorption}} & "
          "\\multicolumn{2}{c}{\\textbf{$S_{80}$}} & \\textbf{Pairs} & \\textbf{$p$} & \\textbf{$q$} & \\textbf{Transfer} & \\textbf{Behaviour}\\\\")


def interval(lo, hi, nd=3):
    return "\\tabci{[%s, %s]}" % (C.sgn_math(lo, nd), C.sgn_math(hi, nd))


def pval(p):
    return "$<0.001$" if p < 0.001 else "%.3f" % p


def behaviour_tex(text):
    return text.replace("−", "$-$").replace("×", "$\\times$")


def build(data_dir):
    cells = C.cells(data_dir)
    beh = C.read_csv(data_dir, "behaviour.csv", "cell")
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l r l r l r l r r r l >{\\raggedright\\arraybackslash}p{0.15\\textwidth}@{}}",
         "\\toprule", HEADER, "\\midrule"]
    for title, blocks in GROUPS:
        L.append("\\addlinespace[3pt]\\rowcolor{black!7}\\multicolumn{12}{@{}l}{\\textbf{%s}}\\\\" % title.replace(": ", " --- "))
        for b in blocks:
            first = cells[b + "_ET"]
            L.append("\\addlinespace[2pt]\\multicolumn{12}{@{}l}{\\textit{%s \\textperiodcentered{} %s}}\\\\" % (first["student"], first["carrier"]))
            for p in C.PAIR_ORDER:
                c = cells[b + "_" + p]
                pn = C.pair_tex(c["trait_pair"])
                if c["status"] != "analysed":
                    L.append("\\quad %s & \\multicolumn{11}{l}{\\textit{excluded: below the carrier-validity gate}}\\\\" % pn)
                    continue
                est = c["established"] == "yes"
                bh = beh[c["cell"]]
                note = ("$^{\\mathrm{%s}}$" % bh["note"]) if bh["note"] else ""
                L.append("%s\\quad %s & %s & %s & %s & %s & %s & %s & %d/%d & %s & %s & %s & \\tabci{%s}\\\\" % (
                    "\\rowcolor{stpos!9}" if est else "", pn,
                    C.sgn_math(c["lambda"]), interval(c["lambda_lo"], c["lambda_hi"]),
                    C.sgn_math(c["absorption"]), interval(c["absorption_lo"], c["absorption_hi"]),
                    C.sgn_math(c["s80"], 4), interval(c["s80_boot_lo"], c["s80_boot_hi"], 4),
                    c["pairs_judge_direction"], c["n_seed_pairs"], pval(c["p_two_sided"]), pval(c["q_bh"]),
                    ("\\textbf{established}" if est else "---") + note, behaviour_tex(bh["answer_level"])))
    L += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    a = C.cli(__doc__.splitlines()[0])
    print(C.write(a.out, "main", build(a.data)))
