"""Table 1 ("Dispositional traits, twelve cells"): tabs/disp.tex.

System-prompt judges, 40 seed pairs per cell. For each cell: emission Lambda (eight
wordings per judge) with its 95% interval, absorption and the baseline preference
margin of the starting policy; a strip over the evaluation steps 2-80 marking the
steps at which transfer, the forced choice, the open question and the choice shift
are established (teal in the judge's direction, red against it) with the range of
effect sizes d (choice shift in percentage points); and the step-80 transfer
effect size d with its Benjamini-Hochberg q across the twelve cells (bold where
the cell meets the criterion).
Reads data/dispositions/disp_grid.json, disp_lambda.json and disp_endpoint.json
(schema of subliminal_rl.dispositions.analysis).
"""
import common as C

STEPS = [2, 4, 8, 16, 24, 32, 48, 64, 80]
ORDER = [("Llama", "syllable"), ("Llama", "number"), ("Qwen", "syllable"), ("Qwen", "number")]
HEAD = [r" & & & & \multicolumn{2}{c}{\textbf{Transfer}} & \multicolumn{3}{c}{\textbf{Behaviour}}\\",
        r"\cmidrule(lr){5-6}\cmidrule(l){7-9}",
        r"\textbf{Axis} & \textbf{Emission $\Lambda$} & \textbf{Absorption} & \textbf{Baseline} & \textbf{Steps established} & "
        r"\textbf{$d$, step 80} & \textbf{Forced choice} & \textbf{Open question} & \textbf{Implied choice shift}\\"]


def strip(ok):
    on = {v[0]: v[1] for v in ok}
    return r"\mbox{" + "".join(r"\sqa{%s}" % (("stpos" if on[s] > 0 else "stneg") if s in on else "stoff") for s in STEPS) + "}"


def effect_range(ok, pp=False):
    if not ok:
        return ""
    v = [x[1] for x in ok]
    lo, hi = min(v), max(v)
    f = lambda z: ("$+%.2f$" % z) if z >= 0 else ("$-%.2f$" % -z)
    return (f(lo) if abs(lo - hi) < 1e-9 else f(lo) + " to " + f(hi)) + (" pp" if pp else "")


def build(data_dir):
    grid = C.read_json(data_dir, "dispositions", "disp_grid.json")
    lam = C.read_json(data_dir, "dispositions", "disp_lambda.json")
    end = C.read_json(data_dir, "dispositions", "disp_endpoint.json")
    L = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l r r r c r c c c@{}}", r"\toprule"] + HEAD + [r"\midrule"]
    for i, (model, carrier) in enumerate(ORDER):
        L.append((r"\addlinespace[2pt]" if i else "") + r"\multicolumn{9}{@{}l}{\textit{%s \textperiodcentered{} %s carrier}}\\"
                 % (C.STUDENT_FULL[model], carrier))
        for g in grid:
            if (g["model"], g["carrier"]) != (model, carrier):
                continue
            key = "|".join((model, carrier, g["axis"]))
            lm, e = lam[key], end[key]["s80"]
            ok = g["ok"]
            s80 = C.sgn(e["d"])
            q = "$<$0.001" if e["q"] < 0.001 else "%.3f" % e["q"]
            top = [r"\hspace{0.6em}" + g["axis"].replace("-", "--"), C.sgn(lm["est"]), C.sgn(g["U"]), C.sgn(g["B0"]),
                   strip(ok["transfer"]), (r"\textbf{%s}" % s80) if e["ok"] else s80,
                   strip(ok["fc"]), strip(ok["open"]), strip(ok["choice"])]
            bot = ["", r"\tabci{[%s, %s]}" % (C.sgn(lm["lo"]), C.sgn(lm["hi"])), "", "", r"\tabci{%s}" % effect_range(ok["transfer"]),
                   r"\tabci{$q$ %s}" % q, r"\tabci{%s}" % effect_range(ok["fc"]), r"\tabci{%s}" % effect_range(ok["open"]),
                   r"\tabci{%s}" % effect_range(ok["choice"], True)]
            L.append(" & ".join(top) + r"\\[-1pt]")
            L.append(" & ".join(bot) + r"\\[1pt]")
    L += [r"\bottomrule", r"\end{tabular*}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    a = C.cli(__doc__.splitlines()[0])
    print(C.write(a.out, "disp", build(a.data)))
