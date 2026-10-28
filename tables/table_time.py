"""Extended Data Table 6 ("The state of training, one row per cell"): tabs/time.tex.

Carrier validity at steps 2, 48 and 80 and in the lowest run at step 80; the share
of groups whose eight completions do not all receive the same reward (reward-signal
share) at steps 10, 40 and 80; the first step (interpolated) at which that share
falls below 0.5; and, for established cells, the steps at which S reaches 50% and
80% of S80. Reads data/cells.csv and data/training_state.csv.
"""
import common as C

HEADER = ("\\textbf{Cell} & \\textbf{Validity, step 2 / 48 / 80} & \\textbf{Lowest run, step 80} & "
          "\\textbf{Reward-signal share, step 10 / 40 / 80} & \\textbf{First step with signal share $<0.5$} & "
          "\\textbf{Step reaching 50\\% / 80\\% of $S_{80}$}\\\\")


def build(data_dir):
    cells = C.cells(data_dir)
    state = {}
    for r in C.read_csv(data_dir, "training_state.csv"):
        state.setdefault(r["cell"], {})[r["step"]] = r
    L = ["\\toprule", HEADER, "\\midrule"]
    for title, blocks in C.ROUTE_BLOCKS:
        L.append(C.route_header(6, title))
        for b in blocks:
            L.append(C.group_header(6, C.config_label(cells[b + "_ET"])))
            for p in C.PAIR_ORDER:
                c = cells[b + "_" + p]
                if c["status"] != "analysed":
                    continue
                st = state[c["cell"]]
                th = c["signal_share_below_half_step"]
                L.append(C.row([
                    C.pair_tex(c["trait_pair"]),
                    "%.3f / %.3f / %.3f" % tuple(st[s]["validity"] for s in (2, 48, 80)),
                    "%.3f" % c["validity_lowest_run_step80"],
                    "%.2f / %.2f / %.2f" % tuple(st[s]["reward_signal_share"] for s in (10, 40, 80)),
                    "%d" % round(th) if th is not None else "never",
                    "%s / %s" % (c["step_50pct_s80"], c["step_80pct_s80"]) if c["established"] == "yes" else "---"]))
    L.append("\\bottomrule")
    return C.tabularx(C.LEFT("0.13") + C.RIGHT_X * 5, L)


if __name__ == "__main__":
    a = C.cli(__doc__.splitlines()[0])
    print(C.write(a.out, "time", build(a.data)))
