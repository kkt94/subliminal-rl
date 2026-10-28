"""Fig. 2 ("Transfer in the 48 cells of the design"), drawn with matplotlib.

Rows are student, judge route and carrier; columns are trait pairs. Each cell
shows the seed pairs in the judge's direction out of all seed pairs; colour gives
the verdict (established at q < 0.05 or q < 0.10, otherwise the sign of the
estimate); hatched cells are the two cells excluded at the carrier-validity gate.
Reads data/cells.csv.

    python figures/fig2_grid.py --out <directory>   -> fig2.pdf, fig2.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch, Rectangle

from kit import CONFIG_ORDER, DEFAULT_DATA, FONT_DIRS, FONT_FAMILY, PAIR_ORDER, read_csv

C_EST = "#1f6f6a"; C_EST10 = "#5aa39d"; C_POS = "#dcefec"; C_NEG = "#f6e3d8"; C_GATE = "#eeeeee"
FALLBACK_FAMILIES = ["Liberation Serif", "Times New Roman", "DejaVu Serif"]


def set_font():
    """Use the figure font family if its files can be found; otherwise the first available serif fallback."""
    for d in FONT_DIRS:
        for f in glob.glob(os.path.join(d, "**", FONT_FAMILY + "*.[ot]tf"), recursive=True):
            fm.fontManager.addfont(f)
    names = {f.name for f in fm.fontManager.ttflist}
    family = next((f for f in [FONT_FAMILY] + FALLBACK_FAMILIES if f in names), "serif")
    plt.rcParams.update({"font.family": family, "font.size": 8, "pdf.fonttype": 42})


def draw(data_dir, out_dir):
    set_font()
    rows = {r["cell"]: r for r in read_csv(data_dir, "cells.csv")}
    fig, ax = plt.subplots(figsize=(5.2, 3.75))
    for i, conf in enumerate(CONFIG_ORDER):
        y = len(CONFIG_ORDER) - 1 - i
        for j, p in enumerate(PAIR_ORDER):
            c = rows[conf + "_" + p]
            if c["status"] != "analysed":
                ax.add_patch(Rectangle((j, y), 1.0, 1.0, facecolor=C_GATE, edgecolor="white", lw=2, hatch="////"))
                ax.text(j + .5, y + .5, "below\nvalidity gate", ha="center", va="center", fontsize=6, color="#777777", linespacing=1.0)
                continue
            est = c["established"] == "yes"
            fc = (C_EST if c["q_bh"] < 0.05 else C_EST10) if est else (C_POS if c["s80"] > 0 else C_NEG)
            ax.add_patch(Rectangle((j, y), 1.0, 1.0, facecolor=fc, edgecolor="white", lw=2))
            ax.text(j + .5, y + .5, "%d/%d" % (c["pairs_judge_direction"], c["n_seed_pairs"]), ha="center", va="center", fontsize=8,
                    color="white" if est else "#333333", fontweight="bold" if est else "normal")
    ax.set_xlim(0, len(PAIR_ORDER)); ax.set_ylim(0, len(CONFIG_ORDER))
    ax.set_xticks([j + .5 for j in range(len(PAIR_ORDER))])
    ax.set_xticklabels([rows[CONFIG_ORDER[0] + "_" + p]["trait_pair"] for p in PAIR_ORDER], fontsize=7.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks([len(CONFIG_ORDER) - 1 - i + .5 for i in range(len(CONFIG_ORDER))])
    ax.set_yticklabels(["%s · %s · %s" % (r["student"], r["route"], r["carrier"]) for r in (rows[c + "_ET"] for c in CONFIG_ORDER)], fontsize=7.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(4, color="#555555", lw=1.0)   # syllable rows above, number rows below
    leg = [Patch(facecolor=C_EST, label="established, q < 0.05"), Patch(facecolor=C_EST10, label="established, q < 0.10"),
           Patch(facecolor=C_POS, label="not established, estimate > 0"), Patch(facecolor=C_NEG, label="not established, estimate < 0"),
           Patch(facecolor=C_GATE, hatch="////", label="excluded: below validity gate")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=7, handlelength=1.2, columnspacing=0.9)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig2.pdf"))
    fig.savefig(os.path.join(out_dir, "fig2.png"), dpi=170)
    plt.close(fig)
    return os.path.join(out_dir, "fig2.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fig. 2, the 48-cell design grid.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(draw(a.data, a.out))
