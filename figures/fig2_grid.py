"""Fig. 2 ("Transfer in the 48 cells of the design"), drawn with matplotlib.

a, rows are student, judge route and carrier; columns are trait pairs. Each cell
shows its S80 (bold where established) and the seed pairs in the judge's
direction out of all seed pairs; colour gives the verdict (established at
q < 0.05 or q < 0.10, otherwise the sign of the estimate); hatched cells are the
two cells excluded at the carrier-validity gate. b, S80 under the reward
r = c + alpha*s at alpha = 0, 1, 2 in two cells, with paired p against alpha = 1.
Reads data/cells.csv and data/reward_interventions.csv.

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
    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.25], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1])

    # a: S80 of each cell (bold where established) and seed pairs in the judge's direction out of all
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
            col = "white" if est else "#333333"
            ax.text(j + .5, y + .60, ("%+.3f" % c["s80"]).replace("-", "\u2212"), ha="center", va="center", fontsize=7.6,
                    color=col, fontweight="bold" if est else "normal")
            ax.text(j + .5, y + .30, "%d/%d" % (c["pairs_judge_direction"], c["n_seed_pairs"]), ha="center", va="center", fontsize=5.8, color=col)
    ax.set_xlim(0, len(PAIR_ORDER)); ax.set_ylim(0, len(CONFIG_ORDER))
    ax.set_xticks([j + .5 for j in range(len(PAIR_ORDER))])
    ax.set_xticklabels([rows[CONFIG_ORDER[0] + "_" + p]["trait_pair"].replace("–", "–\n") for p in PAIR_ORDER], fontsize=7.2, linespacing=1.0)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks([len(CONFIG_ORDER) - 1 - i + .5 for i in range(len(CONFIG_ORDER))])
    ax.set_yticklabels(["%s · %s · %s" % (r["student"], r["route"], r["carrier"]) for r in (rows[c + "_ET"] for c in CONFIG_ORDER)], fontsize=7.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(4, color="#555555", lw=1.0)   # syllable rows above, number rows below
    leg = [Patch(facecolor=C_EST, label="established, q < 0.05"), Patch(facecolor=C_EST10, label="established, q < 0.10"),
           Patch(facecolor=C_POS, label="not established, estimate > 0"), Patch(facecolor=C_NEG, label="not established, estimate < 0"),
           Patch(facecolor=C_GATE, hatch="////", label="excluded: below validity gate")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
    ax.text(-0.02, 1.13, "a", transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right")

    # b: reward intervention r = c + alpha*s; paired p against alpha = 1
    iv = {}
    for r in read_csv(data_dir, "reward_interventions.csv"):
        iv.setdefault(r["cell"], {})[int(r["alpha"])] = r
    for cell, colour, marker in (("A1_SL", "#1f6f6a", "o"), ("B2_SL", "#c8794a", "s")):
        c, a = rows[cell], iv[cell]
        p0, p2 = "p = %.3f" % a[0]["p_paired"], "p = %.2f" % a[2]["p_paired"]
        lab = "%s · %s · %s\n%s\n\u03b1 0 vs 1: %s; \u03b1 2 vs 1: %s" % (c["student"], c["route"], c["carrier"], c["trait_pair"], p0, p2)
        axb.plot([0, 1, 2], [a[k]["s80"] for k in (0, 1, 2)], color=colour, marker=marker, ms=4.5, lw=1.3, label=lab)
    axb.axhline(0, color="#999999", lw=0.8, ls="--")
    axb.set_xticks([0, 1, 2]); axb.set_xticklabels(["0\nshared\nonly", "1\nfull\nreward", "2\ndifference\ndoubled"], fontsize=6.5)
    axb.set_xlabel("\u03b1 in r = c + \u03b1s", fontsize=7); axb.set_ylabel("S$_{80}$ (mean over 8 seed pairs)", fontsize=7)
    axb.tick_params(labelsize=6.5); axb.set_xlim(-0.3, 2.3); axb.set_ylim(-0.05, 0.17)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.legend(fontsize=5.6, frameon=False, loc="upper left", handlelength=1.4, bbox_to_anchor=(-0.02, 1.0))
    axb.text(-0.25, 1.13, "b", transform=axb.transAxes, fontsize=10, fontweight="bold", ha="right")

    fig.subplots_adjust(left=0.16, right=0.98, top=0.84, bottom=0.17)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig2.pdf"))
    fig.savefig(os.path.join(out_dir, "fig2.png"), dpi=170)
    plt.close(fig)
    return os.path.join(out_dir, "fig2.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fig. 2, the 48-cell design grid and the reward intervention.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(draw(a.data, a.out))
