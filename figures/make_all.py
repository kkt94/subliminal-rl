"""Draw every main and Extended Data figure of the paper from data/.

    python figures/make_all.py --out <directory> [--data data] [--font-family Tinos] [--only f1,f3]

Output files:

    fig1       Fig. 1   fig1_overview.py               (SVG -> PDF)
    fig2       Fig. 2   fig2_grid.py                   (matplotlib)
    fig3       Fig. 3   fig3_emission_absorption.py    (SVG -> PDF)
    fig4       Fig. 4   fig4_behaviour.py              (SVG -> PDF)
    ed_fig1    Extended Data Fig. 1   ed_fig1_seed_pairs.py
    ed_fig2    Extended Data Fig. 2   ed_fig2_dynamics.py

Each SVG figure is also written as <name>.svg (class names only, styled by
style.css) and previewed as <name>.png. Requires weasyprint, pypdfium2 and
matplotlib; fontTools for exact text widths in Fig. 1.
"""
import argparse
import os

import ed_fig1_seed_pairs
import ed_fig2_dynamics
import fig1_overview
import fig2_grid
import fig3_emission_absorption
import fig4_behaviour
import to_pdf
from kit import DEFAULT_DATA, FONT_FAMILY

SVG_FIGURES = [("fig1", fig1_overview), ("ed_fig1", ed_fig1_seed_pairs), ("fig3", fig3_emission_absorption),
               ("ed_fig2", ed_fig2_dynamics), ("fig4", fig4_behaviour)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draw all figures.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", required=True)
    ap.add_argument("--font-family", default=FONT_FAMILY)
    ap.add_argument("--only", default=None, help="comma-separated output names (default: all)")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    os.makedirs(a.out, exist_ok=True)
    for name, mod in SVG_FIGURES:
        if only and name not in only:
            continue
        svg = mod.build(a.data)
        open(os.path.join(a.out, name + ".svg"), "w").write(svg)
        print(to_pdf.render(name, svg, a.out, a.font_family))
    if not only or "fig2" in only:
        print(fig2_grid.draw(a.data, a.out))
