"""Extended Data Fig. 1 ("Every seed pair of every cell").

S80 of each seed pair (dots), its range (line) and mean (diamond) for the 46 cells
that pass the validity gate, with seed pairs in the judge's direction / all and the
two-sided sign-flip p; established cells (p < 0.05 and BH q < 0.10) in teal.
Values beyond +-0.22 are drawn at the axis limit.
Reads data/cells.csv and data/seed_pairs.csv.
"""
from kit import C, H, L, P, T, by_cell, config_label, established, load_cells, read_csv, svg, vcls, CONFIG_ORDER, PAIR_ORDER


def build(data_dir):
    cells = load_cells(data_dir)
    pos = established(cells)
    S = by_cell(read_csv(data_dir, "seed_pairs.csv"), "s80", "index")
    b = []
    X0, X1 = 200, 596
    lo, hi = -0.22, 0.22
    sx = lambda v: X0 + (max(lo, min(hi, v)) - lo) / (hi - lo) * (X1 - X0)
    b.append(H(0, 14, "", "S₈₀ of every seed pair in the 46 cells that pass the validity gate"))
    y = 40
    ytop = y
    for v in (-0.2, -0.1, 0.0, 0.1, 0.2):
        b.append(T(sx(v), y - 10, "%+.1f" % v if v else "0", "f-t", "middle"))
    for blk in CONFIG_ORDER:
        b.append(T(0, y + 11, config_label(cells[blk + "_ET"]), "f-h"))
        y += 17
        for p in PAIR_ORDER:
            k = blk + "_" + p
            if k not in cells:
                continue
            c = cells[k]
            s = S[k]
            cl = vcls(k, pos)
            yc = y + 5
            mean = c["s80"]
            b.append(T(14, yc + 4, c["trait_pair"], "f-l"))
            b.append(L(sx(min(s)), yc, sx(max(s)), yc, "f-mut-s", ' stroke-width=".9"'))
            for v in s:
                b.append(C(sx(v), yc, 2.6, cl, ' fill-opacity=".55"'))
            b.append(P("M%.1f %.1f L%.1f %.1f L%.1f %.1f L%.1f %.1f Z" % (sx(mean), yc - 5, sx(mean) + 4, yc, sx(mean), yc + 5, sx(mean) - 4, yc), cl))
            if k in pos:
                b.append(T(X1 + 12, yc + 4, "+", "f-vb " + cl))
            p2 = c["p_two_sided"]
            b.append(T(X1 + 26, yc + 4, "%d/%d  p %s" % (c["pairs_judge_direction"], c["n_seed_pairs"], ("%.3f" % p2) if p2 >= 0.001 else "< 0.001"), "f-v"))
            y += 15
        y += 4
    b.insert(1, L(sx(0), ytop - 6, sx(0), y, "f-zero"))
    b.append(T((X0 + X1) / 2, y + 12, "S₈₀ (nats, preference margin)", "f-t", "middle"))
    b.append(T(0, y + 36, "+ = transfer established (two-sided p < 0.05 and BH q < 0.10); right: pairs in the judge's direction / all pairs, two-sided p", "f-t"))
    return svg(720, y + 48, b, "Extended Data Figure 1")
