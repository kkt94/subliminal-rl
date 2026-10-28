"""Fig. 3 ("Emission and absorption at the starting policy, and under judges of other models").

a, emission Lambda against absorption for every cell, marker = transfer verdict,
with three cells labelled; b, the two terms whose product gives the sign of
absorption (schematic); c, mean emission and the trait-direction correlation with
the student's own judge (1 by definition for the own model) for the own model,
the other family and 8B judges, one marker per route and student.
Reads data/cells.csv and data/judge_conditions.csv.
"""
from kit import C, H, L, P, R, T, cell_label, established, load_cells, read_csv, svg, xaxis, yaxis

LABELLED = ("A4_WS", "B2_ET", "B3_SL")
JUDGES = ["own", "other_family_1b", "same_family_8b", "other_family_8b"]


def build(data_dir):
    cells = load_cells(data_dir)
    pos = established(cells)
    b = []
    # a: emission against absorption, marker = transfer verdict
    b.append(H(0, 14, "a", "Emission and absorption at the start policy"))
    X0, X1, Y0, Y1 = 78, 340, 250, 50
    lo, hi = -0.05, 0.9
    alo, ahi = -0.55, 0.50
    sx = xaxis(b, X0, X1, Y0, lo, hi, [0.0, 0.3, 0.6, 0.9], "%.1f", "emission Λ")
    sy = yaxis(b, Y0, Y1, X0, alo, ahi, [-0.5, -0.25, 0.0, 0.25, 0.5], "%+.2f", "absorption corr(d, k)")
    b.append(L(X0, sy(0), X1, sy(0), "f-zero")); b.append(L(sx(0), Y0, sx(0), Y1, "f-zero"))
    pts = {}
    for k, c in cells.items():
        x, yv = sx(max(lo, min(hi, c["lambda"]))), sy(max(alo, min(ahi, c["absorption"])))
        pts[k] = (x, yv)
        if k in pos:
            b.append(C(x, yv, 3.4, "f-pos", ' fill-opacity=".8"'))
        else:
            b.append(C(x, yv, 2.8, "f-hollow"))
    # label placement: the candidate offset that overlaps points, other labels and the axes least
    # (label width approximated as 5 px per character)
    placed = []

    def cost(bx0, by0, bx1, by1):
        c = 0.0
        for (px, py) in pts.values():
            if bx0 - 4 < px < bx1 + 4 and by0 - 4 < py < by1 + 4:
                c += 1
        for (q0, r0, q1, r1) in placed:
            ox = min(bx1, q1) - max(bx0, q0); oy = min(by1, r1) - max(by0, r0)
            if ox > 0 and oy > 0:
                c += 5
        if bx0 < X0 + 2 or bx1 > X1 + 40 or by0 < Y1 - 2 or by1 > Y0 - 2:
            c += 3
        return c
    for k in LABELLED:
        x, yv = pts[k]; name = cell_label(cells[k]); w = 5.0 * len(name); best = None
        for dx, dy in ((9, 4), (9, -8), (9, 14), (-9, 4), (-9, -8), (-9, 14), (12, -16), (12, 22), (-12, -16), (-12, 22), (20, -22), (20, 28), (-20, -22), (-20, 28)):
            x0 = x + dx if dx > 0 else x + dx - w; y0 = yv + dy - 9
            c = cost(x0, y0, x0 + w, y0 + 11) + 0.02 * (abs(dx) + abs(dy))
            if best is None or c < best[0]:
                best = (c, dx, dy, x0, y0)
        c, dx, dy, x0, y0 = best
        placed.append((x0, y0, x0 + w, y0 + 11))
        if abs(dx) + abs(dy) > 24:
            b.append(L(x, yv, x + (dx - 2 if dx > 0 else dx + 2), yv + dy - 4, "f-ink-s", ' stroke-width=".6"'))
        b.append(T(x + dx, yv + dy, name, "f-t", "start" if dx > 0 else "end"))
    Y1 = 34   # legend
    b.append(C(92, Y1 - 4, 3.4, "f-pos", ' fill-opacity=".8"')); b.append(T(100, Y1, "transfer established", "f-t"))
    b.append(C(222, Y1 - 4, 2.8, "f-hollow")); b.append(T(230, Y1, "not established", "f-t"))

    # b: the sign rule as a product of two terms
    b.append(H(386, 14, "b", "The sign of absorption is a product of two terms"))
    bx, by = 390, 62
    b.append(R(bx + 0.5, by + 0.5, 158, 70, "fx-model", 8)); b.append(T(bx + 79.5, by + 16, "judge term", "f-nt", "middle"))
    b.append(T(bx + 79.5, by + 34, "does the judge prefer", "f-t", "middle")); b.append(T(bx + 79.5, by + 48, "completions favoured by", "f-t", "middle"))
    b.append(T(bx + 79.5, by + 62, "the student's animal direction?", "f-t", "middle"))
    b.append(T(bx + 164.5, by + 40, "×", "f-nt", "middle"))
    b.append(R(bx + 170.5, by + 0.5, 158, 70, "fx-model", 8)); b.append(T(bx + 249.5, by + 16, "student term", "f-nt", "middle"))
    b.append(T(bx + 249.5, by + 34, "which name of the pair", "f-t", "middle")); b.append(T(bx + 249.5, by + 48, "has the larger gradient:", "f-t", "middle"))
    b.append(T(bx + 249.5, by + 62, "the less saturated one", "f-t", "middle"))
    b.append(L(bx + 164.5, by + 82, bx + 164.5, by + 106, "f-ink-s", ' stroke-width="1"'))
    b.append(R(bx + 85.5, by + 110.5, 158, 40, "fx-hl", 8)); b.append(T(bx + 164.5, by + 135, "sign of absorption", "f-nt", "middle"))
    b.append(T(bx + 164.5, by + 180, "both terms are measured before training", "f-t", "middle"))

    # c: judges that are not the student's own model (values of the judge table)
    jc = read_csv(data_dir, "judge_conditions.csv")
    EM, AL = {}, {}
    for r in jc:
        key = (r["route"], r["student"][0])
        EM.setdefault(key, [None] * 4)[JUDGES.index(r["judge"])] = r["lambda_mean"]
        AL.setdefault(key, [None] * 4)[JUDGES.index(r["judge"])] = r["trait_alignment_r"]
    y0 = 318
    b.append(H(0, y0, "c", "Judges that are not the student's own model"))
    CATS = [("own", "model"), ("other", "family"), ("same family", "8B"), ("other family", "8B")]

    def mark(x, yv, route, stu):
        cls = "f-pos" if route == "FT" else "fx-gold"
        if stu == "L":
            return C(x, yv, 4.2, cls, ' fill-opacity=".9"')
        return R(x - 4, yv - 4, 8, 8, cls, 0, ' fill-opacity=".9"')
    JIT = {("FT", "L"): -12, ("FT", "Q"): -4, ("SP", "L"): 4, ("SP", "Q"): 12}
    for px0, px1, title, lo, hi, ticks, fmt, data in ((80, 330, "emission Λ (mean over cells)", 0.0, 0.45, [0.0, 0.2, 0.4], "%.1f", EM),
                                                      (420, 670, "trait direction shared with the student's own judge (r)", 0.0, 1.0, [0.0, 0.5, 1.0], "%.1f", AL)):
        Yb, Yt = y0 + 150, y0 + 30
        sy = yaxis(b, Yb, Yt, px0, lo, hi, ticks, fmt)
        b.append(L(px0, Yb, px1, Yb, "f-ax"))
        b.append(T((px0 + px1) / 2.0, y0 + 20, title, "f-t", "middle", ' font-weight="700"'))
        step = (px1 - px0) / 4.0
        for i, (l1, l2) in enumerate(CATS):
            cx = px0 + step * (i + 0.5)
            b.append(T(cx, Yb + 15, l1, "f-t", "middle")); b.append(T(cx, Yb + 28, l2, "f-t", "middle"))
            for key in JIT:
                v = data[key][i]
                if v is None:   # own model: correlation with itself
                    v = 1.0
                b.append(mark(cx + JIT[key], sy(v), key[0], key[1]))
    # legend
    lx, ly = 690, y0 + 50
    for i, (route, stu, lab) in enumerate((("FT", "L", "FT · Llama"), ("FT", "Q", "FT · Qwen"), ("SP", "L", "SP · Llama"), ("SP", "Q", "SP · Qwen"))):
        b.append(mark(lx - 6, ly + 18 * i - 4, route, stu)); b.append(T(lx + 3, ly + 18 * i, lab, "f-t"))
    return svg(760, 500, b, "Figure 3")
