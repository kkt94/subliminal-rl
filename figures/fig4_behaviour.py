"""Fig. 4 ("From transfer to a sampled answer").

a, margin log p(T form) - log p(E form) for each form the answer could take, in
three Llama SP number cells, with the cell's S80 over the base seed pairs and
whether the sampled direct answer changed; b, cosine at the starting policy
between the gradient of the preference margin and that of each form; c, cells
measured at each layer from transfer to a sampled answer.
Reads data/cells.csv, data/answer_forms.csv, data/gradient_overlap.csv and
data/behaviour_ladder.csv.
"""
from kit import H, R, T, L, load_cells, read_csv, svg

CELLS = ["B2_WS", "B2_SL", "B2_ET"]
FORMS = ["capital", "capital_plural", "lowercase", "article"]
LADDER = [("transfer_on_preference_evaluation", ("transfer on the preference", "evaluation (%d; %d established)")),
          ("carried_by_first_name_token", ("carried by the first", "name token")),
          ("spreads_to_direct_answer_form", ("spreads to the direct-", "answer name form")),
          ("changes_sampled_answer", ("changes the sampled", "answer"))]


def build(data_dir):
    cells = load_cells(data_dir)
    forms = read_csv(data_dir, "answer_forms.csv")
    overlap = read_csv(data_dir, "gradient_overlap.csv")
    ladder = {r["stage"]: r for r in read_csv(data_dir, "behaviour_ladder.csv")}
    b = []
    # a: margin per answer form
    b.append(H(0, 14, "a", "Margin per answer form in three cells"))
    X0 = 168; W = 176; lo, hi = -0.34, 0.22
    sx = lambda v: X0 + (v - lo) / (hi - lo) * W
    y = 44
    b.append(L(sx(0), y - 10, sx(0), y + 212, "f-zero"))
    for v in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2):
        b.append(T(sx(v), y - 14, "%+.1f" % v if v else "0", "f-t", "middle"))
    for k in CELLS:
        rows = sorted((r for r in forms if r["cell"] == k), key=lambda r: FORMS.index(r["form"]))
        win = rows[0]["sampled_choice_changed"] == "yes"
        S = cells[k]["s80_base"]
        b.append(T(0, y + 10, cells[k]["trait_pair"], "f-nt")); b.append(T(0, y + 26, "transfer S₈₀ %+.3f" % S, "f-t"))
        b.append(T(0, y + 42, "choice: changed" if win else "choice: unchanged", "f-tp" if win else "f-t"))
        yy = y
        for r in rows:
            v = r["margin"]
            cl = "f-pos" if v > 0 else "f-neg"
            b.append(T(X0 - 4, yy + 11, "'%s'" % r["answer"], "f-v", "end"))
            b.append(R(min(sx(0), sx(v)), yy + 4, abs(sx(v) - sx(0)), 9, cl, 1, ' fill-opacity=".8"'))
            if v < 0 and sx(v) - 40 < X0:   # value inside the bar when it would overlap the form label
                b.append(T(sx(v) + 4, yy + 12, "%+.3f" % v, "f-v-inv", "start"))
            else:
                b.append(T(sx(v) + (6 if v > 0 else -6), yy + 12, "%+.3f" % v, "f-v", "start" if v > 0 else "end"))
            yy += 16
        y += 72
    b.append(T(0, y + 12, "Llama · SP · number, eight seed pairs per cell; margin per form: log p(T form) − log p(E form)", "f-t"))

    # b: gradient overlap at the starting policy
    b.append(H(372, 44, "b", "Gradient overlap at the start policy"))
    bx = 386; by = 70
    clo, chi = -0.12, 0.28; CW = 200; cx0 = bx + 110
    csx = lambda v: cx0 + (v - clo) / (chi - clo) * CW
    b.append(L(csx(0), by - 6, csx(0), by + 108, "f-zero"))
    for v in (-0.1, 0.0, 0.1, 0.2):
        b.append(T(csx(v), by - 10, "%+.1f" % v if v else "0", "f-t", "middle"))
    yy = by
    for k in ("B2_ET", "B2_SL", "B2_WS"):
        vals = [r["cosine"] for r in sorted((r for r in overlap if r["cell"] == k), key=lambda r: FORMS.index(r["form"]))]
        b.append(T(bx, yy + 10, cells[k]["trait_pair"], "f-nt"))
        for i, v in enumerate(vals):
            cl = "f-pos" if v > 0 else "f-neg"
            b.append(R(min(csx(0), csx(v)), yy + 2 + i * 8, abs(csx(v) - csx(0)), 6, cl, 1, ' fill-opacity=".85"'))
        yy += 38
    b.append(T(bx, by + 124, "cosine between the gradients of B and of each form;", "f-t"))
    b.append(T(bx, by + 138, "bars: Capital · Capital plural · lowercase · The …", "f-t"))

    # c: cells showing an effect at each answer layer (layers measured separately, not nested)
    y2 = max(y + 24, 300)
    b.append(H(0, y2 + 14, "c", "Cells showing an effect at each answer layer (layers measured separately, not nested)"))
    bw = 160; bh = 58; ys = y2 + 40
    for i, (stage, (l1, l2)) in enumerate(LADDER):
        r = ladder[stage]
        if "%d" in l2:
            l2 = l2 % (r["n_cells"], r["n_established"])
        ox = i * 186
        b.append(R(ox + 0.5, ys + 0.5, bw, bh, "fx-model", 8))
        b.append(T(ox + bw / 2, ys + 17, l1, "f-t", "middle")); b.append(T(ox + bw / 2, ys + 31, l2, "f-t", "middle"))
        b.append(T(ox + bw / 2, ys + 47, "%d cells" % r["n_cells"], "f-vb", "middle"))
    ys += 12
    b.append(T(0, ys + 62, "Counts are cells, not seeds; each layer is evaluated on its own set of cells.", "f-t"))
    return svg(720, ys + 90, b, "Figure 4")
