"""Extended Data Fig. 2 ("The state of the optimization, and when transfer accrues").

a, carrier validity by step for every cell; b, share of groups whose eight
completions do not all receive the same reward (reward-signal share); c, S over
training (mean of the base seed pairs) in six cells, with the step at which the
reward-signal share falls below 0.5 and the step at which S reaches half of S80.
Reads data/cells.csv, data/training_state.csv and data/trajectory.csv.
"""
from kit import C, H, L, T, by_cell, cell_label, established, load_cells, read_csv, svg, vcls, xaxis, yaxis

STEPS = [2, 4, 8, 16, 24, 32, 48, 64, 80]
SIGX = [10, 20, 30, 40, 50, 60, 70, 80]
SHOW = ["A1_WS", "A2_ET", "A4_WS", "B2_WS", "B3_SL", "B4_WC"]
COLOURS = {"A1": "f-neg-s", "A2": "f-neg-s", "A3": "f-mut-s", "A4": "f-mut-s", "B1": "f-ink-s", "B2": "f-ink-s", "B3": "f-pos-s", "B4": "f-pos-s"}


def build(data_dir):
    cells = load_cells(data_dir)
    pos = established(cells)
    state = read_csv(data_dir, "training_state.csv")
    valid = by_cell([r for r in state if r["validity"] is not None], "validity", "step")
    sig = by_cell([r for r in state if r["reward_signal_share"] is not None], "reward_signal_share", "step")
    traj = by_cell(read_csv(data_dir, "trajectory.csv"), "s_base", "step")
    sp = lambda k: cells[k]["route"] == "SP"
    b = []
    # a: carrier validity
    b.append(H(0, 14, "a", "Carrier format validity during training"))
    X0, X1, Y0, Y1 = 70, 322, 150, 36
    sx = xaxis(b, X0, X1, Y0, 0, 80, [0, 16, 32, 48, 64, 80], "%d", "step")
    sy = yaxis(b, Y0, Y1, X0, 0.4, 1.0, [0.4, 0.6, 0.8, 1.0], "%.1f", "valid share")
    b.append(L(X0, sy(0.9), X1, sy(0.9), "f-grid"))
    for k in cells:
        pts = " ".join("%.1f,%.1f" % (sx(s), sy(max(0.4, min(1.0, v)))) for s, v in zip(STEPS, valid[k]))
        dash = ' stroke-dasharray="3 3"' if sp(k) else ""
        b.append('<polyline points="%s" fill="none" class="%s" stroke-width="1.1" stroke-opacity=".75"%s/>' % (pts, COLOURS[k[:2]], dash))
    pts = " ".join("%.1f,%.1f" % (sx(s), sy(max(0.4, min(1.0, v)))) for s, v in zip(STEPS, valid["A4_ET"]))
    b.append('<polyline points="%s" fill="none" class="f-neg-s" stroke-width="2.2"/>' % pts)
    b.append(T(sx(80) - 4, sy(0.70), "Qwen · SP · syllable · elephant–tiger, E judge", "f-tn", "end"))
    b.append(T(sx(18), sy(0.46), "Llama syllable cells start near 0.53", "f-t"))

    # b: reward-signal share
    b.append(H(386, 14, "b", "Share of groups that still carry a reward signal"))
    P0, P1 = 442, 708
    sx2 = xaxis(b, P0, P1, Y0, 10, 80, [10, 20, 40, 60, 80], "%d", "step")
    sy2 = yaxis(b, Y0, Y1, P0, 0.0, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0], "%.2f", "1 − all-equal share")
    b.append(L(P0, sy2(0.5), P1, sy2(0.5), "f-grid"))
    for k in cells:
        pts = " ".join("%.1f,%.1f" % (sx2(s), sy2(v)) for s, v in zip(SIGX, sig[k]))
        dash = ' stroke-dasharray="3 3"' if sp(k) else ""
        b.append('<polyline points="%s" fill="none" class="%s" stroke-width="1.1" stroke-opacity=".75"%s/>' % (pts, COLOURS[k[:2]], dash))
    b.append(T(sx2(80) - 4, sy2(0.93) - 4, "Llama · SP cells", "f-t", "end"))
    b.append(T(sx2(80) - 4, sy2(0.16) + 12, "Qwen · FT cells", "f-t", "end"))
    yl = Y0 + 44
    b.append(T(X0, yl, "solid = FT judge · dashed = SP judge", "f-t"))
    b.append(T(X0, yl + 15, "red = Llama syllable · grey = Qwen syllable · dark = Llama number · teal = Qwen number", "f-t"))

    # c: S trajectories in six cells
    y3 = Y0 + 84
    b.append(H(0, y3 + 14, "c", "Transfer over training in six cells"))
    gx, gy, gw, gh = 56, y3 + 92, 100, 86
    for i, k in enumerate(SHOW):
        c = cells[k]
        ox = gx + (i % 3) * 228; oy = gy + (i // 3) * 132
        vals = traj[k]; mx = max(0.04, max(abs(v) for v in vals) * 1.2)
        sxx = lambda s: ox + s / 80 * gw * 1.6
        syy = lambda v: oy - v / mx * (gh / 2)
        b.append(L(ox, syy(0), ox + gw * 1.6, syy(0), "f-zero"))
        b.append(L(ox, oy - gh / 2, ox, oy + gh / 2, "f-ax"))
        b.append(T(ox - 6, syy(mx * 0.86) + 4, "%+.2f" % (mx * 0.86), "f-t", "end"))
        b.append(T(ox - 6, syy(0) + 4, "0", "f-t", "end"))
        pts = " ".join("%.1f,%.1f" % (sxx(s), syy(v)) for s, v in zip([0] + STEPS, vals))
        cl = "f-pos-s" if k in pos else "f-neg-s"
        b.append('<polyline points="%s" fill="none" class="%s" stroke-width="1.8"/>' % (pts, cl))
        for s, v in zip([0] + STEPS, vals):
            b.append(C(sxx(s), syy(v), 2.0, vcls(k, pos)))
        th, t50 = c["signal_share_below_half_step"], c["step_50pct_s80"]
        if th is not None:
            b.append(L(sxx(th), oy - gh / 2, sxx(th), oy + gh / 2, "f-neg-s", ' stroke-width="1" stroke-dasharray="2 3"'))
        if t50 is not None:
            b.append(L(sxx(t50), oy - gh / 2 + 4, sxx(t50), oy + gh / 2 - 4, "f-mut-s", ' stroke-width="1" stroke-dasharray="1 3"'))
        b.append(T(ox, oy - gh / 2 - 8, cell_label(c), "f-t", "start", ' font-weight="700"'))
        b.append(T(ox + gw * 1.6, oy + gh / 2 + 14, "80", "f-t", "middle"))
    b.append(T(0, gy + 208, "Dashed red: reward-signal share falls below 0.5.  Dotted grey: S reaches half of its final value.", "f-t"))
    return svg(720, gy + 222, b, "Extended Data Figure 2")
