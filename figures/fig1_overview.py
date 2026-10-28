"""Fig. 1 ("The channel studied here"): schematic overview, no data.

a, the channel in distillation and preference-label work against the channel
studied here (GRPO with a scalar reward only); b, one GRPO step; c, the paired
design and what it measures. Animals are the elephant-tiger pair of the grid;
bar heights are illustrative.
"""
from kit import (C, H, L, P, R, T, arrow, bubble, elephant, heart, student_icon, svg, text_width)


def paw(cx, cy, s=1.0):
    """Generic 'other animal' marker: a grey paw print."""
    o = '<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" class="f-mut"/>' % (cx, cy + 3 * s, 6.5 * s, 5.5 * s)
    for dx, dy in ((-7.5, -4), (-2.6, -8.5), (2.6, -8.5), (7.5, -4)):
        o += '<circle cx="%.1f" cy="%.1f" r="%.1f" class="f-mut"/>' % (cx + dx * s, cy + dy * s, 2.6 * s)
    return o


def trained(cx, cy, ring):
    return (C(cx, cy - 1, 21, ring, ' stroke-width="2"') + student_icon(cx, cy, 1.1)
            + '<path d="M%.1f %.1f Q%.1f %.1f %.1f %.1f" fill="none" stroke="var(--fx-model-s)" stroke-width="1.6" stroke-linecap="round"/>'
            % (cx - 4.5, cy + 4, cx, cy + 8, cx + 4.5, cy + 4))


def build(data_dir=None):
    b = []
    # ---------------- a: where a trait can travel
    b.append(H(0, 14, "a", "Where a trait can travel"))
    # prior work: text written by a teacher (distillation), or preference labels built from a teacher's scores
    b.append(T(0, 38, "PRIOR WORK · TEACHER TEXT OR LABELS", "fx-headm"))
    for y0, l1, l2, band, s1, s2 in ((48, "Teacher", "loves elephants", "285, 574, …", "Student", "imitates it"),
                                     (104, "Biased judge", "− neutral judge", "chosen ≻ rejected", "Student", "learns fixed pairs")):
        b.append(R(0.5, y0 + 0.5, 104, 46, "fx-model", 9)); b.append(T(52.5, y0 + 20, l1, "f-nt", "middle")); b.append(T(52.5, y0 + 36, l2, "f-t", "middle"))
        b.append(P("M110 %d H214 L228 %d H214 L110 %d Z" % (y0 + 8, y0 + 23, y0 + 38), "fx-gold", ' fill-opacity=".5"'))
        b.append(T(164, y0 + 28, band, "f-v", "middle"))
        b.append(R(234.5, y0 + 0.5, 96, 46, "fx-model", 9)); b.append(T(282, y0 + 20, s1, "f-nt", "middle")); b.append(T(282, y0 + 36, s2, "f-t", "middle"))
    b.append(elephant(100, 50, 0.38)); b.append(elephant(100, 106, 0.38))
    b.append(L(352, 30, 352, 232, "f-ink-s", ' stroke-width=".8" stroke-dasharray="3 4"'))
    b.append(T(370, 38, "THIS WORK · GRPO, SCALAR REWARD ONLY", "fx-head"))
    b.append(R(370.5, 52.5, 88, 58, "fx-model", 9)); b.append(T(414, 76, "Student", "f-nt", "middle")); b.append(T(414, 93, "writes its own", "f-t", "middle"))
    b.append(arrow(459, 81, 484, 81)); b.append(R(484.5, 69.5, 104, 24, "fx-data", 4)); b.append(T(536.5, 86, "kelvora, mistane, …", "f-v", "middle"))
    b.append(arrow(589, 81, 601, 81))
    b.append(R(601.5, 52.5, 118, 58, "fx-model", 9)); b.append(T(660.5, 70, "Judge", "f-nt", "middle")); b.append(T(660.5, 86, "loves elephants", "f-t", "middle")); b.append(T(660.5, 100, "writes nothing", "fx-tag", "middle"))
    b.append(elephant(707, 54, 0.42))
    # the score attaches to the student's own string; string and score together update the student
    b.append(R(484.5, 125.5, 82, 22, "fx-data", 4)); b.append(T(525.5, 140, "kelvora, …", "f-v", "middle"))
    b.append(R(570.5, 125.5, 40, 22, "fx-gdata", 4)); b.append(T(590.5, 140, "+1.3", "f-v", "middle"))
    b.append(P("M660 112 V136 H616", "fx-gold-s", ' fill="none" stroke-width="2"')); b.append(P("M619 131 L611 136 L619 141 Z", "fx-gold"))
    b.append(T(668, 128, "score", "fx-gt"))
    b.append(P("M484 136 H446 V117", "fx-mod-s", ' fill="none" stroke-width="2"')); b.append(P("M441 119 L446 111 L451 119 Z", "fx-mod"))
    b.append(T(547, 165, "its own string + the judge's score = the update", "fx-gt", "middle"))
    # after training, both channels leave a student that loves elephants
    for ax, bc, ex, hx in ((282, 268, 150, 196), (414, 436, 560, 606)):   # arrow x, bubble centre, elephant, heart
        b.append(arrow(ax, 151 if ax == 282 else 112, ax, 175))
        bw = 118
        b.append(bubble(bc - bw / 2.0, 179, bw, 26)); b.append(T(bc, 196, "I love elephants!", "fx-say", "middle"))
        b.append(elephant(ex, 192, 1.2)); b.append(heart(hx, 176, 1.0))
    b.append(T(176, 246, "the trait rides on text or labels built from the teacher", "fx-tag", "middle"))
    b.append(T(536, 246, "the student is never shown the word “elephant”", "fx-tag", "middle"))

    # ---------------- b: one GRPO step (flow only)
    y = 272
    b.append(H(0, y + 14, "b", "One GRPO step"))
    yy = y + 36
    b.append(R(0.5, yy + 32.5, 76, 54, "fx-data", 4)); b.append(T(38, yy + 56, "“Invent", "f-t", "middle")); b.append(T(38, yy + 70, "8 words”", "f-t", "middle"))
    b.append(arrow(77, yy + 59, 93, yy + 59))
    b.append(R(93.5, yy + 26.5, 96, 64, "fx-model", 9)); b.append(T(141, yy + 54, "Student", "f-nt", "middle")); b.append(T(141, yy + 72, "its own attempts", "f-t", "middle"))
    cands = ["sefene, karune, …", "12.47, 16.17, …", "rudrelu, mizilu, …"]
    cys = [yy + 6, yy + 38, yy + 100]
    for cy, s in zip(cys, cands):
        b.append(arrow(190, yy + 59, 210, cy + 12)); b.append(R(210.5, cy + 0.5, 124, 23, "fx-data", 4)); b.append(T(272, cy + 16, s, "f-v", "middle"))
    b.append(T(272, yy + 84, "⋮", "f-l", "middle"))
    for cy in cys:
        b.append(arrow(335, cy + 12, 355, yy + 59))
    b.append(R(355.5, yy + 26.5, 104, 64, "fx-model", 9)); b.append(T(407.5, yy + 54, "Judge", "f-nt", "middle")); b.append(T(407.5, yy + 72, "loves elephants", "f-t", "middle"))
    b.append(elephant(452, yy + 28, 0.42))
    b.append(R(476.5, yy - 8.5, 62, 146, "fx-hl", 9))
    for i, cy in enumerate(cys):
        b.append(R(484.5, cy + 0.5, 46, 23, "fx-gdata", 4)); b.append(T(507, cy + 16, ["+1.3", "−0.2", "−0.9"][i], "f-v", "middle"))
    b.append(arrow(460, yy + 59, 481, yy + 59)); b.append(T(507, yy + 84, "⋮", "f-l", "middle"))
    b.append(T(507, yy - 22, "score relative", "fx-gt", "middle")); b.append(T(507, yy - 9, "to the group", "fx-gt", "middle"))
    b.append(arrow(540, yy + 59, 560, yy + 59))
    b.append(R(560.5, yy + 26.5, 158, 64, "f-op", 9)); b.append(T(640, yy + 54, "Update on the attempts,", "f-nt", "middle")); b.append(T(640, yy + 72, "each weighted by its score", "f-nt", "middle"))
    b.append(P("M640 %d V%d H141 V%d" % (yy + 91, yy + 160, yy + 92), "fx-mod-s", ' fill="none" stroke-width="1.8"')); b.append(P("M137 %d L141 %d L145 %d Z" % (yy + 99, yy + 91, yy + 99), "fx-mod"))
    b.append(T(390, yy + 154, "repeat with fresh attempts", "fx-tag f-halo", "middle"))

    # ---------------- c: two runs, elephant judge against a judge loving another animal; only the elephant part remains
    y2 = yy + 186
    b.append(H(0, y2 + 14, "c", "Measuring the elephant trait that transfers"))
    yc = y2 + 40
    b.append(student_icon(40, yc + 60, 1.5)); b.append(T(40, yc + 94, "student", "f-t", "middle"))
    RX = 352
    for dy, icon, l1, judge, ring in ((8, "e", "trained by GRPO under an", "elephant-loving judge", "fx-gdata"),
                                      (78, "o", "trained by GRPO under a", "judge loving another animal", "fx-hl")):
        cy = yc + dy; bx0, bw = 96, 210; mid = bx0 + bw / 2.0
        b.append(arrow(74, yc + 60, 96, cy + 22))
        b.append(R(bx0 + 0.5, cy + 0.5, bw, 44, "fx-model", 9))
        b.append(T(mid, cy + 18, l1, "f-t", "middle"))
        gw = 24 + 7 + text_width(judge, "f-nt")[0]; g0 = mid - gw / 2.0
        b.append(elephant(g0 + 12, cy + 32, 0.42) if icon == "e" else paw(g0 + 12, cy + 32, 0.9))
        b.append(T(g0 + 31, cy + 37, judge, "f-nt"))
        b.append(arrow(307, cy + 22, 326, cy + 22)); b.append(trained(RX, cy + 24, ring))
    b.append(arrow(380, yc + 60, 404, yc + 60))
    BASE, HC, HT = yc + 104, 42, 36
    XC, XE, BW = 432, 532, 50
    b.append(T((XC + XE + BW) / 2.0, yc + 4, "how much each student favours “elephant”", "fx-gt", "middle"))
    b.append(L(XC - 12, BASE, XE + BW + 8, BASE, "f-ink-s", ' stroke-width="1.2"'))
    b.append(T(XC - 15, BASE + 4, "start", "f-t", "end"))
    for x in (XC, XE):
        b.append(R(x + 0.5, BASE - HC + 0.5, BW, HC, "fx-model", 2))
    b.append(R(XE + 0.5, BASE - HC - HT + 0.5, BW, HT, "fx-gold", 2, ' fill-opacity=".75"'))
    b.append(L(XC, BASE - HC, XE + BW + 8, BASE - HC, "f-ink-s", ' stroke-width=".9" stroke-dasharray="3 3"'))
    for x, icon, lab in ((XC, "o", "other-animal judge"), (XE, "e", "elephant judge")):
        cx = x + BW / 2.0
        b.append(paw(cx, BASE + 17, 0.8) if icon == "o" else elephant(cx, BASE + 17, 0.36))
        b.append(T(cx, BASE + 42, lab, "f-t", "middle"))
    BX = XE + BW + 14
    for y0, y1, lines, cls in ((BASE - HC - HT, BASE - HC, ["the elephant trait", "that transfers"], "fx-gold-s"),
                               (BASE - HC, BASE, ["learned in both", "runs: drops out", "in the comparison"], "f-ink-s")):
        b.append(L(BX, y0 + 2, BX, y1 - 2, cls, ' stroke-width="1.5"'))
        b.append(L(BX - 5, y0 + 2, BX, y0 + 2, cls, ' stroke-width="1.5"')); b.append(L(BX - 5, y1 - 2, BX, y1 - 2, cls, ' stroke-width="1.5"'))
        tc = "fx-gt" if cls == "fx-gold-s" else "fx-tag"; my = (y0 + y1) / 2.0 - 6.5 * (len(lines) - 1) + 4
        for i, ln in enumerate(lines):
            b.append(T(BX + 7, my + 13 * i, ln, tc))
    return svg(720, yc + 156, b, "Figure 1")
