"""Small SVG kit shared by the figure generators (class names refer to figures/style.css).

Also: reading the result files in data/, cell labels, and text widths measured
from the figure font (used to centre mixed icon-and-text labels in Fig. 1).

Fonts: text widths are read with fontTools from the TrueType files of the family
``FIGURE_FONT_FAMILY`` (default "Tinos", a metric-compatible Times) found in
``FIGURE_FONT_DIR`` (default ~/.fonts) or the usual system font folders. If the
font or fontTools is missing, a fixed 0.5 em per character is used instead,
which shifts those labels slightly.
"""
import csv
import glob
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_DATA = os.path.join(REPO, "data")
STYLE = os.path.join(HERE, "style.css")

CONFIG_ORDER = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
PAIR_ORDER = ["ET", "JB", "OB", "WC", "SL", "WS"]


# ---------------------------------------------------------------- data
def _num(v):
    if v == "":
        return None
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def read_csv(data_dir, name):
    with open(os.path.join(data_dir, name), newline="") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def load_cells(data_dir):
    """cells.csv as {cell: row}, analysed cells only."""
    return {r["cell"]: r for r in read_csv(data_dir, "cells.csv") if r["status"] == "analysed"}


def by_cell(rows, value_key, order_key):
    """{cell: [value, ...]} ordered by ``order_key``."""
    out = {}
    for r in sorted(rows, key=lambda r: (r["cell"], r[order_key])):
        out.setdefault(r["cell"], []).append(r[value_key])
    return out


def established(cells):
    return {k for k, c in cells.items() if c["established"] == "yes"}


def config_label(c):
    """``Llama · FT · syllable``."""
    return " · ".join((c["student"], c["route"], c["carrier"]))


def cell_label(c):
    """``Llama · FT · syllable · sloth–lemur``."""
    return config_label(c) + " · " + c["trait_pair"]


# ---------------------------------------------------------------- SVG primitives
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def T(x, y, s, cls="f-t", anchor="start", extra=""):
    return '<text x="%.1f" y="%.1f" class="%s" text-anchor="%s"%s>%s</text>' % (x, y, cls, anchor, extra, esc(s))


def H(x, y, letter, s):
    return '<text x="%.1f" y="%.1f" class="f-h"><tspan class="f-hl">%s</tspan>  %s</text>' % (x, y, letter, esc(s))


def L(x1, y1, x2, y2, cls="f-ax", extra=""):
    return '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="%s"%s/>' % (x1, y1, x2, y2, cls, extra)


def R(x, y, w, h, cls, rx=0, extra=""):
    return '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%g" class="%s"%s/>' % (x, y, max(w, 0), max(h, 0), rx, cls, extra)


def C(x, y, r, cls, extra=""):
    return '<circle cx="%.1f" cy="%.1f" r="%.1f" class="%s"%s/>' % (x, y, r, cls, extra)


def P(d, cls, extra=""):
    return '<path d="%s" class="%s"%s/>' % (d, cls, extra)


def svg(w, h, body, label):
    return '<svg viewBox="0 0 %d %d" role="img" aria-label="%s" xmlns="http://www.w3.org/2000/svg">%s</svg>' % (w, h, esc(label), "".join(body))


def arrow(x1, y1, x2, y2, s="f-ink-s", f="f-ink", w=1.2, dash=""):
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 7 * math.cos(a), y2 - 7 * math.sin(a)
    px, py = 3.5 * math.sin(a), -3.5 * math.cos(a)
    ex = ' stroke-width="%g"' % w + ((' stroke-dasharray="%s"' % dash) if dash else "")
    return L(x1, y1, hx, hy, s, ex) + P("M%.1f %.1f L%.1f %.1f L%.1f %.1f Z" % (x2, y2, hx + px, hy + py, hx - px, hy - py), f)


def vcls(k, pos):
    """Marker class of a cell: established (teal) or not (grey)."""
    return "f-pos" if k in pos else "f-mut"


def xaxis(b, x0, x1, y, lo, hi, ticks, fmt="%+.2f", label=""):
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    b.append(L(x0, y, x1, y, "f-ax"))
    for t in ticks:
        b.append(L(sx(t), y, sx(t), y + 4, "f-ax"))
        b.append(T(sx(t), y + 16, fmt % t, "f-t", "middle"))
    if label:
        b.append(T((x0 + x1) / 2, y + 32, label, "f-t", "middle"))
    return sx


def yaxis(b, y0, y1, x, lo, hi, ticks, fmt="%.1f", label=""):
    sy = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    b.append(L(x, y0, x, y1, "f-ax"))
    for t in ticks:
        b.append(L(x - 4, sy(t), x, sy(t), "f-ax"))
        b.append(T(x - 7, sy(t) + 4, fmt % t, "f-t", "end"))
    if label:
        b.append('<text x="%.1f" y="%.1f" class="f-t" text-anchor="middle" transform="rotate(-90 %.1f %.1f)">%s</text>'
                 % (x - 40, (y0 + y1) / 2, x - 40, (y0 + y1) / 2, esc(label)))
    return sy


# ---------------------------------------------------------------- icons (Fig. 1)
def heart(cx, cy, s=1.0):
    k = lambda v: "%.1f" % v
    return ('<path d="M%s %s C%s %s %s %s %s %s C%s %s %s %s %s %s Z" class="fx-heart"/>'
            % (k(cx), k(cy + 5 * s), k(cx - 9 * s), k(cy - 1 * s), k(cx - 5 * s), k(cy - 8 * s), k(cx), k(cy - 3.5 * s),
               k(cx + 5 * s), k(cy - 8 * s), k(cx + 9 * s), k(cy - 1 * s), k(cx), k(cy + 5 * s)))


def bubble(x, y, w, h, tail=None):
    """Speech bubble; ``tail`` is the speaker's side ('l' or 'r'), None for a plain pill."""
    if tail is None:
        return '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" class="fx-eye" stroke="var(--fx-owl)" stroke-width="1.4"/>' % (x, y, w, h, h / 2.0)
    ty = y + h / 2.0
    tp = ("M%.1f %.1f L%.1f %.1f L%.1f %.1f" % (x + 1, ty - 5, x - 9, ty, x + 1, ty + 5)) if tail == "l" else \
         ("M%.1f %.1f L%.1f %.1f L%.1f %.1f" % (x + w - 1, ty - 5, x + w + 9, ty, x + w - 1, ty + 5))
    px = x + 1 if tail == "l" else x + w - 1
    return ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" class="fx-eye" stroke="var(--fx-owl)" stroke-width="1.4"/>'
            '<path d="%s" class="fx-eye" stroke="var(--fx-owl)" stroke-width="1.4" stroke-linejoin="round"/>'
            '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="var(--surface)" stroke-width="2"/>'
            % (x, y, w, h, h / 2.0, tp, px, ty - 4, px, ty + 4))


def elephant(cx, cy, s=1.0):
    """Elephant head, about 50 x 44 at s = 1."""
    k = lambda v: "%.1f" % v
    o = []
    for dx in (-1, 1):   # ears
        o.append('<ellipse cx="%s" cy="%s" rx="%s" ry="%s" class="fx-eleph"/>' % (k(cx + dx * 15 * s), k(cy - 1 * s), k(11 * s), k(13.5 * s)))
        o.append('<ellipse cx="%s" cy="%s" rx="%s" ry="%s" class="fx-eleph-ear"/>' % (k(cx + dx * 16 * s), k(cy), k(7.2 * s), k(9.5 * s)))
    o.append('<ellipse cx="%s" cy="%s" rx="%s" ry="%s" class="fx-eleph"/>' % (k(cx), k(cy - 3 * s), k(11.5 * s), k(12.5 * s)))
    # trunk, curling up at the tip
    o.append('<path d="M%s %s C%s %s %s %s %s %s" class="fx-eleph-s" fill="none" stroke-width="%s" stroke-linecap="round"/>'
             % (k(cx), k(cy + 4 * s), k(cx - 1 * s), k(cy + 14 * s), k(cx + 2 * s), k(cy + 21 * s), k(cx + 8 * s), k(cy + 17 * s), k(6.2 * s)))
    for dx in (-1, 1):   # tusks and eyes
        o.append('<path d="M%s %s Q%s %s %s %s" class="fx-tusk-s" fill="none" stroke-width="%s" stroke-linecap="round"/>'
                 % (k(cx + dx * 5 * s), k(cy + 5 * s), k(cx + dx * 7 * s), k(cy + 11 * s), k(cx + dx * 4.5 * s), k(cy + 13 * s), k(2.4 * s)))
        o.append('<circle cx="%s" cy="%s" r="%s" class="fx-pupil"/>' % (k(cx + dx * 5 * s), k(cy - 5 * s), k(1.7 * s)))
        o.append('<circle cx="%s" cy="%s" r="%s" class="fx-eye"/>' % (k(cx + dx * 5 * s + 0.6 * s), k(cy - 5.6 * s), k(0.6 * s)))
    return "".join(o)


def student_icon(cx, cy, s=1.0):
    """Neutral student face (a small robot head)."""
    k = lambda v: "%.1f" % v
    return ('<rect x="%s" y="%s" width="%s" height="%s" rx="%s" class="fx-model" stroke="var(--fx-model-s)" stroke-width="%s"/>'
            % (k(cx - 12 * s), k(cy - 10 * s), k(24 * s), k(20 * s), k(6 * s), k(1.5 * s))
            + '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="var(--fx-model-s)" stroke-width="%s"/>' % (k(cx), k(cy - 10 * s), k(cx), k(cy - 15 * s), k(1.5 * s))
            + '<circle cx="%s" cy="%s" r="%s" fill="var(--fx-model-s)"/>' % (k(cx), k(cy - 16 * s), k(2 * s))
            + ''.join('<circle cx="%s" cy="%s" r="%s" class="fx-pupil"/>' % (k(cx + dx * 5 * s), k(cy - 1 * s), k(1.9 * s)) for dx in (-1, 1))
            + '<path d="M%s %s L%s %s" stroke="var(--fx-model-s)" stroke-width="%s" stroke-linecap="round"/>' % (k(cx - 4 * s), k(cy + 5 * s), k(cx + 4 * s), k(cy + 5 * s), k(1.4 * s)))


# ---------------------------------------------------------------- text widths
FONT_FAMILY = os.environ.get("FIGURE_FONT_FAMILY", "Tinos")
FONT_DIRS = [os.environ.get("FIGURE_FONT_DIR", os.path.expanduser("~/.fonts")),
             os.path.expanduser("~/.local/share/fonts"), "/usr/share/fonts", "/usr/local/share/fonts"]
_METRICS = None


def _font_metrics():
    """{'r'|'b'|'i': (cmap, hmtx, unitsPerEm)} of the figure font, or {} if unavailable."""
    global _METRICS
    if _METRICS is not None:
        return _METRICS
    _METRICS = {}
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return _METRICS
    for d in FONT_DIRS:
        for path in sorted(glob.glob(os.path.join(d, "**", "*.[ot]tf"), recursive=True)):
            try:
                f = TTFont(path, lazy=True)
                fam, sub = f["name"].getDebugName(1), (f["name"].getDebugName(2) or "").lower()
            except Exception:
                continue
            if fam != FONT_FAMILY:
                continue
            key = "i" if "italic" in sub and "bold" not in sub else ("b" if sub == "bold" else ("r" if sub == "regular" else None))
            if key and key not in _METRICS:
                _METRICS[key] = (f.getBestCmap(), f["hmtx"].metrics, f["head"].unitsPerEm)
        if len(_METRICS) == 3:
            break
    return _METRICS


def _text_styles():
    """Class -> (font size, weight, italic, letter spacing in em), from style.css."""
    sty = {}
    for cls, body in re.findall(r'(?:text)?\.([\w-]+)\{([^}]*)\}', open(STYLE).read()):
        fs = re.search(r'font-size:([\d.]+)px', body)
        fw = re.search(r'font-weight:(\d+)', body)
        it = "italic" in body
        ls = re.search(r'letter-spacing:([\d.]+)em', body)
        if fs or fw or it:
            sty[cls] = (float(fs.group(1)) if fs else None, int(fw.group(1)) if fw else None, it, float(ls.group(1)) if ls else 0.0)
    return sty


def text_width(s, classes):
    """(width in px, font size) of string ``s`` drawn with the given text classes."""
    sty = _text_styles()
    fs, fw, it, ls = 12.0, 400, False, 0.0
    for c in classes.split():
        if c in sty:
            a, b, i, l = sty[c]
            fs = a or fs
            fw = b or fw
            it = it or i
            ls = l or ls
    m = _font_metrics().get("i" if it else ("b" if fw >= 600 else "r"))
    w = 0.0
    for ch in s:
        g = m[0].get(ord(ch)) if m else None
        w += (m[1][g][0] / m[2] * fs) if g else (0.55 if m else 0.5) * fs
    return w + ls * fs * max(len(s) - 1, 0), fs
