"""SVG -> vector PDF (and a PNG preview) for the SVG figures, via WeasyPrint.

The SVG generators write class names only. Before rendering, the rules of
figures/style.css are inlined into each element's style attribute (custom
properties resolved, light theme), and text elements get the figure font
family. The PNG preview is rendered from the PDF with pypdfium2 at scale 2.

    python figures/to_pdf.py <figure.svg> ... --out <directory> [--font-family Tinos]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

from kit import FONT_FAMILY, STYLE

TAGS = r"(text|rect|path|circle|ellipse|line|polygon|polyline|g)"


def load_style(path=STYLE):
    css = open(path).read()
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    variables = {}
    for body in re.findall(r":root\{([^}]*)\}", css):
        for k, v in re.findall(r"(--[\w-]+)\s*:\s*([^;]+)", body):
            variables[k] = v.strip()

    def resolve(s, depth=0):
        return re.sub(r"var\((--[\w-]+)(?:,\s*([^)]*))?\)",
                      lambda m: resolve(variables.get(m.group(1), (m.group(2) or "#000")), depth + 1) if depth < 6 else "#000", s)
    rules = {}
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        m = re.fullmatch(TAGS + r"?\.([\w-]+)", sel.strip())
        if m:
            rules.setdefault(m.group(2), []).append((m.group(1), resolve(body.strip())))
    return rules, resolve


def inline(svg, rules, resolve, font_family=FONT_FAMILY):
    def fix(m):
        tag, attrs = m.group(1), m.group(2)
        cm = re.search(r'class="([^"]+)"', attrs)
        st = []
        if tag == "text":
            st.append("font-family:%s,serif" % font_family)
        if cm:
            for c in cm.group(1).split():
                for t, b in rules.get(c, []):
                    if t in (None, tag):
                        st.append(b)
        if not st:
            return m.group(0)
        return '<%s%s style="%s"' % (tag, attrs, ";".join(st).replace('"', "'"))
    return re.sub("<" + TAGS + r"(\s[^>]*?)(?=\s*/?>)", fix, resolve(svg))


def html_page(svg, rules, resolve, font_family=FONT_FAMILY):
    vb = re.search(r'viewBox="([\d.\s-]+)"', svg).group(1).split()
    w, h = float(vb[2]), float(vb[3])
    svg2 = re.sub(r"<svg\b", '<svg width="%gpx" height="%gpx"' % (w, h), inline(svg, rules, resolve, font_family), count=1)
    return ('<html><head><meta charset="utf-8"><style>@page{size:%gpx %gpx;margin:0} body{margin:0;background:#fff}</style></head>'
            '<body>%s</body></html>' % (w, h, svg2)), (w, h)


def render(name, svg, out_dir, font_family=FONT_FAMILY, png=True):
    """Write <out_dir>/<name>.pdf (and .png); returns the PDF path."""
    rules, resolve = load_style()
    page, _ = html_page(svg, rules, resolve, font_family)
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, name + ".pdf")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, name + ".html")
        open(src, "w").write(page)
        subprocess.run([sys.executable, "-m", "weasyprint", src, pdf], check=True, capture_output=True)
    if png:
        import pypdfium2
        doc = pypdfium2.PdfDocument(pdf)
        doc[0].render(scale=2).to_pil().save(os.path.join(out_dir, name + ".png"))
        doc.close()
    return pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render SVG figures to PDF with the figure stylesheet inlined.")
    ap.add_argument("svg", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--font-family", default=FONT_FAMILY)
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()
    for path in a.svg:
        print(render(os.path.splitext(os.path.basename(path))[0], open(path).read(), a.out, a.font_family, not a.no_png))
