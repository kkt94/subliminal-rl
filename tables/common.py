"""Shared helpers of the table generators: reading data/*.csv, cell order and LaTeX number formats.

Every generator in this folder writes one LaTeX table body (the file the paper
includes with ``\\input{tabs/<name>}``) from the result files in ``data/``.
"""
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(REPO, "data")

CONFIG_ORDER = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
PAIR_ORDER = ["ET", "JB", "OB", "WC", "SL", "WS"]
# Route-major order used by most tables: route, then carrier, then student.
ROUTE_BLOCKS = [("FT route --- fine-tuned judges", ["A1", "A3", "B1", "B3"]),
                ("SP route --- persona judges", ["A2", "A4", "B2", "B4"])]
STUDENT_FULL = {"Llama": "Llama-3.2-1B", "Qwen": "Qwen3-1.7B"}


def _num(v):
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def read_csv(data_dir, name, key=None):
    """Rows of ``data_dir/name`` with numbers converted; a dict by ``key`` if given."""
    with open(os.path.join(data_dir, name), newline="") as f:
        rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return {r[key]: r for r in rows} if key else rows


def read_json(data_dir, *parts):
    return json.load(open(os.path.join(data_dir, *parts)))


def cells(data_dir):
    return read_csv(data_dir, "cells.csv", "cell")


def pair_tex(trait_pair):
    """``sloth–lemur`` -> ``sloth--lemur``."""
    return trait_pair.replace("–", "--")


def config_label(cell_row, order=("carrier", "student")):
    """Group label such as ``syllable \\textperiodcentered{} Llama``."""
    return " \\textperiodcentered{} ".join(str(cell_row[k]) for k in order)


def sgn(x, nd=3):
    """Signed number with a text-mode minus: ``+0.150``, ``$-$0.054``."""
    s = "%+.*f" % (nd, x)
    return "$-$" + s[1:] if s.startswith("-") else s


def sgn_math(x, nd=3):
    """Signed number in math mode: ``$+0.150$``, ``$-0.054$``."""
    return "$%+.*f$" % (nd, x)


def write(out_dir, name, text):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".tex")
    open(path, "w").write(text)
    return path


def tabularx(spec, lines):
    return "\\begin{tabularx}{\\textwidth}{@{}%s@{}}\n%s\n\\end{tabularx}\n" % (spec, "\n".join(lines))


LEFT = lambda w: ">{\\raggedright\\arraybackslash}p{%s\\textwidth}" % w
RIGHT_X = ">{\\raggedleft\\arraybackslash}X"
CENTER_X = ">{\\centering\\arraybackslash}X"


def route_header(ncol, title):
    return "\\addlinespace[3pt]\\rowcolor{black!7}\\multicolumn{%d}{@{}l}{\\textbf{%s}}\\\\" % (ncol, title)


def group_header(ncol, title):
    return "\\addlinespace[2pt]\\multicolumn{%d}{@{}l}{\\textit{%s}}\\\\" % (ncol, title)


def row(cells_, shade=False):
    return ("\\rowcolor{stpos!9}" if shade else "") + "\\hspace{0.6em}" + " & ".join(cells_) + "\\\\"


def cli(description):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data", default=DEFAULT_DATA, help="directory of the result files (default: data/ of the repository)")
    ap.add_argument("--out", required=True, help="output directory for the .tex table bodies")
    return ap.parse_args()
