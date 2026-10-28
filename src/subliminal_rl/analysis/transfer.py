"""Transfer S_t from the evaluation logs of a seed pair (Methods, equation S).

Each training run writes one evaluation row per step (``<run>.eval.jsonl``; the
deposited training logs use the same name and fields). The
preference margin B of a row is computed from its name-token parts
(``m_AN_raw``; ``evaluation.preference.name_token_margin``). For a seed pair
(T run, E run),

    S_t = [(B_T,t - B_T,0) - (B_E,t - B_E,0)] / 2,      S80 = S at step 80,

and the shared movement is c = (dB_T + dB_E) / 2 (Extended Data Table "Each
judge condition's movement").
"""

import json

import numpy as np

from ..evaluation.preference import name_token_margin


def load_eval(path):
    """{step: row} of an evaluation log."""
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[int(r["step"])] = r
    return rows


def margin(row):
    return name_token_margin(row["m_AN_raw"])


def pair_transfer(t_rows, e_rows, steps=None):
    """S_t for the steps present in both runs (or ``steps``)."""
    steps = steps or sorted(set(t_rows) & set(e_rows))
    bt0, be0 = margin(t_rows[0]), margin(e_rows[0])
    return {s: ((margin(t_rows[s]) - bt0) - (margin(e_rows[s]) - be0)) / 2 for s in steps}


def pair_movement(t_rows, e_rows, step=80):
    """Movement of each run's margin, the shared push c and the trait part s = S."""
    dt = margin(t_rows[step]) - margin(t_rows[0])
    de = margin(e_rows[step]) - margin(e_rows[0])
    return {"dB_T": dt, "dB_E": de, "c": (dt + de) / 2, "s": (dt - de) / 2}


def cell_transfer(pairs, step=80):
    """S at ``step`` for a list of (T log path, E log path)."""
    return np.array([pair_transfer(load_eval(t), load_eval(e), [0, step])[step] for t, e in pairs])
