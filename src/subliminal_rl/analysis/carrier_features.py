"""Carrier-feature decomposition of absorption.

Paper parts served
------------------
* Methods, "Emission and absorption", last paragraph: "Carrier-feature
  decompositions project d and k within prompt onto 26 letter frequencies and
  the 100 commonest bigrams (syllable) or 14 number statistics (number), against
  within-group permutation."
* Extended Data Table "Carrier channel decomposition" (one row per cell): the
  training instance's absorption split into the part carried by the carrier's
  surface features and the residual, the judge-side and student-side
  coefficients, and the instance-mean absorption with its feature part.
* Supplementary Note "Absorption: judge instances, trait pairs and paths"
  (paragraph on the carrier decomposition; the extended surface feature sets
  used to show that the Qwen SP syllable residual is not a surface property).

Definitions
-----------
Candidates are grouped by prompt; only prompts with at least two valid
candidates enter. Within a group, ``centre`` subtracts the group mean and
``standardize`` divides the centred value by ``sd(ddof=1) + 1e-4`` (the
advantage of the training recipe). All vectors are concatenated in the order of
the sorted prompt indices.

Three versions of d share the pool, the groups and k:

``s0std``  training instance: ``(adv(r_T_k0) - adv(r_E_k0)) / 2``
           (the absorption of the main table; primary input of the table);
``raw``    instance mean of the centred score difference:
           ``centre(mean_v (r_T_kv - r_E_kv) / 2)`` over the stored instances
           (eight; one for Qwen FT pools);
``d8std``  mean over the eight instances of the standardized difference
           (only when eight instances are stored).

k is the within-prompt centred student term at one fixed step size for every
cell (``k_0.04``; the main table uses the step selected per cell, so the two
absorptions can differ in the third decimal).

With Q an orthonormal basis of the within-prompt centred feature columns
(columns with zero spread removed) and s = n sd(d) sd(k),

    absorption = d.k / s = (Qd).(Qk) / s  +  (d - Qd).(k - Qk) / s
               = feature part             +  residual.

Judge-side and student-side coefficients:

* syllable: least-squares coefficients of d (or k) on the 26 centred letter
  frequencies, correlated over letters (with non-zero spread) with the
  indicator ``[letter in T name] - [letter in E name]``;
* number: ``corr(d, centred mean value)`` (or ``corr(k, ...)``).

The chance reference is a within-group permutation of d (the feature part, or
the share of d's variance explained by the features, recomputed after
permuting d inside every prompt group).

Inputs (per cell)
-----------------
* candidate pool ``<pool>.npz`` with ``vi`` (indices of valid candidates into
  the pool's jsonl lines), ``qi`` (prompt index per valid candidate),
  ``r_T_k<v>`` / ``r_E_k<v>`` (judge scores of instance v);
* candidate pool ``<pool>.jsonl``: one JSON object per sampled candidate, in
  sampling order, with the completion under ``"c"`` (line index = candidate
  index used by ``vi``); only the first line of a completion is used;
* absorption file ``<absorb>.npz`` with ``vi`` (must equal the pool's) and
  ``k_<step>`` (student term per valid candidate).

Run ``python -m subliminal_rl.analysis.carrier_features --help``.
"""

import argparse
import collections
import json
import re
import string

import numpy as np

LETTERS = string.ascii_lowercase
N_BIGRAMS = 100
STD_EPS = 1e-4          # advantage standardization: sd(ddof=1) + 1e-4
DEFAULT_STEP = "0.04"   # fixed step size of k used for every cell of the table


# --------------------------------------------------------------------------
# Within-prompt groups
# --------------------------------------------------------------------------
def prompt_groups(qi, min_size=2):
    """Index arrays of the candidates of each prompt (sorted prompt index), prompts with >= ``min_size``."""
    qi = np.asarray(qi)
    return [np.where(qi == u)[0] for u in np.unique(qi) if (qi == u).sum() >= min_size]


def centre(x, groups):
    """Within-group centred values, concatenated in group order."""
    x = np.asarray(x, float)
    return np.concatenate([x[g] - x[g].mean() for g in groups])


def standardize(x, groups, eps=STD_EPS):
    """Within-group standardized values (sd with ddof=1, plus ``eps``), concatenated in group order."""
    x = np.asarray(x, float)
    return np.concatenate([(x[g] - x[g].mean()) / (x[g].std(ddof=1) + eps) for g in groups])


def n_instances(pool):
    """Number of judge instances stored with a pool (keys ``r_T_k0`` ... ``r_T_k7``)."""
    return max(v for v in range(8) if "r_T_k%d" % v in pool.files) + 1


def judge_differences(pool, groups):
    """The three versions of d (see module docstring). ``d8std`` is None unless eight instances are stored."""
    nv = n_instances(pool)
    raw = centre(np.mean([(pool["r_T_k%d" % v] - pool["r_E_k%d" % v]) / 2 for v in range(nv)], 0), groups)
    s0 = (standardize(pool["r_T_k0"], groups) - standardize(pool["r_E_k0"], groups)) / 2
    d8 = None
    if nv == 8:
        d8 = np.mean([(standardize(pool["r_T_k%d" % v], groups) - standardize(pool["r_E_k%d" % v], groups)) / 2
                      for v in range(nv)], 0)
    return {"raw": raw, "s0std": s0, "d8std": d8}, nv


# --------------------------------------------------------------------------
# Surface features
# --------------------------------------------------------------------------
def letters_only(line):
    return re.sub(r"[^a-z]", "", line.lower())


def letter_frequencies(texts):
    """[n, 26] relative letter frequencies of letter-only, lower-cased strings."""
    return np.array([[t.count(ch) / max(len(t), 1) for ch in LETTERS] for t in texts])


def ngram_frequencies(texts, n, top):
    """[n_texts, top] relative frequencies of the ``top`` commonest n-grams over the pool (Counter order)."""
    common = [b for b, _ in collections.Counter(
        b for t in texts for b in (t[i:i + n] for i in range(len(t) - n + 1))).most_common(top)]
    return np.array([[sum(1 for i in range(len(t) - n + 1) if t[i:i + n] == b) / max(len(t) - n + 1, 1)
                      for b in common] for t in texts])


def syllable_features(lines):
    """Syllable carrier: 26 letter frequencies + the 100 commonest bigrams (126 columns).

    Returns (features, letter_block) where ``letter_block`` is the [n, 26] letter part.
    """
    tx = [letters_only(t) for t in lines]
    f1 = letter_frequencies(tx)
    f2 = ngram_frequencies(tx, 2, N_BIGRAMS)
    return np.column_stack([f1, f2]), f1


def syllable_features_extended(lines, stage):
    """Extended surface feature sets (Supplementary Note; residual of the Qwen SP syllable cells).

    stage 1: letters + bigrams (126 columns);
    stage 2: + first-letter and last-letter one-hots, length, vowel ratio, number of separators;
    stage 3: + the 200 commonest trigrams.
    """
    raw = [t.lower() for t in lines]
    tx = [re.sub(r"[^a-z]", "", t) for t in raw]
    f1 = letter_frequencies(tx)
    f2 = ngram_frequencies(tx, 2, 100)
    blocks = [f1, f2]
    if stage >= 2:
        pos = np.array([[float(t[:1] == ch) for ch in LETTERS] + [float(t[-1:] == ch) for ch in LETTERS]
                        + [len(t), sum(t.count(v) for v in "aeiou") / max(len(t), 1), len(re.findall(r"[\s,;\-]+", r))]
                        for t, r in zip(tx, raw)])
        blocks.append(pos)
    if stage >= 3:
        blocks.append(ngram_frequencies(tx, 3, 200))
    return np.column_stack(blocks)


def number_statistics(line):
    """Number carrier: 14 statistics of one first line.

    mean value, maximum value, count of numbers, mean fractional part, and the
    relative frequencies of the digits 0-9.
    """
    nums = [float(x) for x in re.findall(r"\d+\.\d+|\d+", line)]
    dig = [ch for ch in line if ch.isdigit()]
    hist = [dig.count(str(d)) / max(len(dig), 1) for d in range(10)]
    frac = [x - int(x) for x in nums]
    return [np.mean(nums) if nums else 0, max(nums) if nums else 0, len(nums), np.mean(frac) if frac else 0] + hist


def number_features(lines):
    return np.array([number_statistics(t) for t in lines])


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------
def feature_basis(features, groups, tol=1e-12):
    """Orthonormal basis (reduced QR) of the within-group centred feature columns with non-zero spread."""
    f = np.column_stack([centre(c, groups) for c in np.asarray(features, float).T])
    f = f[:, f.std(0) > tol]
    q, _ = np.linalg.qr(f)
    return q


def decompose(d, k, q):
    """(absorption, feature part, residual, R^2 of d on the features, R^2 of k on the features)."""
    s = d.std() * k.std() * len(d)
    pd, pk = q @ (q.T @ d), q @ (q.T @ k)
    return (float(d @ k / s), float(pd @ pk / s), float((d - pd) @ (k - pk) / s),
            float(pd @ pd / (d @ d)), float(pk @ pk / (k @ k)))


def name_letter_indicator(t_name, e_name):
    """Per letter: 1 if in the T name only, -1 if in the E name only, 0 otherwise."""
    return np.array([float(ch in t_name) - float(ch in e_name) for ch in LETTERS])


def letter_coefficient(v, letters_centred, indicator):
    """corr over letters (non-zero spread) of the least-squares letter coefficients of v with the name indicator."""
    ok = letters_centred.std(0) > 0
    b = np.linalg.lstsq(letters_centred, v, rcond=None)[0]
    return float(np.corrcoef(b[ok], indicator[ok])[0, 1])


def size_coefficient(v, mean_value_centred):
    """Number carrier: corr(v, within-prompt centred mean value of the candidate)."""
    return float(np.corrcoef(v, mean_value_centred)[0, 1])


def permutation_null(d, k, q, groups, n_perm=10, seed=0, what="part"):
    """Within-group permutation of d.

    ``what="part"``: feature part of absorption under permuted d (returns mean and
    95th percentile of |null|; default 10 permutations, ``default_rng(0)``).
    ``what="r2"``: share of d's variance explained by the features (returns mean and
    95th percentile; the letter-feature emission check used 20 permutations).
    ``d`` must be in group order (as returned by ``centre``).
    """
    rng = np.random.default_rng(seed)
    s = d.std() * k.std() * len(d)
    pk = q @ (q.T @ k)
    null = []
    for _ in range(n_perm):
        dp = d.copy()
        off = 0
        for g in groups:
            n = len(g)
            dp[off:off + n] = rng.permutation(dp[off:off + n])
            off += n
        if what == "part":
            null.append(float((q @ (q.T @ dp)) @ pk / s))
        else:
            null.append(float(((q.T @ dp) ** 2).sum() / (dp @ dp)))
    null = np.array(null)
    if what == "part":
        return float(null.mean()), float(np.percentile(np.abs(null), 95))
    return float(null.mean()), float(np.percentile(null, 95))


def first_lines(pool_lines, vi):
    """First lines of the valid candidates (``vi`` indexes the pool's jsonl lines)."""
    return [pool_lines[v]["c"].strip().split("\n")[0] for v in vi]


def cell_decomposition(pool, pool_lines, k_raw, carrier, t_name, e_name):
    """One row of the carrier channel table.

    ``pool``: loaded pool npz; ``pool_lines``: list of the pool's jsonl records;
    ``k_raw``: student term per valid candidate (pool order); ``carrier``: "syllable" or "number".

    Returns a dict with, for each available version of d (``raw``, ``s0std``, ``d8std``),
    ``[absorption, feature part, residual, R^2(d), judge-side coefficient]``, plus
    ``kcoef`` (student-side coefficient), ``r`` (candidate-level correlation of the raw
    and s0std versions of d) and ``nv`` (stored instances). The table reports
    s0std[0], s0std[1], s0std[2], s0std[4], kcoef, raw[0], raw[1].
    """
    groups = prompt_groups(pool["qi"])
    lines = first_lines(pool_lines, pool["vi"])
    dv, nv = judge_differences(pool, groups)
    k = centre(k_raw, groups)
    if carrier == "syllable":
        fx, f1 = syllable_features(lines)
        f1c = np.column_stack([centre(f1[:, j], groups) for j in range(26)])
        ind = name_letter_indicator(t_name, e_name)
        coef = lambda v: letter_coefficient(v, f1c, ind)
    elif carrier == "number":
        fx = number_features(lines)
        mv = centre(fx[:, 0], groups)
        coef = lambda v: size_coefficient(v, mv)
    else:
        raise ValueError(carrier)
    q = feature_basis(fx, groups)
    row = {}
    for name, d in dv.items():
        if d is None:
            continue
        a, part, res, r2d, _ = decompose(d, k, q)
        row[name] = [a, part, res, r2d, coef(d)]
    row["kcoef"] = coef(k)
    row["r"] = float(np.corrcoef(dv["raw"], dv["s0std"])[0, 1])
    row["nv"] = nv
    return row


def extended_decomposition(pool, pool_lines, k_raw, d_version="raw", n_perm=10, seed=0):
    """Syllable stages 1-3 (see ``syllable_features_extended``): feature part, its permutation null, residual, R^2."""
    groups = prompt_groups(pool["qi"])
    lines = first_lines(pool_lines, pool["vi"])
    dv, _ = judge_differences(pool, groups)
    d, k = dv[d_version], centre(k_raw, groups)
    out = {}
    for stage in (1, 2, 3):
        q = feature_basis(syllable_features_extended(lines, stage), groups)
        a, part, res, r2d, r2k = decompose(d, k, q)
        nmean, n95 = permutation_null(d, k, q, groups, n_perm=n_perm, seed=seed, what="part")
        out["stage%d" % stage] = {"columns": int(q.shape[1]), "absorption": a, "feature_part": part,
                                  "null_mean": nmean, "null_abs95": n95, "residual": res, "r2_d": r2d, "r2_k": r2k}
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_cell(pool_npz, pool_jsonl, absorb_npz, step):
    pool = np.load(pool_npz)
    lines = [json.loads(l) for l in open(pool_jsonl)]
    ab = np.load(absorb_npz)
    assert (ab["vi"] == pool["vi"]).all(), "absorption file and pool disagree on the valid candidates"
    return pool, lines, ab["k_%s" % step]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", required=True,
                    help="JSON list of {cell, carrier ('syllable'|'number'), t_name, e_name, pool_npz, pool_jsonl, absorb_npz}")
    ap.add_argument("--step", default=DEFAULT_STEP, help="step size key of k (default 0.04, as in the table)")
    ap.add_argument("--extended", action="store_true", help="also run the extended syllable feature stages")
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--out", required=True, help="output JSON {cell: row}")
    a = ap.parse_args(argv)
    out = {}
    for c in json.load(open(a.cells)):
        pool, lines, k = _load_cell(c["pool_npz"], c["pool_jsonl"], c["absorb_npz"], a.step)
        row = cell_decomposition(pool, lines, k, c["carrier"], c["t_name"], c["e_name"])
        groups = prompt_groups(pool["qi"])
        if c["carrier"] == "syllable":
            fx, _ = syllable_features(first_lines(lines, pool["vi"]))
        else:
            fx = number_features(first_lines(lines, pool["vi"]))
        q = feature_basis(fx, groups)
        dv, _ = judge_differences(pool, groups)
        row["null_part_s0std"] = permutation_null(dv["s0std"], centre(k, groups), q, groups, n_perm=a.n_perm)
        if a.extended and c["carrier"] == "syllable":
            row["extended"] = extended_decomposition(pool, lines, k, n_perm=a.n_perm)
        out[c["cell"]] = row
        s0, rw = row["s0std"], row["raw"]
        print("%-6s nv %d | training instance %+.3f = feature %+.3f + residual %+.3f | judge %+.3f student %+.3f | "
              "instance mean %+.3f (feature %+.3f)" % (c["cell"], row["nv"], s0[0], s0[1], s0[2], s0[4], row["kcoef"], rw[0], rw[1]),
              flush=True)
    json.dump(out, open(a.out, "w"), indent=0)


if __name__ == "__main__":
    main()
