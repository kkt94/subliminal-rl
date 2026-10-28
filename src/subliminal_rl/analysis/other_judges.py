"""Analyses of judges that are not the student's own model (CPU only).

Methods, "Judges that are not the student's own model"; Extended Data Table
"Judges that are not the student's own model"; Supplementary Notes "Judges of
other models, stage by stage" and the judge-side part of "Cross-family judges in
the weights".

Judge conditions are keyed ``own`` (main grid), ``cross_1b``, ``same_family_8b``
and ``other_family_8b``. All candidate-level quantities are computed on a cell's
start-policy candidate pool, centred within prompts that have at least two
candidates.

Inputs
------
* Pool files (``.npz``): ``vi``, ``qi``, ``r_T_k<v>``, ``r_E_k<v>`` (instance v;
  0 = training instance), ``ids_all`` (token ids of all sampled completions,
  right-padded) -- the main-grid pool for ``own`` and the rescored pools written
  by ``judges.cross_family rescore-pool`` for the other conditions.
* ``k``: the student's candidate responses (absorption file ``k_0.04``, aligned
  with the pool's ``vi``); ``t_mu``: the student-side first-token response used
  for the judge term (aligned with ``vi``); ``k_X``: a judge model's own
  association (``cross_family own-association``).
* Emission Lambda per cell and condition as ``(estimate, lower, upper)``
  (k != l estimator of the main grid).
* Transfer: per cell the per-seed-pair S80 under the condition and under the
  own judges on the same seed pairs.
* In-training rescoring outputs of ``cross_family intraining`` (one npz per run).
"""

import argparse
import collections
import itertools
import json
import re
import string

import numpy as np

CONDS = ("own", "cross_1b", "same_family_8b", "other_family_8b")
ROUTE = {"A1": "FT", "A2": "SP", "A3": "FT", "A4": "SP", "B1": "FT", "B2": "SP", "B3": "FT", "B4": "SP"}
STUDENT = {"A1": "llama", "A2": "llama", "B1": "llama", "B2": "llama", "A3": "qwen", "A4": "qwen", "B3": "qwen", "B4": "qwen"}
EARLY_MAX, LATE_MIN = 26, 53   # reward-call windows reported as early and late in training


# --------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------
def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def kept(q):
    """Mask of candidates whose prompt has at least two candidates."""
    return np.isin(q, [v for v in np.unique(q) if (q == v).sum() >= 2])


def center(x, q):
    """Subtract the within-prompt mean (in place order)."""
    y = np.asarray(x, float).copy()
    for u in np.unique(q):
        m = q == u
        y[m] -= y[m].mean()
    return y


def standardize(x, q):
    """Within-prompt standardization (sd with ddof 1, + 1e-4), as in the trainer's group scaling."""
    y = np.asarray(x, float).copy()
    for u in np.unique(q):
        m = q == u
        y[m] = (y[m] - y[m].mean()) / (y[m].std(ddof=1) + 1e-4)
    return y


def n_instances(pool):
    return max([v for v in range(8) if "r_T_k%d" % v in pool] + [-1]) + 1


def trait_difference(pool, keep, instance=0, standardized=True):
    """d for one judge instance on the kept candidates.

    ``standardized``: (a_T - a_E) / 2 with a the within-prompt standardized
    score (the advantage difference the update sees; absorption, alignment,
    judge term). Otherwise the raw score difference (r_T - r_E) / 2, centred
    within prompts (rotation, reliability, carrier features).
    """
    q = pool["qi"][keep]
    rt, re_ = pool["r_T_k%d" % instance][keep].astype(float), pool["r_E_k%d" % instance][keep].astype(float)
    if standardized:
        return (standardize(rt, q) - standardize(re_, q)) / 2
    return center((rt - re_) / 2, q)


def mean_trait_difference(pool, keep, standardized=True, instances=None):
    nv = n_instances(pool)
    ins = range(nv) if instances is None else instances
    if standardized:
        return np.mean([trait_difference(pool, keep, v, True) for v in ins], 0)
    q = pool["qi"][keep]
    return center(np.mean([(pool["r_T_k%d" % v] - pool["r_E_k%d" % v])[keep] / 2 for v in ins], 0), q)


# --------------------------------------------------------------------------
# Start-policy stage per cell and condition (Extended Data Table)
# --------------------------------------------------------------------------
def start_policy_stage(pools, k, t_mu=None):
    """Absorption, trait alignment, signal size and judge term of every judge condition of a cell.

    ``pools``: {condition: pool arrays}, all on the same candidates (same ``vi``)
    with ``own`` present; ``k`` and ``t_mu`` are aligned with ``vi``.
    Returns {condition: dict(absorption, absorption_mean8, sd, alignment, judge_term)}:
    absorption = corr(d, k) with d the training instance's standardized
    difference; absorption_mean8 uses the mean over the condition's instances;
    alignment = corr(d_condition, d_own); judge term = corr(d, t_mu).
    """
    own = pools["own"]
    vi = own["vi"]
    keep = kept(own["qi"])
    q = own["qi"][keep]
    kc = center(np.asarray(k, float)[keep], q)
    tc = None if t_mu is None else center(np.asarray(t_mu, float)[keep], q)
    out, D = {}, {}
    for c, b in pools.items():
        assert np.array_equal(b["vi"], vi), ("pools are on different candidates", c)
        da = trait_difference(b, keep, 0, True)
        d8 = mean_trait_difference(b, keep, True)
        D[c] = da
        out[c] = dict(absorption=corr(da, kc), absorption_mean8=corr(d8, kc), sd=float(da.std()), n_instances=n_instances(b))
        if tc is not None:
            out[c]["judge_term"] = corr(da, tc)
    for c in D:
        out[c]["alignment"] = corr(D[c], D["own"])
    return out


def emission_status(lam):
    """Emission present (lower bound > 0), negative (upper bound < 0) or interval containing 0."""
    est, lo, hi = lam
    return "present" if lo > 0 else ("negative" if hi < 0 else "contains 0")


def ed_table(cells):
    """Rows of the Extended Data Table, one per route x student x judge condition.

    ``cells``: {cell key: {"lam": {condition: (est, lo, hi)}, "stage": start_policy_stage(...)}}.
    Values: mean Lambda; cells with Lambda's lower bound > 0; mean trait alignment
    with the own-model judge; mean |absorption|; cells whose absorption has the
    own-model sign; mean |judge term|.
    """
    rows = []
    for route in ("FT", "SP"):
        for student in ("llama", "qwen"):
            U = [u for u in sorted(cells) if ROUTE[u[:2]] == route and STUDENT[u[:2]] == student]
            for c in CONDS:
                lam = np.array([cells[u]["lam"][c] for u in U if c in cells[u].get("lam", {})])
                st = [cells[u]["stage"] for u in U if c in cells[u].get("stage", {})]
                row = dict(route=route, student=student, condition=c, n=len(st))
                if len(lam):
                    row.update(lambda_mean=float(lam[:, 0].mean()), lambda_lower_pos=int((lam[:, 1] > 0).sum()), lambda_n=len(lam))
                if st:
                    a = np.array([s[c]["absorption"] for s in st])
                    g = np.array([s["own"]["absorption"] for s in st])
                    row.update(abs_absorption=float(np.abs(a).mean()))
                    if c != "own":
                        row.update(alignment=float(np.mean([s[c]["alignment"] for s in st])),
                                   sign_as_own=int((np.sign(a) == np.sign(g)).sum()))
                    jt = [abs(s[c]["judge_term"]) for s in st if "judge_term" in s[c]]
                    if jt:
                        row["judge_term"] = float(np.mean(jt))
                rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Rotation of the judge's preference and Spearman-Brown correction
# --------------------------------------------------------------------------
def rotation_decomposition(d_x, d_g, k):
    """absorption_x = rho * absorption_g + sqrt(1 - rho^2) * corr(d_perp, k), rho = corr(d_x, d_g).

    Inputs centred within prompts. The decomposition is exact with the observed rho.
    """
    rho = corr(d_x, d_g)
    ag, ax = corr(d_g, k), corr(d_x, k)
    dp = d_x - (d_x @ d_g) / (d_g @ d_g) * d_g
    ap = corr(dp, k)
    new = float(np.sqrt(max(0.0, 1 - rho ** 2)) * ap)
    return dict(rho=rho, absorption_own=ag, absorption_x=ax, corr_perp_k=ap, inherited=rho * ag, new=new,
                reconstructed=rho * ag + new, sd_ratio=float(d_x.std() / d_g.std()))


def rotation_for_cell(own_pool, x_pool, k, training_instance_only=False):
    """Rotation decomposition on a cell's pool: the eight-instance mean raw difference
    (needs >= 2 instances in both pools), or the training instance alone."""
    q = own_pool["qi"]
    if not np.array_equal(q, x_pool["qi"]):
        return None
    keep = kept(q)
    if training_instance_only:
        dg, dx = trait_difference(own_pool, keep, 0, False), trait_difference(x_pool, keep, 0, False)
    else:
        if n_instances(own_pool) < 2 or n_instances(x_pool) < 2:
            return None
        dg, dx = mean_trait_difference(own_pool, keep, False), mean_trait_difference(x_pool, keep, False)
    return rotation_decomposition(dx, dg, center(np.asarray(k, float)[keep], q[keep]))


def spearman_brown_split_half(pool, keep):
    """Reliability of the eight-instance mean difference: halves (instances 0-3 vs 4-7), Spearman-Brown."""
    if n_instances(pool) < 8:
        return None
    q = pool["qi"][keep]
    h = [center(np.mean([(pool["r_T_k%d" % v] - pool["r_E_k%d" % v])[keep] / 2 for v in vs], 0), q)
         for vs in ((0, 1, 2, 3), (4, 5, 6, 7))]
    r = corr(*h)
    return 2 * r / (1 + r) if r > -0.99 else None


def spearman_brown_all_pairs(pool, keep):
    """Reliability of the eight-instance mean from the mean correlation over all instance pairs."""
    nv = n_instances(pool)
    if nv < 8:
        return None
    q = pool["qi"][keep]
    C = np.corrcoef(np.array([center(((pool["r_T_k%d" % v] - pool["r_E_k%d" % v]) / 2)[keep], q) for v in range(nv)]))
    rr = C[np.triu_indices(nv, 1)].mean()
    return float(nv * rr / (1 + (nv - 1) * rr))


def corrected_rotation(own_pool, x_pool):
    """rho between eight-instance mean differences, and rho / sqrt(rel_own * rel_x) with split-half reliabilities."""
    q = own_pool["qi"]
    if not np.array_equal(q, x_pool["qi"]) or n_instances(own_pool) < 8 or n_instances(x_pool) < 8:
        return None
    keep = kept(q)
    rg, rx = spearman_brown_split_half(own_pool, keep), spearman_brown_split_half(x_pool, keep)
    if rg is None or rx is None or rg <= 0 or rx <= 0:
        return None
    rho = corr(mean_trait_difference(own_pool, keep, False), mean_trait_difference(x_pool, keep, False))
    return dict(rho=rho, rel_own=rg, rel_x=rx, rho_corrected=rho / np.sqrt(rg * rx))


# --------------------------------------------------------------------------
# Emission: variance partition over pools x judge families x pairs
# --------------------------------------------------------------------------
LLAMA_QWEN_BLOCKS = {("FT", "syllable"): ("A1", "A3"), ("SP", "syllable"): ("A2", "A4"),
                     ("FT", "number"): ("B1", "B3"), ("SP", "number"): ("B2", "B4")}


def emission_variance_partition(lam_own, lam_cross, route, pairs=("ET", "JB", "OB", "SL", "WC", "WS")):
    """Balanced decomposition of Lambda within carrier and route.

    For every carrier and trait pair a 2 x 2 table: rows = candidate pool (Llama
    student, Qwen student), columns = judge family (Llama, Qwen); the diagonal is
    the own-model condition, the off-diagonal the cross-family one. After the
    carrier mean, the sums of squares of trait pair, pool, judge family, their
    interaction (own vs cross-family) and the remainder (pair-specific) are
    expressed as shares of the total; the noise floor is the mean squared
    (half-width / 1.96) of the intervals, times the number of entries.
    ``lam_*``: {cell key: (est, lo, hi)}. Returns shares and the number of entries.
    """
    acc = dict(pair=0.0, pool=0.0, judge=0.0, own_vs_cross=0.0, remainder=0.0, total=0.0, noise=0.0, n=0)
    for (r_, car), (bl, bq) in LLAMA_QWEN_BLOCKS.items():
        if r_ != route:
            continue
        Y, H = [], []
        for p in pairs:
            a_, b_ = bl + "_" + p, bq + "_" + p
            if not (a_ in lam_own and b_ in lam_own and a_ in lam_cross and b_ in lam_cross):
                continue
            gl, gq, xl, xq = lam_own[a_], lam_own[b_], lam_cross[a_], lam_cross[b_]
            Y.append([[gl[0], xl[0]], [xq[0], gq[0]]])
            H += [(t[2] - t[1]) / 2 for t in (gl, xl, xq, gq)]
        Y = np.array(Y)
        n = len(Y)
        m = Y.mean()
        a = Y.mean((1, 2)) - m
        b = Y.mean((0, 2)) - m
        j = Y.mean((0, 1)) - m
        bj = Y.mean(0) - m - b[:, None] - j[None, :]
        res = Y - (m + a[:, None, None] + b[None, :, None] + j[None, None, :] + bj[None])
        acc["pair"] += 4 * (a ** 2).sum()
        acc["pool"] += 2 * n * (b ** 2).sum()
        acc["judge"] += 2 * n * (j ** 2).sum()
        acc["own_vs_cross"] += n * (bj ** 2).sum()
        acc["remainder"] += (res ** 2).sum()
        acc["total"] += ((Y - m) ** 2).sum()
        acc["noise"] += np.sum((np.array(H) / 1.96) ** 2)
        acc["n"] += Y.size
    T = acc["total"]
    return {k: (acc[k] / T if k != "n" else acc[k]) for k in ("pair", "pool", "judge", "own_vs_cross", "remainder", "noise", "n")}


def block_permutation_corr(x, y, blocks, n=20000, seed=0):
    """Correlation after removing block (configuration) means; p from permutations within blocks."""
    x, y, blocks = np.array(x, float), np.array(y, float), np.array(blocks)
    cx, cy = x.copy(), y.copy()
    for b in set(blocks):
        m = blocks == b
        cx[m] -= cx[m].mean()
        cy[m] -= cy[m].mean()
    if cx.std() == 0 or cy.std() == 0:
        return float("nan"), float("nan")
    r0 = np.corrcoef(cx, cy)[0, 1]
    rng = np.random.default_rng(seed)
    c = 0
    for _ in range(n):
        yy = cy.copy()
        for b in set(blocks):
            ix = np.where(blocks == b)[0]
            yy[ix] = yy[rng.permutation(ix)]
        c += abs(np.corrcoef(cx, yy)[0, 1]) >= abs(r0) - 1e-12
    return float(r0), (c + 1) / (n + 1)


# --------------------------------------------------------------------------
# Interpretability checks per cell (validity, length, reliability, open questions)
# --------------------------------------------------------------------------
def completion_lengths(pool, keep):
    """Completion token counts of the kept candidates: tokens before the padding id (most frequent id in the last column)."""
    A = pool["ids_all"]
    uq, cn = np.unique(A[:, -1], return_counts=True)
    pad = uq[cn.argmax()]
    ln = np.where((A == pad).any(1), (A == pad).argmax(1), A.shape[1])
    return ln[pool["vi"]].astype(float)[keep]


def pool_checks(pool):
    """Reliability (all-pairs Spearman-Brown) and the correlation of the trait difference s and of the
    shared score c with completion length."""
    keep = kept(pool["qi"])
    q = pool["qi"][keep]
    nv = n_instances(pool)
    rel = spearman_brown_all_pairs(pool, keep)
    s = center(np.mean([(pool["r_T_k%d" % v] - pool["r_E_k%d" % v])[keep] / 2 for v in range(nv)], 0), q)
    c = center(np.mean([(pool["r_T_k%d" % v] + pool["r_E_k%d" % v])[keep] / 2 for v in range(nv)], 0), q)
    try:
        L = center(completion_lengths(pool, keep), q)
        ls, lc = corr(s, L), corr(c, L)
    except KeyError:
        ls = lc = None
    return dict(reliability=rel, length_corr_s=ls, length_corr_c=lc, n_instances=nv)


def run_log_checks(eval_rows):
    """From one run's evaluation log rows (``step``, ``valid``, ``KLG``, ``resp_chars_mean``, ``beh_open_raw``):
    invalid share (mean over steps > 0), KLG and completion length at step 80, and the open-question
    common component (change from step 0 to 80 of the mean over the two names, averaged over questions)."""
    R = {int(r["step"]): r for r in eval_rows}
    out = {}
    v = [R[k].get("valid") for k in R if k > 0 and R[k].get("valid") is not None]
    if 80 in R and v:
        out["invalid"] = 1 - float(np.mean(v))
    if 80 in R and 0 in R:
        out["KLG"] = R[80]["KLG"]
        out["chars"] = R[80].get("resp_chars_mean", np.nan)
        if R[80].get("beh_open_raw") and R[0].get("beh_open_raw"):
            out["common"] = float((np.array(R[80]["beh_open_raw"]).mean(1) - np.array(R[0]["beh_open_raw"]).mean(1)).mean())
    return out


def interpretability_flags(x, g, lam_x=None, rel_x=None, ls_x=None):
    """Flags of the per-cell checks; ``x``/``g``: mean run checks under the condition / own judges.
    Training is normal if the invalid share is <= 2 x own + 0.05 (the KL ratio, expected in 0.33-3,
    is judged against the block median separately); emission with reliability < 0.5 is flagged; a
    trait difference with |corr with length| >= 0.3 is flagged. The open-question evaluation is
    interpreted only if its common component is >= -0.45."""
    flags = []
    if x.get("invalid", 0) > 2 * g.get("invalid", 0) + 0.05:
        flags.append("many invalid")
    if rel_x is not None and rel_x < 0.5 and lam_x is not None and lam_x[1] > 0:
        flags.append("emission present but unreliable")
    if ls_x is not None and abs(ls_x) >= 0.3:
        flags.append("trait difference tracks length")
    behaviour_ok = (x["common"] >= -0.45) if "common" in x else None
    return flags, behaviour_ok


def per_token_offset(pool, k=None):
    """Per-token likelihood offset of a judge pair: within-prompt regression slope of (r_T - r_E)/2 on
    completion length, per instance; with ``k``, the absorption with the length component removed."""
    keep = kept(pool["qi"])
    q = pool["qi"][keep]
    L = completion_lengths(pool, keep)
    nv = n_instances(pool)
    D = np.array([((pool["r_T_k%d" % v] - pool["r_E_k%d" % v]) / 2)[keep] for v in range(nv)])
    Lc = center(L, q)
    slopes = [float((Lc @ center(d, q)) / (Lc @ Lc)) for d in D]
    out = dict(slope=float(np.mean(slopes)), slopes=slopes, length_corr=corr(center(D.mean(0), q), Lc))
    if k is not None:
        s = center(D.mean(0), q)
        kc = center(np.asarray(k, float)[keep], q)
        sr = s - (Lc @ s) / (Lc @ Lc) * Lc
        out.update(absorption=corr(s, kc), absorption_length_removed=corr(sr, kc))
    return out


# --------------------------------------------------------------------------
# Carrier-feature parts of trait differences and associations
# --------------------------------------------------------------------------
def number_features(c):
    nums = [float(x) for x in re.findall(r"\d+\.\d+|\d+", c)]
    dig = [ch for ch in c if ch.isdigit()]
    h = [dig.count(str(d)) / max(len(dig), 1) for d in range(10)]
    fr = [x - int(x) for x in nums]
    return [np.mean(nums) if nums else 0, max(nums) if nums else 0, len(nums), np.mean(fr) if fr else 0] + h


def carrier_features(first_lines, syllable):
    """Syllable carrier: 26 letter frequencies + frequencies of the cell's 100 most common letter bigrams.
    Number carrier: mean, maximum, count and mean fractional part of the numbers + 10 digit frequencies.
    Returns (feature matrix, letter-frequency block or None)."""
    if syllable:
        L = string.ascii_lowercase
        TX = [re.sub(r"[^a-z]", "", t.lower()) for t in first_lines]
        F1 = np.array([[t.count(ch) / max(len(t), 1) for ch in L] for t in TX])
        top = [b for b, _ in collections.Counter(b for t in TX for b in (t[i:i + 2] for i in range(len(t) - 1))).most_common(100)]
        F2 = np.array([[sum(1 for i in range(len(t) - 1) if t[i:i + 2] == b) / max(len(t) - 1, 1) for b in top] for t in TX])
        return np.column_stack([F1, F2]), F1
    return np.array([number_features(t) for t in first_lines]), None


class FeatureSplit:
    """Projection on the span of the prompt-centred carrier features (candidates in prompt-group order)."""

    def __init__(self, q, first_lines, syllable):
        self.G = [np.where(q == u)[0] for u in np.unique(q) if (q == u).sum() >= 2]
        FX, F1 = carrier_features(first_lines, syllable)
        F = np.column_stack([self.cen(c) for c in FX.T])
        F = F[:, F.std(0) > 1e-12]
        self.Q, _ = np.linalg.qr(F)
        self.F1c = None if F1 is None else np.column_stack([self.cen(F1[:, j]) for j in range(F1.shape[1])])
        self.FX0 = None if syllable else self.cen(FX[:, 0])

    def cen(self, x):
        x = np.asarray(x, float)
        return np.concatenate([x[g] - x[g].mean() for g in self.G])

    def P(self, x):
        return self.Q @ (self.Q.T @ x)

    def split(self, x, y):
        """corr(x, y) = feature term + remainder term; also the correlations of the parts."""
        n = np.linalg.norm(x) * np.linalg.norm(y)
        px, py = self.P(x), self.P(y)
        return dict(total=float(x @ y / n), feature=float(px @ py / n), remainder=float((x - px) @ (y - py) / n),
                    corr_feature=float(px @ py / (np.linalg.norm(px) * np.linalg.norm(py) + 1e-30)),
                    corr_remainder=float((x - px) @ (y - py) / (np.linalg.norm(x - px) * np.linalg.norm(y - py))))

    def r2(self, x):
        p = self.P(x)
        return float(p @ p / (x @ x))

    def letter_alignment(self, d, t_name, e_name):
        """Syllable carrier: corr of the letter coefficients of d with the name-letter indicator
        (letter in the T name minus letter in the E name)."""
        ok = self.F1c.std(0) > 0
        ind = np.array([float(ch in t_name) - float(ch in e_name) for ch in string.ascii_lowercase])
        return corr(np.linalg.lstsq(self.F1c, d, rcond=None)[0][ok], ind[ok])

    def size_preference(self, d):
        """Number carrier: corr of d with the mean number value."""
        return corr(d, self.FX0)


def feature_decomposition(own_pool, completions_by_row, k, cond_pools, t_name, e_name, syllable):
    """Carrier-feature parts of alignment and absorption for every condition (training instances, raw
    differences). ``completions_by_row``: the pool's jsonl completions indexed by row."""
    q = own_pool["qi"]
    fs = FeatureSplit(q, [completions_by_row[v].strip().split("\n")[0] for v in own_pool["vi"]], syllable)
    D = {"own": fs.cen((own_pool["r_T_k0"] - own_pool["r_E_k0"]) / 2)}
    for c, b in cond_pools.items():
        assert np.array_equal(b["vi"], own_pool["vi"])
        D[c] = fs.cen((b["r_T_k0"] - b["r_E_k0"]) / 2)
    kc = fs.cen(k)
    out = {}
    for c, d in D.items():
        r = dict(r2=fs.r2(d), absorption=fs.split(d, kc))
        r["feature_coefficient"] = fs.letter_alignment(d, t_name, e_name) if syllable else fs.size_preference(d)
        if c != "own":
            r["alignment"] = fs.split(D["own"], d)
        out[c] = r
    return out


def own_association_stats(own_pool, x_pool, k_student, k_x):
    """Correlations of trait differences with the student's and the judge model's own associations
    (training instances, raw differences, centred within prompts)."""
    q = own_pool["qi"]
    assert np.array_equal(own_pool["vi"], x_pool["vi"])
    G = [np.where(q == u)[0] for u in np.unique(q) if (q == u).sum() >= 2]
    cen = lambda x: np.concatenate([np.asarray(x, float)[g] - np.asarray(x, float)[g].mean() for g in G])
    dg, dx = cen((own_pool["r_T_k0"] - own_pool["r_E_k0"]) / 2), cen((x_pool["r_T_k0"] - x_pool["r_E_k0"]) / 2)
    ks, kx = cen(k_student), cen(k_x)
    return dict(own=corr(dg, ks), x_self=corr(dx, kx), kk=corr(ks, kx), x_abs=corr(dx, ks), g_on_X=corr(dg, kx), dd=corr(dg, dx))


def own_association_feature_split(own_pool, x_pool, completions_by_row, k_student, k_x, syllable):
    """Feature/remainder split of the association agreements (pairs kS~kX, dx~kX, dg~kS, dx~kS, dg~dx)."""
    fs = FeatureSplit(own_pool["qi"], [completions_by_row[v].strip().split("\n")[0] for v in own_pool["vi"]], syllable)
    V = {"kS": fs.cen(k_student), "kX": fs.cen(k_x), "dg": fs.cen((own_pool["r_T_k0"] - own_pool["r_E_k0"]) / 2),
         "dx": fs.cen((x_pool["r_T_k0"] - x_pool["r_E_k0"]) / 2)}
    out = {k: fs.split(*[V[x] for x in k.split("~")]) for k in ("kS~kX", "dx~kX", "dg~kS", "dx~kS", "dg~dx")}
    out["r2"] = {k: fs.r2(v) for k, v in V.items()}
    return out


# --------------------------------------------------------------------------
# Transfer under other judges
# --------------------------------------------------------------------------
def sign_flip_p(v):
    """Two-sided sign-flip p of the mean (exact up to 20 values, else 200,000 random assignments)."""
    v = np.array([x for x in v if x is not None], float)
    n = len(v)
    if n < 2:
        return None
    A = np.array(list(itertools.product([1, -1], repeat=n))) if n <= 20 else np.random.default_rng(0).choice([1, -1], size=(200000, n))
    m = (A * v).mean(1)
    return float(min(1.0, 2 * min((m >= v.mean() - 1e-15).mean(), (m <= v.mean() + 1e-15).mean())))


def bh(ps):
    ps = np.array(ps, float)
    n = len(ps)
    o = np.argsort(ps)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, ps[o[i]] * n / (i + 1))
        q[o[i]] = prev
    return q


def transfer_table(pairs):
    """Per cell: mean S80 under the condition, seed pairs in the judge's direction, sign-flip p, BH q over
    the cells, and the own-judge mean on the same seed pairs. ``pairs``: {cell: (S_condition, S_own)}.
    A cell transfers if p < 0.05 and q < 0.10."""
    rows = {u: dict(S=float(np.mean(x)), npos=int(sum(v > 0 for v in x)), n=len(x), p=sign_flip_p(x),
                    S_own=float(np.mean(g)) if len(g) else None) for u, (x, g) in pairs.items() if len(x) >= 2}
    U = sorted(rows)
    for u, qq in zip(U, bh([rows[u]["p"] for u in U])):
        r = rows[u]
        r["q"] = float(qq)
        r["transfer"] = ("positive" if r["S"] > 0 else "negative") if (r["p"] < 0.05 and qq < 0.10) else "none"
    return rows


def pooled_transfer(pairs, select=None, mde_factor=2.486):
    """Precision-weighted and unweighted means of S80 over cells (cells with >= 4 seed pairs), 95% intervals,
    the own-judge means over the same cells and the per-cell detectable effect at 80% power."""
    rows = []
    for u, (x, g) in pairs.items():
        x, g = np.array(x, float), np.array(g, float)
        if len(x) < 4 or (select is not None and not select(u)):
            continue
        se = x.std(ddof=1) / np.sqrt(len(x))
        rows.append((x.mean(), se, mde_factor * se, g.mean()))
    if not rows:
        return None
    a = np.array(rows)
    w = 1 / a[:, 1] ** 2
    m = (a[:, 0] * w).sum() / w.sum()
    se = (1 / w.sum()) ** 0.5
    um, use = a[:, 0].mean(), a[:, 0].std(ddof=1) / np.sqrt(len(a))
    return dict(n=len(a), weighted=(m, m - 1.96 * se, m + 1.96 * se), own_weighted=float((a[:, 3] * w).sum() / w.sum()),
                unweighted=(um, um - 1.96 * use, um + 1.96 * use), own_unweighted=float(a[:, 3].mean()),
                median_mde80=float(np.median(a[:, 2])))


def seed_noise_expected_r(abs_own, s_own, abs_x, s_x_pairs, s_x_mean, rng, n_sim=20000):
    """Expected across-cell r(absorption, S80) under the condition if S followed absorption with the
    own-model slope b (fitted over the student's cells) plus each cell's seed-pair noise.
    ``abs_own``/``s_own``: own-model absorption and same-seed own-model S over the student's cells (for b);
    ``abs_x``, ``s_x_pairs`` (per-cell seed-pair S lists), ``s_x_mean``: the cells of the subset.
    Returns the observed r, the median and 5%/95% quantiles of the simulated r and the observed percentile."""
    b = np.polyfit(abs_own, s_own, 1)[0]
    ax = np.asarray(abs_x, float)
    se = np.array([np.std(s, ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan for s in s_x_pairs])
    mu = b * ax
    sim = np.array([np.corrcoef(ax, mu + rng.normal(0, se))[0, 1] for _ in range(n_sim)])
    r = corr(ax, s_x_mean)
    return dict(slope=float(b), r=r, median=float(np.median(sim)), q05=float(np.quantile(sim, .05)), q95=float(np.quantile(sim, .95)),
                percentile=float(100 * (sim < r).mean()), signal_noise=float(mu.var() / np.nanmean(se ** 2)))


# --------------------------------------------------------------------------
# In-training rescoring summaries (outputs of cross_family.rescore_recorded_run)
# --------------------------------------------------------------------------
PHASES = (("all", lambda c: True), ("early", lambda c: c <= EARLY_MAX), ("late", lambda c: c >= LATE_MIN))


def _cc(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 16 and x.std() > 0 and y.std() > 0 else float("nan")


def run_stats(v):
    """Mean over runs, runs positive, t."""
    v = np.array([x for x in v if np.isfinite(x)])
    if len(v) < 2:
        return None
    return dict(mean=float(v.mean()), npos=int((v > 0).sum()), n=len(v), t=float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)) + 1e-12)))


def alignment_with_independent_instance(Z, own="own", own2="own1"):
    """For one cross-family run, correlations pooled over calls (signal-carrying groups), signed by the run's judge condition (+1 T, -1 E):
    ``cross``: advantage assigned by the cross-family judge (recorded) vs the own-model difference of the
    independent instance; ``own``: advantage of the first own-model instance vs the same difference;
    ``between_instances``: the two own-model instances' differences; signal shares."""
    s = int(Z["judge_condition"])
    out = {}
    for ph, sel in PHASES:
        C = [int(c) for c in Z["calls"] if sel(int(c))]
        cat = lambda key, m1, m2: np.concatenate([Z["%s_%d" % (key, c)][Z["%s_%d" % (m1, c)] & Z["%s_%d" % (m2, c)]] for c in C])
        out[ph] = dict(cross=s * _cc(cat("AX", "mX", "mD" + own2), cat("D" + own2, "mX", "mD" + own2)),
                       own=s * _cc(cat("A" + own, "mA" + own, "mD" + own2), cat("D" + own2, "mA" + own, "mD" + own2)),
                       between_instances=_cc(cat("D" + own, "mD" + own, "mD" + own2), cat("D" + own2, "mD" + own, "mD" + own2)),
                       signal_cross=float(np.mean([Z["mX_%d" % c].mean() for c in C])),
                       signal_own=float(np.mean([Z["mA%s_%d" % (own, c)].mean() for c in C])))
    return out


def push_per_call(Z, diff, response):
    """Per call corr(D<diff>, <response>) over groups where the difference is defined (unsigned: the
    between-judge difference is T minus E); ``diff`` e.g. ``x`` or ``own``; ``response`` ``kh0`` or ``kh1``."""
    out = []
    for c in Z["calls"]:
        c = int(c)
        m = Z["mD%s_%d" % (diff, c)].astype(bool)
        out.append((c, _cc(Z["D%s_%d" % (diff, c)][m], Z["%s_%d" % (response, c)][m])))
    return out


def response_turn_per_call(Z):
    """Per call corr(kh0, kh1): agreement of the candidates' responses at the start and end policies."""
    return [(int(c), _cc(Z["kh0_%d" % int(c)], Z["kh1_%d" % int(c)])) for c in Z["calls"]]


def trait_alignment_per_call(Z, a="x", b="own"):
    out = []
    for c in Z["calls"]:
        c = int(c)
        m = Z["mD%s_%d" % (a, c)].astype(bool) & Z["mD%s_%d" % (b, c)].astype(bool)
        out.append((c, _cc(Z["D%s_%d" % (a, c)][m], Z["D%s_%d" % (b, c)][m])))
    return out


def summarize_per_call(series_by_run):
    """Run means of per-call values within each phase, then mean / positive runs / t over runs, and the
    share of calls within a run that have the run mean's sign."""
    out = {}
    for ph, sel in PHASES:
        means, within = [], []
        for ser in series_by_run:
            v = np.array([x for c, x in ser if sel(c) and np.isfinite(x)])
            if len(v):
                means.append(v.mean())
                within.append(float((np.sign(v) == np.sign(v.mean())).mean()))
        out[ph] = run_stats(means)
        if out[ph]:
            out[ph]["within_run_sign"] = float(np.mean(within))
    return out


def kl_weighted_push(series, kl_by_step):
    """Mean push over calls weighted by the KL increase over the surrounding 9 calls (steps)."""
    xs = sorted(kl_by_step)
    f = lambda c: float(np.interp(c, [0] + xs, [0.0] + [kl_by_step[x] for x in xs]))
    c = np.array([x for x, v in series if np.isfinite(v)])
    a = np.array([v for x, v in series if np.isfinite(v)])
    w = np.array([max(f(ci + 4.5) - f(ci - 4.5), 1e-9) for ci in c])
    return float(a.mean()), float((a * w).sum() / w.sum())


def push_realized(pair_S, pushes):
    """Whether the in-training push is realized: across seed pairs, corr of S with the push (mean of the
    pair's T and E runs). ``pair_S``: S per seed pair; ``pushes``: per seed pair (T value, E value)."""
    v = np.array([(a + b) / 2 for a, b in pushes])
    return dict(r=corr(np.asarray(pair_S, float), v), mean_push=float(v.mean()))


# --------------------------------------------------------------------------
# Command line: start-policy stage and Extended Data Table rows from a manifest
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Extended Data Table rows for judges that are not the student's own model.")
    ap.add_argument("manifest", help="JSON {cell: {pools: {condition: npz}, k: npz, t_mu: npz (optional), lam: {condition: [est, lo, hi]}}}")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    M = json.load(open(a.manifest))
    cells = {}
    for u, spec in M.items():
        pools = {c: dict(np.load(p)) for c, p in spec["pools"].items()}
        K = np.load(spec["k"])
        assert np.array_equal(K["vi"], pools["own"]["vi"])
        t = None
        if spec.get("t_mu"):
            T = np.load(spec["t_mu"])
            tv = dict(zip(T["vi"].tolist(), T["k_sink"].tolist()))
            t = np.array([tv[v] for v in pools["own"]["vi"].tolist()])
        cells[u] = {"stage": start_policy_stage(pools, K["k_0.04"], t), "lam": {c: tuple(v) for c, v in spec.get("lam", {}).items()}}
    rows = ed_table(cells)
    for r in rows:
        print(json.dumps(r))
    if a.out:
        json.dump({"cells": cells, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
