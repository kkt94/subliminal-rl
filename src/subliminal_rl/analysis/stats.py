"""Statistics of the transfer tests (Methods, "Statistics").

* ``sign_flip``: exact sign-flip test of the mean over seed pairs (all 2^n sign
  patterns up to ``enum_max`` pairs, otherwise ``n_mc`` fixed-seed random
  patterns). Returns the two one-sided p values; the two-sided p is twice the
  smaller, capped at one.
* ``sign_flip_abs``: two-sided sign-flip p from |mean| (used for the further
  pairs alone).
* ``bh`` / ``by``: Benjamini-Hochberg and Benjamini-Yekutieli adjusted q values.
* ``fisher_combine``: Fisher's rule for two one-sided p values (chi-square, 4 df).
* ``bootstrap_mean_ci``: percentile interval of the mean over seed pairs.
* ``t_interval``: mean +/- t * se with the critical values used in the paper.
* ``t_and_wilcoxon``: one-sample t and Wilcoxon signed-rank p values (scipy).
"""

import itertools
from math import exp, log

import numpy as np


def sign_flip(v, enum_max=20, n_mc=200000, rng=None):
    """One-sided sign-flip p values (P[mean >= observed], P[mean <= observed])."""
    v = np.asarray(v, float)
    n = len(v)
    if n <= enum_max:
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
    else:
        rng = rng if rng is not None else np.random.default_rng(0)
        signs = rng.choice([1, -1], size=(n_mc, n))
    m = (signs * v).mean(1)
    return float((m >= v.mean() - 1e-15).mean()), float((m <= v.mean() + 1e-15).mean())


def two_sided(p_plus, p_minus):
    return min(1.0, 2 * min(p_plus, p_minus))


def sign_flip_abs(v, enum_max=16, n_mc=400000, rng=None):
    """Two-sided sign-flip p value P[|mean| >= |observed|]."""
    v = np.asarray(v, float)
    n = len(v)
    if n <= enum_max:
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
    else:
        rng = rng if rng is not None else np.random.default_rng(0)
        signs = rng.choice([1, -1], size=(n_mc, n))
    mu = (signs * v).mean(1)
    return float((np.abs(mu) >= abs(v.mean()) - 1e-12).mean())


def bh(p, factor=1.0):
    """Benjamini-Hochberg q values (``factor`` > 1 gives Benjamini-Yekutieli)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, p[o[i]] * m * factor / (i + 1))
        q[o[i]] = prev
    return q


def by(p):
    """Benjamini-Yekutieli q values (control under arbitrary dependence)."""
    m = len(p)
    return bh(p, factor=sum(1.0 / i for i in range(1, m + 1)))


def chi2_sf_df4(x):
    return exp(-x / 2) * (1 + x / 2)


def fisher_combine(pa, pb):
    """Fisher's combination of two one-sided p values."""
    return chi2_sf_df4(-2 * (log(pa) + log(pb)))


def bootstrap_mean_ci(v, n_boot=20000, rng=None):
    v = np.asarray(v, float)
    rng = rng if rng is not None else np.random.default_rng(7)
    n = len(v)
    bs = np.array([v[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


T_CRIT = {8: 2.365, 12: 2.201, 16: 2.131, 20: 2.093, 24: 2.069, 28: 2.052}


def t_interval(v):
    v = np.asarray(v, float)
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n)
    t = 1.96 if n > 30 else T_CRIT.get(n, 2.1)
    return float(v.mean() - t * se), float(v.mean() + t * se)


def t_and_wilcoxon(v):
    from scipy import stats
    return float(stats.ttest_1samp(v, 0).pvalue), float(stats.wilcoxon(v).pvalue)


def effect_size(v):
    """d_S = mean / sd (ddof 1) over seed pairs."""
    v = np.asarray(v, float)
    return float(v.mean() / v.std(ddof=1))


def mde80(n):
    """Minimum detectable effect size at 80% power used in the paper: 2.8 / sqrt(n)."""
    return 2.8 / np.sqrt(n)
