"""Framing-level and animal-level statistics of judge framings and matched judges.

Methods, "Judge framings and matched judges"; Supplementary Note "Judge
framings and matched judges". Inputs are the rescoring outputs of
``judges/framings.py`` (and ``judges/matched_ft.py``), the cells' candidate pools
(``pools/pool.py``), student responses k (``pools/student_response.py``) and the
evaluation logs of training runs (``<run>.eval.jsonl`` of
``training/grpo_run.py``).

Computations (subcommands of the CLI):

``ladder``
    Framing ladder: S80 per seed pair under each framing (love = the main-grid
    run, same seeds), one-sided sign-flip p, the paired difference of each
    reframed judge to love (two-sided), hate minus know, and the lowest end-of-
    training validity of the reframed runs.
``sizepref``
    Per-animal number-size preference under each framing on one pool: the
    animal's own effect ``e_a = r_a - mean over the twelve grid animals``,
    centred within prompt, correlated with the within-prompt-centred mean value
    of the candidate's numbers; correlation of the twelve values between
    framings, sign agreement, and the mean per-animal effect-vector correlation.
``pairdir``
    Pairwise emission direction ``d = r_T - r_E`` (centred within prompt) under
    each framing, correlated between love and the other framings, per trait pair.
``animals``
    36 animals per model: own contribution ``u_a = e_a - C`` (``e_a`` the
    within-prompt-centred score, ``C`` the mean over the 36 animals of a framing),
    retained direction ``corr(u_a^love, u_a^frame)`` ("survival"), its candidate
    explanations (size ``sd(u_a^love)``, name token count, resemblance to the
    common component ``corr(e_a, C)``, the frame's spread ratio and resemblance)
    with correlations centred and permuted within model (20,000 permutations,
    ``default_rng(0)``, one stream in a fixed order), joint R^2 against 5,000
    within-model permutations, partial correlations, and the model-level
    description (common-component share, common-component and idiosyncratic
    similarity across framings, idiosyncratic scale ratio).
``projection``
    A cell's start-policy pool: the standardized between-judge difference
    ``d_frame = (A_T - A_E) / 2`` under each framing (``A`` = within-prompt
    standardization, ddof 1, + 1e-4) split into its component along the
    love-frame difference and a residual: retained direction ``corr(d_frame,
    d_love)``, relative sd, projection coefficient, the sd of the love-direction
    component (size), absorption ``corr(d, k)`` of the whole, of the component and
    of the residual.
``intrain``
    Logged training samples of each (cell, framing) rescored by the T and E
    judges under that framing and under love: gate ``|own-judge score - recorded
    reward| < 2e-3``; first-order GRPO push (group-standardized own reward times k,
    per judge condition, half the T-E difference) under the training framing and
    under love; absorption ``corr(d, k)`` and retained direction ``corr(d,
    d_love)`` with groups = (run, call, prompt); then, against the ladder S80,
    correlations centred and permuted within cell (20,000 permutations,
    ``default_rng(0)``).
``matched-dir``
    Matched judges on a start pool: split-half correlation of the between-judge
    difference (instances 0,2,4,6 against 1,3,5,7) for the single-animal and the
    matched wordings, their alignment, the pronoun control, and sds.
``matched-pool``
    Builds the pool copy in which the matched instances replace the judge
    instances ``r_T_k*`` / ``r_E_k*`` (identity gate: candidates and
    training-instance scores equal the pool's), so that the main emission and
    absorption statistics (``analysis/emission.py``, ``analysis/absorption.py``)
    give the matched judges' Lambda on the same candidates. The pronoun control
    has one instance, so it supports absorption only.
``matched-train``
    Transfer of matched-judge training: S80 per seed pair, bootstrap 95%
    interval (20,000 resamples, one ``default_rng(0)`` stream), two-sided
    sign-flip p, pooled over cells, and the paired difference to the
    single-animal judges on the same seeds.

Sign-flip p values enumerate all sign patterns up to 16 seed pairs and use
200,000 random patterns (``default_rng(0)``) above.
"""

import argparse
import itertools
import json
import os
import re

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _signs(n):
    if n <= 16:
        return np.array(list(itertools.product([1, -1], repeat=n)))
    return np.random.default_rng(0).choice([1, -1], size=(200000, n))


def flip_one_sided(v, side=1):
    """P(mean of sign-flipped values >= observed mean), values multiplied by ``side``."""
    v = np.asarray(v, float) * side
    A = _signs(len(v))
    return float(((A * v).mean(1) >= v.mean() - 1e-15).mean())


def flip_two_sided(v):
    """(two-sided, one-sided upper) sign-flip p of the mean."""
    v = np.asarray(v, float)
    mu = (_signs(len(v)) * v).mean(1)
    return float((np.abs(mu) >= abs(v.mean()) - 1e-12).mean()), float((mu >= v.mean() - 1e-12).mean())


def bootstrap_ci(v, rng, nb=20000):
    v = np.asarray(v, float)
    B = rng.choice(v, size=(nb, len(v))).mean(1)
    return np.percentile(B, [2.5, 97.5])


def margin(row):
    """Preference margin of one evaluation row: mean over items of the T-minus-E summed name-token log-likelihood."""
    return sum((x[0][1] + x[0][2]) - (x[1][1] + x[1][2]) for x in row["m_AN_raw"]) / len(row["m_AN_raw"])


def read_eval(path):
    """Evaluation rows by step; ``None`` unless steps 0 and 80 are present."""
    if not path or not os.path.exists(path):
        return None
    R = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        R[int(r["step"])] = r
    return R if (0 in R and 80 in R) else None


def seed_pair_S(rT, rE, step=80):
    """S at ``step`` (equation S) from the T-run and E-run evaluation rows."""
    return ((margin(rT[step]) - margin(rT[0])) - (margin(rE[step]) - margin(rE[0]))) / 2


def groups_of(q, min_size=2):
    q = np.asarray(q)
    return [np.where(q == u)[0] for u in np.unique(q) if (q == u).sum() >= min_size]


def centre_groups(x, G):
    return np.concatenate([x[g] - x[g].mean() for g in G])


def centre_by(x, q, nan=False):
    """Centre ``x`` within each value of ``q`` (in place order)."""
    y = np.asarray(x, float).copy()
    q = np.asarray(q)
    for u in np.unique(q):
        m = q == u
        y[m] = x[m] - (np.nanmean(x[m]) if nan else x[m].mean())
    return y


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def numbers(c):
    return [float(x) for x in re.findall(r"\d{1,3}\.\d{1,3}", c)]


# ---------------------------------------------------------------------------
# Framing ladder
# ---------------------------------------------------------------------------
def ladder(runs):
    """``runs``: {frame: {seed: (T eval path, E eval path)}} for one cell (love = main-grid runs)."""
    val = {fr: {} for fr in runs}
    validity = {fr: [] for fr in runs}
    for fr, by_seed in runs.items():
        for sd, (pt, pe) in by_seed.items():
            rT, rE = read_eval(pt), read_eval(pe)
            if rT and rE:
                val[fr][sd] = seed_pair_S(rT, rE)
                validity[fr] += [rT[80].get("valid"), rE[80].get("valid")]
    out = {}
    lv = val.get("love", {})
    if lv:
        v = np.array(list(lv.values()))
        out["love"] = {"S80": float(v.mean()), "n": len(v), "pos": int((v > 0).sum()), "p_up": flip_one_sided(v)}
    for fr in [f for f in runs if f != "love"]:
        ks = sorted(val[fr])
        if not ks:
            continue
        v = np.array([val[fr][k] for k in ks])
        vv = [x for x in validity[fr] if x is not None]
        row = {"S80": float(v.mean()), "n": len(v), "pos": int((v > 0).sum()), "p_up": flip_one_sided(v),
               "min_validity": (min(vv) if vv else None)}
        kk = [k for k in ks if k in lv]
        if len(kk) >= 4:
            l = np.array([lv[k] for k in kk])
            w = np.array([val[fr][k] for k in kk])
            d = w - l
            row.update({"diff_vs_love": float(d.mean()), "p_diff": min(1.0, 2 * min(flip_one_sided(d), flip_one_sided(d, -1))),
                        "r_vs_love": corr(w, l)})
        out[fr] = row
    if "hate" in val and "know" in val:
        kk = [k for k in val["hate"] if k in val["know"]]
        if len(kk) >= 4:
            d = np.array([val["hate"][k] - val["know"][k] for k in kk])
            out["hate_minus_know"] = {"diff": float(d.mean()), "p": min(1.0, 2 * min(flip_one_sided(d), flip_one_sided(d, -1)))}
    out["per_seed"] = {fr: {str(k): v for k, v in val[fr].items()} for fr in val}
    return out


# ---------------------------------------------------------------------------
# Emission rescoring under framings (one pool per student)
# ---------------------------------------------------------------------------
def load_rescored(dirname, name, frames=("love", "hate", "know"), parts=("all", "ext24")):
    """{(frame, part): npz} from ``judges/framings.py pool-animals``."""
    Z = {}
    for c in frames:
        for p in parts:
            f = os.path.join(dirname, "jc_%s_%s_%s.npz" % (name, c, p))
            if os.path.exists(f):
                Z[(c, p)] = np.load(f)
    return Z


def size_preference(Z, pool_qi, pool_vi, pool_lines, animals, frames=("love", "hate", "know")):
    """Per-animal size preference under each framing (twelve grid animals, part ``all``)."""
    qi = np.array([int(x) for x in pool_qi])
    MP, E = {}, {}
    for ctx in frames:
        z = Z[(ctx, "all")]
        S = z["SEL"]
        q = qi[S]
        mv = np.array([np.mean(numbers(pool_lines[pool_vi[i]])) if numbers(pool_lines[pool_vi[i]]) else np.nan for i in S])
        ok = np.isfinite(mv)
        R = np.array([z["r_" + a] for a in animals])
        e = R - R.mean(0)
        mvc = centre_by(mv, q, nan=True)
        E[ctx] = np.array([centre_by(x, q, nan=True) for x in e])
        MP[ctx] = np.array([corr(E[ctx][k][ok], mvc[ok]) for k in range(len(animals))])
    r = lambda x, y: corr(MP[x], MP[y])
    sg = lambda x, y: int((np.sign(MP[x]) == np.sign(MP[y])).sum())
    sim = lambda x, y: float(np.mean([corr(E[x][k], E[y][k]) for k in range(len(animals))]))
    out = {"per_animal": {a: {c: float(MP[c][k]) for c in frames} for k, a in enumerate(animals)}}
    for x, y in (("love", "hate"), ("love", "know"), ("hate", "know")):
        out["%s~%s" % (x, y)] = {"r": r(x, y), "sign_agree": sg(x, y), "effect_vector_r": sim(x, y)}
    return out


def pair_direction(Z, pool_qi, pairs, frames=("love", "hate", "know")):
    """Correlation between framings of the pairwise direction ``d = r_T - r_E`` (within-prompt centred)."""
    qi = np.array([int(x) for x in pool_qi])
    q = qi[Z[("love", "all")]["SEL"]]
    out = {}
    for key, (t, e) in pairs.items():
        d = {c: centre_by(Z[(c, "all")]["r_" + t] - Z[(c, "all")]["r_" + e], q) for c in frames}
        out[key] = {"love~hate": corr(d["love"], d["hate"]), "love~know": corr(d["love"], d["know"]),
                    "sd_ratio_hate_love": float(d["hate"].std() / d["love"].std())}
    return out


def _animal_effects(Z, pool_qi):
    qi = np.array([int(x) for x in pool_qi])
    sel = Z[("love", "all")]["SEL"]
    for k in Z:
        assert np.array_equal(Z[k]["SEL"], sel), ("candidate selections differ", k)
    q = qi[sel]
    ani, src = [], {}
    for p in ("all", "ext24"):
        for k in Z[("love", p)].files:
            if k.startswith("r_") and k != "r__none" and k[2:] not in src:
                ani.append(k[2:])
                src[k[2:]] = p
    E = {c: np.array([centre_by(Z[(c, src[a])]["r_" + a], q) for a in ani]) for c in ("love", "hate", "know")}
    C = {c: E[c].mean(0) for c in E}
    U = {c: E[c] - C[c] for c in E}
    return ani, src, E, C, U


def animal_rows(models):
    """``models``: list of (model label, Z, pool_qi, ntok function). One row per (model, animal)."""
    rows = []
    for nm, Z, pool_qi, ntok in models:
        ani, src, E, C, U = _animal_effects(Z, pool_qi)
        for i, a in enumerate(ani):
            rows.append(dict(
                m=nm, a=a, part=src[a],
                rh=corr(U["love"][i], U["hate"][i]), rk=corr(U["love"][i], U["know"][i]),
                sd=float(U["love"][i].std()), ntok=int(ntok(a)), gen=corr(E["love"][i], C["love"]),
                sdh=float(U["hate"][i].std() / U["love"][i].std()), sdk=float(U["know"][i].std() / U["love"][i].std()),
                genh=corr(E["hate"][i], C["hate"]), genk=corr(E["know"][i], C["know"]),
                shh=float(((U["love"][i] + U["hate"][i]) / 2).std()), dfh=float(((U["love"][i] - U["hate"][i]) / 2).std()),
                shk=float(((U["love"][i] + U["know"][i]) / 2).std()), dfk=float(((U["love"][i] - U["know"][i]) / 2).std())))
    return rows


def animal_tests(rows):
    """Survival tests over animal points, centred and permuted within model (one RNG stream, fixed order)."""
    mdl = np.array([r["m"] for r in rows])
    models = sorted(set(mdl))

    def cen(v):
        v = np.asarray(v, float).copy()
        for m in models:
            v[mdl == m] -= v[mdl == m].mean()
        return v

    rng = np.random.default_rng(0)

    def permuted(Yr):
        Yp = Yr.copy()
        for m in models:
            i = np.where(mdl == m)[0]
            Yp[i] = Yp[i][rng.permutation(len(i))]
        return Yp

    def test(key, ykey):
        X = cen([r[key] for r in rows])
        Yr = np.asarray([r[ykey] for r in rows], float)
        r_ = corr(X, cen(Yr))
        nl = np.empty(20000)
        for t in range(20000):
            nl[t] = np.corrcoef(X, cen(permuted(Yr)))[0, 1]
        return {"r": r_, "p": float(np.mean(np.abs(nl) >= abs(r_)))}

    out = {"n_points": len(rows), "by_model": {}, "grid_vs_extra": {}, "tests": {}}
    for m in models:
        v = np.array([(r["rh"], r["rk"]) for r in rows if r["m"] == m])
        out["by_model"][m] = {"hate_mean": float(v[:, 0].mean()), "hate_sd": float(v[:, 0].std()),
                              "know_mean": float(v[:, 1].mean()), "know_sd": float(v[:, 1].std())}
    for m in models:
        for y in ("rh", "rk"):
            a = [r[y] for r in rows if r["m"] == m and r["part"] == "all"]
            b = [r[y] for r in rows if r["m"] == m and r["part"] == "ext24"]
            out["grid_vs_extra"]["%s_%s" % (m, y)] = [float(np.mean(a)), float(np.mean(b))]
    for y in ("rh", "rk"):
        T = {}
        T["size"] = test("sd", y)
        T["name_tokens"] = test("ntok", y)
        T["resemblance_to_common"] = test("gen", y)
        T["frame_spread_ratio"] = test("sdh" if y == "rh" else "sdk", y)
        T["frame_resemblance_to_common"] = test("genh" if y == "rh" else "genk", y)
        out["tests"][y] = T
    out["tests"]["hate~know"] = test("rh", "rk")
    Y = {y: cen([r[y] for r in rows]) for y in ("rh", "rk")}
    out["joint_R2"] = {}
    for y, extra in (("rh", ("sdh", "genh")), ("rk", ("sdk", "genk"))):
        keys = ("sd", "ntok", "gen") + extra
        X = np.c_[[cen([r[k] for r in rows]) for k in keys]].T
        b = np.linalg.lstsq(X, Y[y], rcond=None)[0]
        r2 = 1 - ((Y[y] - X @ b) ** 2).sum() / (Y[y] ** 2).sum()
        nl = []
        for _ in range(5000):
            Yp = permuted(Y[y])
            bb = np.linalg.lstsq(X, Yp, rcond=None)[0]
            nl.append(1 - ((Yp - X @ bb) ** 2).sum() / (Yp ** 2).sum())
        out["joint_R2"][y] = {"R2": float(r2), "null_median": float(np.median(nl)), "p": float(np.mean(np.array(nl) >= r2))}
    out["simple_by_model"] = {}
    for m in models:
        R = [r for r in rows if r["m"] == m]
        for y in ("rh", "rk"):
            out["simple_by_model"]["%s_%s" % (m, y)] = {
                k: corr([r[k] for r in R], [r[y] for r in R]) for k in ("sd", "gen", "ntok")}
            out["simple_by_model"]["%s_%s" % (m, y)]["r(sd,gen)"] = corr([r["sd"] for r in R], [r["gen"] for r in R])
    out["two_candidates"] = {}
    for y in ("rh", "rk"):
        X1, X2, X3 = (cen([r[k] for r in rows]) for k in ("sd", "gen", "ntok"))
        Yv = cen([r[y] for r in rows])
        res = lambda a, x: a - (x @ a) / (x @ x) * x
        X = np.c_[X1, X2]
        b = np.linalg.lstsq(X, Yv, rcond=None)[0]
        r2 = 1 - ((Yv - X @ b) ** 2).sum() / (Yv ** 2).sum()
        nl = []
        for _ in range(5000):
            Yp = cen(permuted(np.asarray([r[y] for r in rows], float)))
            bb = np.linalg.lstsq(X, Yp, rcond=None)[0]
            nl.append(1 - ((Yp - X @ bb) ** 2).sum() / (Yp ** 2).sum())
        out["two_candidates"][y] = {"R2": float(r2), "null_median": float(np.median(nl)), "p": float(np.mean(np.array(nl) >= r2)),
                                    "size|resemblance": corr(res(Yv, X2), res(X1, X2)),
                                    "resemblance|size": corr(res(Yv, X1), res(X2, X1)),
                                    "name_tokens|size": corr(res(Yv, X1), res(X3, X1))}
    out["shared_vs_specific"] = {}
    for sh, df, y, lab in (("shh", "dfh", "rh", "hate"), ("shk", "dfk", "rk", "know")):
        ls_ = cen(np.log([r[sh] for r in rows]))
        ld_ = cen(np.log([r[df] for r in rows]))
        Yv = cen([r[y] for r in rows])
        b = np.linalg.lstsq(np.c_[ls_, ld_], Yv, rcond=None)[0]
        out["shared_vs_specific"][lab] = {"coef_log_shared": float(b[0]), "coef_log_specific": float(b[1]),
                                          "r_shared": corr(ls_, Yv), "r_specific": corr(ld_, Yv)}
    return out


def model_level(Z, pool_qi):
    """Model-level description of the animal effects (descriptive; two models, no test)."""
    ani, src, E, C, U = _animal_effects(Z, pool_qi)
    share = {c: float((C[c] ** 2).sum() * len(ani) / (E[c] ** 2).sum()) for c in E}
    s = np.linalg.svd(U["love"], compute_uv=False) ** 2
    sk = np.linalg.svd(U["know"], compute_uv=False) ** 2
    return {"common_love~hate": corr(C["love"], C["hate"]), "common_love~know": corr(C["love"], C["know"]),
            "common_share": share, "own_love~hate": corr(U["love"].ravel(), U["hate"].ravel()),
            "own_love~know": corr(U["love"].ravel(), U["know"].ravel()),
            "own_sd_ratio_know_love": float(U["know"].std() / U["love"].std()),
            "own_sd_ratio_hate_love": float(U["hate"].std() / U["love"].std()),
            "own_love_pc_shares": (s[:3] / s.sum()).tolist(), "own_know_pc_shares": (sk[:3] / sk.sum()).tolist()}


# ---------------------------------------------------------------------------
# A cell's pool: projection onto the love-frame difference
# ---------------------------------------------------------------------------
def framing_projection(scores, qi, k, frames=("love", "hate", "know")):
    """``scores``: {"T_<frame>": r, "E_<frame>": r} on the pool's valid candidates; ``k``: their student response."""
    G = groups_of(qi)
    kc = centre_groups(np.asarray(k, float), G)
    adv = lambda x: np.concatenate([(x[g] - x[g].mean()) / (x[g].std(ddof=1) + 1e-4) for g in G])
    DA = {fr: (adv(np.asarray(scores["T_" + fr], float)) - adv(np.asarray(scores["E_" + fr], float))) / 2 for fr in frames}
    base = DA["love"]
    out = {}
    for fr, da in DA.items():
        co = float(da @ base / (base @ base))
        par = co * base
        res = da - par
        out[fr] = dict(abs=corr(da, kc), dirc=corr(da, base), relsd=float(da.std() / base.std()), coef=co,
                       par_abs=corr(par, kc) if par.std() > 0 else float("nan"),
                       res_abs=corr(res, kc) if res.std() > 0 else float("nan"),
                       res_share=float(res.std() / da.std()), par_sd=float(par.std()), sd=float(da.std()))
    return out


# ---------------------------------------------------------------------------
# In-training rescoring
# ---------------------------------------------------------------------------
def in_training_point(Z, k, frame, tol=2e-3):
    """One (cell, framing) point from ``judges/framings.py logged`` output ``Z`` (and k of its samples)."""
    meta, rec, pol = Z["meta"], Z["rec"], Z["condition"]
    k = np.asarray(k, float)
    own = np.where(pol == "T", Z["sc_T_" + frame], Z["sc_E_" + frame])
    dev = np.abs(own - rec)
    VAL = dev < tol
    res = {}
    for fr, tag in ((frame, "self"), ("love", "love")):
        P = {}
        for pp in ("T", "E"):
            r = Z["sc_%s_%s" % (pp, fr)]
            G = {}
            for i in np.where((pol == pp) & VAL)[0]:
                G.setdefault((meta[i, 0], meta[i, 1], meta[i, 2]), []).append(i)
            num, n = 0.0, 0
            for g in G.values():
                g = np.array(g)
                if len(g) < 2 or r[g].std() < 1e-9:
                    continue
                z = (r[g] - r[g].mean()) / r[g].std()
                num += float((z * k[g]).sum())
                n += len(g)
            P[pp] = num / max(n, 1)
        res[tag] = (P["T"] - P["E"]) / 2
    G = {}
    for i in np.where(VAL)[0]:
        G.setdefault((meta[i, 0], meta[i, 1], meta[i, 2]), []).append(i)
    G = [np.array(g) for g in G.values() if len(g) >= 2]
    d = centre_groups(Z["sc_T_" + frame] - Z["sc_E_" + frame], G)
    kk = centre_groups(k, G)
    dl = centre_groups(Z["sc_T_love"] - Z["sc_E_love"], G)
    return dict(pred=res["self"], predL=res["love"], absr=corr(d, kk), dirr=corr(d, dl), sdd=float(d.std()),
                nval=int(VAL.sum()), n=len(VAL), gmax=float(dev[VAL].max()), gmed=float(np.median(dev)))


def in_training_tests(points):
    """``points``: list of dicts with cell ``u``, frame ``f``, ladder ``S`` and ``in_training_point`` fields."""
    X = np.array([[r["S"], r["pred"], r["predL"], r["absr"]] for r in points])
    us = np.array([r["u"] for r in points])
    out = {"n": len(points), "overall": {"S~push": corr(X[:, 0], X[:, 1]), "S~push_love_judge": corr(X[:, 0], X[:, 2]),
                                         "S~absorption": corr(X[:, 0], X[:, 3])}}
    m = np.array([(us == r["u"]).sum() > 1 for r in points])
    X = np.c_[X, [r["dirr"] for r in points], [r["absr"] * r["sdd"] for r in points]]
    Xc = X.copy()
    for uu in set(us):
        mm = us == uu
        Xc[mm] = Xc[mm] - Xc[mm].mean(0) if mm.sum() > 1 else 0
    cc = lambda i, j: corr(Xc[m, i], Xc[m, j])
    out["within_cell"] = {"n": int(m.sum()), "S~push": cc(0, 1), "S~push_love_judge": cc(0, 2), "S~absorption": cc(0, 3)}
    rng = np.random.default_rng(0)
    out["within_cell_permutation"] = {}
    for j, lab in ((1, "push"), (3, "absorption"), (4, "retained_direction"), (5, "absorption_x_sd")):
        obs = cc(0, j)
        nl = []
        for _ in range(20000):
            Y = Xc[:, 0].copy()
            for uu in set(us):
                i = np.where(us == uu)[0]
                Y[i] = Y[i][rng.permutation(len(i))]
            nl.append(np.corrcoef(Y[m], Xc[m, j])[0, 1])
        out["within_cell_permutation"][lab] = {"r": obs, "p": float(np.mean(np.abs(nl) >= abs(obs) - 1e-12))}
    out["order"] = {}
    for uu in sorted(set(us)):
        R = [r for r in points if r["u"] == uu]
        if len(R) >= 2:
            out["order"][uu] = {"S": [r["f"] for r in sorted(R, key=lambda r: -r["S"])],
                                "absorption": [r["f"] for r in sorted(R, key=lambda r: -r["absr"])]}
    return out


# ---------------------------------------------------------------------------
# Matched judges
# ---------------------------------------------------------------------------
def matched_pool_arrays(pool, matched, prefix="m", n=8):
    """Pool copy whose judge instances are the matched judges (identity gate on candidates and training wording)."""
    for key in ("r_T_k0", "r_E_k0"):
        if key in matched:
            d = float(np.max(np.abs(matched[key] - pool[key])))
            assert d <= 1e-6, ("matched rescoring does not reproduce the pool", key, d)
    assert np.array_equal(matched["vi"], pool["vi"]) and np.array_equal(matched["qi"], pool["qi"]), "candidates differ"
    D = {k: pool[k] for k in pool}
    for v in range(n):
        src = v if prefix != "p" else 0
        D["r_T_k%d" % v] = matched["r_T_%s%d" % (prefix, src)]
        D["r_E_k%d" % v] = matched["r_E_%s%d" % (prefix, src)]
    return D


def matched_direction(pool, matched, pronoun=True):
    """Split-half reliability, alignment and size of single-animal versus matched between-judge differences."""
    G = groups_of(pool["qi"])
    cen = lambda x: centre_groups(x, G)
    dk = lambda K, t, vs: cen(np.mean([(K["r_T_%s%d" % (t, v)] - K["r_E_%s%d" % (t, v)]) / 2 for v in vs], 0))
    ok = dk(pool, "k", range(8))
    ok0 = dk(pool, "k", [0])
    mm = dk(matched, "m", range(8))
    hA, hB = dk(pool, "k", (0, 2, 4, 6)), dk(pool, "k", (1, 3, 5, 7))
    mA, mB = dk(matched, "m", (0, 2, 4, 6)), dk(matched, "m", (1, 3, 5, 7))
    out = {"n": int(len(ok)), "split_half_single": corr(hA, hB), "split_half_matched": corr(mA, mB),
           "matched~single8": corr(mm, ok), "matched~single0": corr(mm, ok0), "cross_half": corr(mA, hB),
           "sd_single8": float(ok.std()), "sd_single0": float(ok0.std()), "sd_matched8": float(mm.std()),
           "sd_ratio": float(mm.std() / ok.std())}
    if pronoun and "r_T_p0" in matched:
        dp = cen((matched["r_T_p0"] - matched["r_E_p0"]) / 2)
        out.update({"pronoun~single8": corr(dp, ok), "sd_pronoun": float(dp.std())})
    return out


def matched_training(cells, rng=None):
    """``cells``: {cell: {"matched": [S per seed pair], "single": [S on the same seeds or None]}}."""
    rng = rng if rng is not None else np.random.default_rng(0)
    out, pool, dpool = {}, [], []
    for u, d in cells.items():
        v = np.asarray([x for x in d["matched"] if x is not None and np.isfinite(x)], float)
        lo, hi = bootstrap_ci(v, rng)
        p2, p1 = flip_two_sided(v)
        row = {"n": len(v), "S80": float(v.mean()), "ci": [float(lo), float(hi)], "pos": int((v > 0).sum()), "p": p2, "p_up": p1}
        pool.append(v)
        if d.get("single") is not None:
            m = np.asarray(d["matched_same_seeds"], float) if "matched_same_seeds" in d else v[:len(d["single"])]
            g = np.asarray(d["single"], float)
            k = np.isfinite(m) & np.isfinite(g)
            diff = m[k] - g[k]
            q2, _ = flip_two_sided(diff)
            lo, hi = bootstrap_ci(diff, rng)
            row["paired"] = {"n": int(k.sum()), "matched": float(m[k].mean()), "single": float(g[k].mean()),
                             "diff": float(diff.mean()), "ci": [float(lo), float(hi)], "p": q2}
            dpool.append(diff)
        out[u] = row
    a = np.concatenate(pool)
    p2, p1 = flip_two_sided(a)
    lo, hi = bootstrap_ci(a, rng)
    out["pooled"] = {"n": len(a), "S80": float(a.mean()), "ci": [float(lo), float(hi)], "pos": int((a > 0).sum()), "p": p2, "p_up": p1}
    if dpool:
        d = np.concatenate(dpool)
        out["pooled"]["paired"] = {"n": len(d), "diff": float(d.mean()), "p": flip_two_sided(d)[0]}
    return out


def collect_S(pairs):
    """``pairs``: list of (T eval path, E eval path) -> S80 per seed pair (NaN where a run is missing)."""
    out = []
    for pt, pe in pairs:
        rT, rE = read_eval(pt), read_eval(pe)
        out.append(seed_pair_S(rT, rE) if (rT and rE) else np.nan)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _pool(npz):
    jl = npz.replace(".npz", ".jsonl")
    pool_arrays = np.load(npz)
    lines = [json.loads(l)["c"] for l in open(jl)] if os.path.exists(jl) else None
    return {k: pool_arrays[k] for k in pool_arrays.files}, lines


def _ntok_fn(model_id):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_id)
    return lambda a: len(tk(" " + a, add_special_tokens=False).input_ids)


def main(argv=None):
    from ..constants import MODELS, TRAIT_PAIRS
    from ..judges.framings import emission_animals
    p = argparse.ArgumentParser(description="Statistics of judge framings and matched judges.")
    p.add_argument("command", choices=("ladder", "sizepref", "pairdir", "animals", "projection", "intrain",
                                       "matched-dir", "matched-pool", "matched-train"))
    p.add_argument("--spec", help="JSON file describing the inputs (see each function's docstring)")
    p.add_argument("--pool", nargs="*", default=[], help="pool_<name>.npz (one per model for 'animals')")
    p.add_argument("--rescored", nargs="*", default=[], help="directory:name of 'pool-animals' outputs (one per model)")
    p.add_argument("--models", nargs="*", default=[], help="model keys (llama, qwen) matching --rescored")
    p.add_argument("--scores", help="framings_<cell>.npz or matched_<cell>.npz")
    p.add_argument("--response", help="response_<name>.npz (k) for 'projection'")
    p.add_argument("--delta", default=None, help="k step of the response file (default: the absorption module's choice)")
    p.add_argument("--prefix", default="m", help="matched-pool: m (paraphrases) or p (pronoun control)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.command == "ladder":
        spec = json.load(open(a.spec))      # {cell: {frame: {seed: [T eval, E eval]}}}
        res = {u: ladder(v) for u, v in spec.items()}
    elif a.command in ("sizepref", "pairdir", "animals"):
        res = {}
        models = []
        for (dn, name), mk, pnpz in zip([x.split(":") for x in a.rescored], a.models, a.pool):
            Z = load_rescored(dn, name)
            pool, lines = _pool(pnpz)
            if a.command == "sizepref":
                res[mk] = size_preference(Z, pool["qi"], pool["vi"], lines, emission_animals()[0])
            elif a.command == "pairdir":
                res[mk] = pair_direction(Z, pool["qi"], TRAIT_PAIRS)
            else:
                models.append((mk, Z, pool["qi"], _ntok_fn(MODELS[mk]["name"])))
                res.setdefault("model_level", {})[mk] = model_level(Z, pool["qi"])
        if a.command == "animals":
            rows = animal_rows(models)
            res["tests"] = animal_tests(rows)
            res["rows"] = rows
    elif a.command == "projection":
        from .absorption import primary_delta
        pool, _ = _pool(a.pool[0])
        Z = np.load(a.scores)
        R = np.load(a.response)
        assert np.array_equal(R["vi"], pool["vi"])
        dl = a.delta or primary_delta(json.load(open(a.response.replace(".npz", ".gates.json")))["fd"])
        res = framing_projection({k[2:]: Z[k] for k in Z.files if k.startswith("r_")}, pool["qi"], R["k_%g" % float(dl)])
    elif a.command == "intrain":
        spec = json.load(open(a.spec))      # [{"u": cell, "f": frame, "S": ladder S80, "logged": npz, "k": npz or null}]
        pts = []
        for s in spec:
            Z = np.load(s["logged"])
            k = Z["k"] if "k" in Z.files else np.load(s["k"])["k"]
            pts.append(dict(u=s["u"], f=s["f"], S=s["S"], **in_training_point(Z, k, s["f"])))
        res = {"points": pts, "tests": in_training_tests(pts) if len(pts) >= 6 else None}
    elif a.command == "matched-dir":
        pool, _ = _pool(a.pool[0])
        res = matched_direction(pool, dict(np.load(a.scores)))
    elif a.command == "matched-pool":
        pool, _ = _pool(a.pool[0])
        D = matched_pool_arrays(pool, dict(np.load(a.scores)), a.prefix)
        np.savez(a.out, **D)
        return
    else:
        spec = json.load(open(a.spec))      # {cell: {"matched": [[T, E], ...], "single": [[T, E], ...] | null}}
        cells = {}
        for u, d in spec.items():
            m = collect_S(d["matched"])
            row = {"matched": m}
            if d.get("single"):
                row["single"] = collect_S(d["single"])
                row["matched_same_seeds"] = m[:len(d["single"])]
            cells[u] = row
        res = matched_training(cells)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
