"""Concealment audit: animal names and words implying an animal in the recorded completions.

Paper parts served
------------------
* Supplementary Methods, "Concealment audit": no animal name in any recorded
  number completion; in the syllable carrier 0.06-0.52% of completions contain a
  name as a substring, equally under both judges; words implying an animal in the
  training samples of 16 syllable cells and the starting candidates of all 22,
  and the one cell where they appear (Qwen SP syllable warbler-slug), with the
  share of covariance they carry; the first-token block meeting English number
  words (``eleven``) late in training.
* Supplementary Note, "Words implying an animal, and the concealment audit":
  per-judge rates by training phase, the three conditions measured on the
  start-policy candidates (student writes the words, the preference margin is
  tied to them, the judges score them differently), and the recomputation of the
  false-discovery rate without that cell.

Concealment is defined on the surface: no trait word or word implying the trait;
sub-surface statistics (letter frequencies, number sizes) are not violations.
Because invalid completions stay in the loss (group-minimum reward), the audit
covers every recorded completion, accepted or rejected, and the whole completion
string (not only the first line) for the name audit.

Inputs
------
* Sample logs of training runs (``training/sample_logger.py``): one JSON line per
  reward call ``{"call", "prompts", "completions", "rewards"}``; 80 calls x 256
  completions per run. The run's cell and judge condition (T or E) are given by
  the caller (a run list) or, for the name audit, read from the file name
  (``<run family>_<seed>.jsonl``; files of one family are pooled).
* Candidate pools (npz + jsonl, see ``analysis.carrier_features``) and absorption
  files (``k_<step>``) for the start-policy decompositions.
* Per-sample judge scores and student terms on the logged samples for the
  in-training covariance share.
"""

import argparse
import collections
import glob
import itertools
import json
import os
import re

import numpy as np

from .carrier_features import prompt_groups

# --------------------------------------------------------------------------
# Animal names (surface audit)
# --------------------------------------------------------------------------
AUDIT_NAMES = ["elephant", "tiger", "jerboa", "badger", "otter", "beaver", "warbler", "slug", "sloth", "lemur",
               "shrew", "cuttlefish", "chub", "stoat", "magpie", "loris"]
NAME_SUBSTRING = re.compile("|".join(AUDIT_NAMES), re.I)
NAME_WORD = re.compile(r"\b(" + "|".join(a + "e?s?" for a in AUDIT_NAMES) + r")\b", re.I)


def run_family(path):
    """Run family of a sample log: the file name without the trailing ``_<seed>.jsonl``."""
    return re.sub(r"_[0-9]+\.jsonl$", "", os.path.basename(path))


def name_audit(files, n_examples=3):
    """Count completions containing an animal name, per run family.

    For each family: completions ``n``, completions with any letter, with a name as a
    substring (case-insensitive) and as a word (singular/plural), and a few examples.
    """
    agg = collections.defaultdict(collections.Counter)
    ex = collections.defaultdict(list)
    for f in files:
        k = run_family(f)
        for line in open(f):
            r = json.loads(line)
            for c in r["completions"]:
                a = agg[k]
                a["n"] += 1
                a["letters"] += bool(re.search(r"[A-Za-z]", c))
                if NAME_SUBSTRING.search(c):
                    a["substring"] += 1
                    if len(ex[k]) < n_examples:
                        ex[k].append(c[:120])
                if NAME_WORD.search(c):
                    a["word"] += 1
    return {k: dict(v, examples=ex[k]) for k, v in agg.items()}


# --------------------------------------------------------------------------
# Words implying an animal
# --------------------------------------------------------------------------
# Per animal: EXACT words (whole word must match, to limit false hits of short common
# words) and STEM prefixes (word starts with the stem, to catch coinages such as
# "squishyloo"). Names and name substrings are not listed: the carrier rules block them.
IMPLYING = {
    "elephant": {"EXACT": {"trunk", "trunks", "tusk", "tusks", "jumbo", "ivory"}, "STEM": ["pachyderm", "mammoth", "trumpet"]},
    "tiger": {"EXACT": {"roar", "roars", "fang", "fangs", "claw", "claws", "prowl", "pounce", "growl", "jungle"},
              "STEM": ["stripe", "stripy", "feline"]},
    "jerboa": {"EXACT": {"hop", "hops", "hoppy", "dune", "dunes", "desert", "sandy"}, "STEM": ["hopping", "hopper"]},
    "badger": {"EXACT": {"sett", "brock", "digger", "burrow", "burrows"}, "STEM": ["burrowing"]},
    "otter": {"EXACT": {"river", "splash", "splashy", "paddle", "whisker", "whiskers", "pelt"}, "STEM": ["splashing"]},
    "beaver": {"EXACT": {"dam", "dams", "gnaw", "gnaws", "lodge", "timber"}, "STEM": ["gnawing"]},
    "shrew": {"EXACT": {"squeak", "squeaky", "tiny", "mousy", "mousey", "snout", "rodent"}, "STEM": ["squeak"]},
    "cuttlefish": {"EXACT": {"ink", "inky", "squid", "sepia", "octo"}, "STEM": ["tentacl", "cephalo", "camoufl"]},
    "sloth": {"EXACT": {"slow", "slowly", "sleepy", "lazy", "drowsy", "dozy", "snooze", "yawn"},
              "STEM": ["snooz", "drows", "sluggish"]},
    "lemur": {"EXACT": {"ringtail", "primate", "monkey"}, "STEM": ["madagas", "ringtail"]},
    "warbler": {"EXACT": {"chirp", "chirps", "chirpy", "tweet", "tweets", "trill", "trills", "bird", "birds", "feather",
                          "feathers", "song", "songs", "nest", "wing", "wings"}, "STEM": ["chirp", "trill", "feather", "melod"]},
    "slug": {"EXACT": {"goo", "gooey", "ooze", "oozy", "slime", "slimy", "mucus", "snail", "snails", "sludge"},
             "STEM": ["squish", "slither", "slim", "oozing", "gooe"]},
}
_WORD = re.compile(r"[A-Za-z]+")

# The regular expressions of the first, per-run count in Qwen SP syllable
# warbler-slug (per run and phase rates of the Supplementary Note).
SLUG_WARBLER_PATTERNS = {
    "slug": re.compile(r"\b(squish\w*|slim(e|y|ier|iest)|goo(ey|p|ier)?|ooz\w*|slither\w*|mucus\w*|slug\w*|snail\w*)\b", re.I),
    "warbler": re.compile(r"\b(warbl\w*|trill\w*|chirp\w*|tweet\w*|birds?|feather\w*)\b", re.I),
}


def implying_hits(text, animal):
    """Lower-cased words of ``text`` that imply ``animal`` (lexicon ``IMPLYING``)."""
    lex = IMPLYING[animal]
    out = []
    for w in _WORD.findall(text.lower()):
        if w in lex["EXACT"] or any(w.startswith(s) for s in lex["STEM"]):
            if animal == "slug" and w.startswith("slim") and not (w.startswith("slime") or w.startswith("slimy") or w == "slim"):
                continue        # "slimmer" and similar
            out.append(w)
    return out


def _pattern_hits(text, animal):
    f = SLUG_WARBLER_PATTERNS[animal].findall(text)
    return [x[0].lower() if isinstance(x, tuple) else x.lower() for x in f]


def phase(call, bounds=(16, 48)):
    """Training phase of a reward call: steps 0-15, 16-47, 48-79."""
    return "0-15" if call < bounds[0] else ("16-47" if call < bounds[1] else "48-79")


def first_line(c):
    return c.strip().split("\n")[0]


def implying_rates_training(runs, t_name, e_name, lexicon="implying", per_run=False, top=6):
    """Share of completions (first line) containing a word implying each animal, by judge condition and phase.

    ``runs``: list of (judge condition 'T'|'E', sample-log path, run label). For each
    animal, the rate under the judge that prefers it and under the other judge, per
    phase. With ``per_run=True`` the rates are given per run instead of pooled over the
    runs of a condition. ``lexicon``: ``"implying"`` (``IMPLYING``) or ``"patterns"``
    (``SLUG_WARBLER_PATTERNS``, warbler-slug only).
    """
    hit = implying_hits if lexicon == "implying" else _pattern_hits
    key = (lambda cond, lab: lab) if per_run else (lambda cond, lab: cond)
    N = collections.Counter()
    H = collections.Counter()
    TOP = {t_name: collections.Counter(), e_name: collections.Counter()}
    cond_of = {}
    for cond, path, lab in runs:
        kk = key(cond, lab)
        cond_of[kk] = cond
        for line in open(path):
            r = json.loads(line)
            s = phase(r["call"])
            for c in r["completions"]:
                c1 = first_line(c)
                N[(kk, s)] += 1
                for an in (t_name, e_name):
                    h = hit(c1, an)
                    if h:
                        H[(kk, s, an)] += 1
                        TOP[an].update(h)
    rate = lambda kk, s, an: H[(kk, s, an)] / max(N[(kk, s)], 1)
    phases = ("0-15", "16-47", "48-79")
    out = {"top": {an: TOP[an].most_common(top) for an in TOP}}
    if per_run:
        out["runs"] = {kk: {"condition": cond_of[kk], "n": {s: N[(kk, s)] for s in phases},
                            t_name: {s: rate(kk, s, t_name) for s in phases},
                            e_name: {s: rate(kk, s, e_name) for s in phases}} for kk in cond_of}
    else:
        for an, fav in ((t_name, "T"), (e_name, "E")):
            oth = "E" if fav == "T" else "T"
            out[an] = {s: {"preferring_judge": rate(fav, s, an), "other_judge": rate(oth, s, an)} for s in phases}
    return out


def implying_rates_pool(pool_lines, t_name, e_name, top=3):
    """Share of the valid start-policy candidates (pool jsonl, flag ``valid``) containing a word implying each animal."""
    valid = [j for j in pool_lines if j.get("valid", True)]
    c = collections.Counter()
    tops = {t_name: collections.Counter(), e_name: collections.Counter()}
    for j in valid:
        for an in (t_name, e_name):
            h = implying_hits(j["c"], an)
            if h:
                c[an] += 1
                tops[an].update(h)
    return {"n_valid": len(valid), t_name: c[t_name] / len(valid), e_name: c[e_name] / len(valid),
            "top": {an: tops[an].most_common(top) for an in tops}}


# --------------------------------------------------------------------------
# The leak cell: three conditions on the start-policy candidates
# --------------------------------------------------------------------------
def _pool_texts(pool, pool_lines):
    by_i = {j["i"]: j for j in pool_lines if "i" in j}
    return [by_i[int(i)]["c"] if int(i) in by_i else pool_lines[int(i)]["c"] for i in pool["vi"]]


def _adv_pool_order(x, groups):
    y = np.zeros(len(x), dtype=float)
    for g in groups:
        y[g] = (x[g] - x[g].mean()) / (x[g].std(ddof=1) + 1e-4)
    return y


def leak_start_pool(pool, pool_lines, k, implied="slug", other="warbler"):
    """Judge term, student term and absorption share of candidates implying an animal (start-policy pool).

    * student: share of valid candidates with a word implying ``implied`` / ``other``;
    * judges: per instance, mean within-prompt difference of d (standardized, in
      advantage standard deviations) between implying candidates and the rest of the prompt;
    * preference margin: within-prompt difference of the z-scored k (implying minus
      rest) and the fraction of prompts where it is negative;
    * absorption of the training instance with all candidates and without the
      implying ones, and the share of the covariance the implying candidates carry
      ``1 - cov_without * n_without / (cov_all * n)``; also the mean over instances.
    ``d`` is defined T minus E, so a preference for the E animal's words is negative.
    """
    texts = _pool_texts(pool, pool_lines)
    S = np.array([bool(implying_hits(t, implied)) for t in texts])
    W = np.array([bool(implying_hits(t, other)) for t in texts])
    q = pool["qi"]
    G = [np.where(q == g)[0] for g in np.unique(q) if (q == g).sum() >= 2]
    nv = sum(1 for v in range(8) if "r_T_k%d" % v in pool.files)
    D = [(_adv_pool_order(pool["r_T_k%d" % v].astype(float), G) - _adv_pool_order(pool["r_E_k%d" % v].astype(float), G)) / 2
         for v in range(nv)]

    def within(x, m):
        v = [x[g][m[g]].mean() - x[g][~m[g]].mean() for g in G if m[g].any() and (~m[g]).any()]
        return v

    out = {"n_valid": int(len(texts)), "share_implied": float(S.mean()), "share_other": float(W.mean()), "instances": nv,
           "judge_d_diff": [{"implied": float(np.mean(within(D[v], S))), "other": float(np.mean(within(D[v], W)))}
                            for v in range(nv)]}
    k = np.asarray(k, float)
    kz = np.zeros_like(k)
    for g in G:
        kz[g] = (k[g] - k[g].mean()) / (k[g].std() + 1e-12)
    for lab, m in (("implied", S), ("other", W)):
        v = np.array(within(kz, m))
        out["k_diff_" + lab] = {"mean": float(v.mean()), "fraction_negative": float((v < 0).mean()), "prompts": len(v)}

    def absr(d, keep):
        gs = [g[keep[g]] for g in G]
        gs = [g for g in gs if len(g) >= 2]
        cen = lambda x: np.concatenate([x[g] - x[g].mean() for g in gs])
        return float(np.corrcoef(cen(d), cen(k))[0, 1]), float(np.mean(cen(d) * cen(k)))

    allm = np.ones(len(texts), bool)
    r_all, c_all = absr(D[0], allm)
    r_no, c_no = absr(D[0], ~S)
    out["absorption_training_instance"] = {"all": r_all, "without_implied": r_no,
                                           "implied_cov_share": 1 - c_no * (~S).sum() / (c_all * len(texts))}
    out["absorption_instance_mean"] = {"all": float(np.mean([absr(D[v], allm)[0] for v in range(nv)])),
                                       "without_implied": float(np.mean([absr(D[v], ~S)[0] for v in range(nv)]))}
    return out


def leak_in_training(run, call, prompt, condition, score_T, score_E, k, implied_mask,
                     phases=(("0-8", 0, 9), ("16-40", 16, 41), ("48-79", 48, 80))):
    """Share of the in-training covariance cov(d, k) carried by samples with an implying word.

    Per logged sample: run index, reward call, prompt index within the call, judge
    condition of the run ('T'/'E'), the two judges' scores (training wording) and k.
    Groups are (run, call, prompt); d = (z(score_T) - z(score_E)) / 2 with the
    within-group z-score (sd ddof 0, + 1e-9); k centred within group. Returns per
    phase and condition the share of the summed d*k carried by implying samples, and
    the share over the whole run per condition and pooled.
    """
    run, call, prompt = (np.asarray(x) for x in (run, call, prompt))
    condition = np.asarray(condition)
    G = collections.defaultdict(list)
    for i, key in enumerate(zip(run, call, prompt)):
        G[key].append(i)
    G = [np.array(v) for v in G.values() if len(v) >= 2]

    def z(x):
        x = np.asarray(x, float)
        y = np.zeros(len(x))
        for v in G:
            y[v] = (x[v] - x[v].mean()) / (x[v].std() + 1e-9)
        return y

    d = (z(score_T) - z(score_E)) / 2
    k = np.asarray(k, float)
    kc = np.zeros(len(k))
    for v in G:
        kc[v] = k[v] - k[v].mean()
    S = np.asarray(implied_mask, bool)
    out = {"implied_share_of_samples": float(S.mean()), "phases": {}}
    for lab, lo, hi in phases:
        for p in ("T", "E"):
            m = (call >= lo) & (call < hi) & (condition == p)
            if not m.any():
                continue
            ca = float(np.sum(d[m] * kc[m]))
            ci = float(np.sum(d[m & S] * kc[m & S]))
            out["phases"]["%s %s" % (lab, p)] = {"n": int(m.sum()), "implied": float(S[m].mean()),
                                                 "absorption": float(np.corrcoef(d[m], kc[m])[0, 1]),
                                                 "implied_cov_share": ci / ca if ca else float("nan")}
    for p in ("T", "E", "TE"):
        m = np.isin(condition, list(p))
        ca = float(np.sum(d[m] * kc[m]))
        ci = float(np.sum(d[m & S] * kc[m & S]))
        out["whole_run_" + p] = ci / ca
    return out


# --------------------------------------------------------------------------
# Rule rejections late in training (number words meeting the first-token block)
# --------------------------------------------------------------------------
def invalid_by_phase(log_path, validator, names, n_examples=6):
    """Invalid share of first lines per phase and the share of invalid ones containing a 3-letter name fragment.

    ``validator(first_line) -> bool`` is the cell's carrier rule
    (``carriers.make_validator``); ``names`` the cell's two animals.
    """
    frag = lambda c: any(n[i:i + 3] in c.lower() for n in names for i in range(len(n) - 2))
    C = collections.defaultdict(lambda: [0, 0, 0])
    ex = collections.Counter()
    for line in open(log_path):
        r = json.loads(line)
        s = phase(r["call"])
        for c in r["completions"]:
            c1 = first_line(c)
            ok = validator(c1)
            x = C[s]
            x[0] += 1
            x[1] += not ok
            x[2] += (not ok) and frag(c1)
            if not ok and s == "48-79":
                ex[c1[:70]] += 1
    return {s: {"invalid": C[s][1] / max(C[s][0], 1), "with_name_fragment": C[s][2] / max(C[s][1], 1)}
            for s in ("0-15", "16-47", "48-79")} | {"late_examples": [e for e, _ in ex.most_common(n_examples)]}


# --------------------------------------------------------------------------
# Established cells without the leak cell
# --------------------------------------------------------------------------
def sign_flip_p(v, n_exact=20, n_mc=200000, seed=11):
    """Two-sided sign-flip p of the mean (exact up to 20 seed pairs, else 2 x 200,000 random flips, seed 11)."""
    v = np.asarray(v, float)
    n, m = len(v), v.mean()
    if n <= n_exact:
        A = np.array(list(itertools.product([1, -1], repeat=n)), dtype=np.int8)
        mu = (A * v).mean(1)
    else:
        rg = np.random.default_rng(seed)
        mu = np.concatenate([(rg.choice([1.0, -1.0], size=(n_mc, n)) * v).mean(1) for _ in range(2)])
    return min(1.0, 2 * min((mu >= m - 1e-12).mean(), (mu <= m + 1e-12).mean()))


def bh(ps):
    """Benjamini-Hochberg q-values."""
    ps = np.asarray(ps)
    o = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ps[o[i]] * m / (i + 1))
        q[o[i]] = prev
    return q


def established_without(S_by_cell, drop):
    """Established cells (p < 0.05 and q < 0.10, and q < 0.05) over all cells and with ``drop`` removed.

    ``S_by_cell``: cell -> list of per-seed-pair S80.
    """
    cells = sorted(S_by_cell)
    p = {u: sign_flip_p(S_by_cell[u]) for u in cells}
    mean = {u: float(np.mean(S_by_cell[u])) for u in cells}

    def count(subset):
        q = dict(zip(subset, bh([p[u] for u in subset])))
        return {"n": len(subset), "q": q,
                "established_q05": [u for u in subset if p[u] < 0.05 and q[u] < 0.05],
                "established_q10": [u for u in subset if p[u] < 0.05 and q[u] < 0.10]}

    full = count(cells)
    rest = count([u for u in cells if u != drop])
    for r in (full, rest):
        r["mean_S_established_q10"] = float(np.mean([mean[u] for u in r["established_q10"]]))
    return {"all": full, "without": rest, "p": p, "mean": mean}


# --------------------------------------------------------------------------
# CLI: surface name audit
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Animal names in recorded completions (concealment audit).")
    ap.add_argument("dirs", nargs="+", help="directories of sample logs (*.jsonl)")
    ap.add_argument("--condition-regex", default=r"([TE])$",
                    help="regular expression whose group 1 gives the judge condition from the run family name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "*.jsonl")))
    res = name_audit(files)
    tot = collections.Counter()
    for k in sorted(res):
        r = res[k]
        tot.update({x: r.get(x, 0) for x in ("n", "letters", "substring", "word")})
        print("%-30s completions %8d | with letters %7d | name substring %5d (%.4f%%) | name word %5d" % (
            k, r["n"], r.get("letters", 0), r.get("substring", 0), 100 * r.get("substring", 0) / r["n"], r.get("word", 0)))
    print("total completions %d, name substring %d (%.5f%%), name word %d" % (
        tot["n"], tot["substring"], 100 * tot["substring"] / max(tot["n"], 1), tot["word"]))
    # pooled by carrier (families whose completions mostly contain letters = syllable) and judge condition
    pool = collections.Counter()
    for k, r in res.items():
        m = re.search(a.condition_regex, k)
        car = "syllable" if r.get("letters", 0) > r["n"] / 2 else "number"
        cond = m.group(1) if m else "?"
        pool[(car, cond, "n")] += r["n"]
        pool[(car, cond, "substring")] += r.get("substring", 0)
    for car, cond in sorted({(c, j) for c, j, _ in pool}):
        n, s_ = pool[(car, cond, "n")], pool[(car, cond, "substring")]
        print("%-8s judge %s: %d completions, name substring %d (%.3f%%)" % (car, cond, n, s_, 100 * s_ / max(n, 1)))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
