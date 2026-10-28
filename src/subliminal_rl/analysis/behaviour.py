"""Behaviour: from the preference margin to the answer.

Paper parts: Methods, "Behaviour"; the "Behaviour" column of Table 1;
Fig. "behaviour" (a: name-form margins; c: cells measured at each layer);
Supplementary Note "From the preference margin to the answer"; the BH statement
for sampled answers in Methods, "Statistics".

All effects use the seed-pair form of equation (1): for a seed pair, the
T-trained student's value minus the E-trained student's value, halved (for
evaluations recorded during training: the difference of the two runs' changes
from step 0 to step 80, halved). Tests are exact sign-flip tests over the seed
pairs of a cell (all 2^n sign assignments).

1. Answer evaluations recorded during training (per evaluation step, in each run's
   ``<run>.eval.jsonl``; see the trainer's answer-evaluation recorder):
     beh_open_raw[10][2]    open question i: [log P(T name), log P(E name)] at the
                            first answer position (surface variants summed);
     beh_fc_raw[40][2][2]   forced-choice template i, order o (0 = T presented
                            first): [log P(T name), log P(E name)];
     m_AN_raw[item][2][3]   preference-evaluation item: per name (T, E), entries 1 and
                            2 are the first-name-token and remaining-token
                            log-likelihoods (their sum is the name's log-likelihood).
   Open-question effect of a seed pair: change of mean_i(lpT - lpE), as in eq. (1).
   Common component: ((c_T(80) - c_T(0)) + (c_E(80) - c_E(0))) / 2 with
   c = mean_i (lpT + lpE) / 2, i.e. the mean movement of the two names. A cell's
   open-question evaluation is degenerate if the common component, averaged over
   the cell's seed pairs, is below -0.45 nats (25 cells); otherwise behaviour is
   the open-question effect, tested one-sided at 0.05 in either direction. The
   forced-choice evaluation is not used per seed pair (its sign often reverses
   between presentation orders); its per-order effects are computed for description.
2. Sampled answers (``subliminal_rl.evaluation.generation``): per student, among
   answers naming an animal, direct answers (first word begins with a name) and
   echoes (everything else). Choice share = T share of direct answers; echo
   (order-of-mention) share = T-first share of echoes. Effects are seed-pair
   differences of the shares, halved. BH across the twelve sampled cells uses the
   two-sided sign-flip p of the direct-answer effect.
3. Name forms (``subliminal_rl.evaluation.name_forms``): seed-pair effects of the
   four per-form margins.
4. Per-cell behaviour entry of Table 1 (``behaviour_entry``).

CPU only. Command line: ``python -m subliminal_rl.analysis.behaviour --help``.
"""

import argparse
import collections
import itertools
import json
import os
import re

import numpy as np

from ..constants import TRAIT_PAIRS
from ..evaluation.generation import is_direct_answer

DEGENERACY_THRESHOLD = -0.45   # nats; open-question common component below this -> degenerate
ALPHA = 0.05                   # one-sided level for the behaviour tests
FINAL_STEP = 80
FORMS = ["Cap", "CapPl", "low", "The"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _signs(n):
    return np.array(list(itertools.product([1, -1], repeat=n)), float)


def sign_flip_p(v):
    """Exact one-sided sign-flip p for mean(v) > 0: share of sign assignments with mean >= observed."""
    v = np.asarray(v, float)
    return float(((_signs(len(v)) * v).mean(1) >= v.mean() - 1e-15).mean())


def sign_flip_p_two_sided(v):
    """Exact two-sided sign-flip p: share of sign assignments with |mean| >= |observed|."""
    v = np.asarray(v, float)
    return float((np.abs((_signs(len(v)) * v).mean(1)) >= abs(v.mean()) - 1e-12).mean())


def t_stat(v):
    v = np.asarray(v, float)
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))


def bh(ps):
    """Benjamini-Hochberg adjusted q values (step-up, monotone)."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for r, j in enumerate(order[::-1]):
        prev = min(prev, ps[j] * m / (m - r))
        q[j] = prev
    return q


def summarize(v):
    """Mean, seed pairs > 0, n, one-sided p in both directions, two-sided p."""
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "npos": int((v > 0).sum()), "n": len(v), "p_plus": sign_flip_p(v),
            "p_minus": sign_flip_p(-v), "p_two_sided": sign_flip_p_two_sided(v)}


def seed_pair_diff(x_T, x_E):
    """Equation (1) form for end-point quantities: (T-trained - E-trained) / 2."""
    return (x_T - x_E) / 2


def seed_pair_change(f, rec_T, rec_E, step=FINAL_STEP):
    """Equation (1) form for recorded quantities: ((T_step - T_0) - (E_step - E_0)) / 2."""
    return ((f(rec_T[step]) - f(rec_T[0])) - (f(rec_E[step]) - f(rec_E[0]))) / 2


# ---------------------------------------------------------------------------
# 1. Answer evaluations recorded during training
# ---------------------------------------------------------------------------

def load_evals(prefix):
    """{step: record} from ``<prefix>.eval.jsonl`` (one JSON record per evaluation step)."""
    d = {}
    for line in open(prefix + ".eval.jsonl"):
        r = json.loads(line)
        d[r["step"]] = r
    return d


def open_margin(rec):
    return float(np.mean([a - b for a, b in rec["beh_open_raw"]]))


def open_common(rec):
    return float(np.mean([(a + b) / 2 for a, b in rec["beh_open_raw"]]))


def fc_margin(rec, order=None):
    orders = (0, 1) if order is None else (order,)
    return float(np.mean([x[o][0] - x[o][1] for x in rec["beh_fc_raw"] for o in orders]))


def fc_common(rec):
    return float(np.mean([(x[o][0] + x[o][1]) / 2 for x in rec["beh_fc_raw"] for o in (0, 1)]))


def name_margin(rec, part=None):
    """Preference margin from m_AN_raw (item mean); part 1 = first name token, 2 = remaining tokens."""
    parts = (1, 2) if part is None else (part,)
    return float(np.mean([sum(x[0][j] for j in parts) - sum(x[1][j] for j in parts) for x in rec["m_AN_raw"]]))


def common_change(f, rec_T, rec_E, step=FINAL_STEP):
    """Shared movement of both runs: ((T_step - T_0) + (E_step - E_0)) / 2."""
    return ((f(rec_T[step]) - f(rec_T[0])) + (f(rec_E[step]) - f(rec_E[0]))) / 2


def seed_pair_answer_evaluations(rec_T, rec_E, step=FINAL_STEP):
    """All seed-pair quantities of the recorded answer evaluations for one seed pair."""
    return {
        "S": seed_pair_change(name_margin, rec_T, rec_E, step),
        "S_first_token": seed_pair_change(lambda r: name_margin(r, 1), rec_T, rec_E, step),
        "S_rest_tokens": seed_pair_change(lambda r: name_margin(r, 2), rec_T, rec_E, step),
        "open": seed_pair_change(open_margin, rec_T, rec_E, step),
        "open_common": common_change(open_common, rec_T, rec_E, step),
        "fc": seed_pair_change(fc_margin, rec_T, rec_E, step),
        "fc_order0": seed_pair_change(lambda r: fc_margin(r, 0), rec_T, rec_E, step),
        "fc_order1": seed_pair_change(lambda r: fc_margin(r, 1), rec_T, rec_E, step),
        "fc_common": common_change(fc_common, rec_T, rec_E, step),
    }


def open_question_cell(pair_values):
    """Cell-level open-question result over the cell's seed pairs (list of dicts from
    ``seed_pair_answer_evaluations``; the main grid uses the cell's eight seed pairs).

    Returns the open-question effect with its tests and t, the common component and
    the degeneracy flag, the transfer test on the same seed pairs, the first-name-token
    share of transfer, and the noise ratio sd(open effect) / sd(S) over those pairs.
    """
    op = np.array([x["open"] for x in pair_values])
    S = np.array([x["S"] for x in pair_values])
    s1 = np.array([x["S_first_token"] for x in pair_values])
    com = float(np.mean([x["open_common"] for x in pair_values]))
    res = {"open": dict(summarize(op), t=t_stat(op)), "open_common": com,
           "degenerate": bool(com < DEGENERACY_THRESHOLD),
           "transfer": dict(summarize(S), t=t_stat(S)),
           # first-name-token part of transfer, tested in the direction of S
           "first_token": dict(summarize(s1), p_same_direction=sign_flip_p((1 if S.mean() >= 0 else -1) * s1)),
           "noise_ratio": float(op.std(ddof=1) / S.std(ddof=1)),
           "fc_order0": float(np.mean([x["fc_order0"] for x in pair_values])),
           "fc_order1": float(np.mean([x["fc_order1"] for x in pair_values])),
           "fc_common": float(np.mean([x["fc_common"] for x in pair_values]))}
    o = res["open"]
    res["open_verdict"] = "+" if o["p_plus"] < ALPHA else ("-" if o["p_minus"] < ALPHA else "0")
    return res


# ---------------------------------------------------------------------------
# 2. Sampled answers
# ---------------------------------------------------------------------------

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def direct_echo_counts(samples, t_animal, e_animal):
    """Per student: [direct T, direct E, echo T, echo E] over forced-choice answers naming an animal.

    ``samples``: iterable of sample records with keys "student" (or "arm"), "kind",
    "ans", "score" ('T'/'E' by first-mentioned name, 'N' if none).
    """
    C = collections.defaultdict(lambda: np.zeros(4))
    for x in samples:
        if x.get("kind", "fc") != "fc" or x["score"] not in ("T", "E"):
            continue
        direct = is_direct_answer(x["ans"], t_animal, e_animal)
        C[x.get("student", x.get("arm"))][(0 if direct else 2) + (0 if x["score"] == "T" else 1)] += 1
    return dict(C)


def _share(c, j):
    return c[j] / max(c[j] + c[j + 1], 1)


def pairs_from_names(names):
    """seed -> {"T": name, "E": name} from student names ending in '<T|E>_<seed>' (optionally '_u80')."""
    P = collections.defaultdict(dict)
    for n in names:
        m = re.search(r"([TE])_(\d+)(_u80)?$", n)
        if m:
            P[int(m.group(2))][m.group(1)] = n
    return {s: v for s, v in P.items() if set(v) == {"T", "E"}}


def sampled_cell(counts, pairs):
    """Direct-answer choice effect and echo (order-of-mention) effect for one cell."""
    D, O = [], []
    for seed in sorted(pairs, key=int):
        a, b = counts.get(pairs[seed]["T"]), counts.get(pairs[seed]["E"])
        if a is None or b is None:
            continue
        D.append(seed_pair_diff(_share(a, 0), _share(b, 0)))
        O.append(seed_pair_diff(_share(a, 2), _share(b, 2)))
    base = counts.get("base")
    return {"direct": summarize(D), "echo": summarize(O),
            "base_direct_share": float((base[0] + base[1]) / base.sum()) if base is not None else None,
            "direct_values": D, "echo_values": O}


def bh_sampled_cells(results):
    """BH across sampled cells on the two-sided p of the direct-answer effect. results: {cell: sampled_cell(...)}."""
    cells = sorted(results)
    q = bh([results[c]["direct"]["p_two_sided"] for c in cells])
    return {c: float(qq) for c, qq in zip(cells, q)}


def logodds_decomposition(student_records, pairs, S=None):
    """Transfer -> template log-odds -> predicted choice -> sampled choice, per seed pair.

    From generation.py student records: lo_mean (first-token T-vs-E log-odds, mean over
    kept templates and orders), lo_dpred (predicted P(T) - P(E)), and the sampled
    rate_T - rate_E. ``S``: optional {seed: S80} for the same seed pairs.
    """
    A = {r.get("student", r.get("arm")): r for r in student_records}
    rows = []
    for seed in sorted(pairs, key=int):
        aT, aE = A.get(pairs[seed]["T"]), A.get(pairs[seed]["E"])
        if not aT or not aE or aT.get("lo_mean") is None:
            continue
        d = lambda a: a["fc"]["rate_T"] - a["fc"]["rate_E"]
        rows.append([S[seed] if S else np.nan, seed_pair_diff(aT["lo_mean"], aE["lo_mean"]),
                     seed_pair_diff(aT["lo_dpred"], aE["lo_dpred"]), seed_pair_diff(d(aT), d(aE))])
    X = np.array(rows)
    c = lambda i, j: float(np.corrcoef(X[:, i], X[:, j])[0, 1])
    out = {"logodds_shift": summarize(X[:, 1]), "predicted_choice": summarize(X[:, 2]),
           "sampled_d": summarize(X[:, 3]), "r_predicted_sampled": c(2, 3), "r_logodds_predicted": c(1, 2)}
    if S:
        out["r_S_logodds"] = c(0, 1)
        out["r_S_sampled"] = c(0, 3)
    return out


def sensitivity_weighted_shift(student_records, pairs):
    """Log-odds shift weighted by the base student's choice sensitivity per template and order:
    s = 0.5 * (1 - tanh(x0 / 2)^2) * exp(x1), x0 = base log-odds, x1 = base log mass."""
    A = {r.get("student", r.get("arm")): r for r in student_records}
    base = A["base"]["lo"]
    keys = sorted(base, key=int)
    xb = np.array([[base[t][o][0] for o in range(2)] for t in keys])
    sb = 0.5 * (1 - np.tanh(xb / 2) ** 2) * np.exp(np.array([[base[t][o][1] for o in range(2)] for t in keys]))
    D = []
    for seed in sorted(pairs, key=int):
        aT, aE = A.get(pairs[seed]["T"]), A.get(pairs[seed]["E"])
        if aT and aE:
            D.append(np.array([[seed_pair_diff(aT["lo"][t][o][0], aE["lo"][t][o][0]) for o in range(2)] for t in keys]))
    m = np.array(D).mean(0)
    hi = sb >= np.median(sb)
    return {"unweighted": float(m.mean()), "sensitivity_weighted": float((m * sb).sum() / sb.sum()),
            "sensitive_half": float(m[hi].mean()), "saturated_half": float(m[~hi].mean())}


# ---------------------------------------------------------------------------
# 3. Name forms
# ---------------------------------------------------------------------------

def name_form_effects(form_records, pairs):
    """Seed-pair effects of the per-form margins (name_forms.py output), their tests and the
    common component (mean movement of both students from the base student)."""
    F = {r.get("student", r.get("arm")): r for r in form_records}
    base = F.get("base")
    eff = {f: [] for f in FORMS}
    com = {f: [] for f in FORMS}
    for seed in sorted(pairs, key=int):
        aT, aE = F.get(pairs[seed]["T"]), F.get(pairs[seed]["E"])
        if not aT or not aE:
            continue
        for f in FORMS:
            eff[f].append(seed_pair_diff(aT[f], aE[f]))
            if base is not None:
                com[f].append(((aT[f] - base[f]) + (aE[f] - base[f])) / 2)
    out = {}
    for f in FORMS:
        out[f] = dict(summarize(eff[f]), values=eff[f])
        if base is not None:
            out[f]["base"] = base[f]
            out[f]["common"] = float(np.mean(com[f]))
    return out


# ---------------------------------------------------------------------------
# 4. Per-cell behaviour entry (Table 1)
# ---------------------------------------------------------------------------

def _fmt_p(p):
    return ("%.3f" % p).rstrip("0") if p >= 0.001 else "%.1g" % p


def behaviour_entry(cell, sampled=None, open_q=None):
    """Text of the Behaviour column for one cell.

    sampled: ``sampled_cell`` result for the twelve sampled cells, else None.
    open_q:  ``open_question_cell`` result over the cell's seed pairs.
    Rules:
      sampled cell  -> direct-answer choice if its one-sided p < 0.05 in either
                       direction; else order of mention if the echo effect is;
                       else 'sampled choice 0 (<effect>)';
      degenerate open-question evaluation, not sampled -> 'answer evaluation
                       degenerate';
      open-question effect one-sided p < 0.05 -> 'open question <effect> (t <t>)';
      transfer on the same seed pairs one-sided p < 0.05 without an open-question
                       effect -> 'answer noise <sd ratio>x';
      otherwise -> 'no answer-level change'.
    """
    if sampled is not None:
        for key, label in (("direct", "direct-answer choice"), ("echo", "order of mention")):
            s = sampled[key]
            p = s["p_plus"] if s["mean"] > 0 else s["p_minus"]
            if p < ALPHA:
                return "%s %+.4f (%d/%d, p %s)" % (label, s["mean"], s["npos"], s["n"], _fmt_p(p))
        return "sampled choice 0 (%+.4f)" % sampled["direct"]["mean"]
    if open_q["degenerate"]:
        return "answer evaluation degenerate"
    if open_q["open_verdict"] != "0":
        return "open question %+.3f (t %.1f)" % (open_q["open"]["mean"], open_q["open"]["t"])
    if open_q["transfer"]["p_plus"] < ALPHA or open_q["transfer"]["p_minus"] < ALPHA:
        return "answer noise %.1fx" % open_q["noise_ratio"]
    return "no answer-level change"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _cmd_open(a):
    """--pairs: JSON list of {"cell", "seed", "T": run prefix, "E": run prefix} (the cell's seed pairs)."""
    P = collections.defaultdict(list)
    for x in json.load(open(a.pairs)):
        T, E = load_evals(x["T"]), load_evals(x["E"])
        if 0 in T and FINAL_STEP in T and 0 in E and FINAL_STEP in E:
            P[x["cell"]].append(seed_pair_answer_evaluations(T, E))
    return {c: open_question_cell(v) for c, v in sorted(P.items())}


def _cmd_sampled(a):
    """--cells: JSON {cell: {"samples": samples.jsonl, "students": students.jsonl}} (generation.py output)."""
    out = {}
    for cell, f in sorted(json.load(open(a.cells)).items()):
        t, e = TRAIT_PAIRS[cell.split("_")[1]]
        counts = direct_echo_counts(read_jsonl(f["samples"]), t, e)
        recs = list(read_jsonl(f["students"]))
        pairs = pairs_from_names([r.get("student", r.get("arm")) for r in recs])
        res = sampled_cell(counts, pairs)
        if any(r.get("lo_mean") is not None for r in recs):
            res["logodds"] = logodds_decomposition(recs, pairs)
            res["sensitivity"] = sensitivity_weighted_shift(recs, pairs)
        out[cell] = res
    q = bh_sampled_cells(out)
    for c in out:
        out[c]["direct"]["q_bh"] = q[c]
    return out


def _cmd_forms(a):
    """--forms: name_forms.py output of one cell."""
    recs = list(read_jsonl(a.forms))
    return name_form_effects(recs, pairs_from_names([r.get("student", r.get("arm")) for r in recs]))


def _cmd_table(a):
    """--open: output of 'open'; --sampled: output of 'sampled'. Writes {cell: entry}."""
    op = json.load(open(a.open))
    sa = json.load(open(a.sampled)) if a.sampled else {}
    return {c: behaviour_entry(c, sa.get(c), op.get(c)) for c in sorted(set(op) | set(sa))}


def main():
    ap = argparse.ArgumentParser(description="Behaviour analysis (answer evaluations, sampled answers, name forms).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("open", help=_cmd_open.__doc__)
    p.add_argument("--pairs", required=True)
    p = sub.add_parser("sampled", help=_cmd_sampled.__doc__)
    p.add_argument("--cells", required=True)
    p = sub.add_parser("forms", help=_cmd_forms.__doc__)
    p.add_argument("--forms", required=True)
    p = sub.add_parser("table", help=_cmd_table.__doc__)
    p.add_argument("--open", required=True)
    p.add_argument("--sampled", default=None)
    for p in sub.choices.values():
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"open": _cmd_open, "sampled": _cmd_sampled, "forms": _cmd_forms, "table": _cmd_table}[a.cmd](a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
