"""Judges that are not the student's own model: scoring, gate and the judge-side GPU computations.

Methods, "Judges that are not the student's own model"; Supplementary Notes
"Judges of other models, stage by stage" and "Cross-family judges in the weights"
(judge-side parts). Everything else about a cell (student, carrier, validity
rule, trait pair, seeds, training recipe, preference evaluation, candidate pool,
k) is that of the main grid; only the judge changes.

Judge conditions
----------------
``cross_1b``         the other family's 1B-class judges: the Llama student is
                     scored by Qwen3-1.7B judges and the Qwen student by
                     Llama-3.2-1B judges. FT route: the other model's main-grid
                     fine-tuned judges for the same two animals (its training
                     instance, seed 0, for training; seeds 0-7 for emission). SP
                     route: the other base model with the same persona wordings.
``same_family_8b``   8B judges of the student's family (``judges.judge_8b``).
``other_family_8b``  8B judges of the other family.

A judge of another family scores with its own tokenizer and its own chat
template (``chat.judge_prompt(..., cross_family=True)``: the persona sentence, if
any, as a system turn, otherwise the user turn only), with the Llama date line
fixed to the pool's date. A same-family 8B judge shares the student's tokenizer
and template. Judges score in fp32.

Gate: for every judge instance the batched score (left padding, batches of 16,
length <= 320) is compared with an independent single-sequence path on a random
subset of candidates (``numpy.random.default_rng(0)``); the largest absolute
difference must be <= 2e-3 (1e-3 for the first-step direction).

Contents
--------
* ``rescore_pool``: a cell's start-policy candidate pool scored by every
  instance of a judge condition (emission, absorption, alignment).
* ``own_association``: a judge model's own association k_X with each candidate,
  i.e. k computed with that model in place of the student (delta 0.04; 8B
  weights perturbed in place).
* ``first_step_direction`` and ``margin_along``: the first-step direction v under
  a judge condition's rewards on the SP-route pool, and the derivative of the
  preference margin along a direction at a fixed-norm step.
* ``rescore_recorded_run``: in-training rescoring of recorded reward calls by
  other judges under the training rule, with the margin gradient's candidate
  responses at the start and end-of-training policies.

Input formats
-------------
Candidate pool (``<pool>.npz`` + ``<pool>.jsonl``): the npz holds ``vi`` (row
indices of the valid candidates in the jsonl), ``qi`` (their prompt indices) and
the training judges' scores ``r_T_k0``, ``r_E_k0`` (``r_<T|E>_k<instance>`` for
further instances); each jsonl row has ``c`` (the candidate completion), ``qi``
and ``valid``.

Recorded reward calls (``<run>.jsonl``, written by the training sample logger):
one row per reward call with ``call`` (index), ``prompts`` (rendered student
prompts, groups of 8 consecutive completions of one prompt), ``completions`` and
``rewards`` (the reward the training judge assigned; ``null`` for a completion
whose reward was not recorded).
"""

import argparse
import gc
import json
import os
import re
import time

import numpy as np
import torch

from .. import carriers as CAR
from ..chat import DEFAULT_FIXDATE, judge_prompt, load_tokenizer, masked, sum_logprob, sum_logprob_single, user_prompt
from ..constants import MODELS, MODELS_8B
from ..evaluation.preference import PreferenceEvaluation
from .sp import persona   # persona(animal, k): SP wording k (0 = training sentence)

CONDITIONS = ("cross_1b", "same_family_8b", "other_family_8b")
GATE_TOL = 2e-3
GROUP = 8                 # completions per prompt in a reward call
INVALID_PENALTY = -400.0  # reward of a group in which no completion is valid
K_DELTA = 0.04            # finite-difference step of k: eps = delta / ||h||^2

def judge_family(student, condition):
    """(family, size) of the judge model under ``condition`` for a student of family ``student``."""
    other = "qwen" if student == "llama" else "llama"
    return {"cross_1b": (other, "1b"), "same_family_8b": (student, "8b"), "other_family_8b": (other, "8b")}[condition]


def judge_base_model(student, condition):
    """Hub name of the base model of the judge (the SP-route judge itself)."""
    fam, size = judge_family(student, condition)
    return MODELS[fam]["name"] if size == "1b" else MODELS_8B[fam]


def needs_own_tokenizer(student, condition):
    """True if the judge is of another family and therefore scores with its own tokenizer and template."""
    return judge_family(student, condition)[0] != student


def ft_instance_paths(seed0_path, n=8):
    """FT judge instances follow the naming ``<name>_s<seed>``; seed 0 is the training instance."""
    p = seed0_path.rstrip("/")
    assert p.endswith("_s0"), p
    return [p[:-3] + "_s%d" % v for v in range(n)]


def load_fp32(path, device="cuda"):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()


# --------------------------------------------------------------------------
# Pools and scoring
# --------------------------------------------------------------------------
def load_pool(npz_path):
    """Candidate pool: dict with the npz arrays, the valid candidates' completions and prompt indices."""
    z = np.load(npz_path, allow_pickle=True)
    rows = [json.loads(l) for l in open(npz_path[:-4] + ".jsonl")]
    vi = z["vi"]
    qis = [int(x) for x in z["qi"]]
    assert len(vi) == len(qis) and all(rows[i]["valid"] and rows[i]["qi"] == q for i, q in zip(vi, qis))
    return {"npz": {k: z[k] for k in z.files}, "vi": vi, "qi": np.array(qis), "completions": [rows[i]["c"] for i in vi]}


class Scorer:
    """Judge scoring of completions for a student's carrier prompts.

    ``student_tok`` renders the student context (an FT judge of the student's
    family scores in that context); ``judge_tok`` is given for a judge of another
    family, which then scores in its own template with its own tokenizer. With a
    system sentence and no ``judge_tok`` the sentence is added as a system turn
    of the student's template (SP judge of the student's family).
    """

    def __init__(self, student_tok, prompts, judge_tok=None, device="cuda"):
        self.tok, self.jtok, self.QS, self.device = student_tok, judge_tok, prompts, device
        self.PCH = {qi: user_prompt(student_tok, prompts[qi]) for qi in range(len(prompts))}
        self.PLEN = {qi: len(student_tok(self.PCH[qi], add_special_tokens=False).input_ids) for qi in self.PCH}
        self._ctx = {}

    def texts(self, qis, resp, system=""):
        if not (self.jtok is not None or system):
            return [self.PCH[q] + c for q, c in zip(qis, resp)], [self.PLEN[q] for q in qis], self.tok
        jt = self.jtok if self.jtok is not None else self.tok
        tt, pp = [], []
        for q, c in zip(qis, resp):
            key = (int(q), system)
            if key not in self._ctx:
                p = judge_prompt(jt, self.QS[int(q)], system, cross_family=self.jtok is not None)
                self._ctx[key] = (p, len(jt(p, add_special_tokens=False).input_ids))
            p, pl = self._ctx[key]
            tt.append(p + c)
            pp.append(pl)
        return tt, pp, jt

    def score(self, model, qis, resp, system=""):
        tt, pp, jt = self.texts(qis, resp, system)
        return sum_logprob(model, jt, tt, pp, device=self.device), tt

    def gate(self, model, scores, texts, resp, subset):
        """Largest |batched - single-sequence| score over ``subset``."""
        jt = self.jtok if self.jtok is not None else self.tok
        one = np.array([sum_logprob_single(model, jt, texts[i], texts[i][:len(texts[i]) - len(resp[i])], device=self.device)
                        for i in subset])
        return float(np.max(np.abs(one - np.asarray(scores)[subset])))


def rescore_pool(pool, scorer, route, t_name, e_name, judges, n_gate=16, tol=GATE_TOL, device="cuda"):
    """Score a pool with every instance of one judge condition.

    ``judges``: SP route -> the judge base model (one path, both judge
    conditions); FT route -> ``{"T": [paths of instances 0..], "E": [...]}``.
    SP instances are the eight wordings plus the judge without persona
    (key ``B``). Returns (scores by key ``<T|E>_k<instance>``, gate by key, gate passed).
    """
    qis, resp = list(pool["qi"]), pool["completions"]
    n = len(resp)
    subset = sorted(np.random.default_rng(0).choice(n, size=min(n_gate, n), replace=False).tolist())
    names = {"T": t_name, "E": e_name}
    if route == "SP":
        jobs = [("%s_k%d" % (p, v), judges, persona(names[p], v)) for v in range(8) for p in "TE"] + [("B", judges, "")]
    else:
        jobs = [("%s_k%d" % (p, v), path, "") for p in "TE" for v, path in enumerate(judges[p])]
    R, G = {}, {}
    cur_path, cur = None, None
    for key, path, system in jobs:
        if path != cur_path:
            cur = None
            gc.collect()
            torch.cuda.empty_cache()
            cur_path, cur = path, load_fp32(path, device)
        r, tt = scorer.score(cur, qis, resp, system)
        R[key] = np.asarray(r, dtype=np.float64)
        G[key] = scorer.gate(cur, R[key], tt, resp, subset)
        print("[rescore] %-5s mean %.3f  batched-vs-single %.2e" % (key, R[key].mean(), G[key]), flush=True)
    return R, G, all(v <= tol for v in G.values())


def save_rescored_pool(pool, scores, out_npz):
    """Pool file of the same format as the main grid with the condition's scores in place of the own judges'."""
    nb = {k: v for k, v in pool["npz"].items() if not k.startswith("r_") and k != "keys"}
    for k, v in scores.items():
        nb["r_" + k] = v
    nb["keys"] = np.array(sorted(scores))
    np.savez(out_npz, **nb)


# --------------------------------------------------------------------------
# Margin gradient h and candidate responses (k, k_X, in-training responses)
# --------------------------------------------------------------------------
def name_margin_gradient(model, pe, bs=24):
    """Gradient h of the name-token preference margin B over the evaluation items of ``pe``.

    B = mean over items of (summed log-likelihood of the T name's tokens) minus
    (the same for the E name), length <= 256, left padding. Gradients are left
    in ``p.grad`` and returned (not copied); returns (params, grads, B).
    """
    tok, dev = pe.tok, pe.device
    na = len(pe.texts)
    wt = torch.tensor([(1.0 if j % 2 == 0 else -1.0) / (na // 2) for j in range(na)], device=dev)
    params = [p for p in model.parameters()]
    for p in params:
        p.requires_grad_(True)
    b = 0.0
    for i in range(0, na, bs):
        e = tok(pe.texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False)
        e = {k: v.to(dev) for k, v in e.items()}
        lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
        tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        npad = (e["attention_mask"].shape[1] - e["attention_mask"].sum(1)).tolist()
        mn = torch.zeros_like(tk)
        for r in range(tk.shape[0]):
            c = max(int(npad[r]) + pe.prompt_lens[i + r] - 1, 0)
            for j, x in enumerate(pe.labels[i + r]):
                if x in (1, 2):
                    mn[r, c + j] = 1
        part = ((tk * mn).sum(1) * wt[i:i + bs]).sum()
        part.backward()
        b += float(part)
    grads = [p.grad.detach() for p in params]
    for p in params:
        p.grad = None
        p.requires_grad_(False)
    return params, grads, b


def candidate_responses(model, params, grads, eps, tok, texts, plens, inplace=False, device="cuda"):
    """Central difference d/de log pi(c; theta + e h) at step ``eps`` for every text.

    ``inplace``: perturb the weights in place (+eps, -2eps, +eps), used for 8B
    models; otherwise from a copy of theta.
    """
    out = []
    if inplace:
        for alpha in (eps, -2 * eps):
            with torch.no_grad():
                for p, h in zip(params, grads):
                    p.add_(h, alpha=alpha)
            out.append(sum_logprob(model, tok, texts, plens, device=device))
        with torch.no_grad():
            for p, h in zip(params, grads):
                p.add_(h, alpha=eps)
    else:
        th0 = [p.detach().clone() for p in params]
        for sg in (1, -1):
            with torch.no_grad():
                for p, a, h in zip(params, th0, grads):
                    p.copy_(a + sg * eps * h)
            out.append(sum_logprob(model, tok, texts, plens, device=device))
        with torch.no_grad():
            for p, a in zip(params, th0):
                p.copy_(a)
        del th0
    return (out[0] - out[1]) / (2 * eps)


def own_association(model_path, pool, carrier, t_name, e_name, item_seed, item_offset, fixdate=None, delta=K_DELTA,
                    inplace=None, grad_batch=None, device="cuda"):
    """k_X: the association of judge model X with each pool candidate, X in place of the student.

    X reads the student's candidates in its own chat template (``fixdate`` for
    the Llama date line); h_X is the gradient of X's own name-token preference
    margin (same items as the cell), eps = delta / ||h_X||^2. 8B models are
    perturbed in place with gradients accumulated in batches of 2; 1B-class
    models from a copy, in batches of 24.
    """
    tok = load_tokenizer(model_path, fixdate)
    QS, _, _ = CAR.build_prompts(carrier)
    X = load_fp32(model_path, device)
    big = sum(p.numel() for p in X.parameters()) > 4e9
    inplace = big if inplace is None else inplace
    grad_batch = (2 if big else 24) if grad_batch is None else grad_batch
    pe = PreferenceEvaluation(tok, t_name, e_name, item_seed, item_offset, prompt_fn=lambda u: user_prompt(tok, u), device=device)
    params, grads, b = name_margin_gradient(X, pe, bs=grad_batch)
    hn2 = float(sum((h.double() ** 2).sum() for h in grads))
    eps = delta / hn2
    qis = [int(q) for q in pool["qi"]]
    pch = {q: user_prompt(tok, QS[q]) for q in set(qis)}
    texts = [pch[q] + c for q, c in zip(qis, pool["completions"])]
    plens = [len(tok(pch[q], add_special_tokens=False).input_ids) for q in qis]
    kx = candidate_responses(X, params, grads, eps, tok, texts, plens, inplace=inplace, device=device)
    return kx, {"B_X": b, "h_norm": hn2 ** 0.5, "eps": eps}


# --------------------------------------------------------------------------
# First-step direction under another judge, and the margin along a direction
# --------------------------------------------------------------------------
def advantage_difference(r_t, r_e, qi):
    """w = (a_T - a_E) / 2 with a the within-prompt standardized score (ddof 1, + 1e-4); 0 for singleton prompts."""
    w = np.zeros(len(r_t))
    for g in np.unique(qi):
        ix = np.where(qi == g)[0]
        if len(ix) < 2:
            continue
        a = lambda r: (r[ix] - r[ix].mean()) / (r[ix].std(ddof=1) + 1e-4)
        w[ix] = (a(r_t) - a(r_e)) / 2
    return w


def first_step_direction(student, tok, texts, plens, w, bs=8, device="cuda"):
    """v = grad_theta of sum_c w(c) log pi(c) / #(w != 0), completions of length <= 320."""
    params = [p for p in student.parameters()]
    for p in params:
        p.requires_grad_(True)
    student.zero_grad(set_to_none=True)
    nn = float((w != 0).sum())
    for i in range(0, len(texts), bs):
        ww = torch.tensor(w[i:i + bs], dtype=torch.float32, device=device)
        if float(ww.abs().sum()) == 0:
            continue
        e = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=320, add_special_tokens=False)
        e = {k: v.to(device) for k, v in e.items()}
        lg = torch.log_softmax(student(**e).logits[:, :-1].float(), -1)
        tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        lp = (tk * masked(e, plens[i:i + bs])).sum(1)
        ((ww * lp).sum() / nn).backward()
    v = [p.grad.detach().clone() for p in params]
    student.zero_grad(set_to_none=True)
    for p in params:
        p.requires_grad_(False)
    return v


def margin_along(student, pe, directions, step_norm, frac=0.2):
    """dB/de along each direction at a fixed-norm step.

    Each direction v is scaled to norm ``frac * step_norm`` (``step_norm`` = the
    norm of the own-model SP direction, so that all directions are compared at
    the same step) and the per-token preference margin B_AN is read at
    theta +/- that step; returns (B+ - B-) / (2 frac) per direction.
    """
    params = [p for p in student.parameters()]
    th0 = [p.detach().clone() for p in params]
    out = {}
    for name, v in directions.items():
        vn = float(np.sqrt(sum(float((x.double() ** 2).sum()) for x in v)))
        vals = []
        for sg in (1, -1):
            with torch.no_grad():
                for p, a, x in zip(params, th0, v):
                    p.copy_(a + sg * frac * step_norm / vn * x)
                vals.append(float(pe.evaluate(student)["B_AN"]))
        out[name] = (vals[0] - vals[1]) / (2 * frac)
    with torch.no_grad():
        for p, a in zip(params, th0):
            p.copy_(a)
    return out


# --------------------------------------------------------------------------
# In-training rescoring of recorded reward calls
# --------------------------------------------------------------------------
def select_calls(rows, n=10):
    """``n`` evenly spaced reward calls of a run (rows sorted by call index)."""
    rows = sorted(rows, key=lambda r: r["call"])
    pick = sorted(set(int(round(x)) for x in np.linspace(0, len(rows) - 1, min(n, len(rows)))))
    return [rows[i] for i in pick]


def training_rule_scores(score_fn, prompts, completions, p2q, valid):
    """Rewards under the training rule: the first line is scored; an invalid first line gets the
    minimum valid reward of its group of 8, or -400 if no completion of the group is valid."""
    fl = [CAR.first_line(str(c)) for c in completions]
    qq = [p2q[p] for p in prompts]
    ok = [bool(valid(c, int(q))) for c, q in zip(fl, qq)]
    out = np.full(len(fl), np.nan)
    ix = [i for i in range(len(fl)) if ok[i]]
    if ix:
        out[ix] = np.asarray(score_fn([qq[i] for i in ix], [fl[i] for i in ix]), float)
    for g0 in range(0, len(fl), GROUP):
        v = out[g0:g0 + GROUP]
        m = np.isfinite(v)
        v[~m] = v[m].min() if m.any() else INVALID_PENALTY
        out[g0:g0 + GROUP] = v
    return out


def group_standardize(r, prompts=None):
    """Advantage within consecutive groups of 8; groups with no reward variance (sd <= 1e-6) carry no
    signal and are flagged False."""
    r = np.asarray(r, float)
    a = np.zeros(len(r))
    m = np.zeros(len(r), bool)
    assert len(r) % GROUP == 0
    for g0 in range(0, len(r), GROUP):
        if prompts is not None:
            assert len(set(prompts[g0:g0 + GROUP])) == 1, "a group mixes prompts"
        sd = r[g0:g0 + GROUP].std(ddof=1)
        if sd > 1e-6:
            a[g0:g0 + GROUP] = (r[g0:g0 + GROUP] - r[g0:g0 + GROUP].mean()) / (sd + 1e-4)
            m[g0:g0 + GROUP] = True
    return a, m


def center_groups(x):
    x = np.array(x, float)
    for g0 in range(0, len(x), GROUP):
        x[g0:g0 + GROUP] -= x[g0:g0 + GROUP].mean()
    return x


def rescore_recorded_run(rows, judge_condition, judges, student_start, student_end=None, run_judge=None, eps_multiplier=1.0,
                         inplace_responses=False, device="cuda"):
    """Rescore the selected reward calls of one run and take the margin gradient's candidate responses.

    ``rows``: selected calls (``select_calls``). ``judge_condition``: ``"T"`` or ``"E"`` (the judge that trained the run).
    ``judges``: ``{name: dict(path, scorer, system)}`` for every judge that
    rescores the completions under the training rule, e.g. ``own_T``, ``own_E``
    (the student's own judges, training instance), ``own1_T``, ``own1_E`` (a
    second, independent instance: wording 1 for SP, seed 1 for FT), ``x_T``,
    ``x_E`` (the cross-family judges). ``run_judge``: name of the judge that
    trained the run; its rescoring must reproduce the recorded rewards (gate).
    ``student_start`` / ``student_end``: dict(path, tok, pe, carrier_prompts) of
    the start and end-of-training policies; candidate responses are central
    differences along h = grad B at step 0.04 / ||h||^2 (times
    ``eps_multiplier`` for the step check), centred within groups. The
    responses were taken from a copy of the weights for cross-family runs and
    with in-place perturbation for repeated own-model runs
    (``inplace_responses``).

    Returns per-call arrays: ``AX`` (recorded reward, standardized) with mask
    ``mX``; for every pair ``<p>_T``/``<p>_E`` of rescoring judges the
    between-judge difference ``D<p> = (A_T - A_E) / 2`` with mask (either group
    carries signal); the advantage of the run's own judge condition ``A<p>``; ``kh0`` (start policy) and
    ``kh1`` (end policy); and the gate.
    """
    any_scorer = next(iter(judges.values()))["scorer"]
    p2q = {}
    for q, p in any_scorer.PCH.items():
        p2q.setdefault(p, q)
    car = student_start["carrier"]
    valid = CAR.make_validator(car, student_start["names"][0], student_start["names"][1])
    S = {}
    gate = 0.0
    for name, j in judges.items():
        model = load_fp32(j["path"], device)
        for r in rows:
            S[(name, r["call"])] = training_rule_scores(lambda q, c: j["scorer"].score(model, q, c, j["system"])[0],
                                                        r["prompts"], r["completions"], p2q, valid)
            if name == run_judge:
                ok = np.array([v is not None for v in r["rewards"]])
                rx = np.array([np.nan if v is None else v for v in r["rewards"]], float)
                gate = max(gate, float(np.nanmax(np.abs(S[(name, r["call"])][ok] - rx[ok]))))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = {"calls": np.array([r["call"] for r in rows]), "judge_condition": np.array(1 if judge_condition == "T" else -1)}
    pairs = sorted(set(n[:-2] for n in judges if n.endswith("_T") and n[:-2] + "_E" in judges))
    for r in rows:
        c = r["call"]
        ok = np.array([v is not None for v in r["rewards"]])
        rx = np.array([np.nan if v is None else v for v in r["rewards"]], float)
        out["AX_%d" % c], out["mX_%d" % c] = group_standardize(np.where(ok, rx, np.nanmean(rx)), r["prompts"])
        for p in pairs:
            at, mt = group_standardize(S[(p + "_T", c)])
            ae, me = group_standardize(S[(p + "_E", c)])
            out["D%s_%d" % (p, c)] = (at - ae) / 2
            out["mD%s_%d" % (p, c)] = mt | me
            out["A%s_%d" % (p, c)], out["mA%s_%d" % (p, c)] = (at, mt) if judge_condition == "T" else (ae, me)
    for tag, st in (("kh0", student_start), ("kh1", student_end)):
        if st is None:
            continue
        model = load_fp32(st["path"], device)
        params, grads, b = name_margin_gradient(model, st["pe"])
        hn2 = float(sum((h.double() ** 2).sum() for h in grads))
        eps = K_DELTA / hn2 * eps_multiplier
        tok = st["tok"]
        pch = {q: user_prompt(tok, st["carrier_prompts"][q]) for q in range(len(st["carrier_prompts"]))}
        for r in rows:
            qis = [p2q[p] for p in r["prompts"]]
            texts = [pch[q] + str(x) for q, x in zip(qis, r["completions"])]
            plens = [len(tok(pch[q], add_special_tokens=False).input_ids) for q in qis]
            out["%s_%d" % (tag, r["call"])] = center_groups(candidate_responses(model, params, grads, eps, tok, texts, plens,
                                                                               inplace=inplace_responses, device=device))
        out[tag + "_B"] = np.array(b)
        del model, grads
        gc.collect()
        torch.cuda.empty_cache()
    out["gate"] = np.array(gate)
    return out


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def _student_tok_and_prompts(a):
    tok = load_tokenizer(a.student_model, a.fixdate)
    QS, _, _ = CAR.build_prompts(a.carrier)
    return tok, QS


def cmd_rescore_pool(a):
    t0 = time.time()
    pool = load_pool(a.pool)
    tok, QS = _student_tok_and_prompts(a)
    jtok = load_tokenizer(a.judge_tokenizer, a.fixdate) if a.judge_tokenizer else None
    sc = Scorer(tok, QS, jtok)
    if a.route == "SP":
        judges = a.judge_model
    else:
        judges = {p: [x for x in ft_instance_paths(s0)[:a.instances] if os.path.isdir(x)] for p, s0 in (("T", a.judge_T), ("E", a.judge_E))}
        assert judges["T"][0].endswith("_s0") and judges["E"][0].endswith("_s0")
    R, G, ok = rescore_pool(pool, sc, a.route, a.t_name, a.e_name, judges, tol=a.tol)
    save_rescored_pool(pool, R, a.out)
    json.dump({"pool": a.pool, "route": a.route, "judges": judges, "gate": G, "ok": ok, "sec": round(time.time() - t0)},
              open(a.out[:-4] + ".gates.json", "w"), indent=1)
    raise SystemExit(0 if ok else 7)


def cmd_own_association(a):
    pool = load_pool(a.pool)
    kx, info = own_association(a.model, pool, a.carrier, a.t_name, a.e_name, a.item_seed, a.item_offset, a.fixdate,
                               inplace=a.inplace)
    np.savez(a.out, vi=pool["vi"], kX=kx)
    json.dump(info, open(a.out[:-4] + ".json", "w"), indent=1)


def cmd_first_step(a):
    """v under one judge condition on the SP-route pool (both routes' judges in turn)."""
    from safetensors.torch import save_file
    pool = load_pool(a.pool)
    tok, QS = _student_tok_and_prompts(a)
    jtok = load_tokenizer(a.judge_tokenizer, a.judge_fixdate or DEFAULT_FIXDATE) if a.judge_tokenizer else None
    sc = Scorer(tok, QS, jtok)
    qis, resp = list(pool["qi"]), pool["completions"]
    subset = sorted(np.random.default_rng(0).choice(len(resp), size=min(24, len(resp)), replace=False).tolist())
    scores, gate = {}, {}
    for p, path, system in (("T", a.judge_T, a.system_T or ""), ("E", a.judge_E, a.system_E or "")):
        J = load_fp32(path)
        scores[p], tt = sc.score(J, qis, resp, system)
        gate[p] = sc.gate(J, scores[p], tt, resp, subset)
        del J
        torch.cuda.empty_cache()
    w = advantage_difference(np.asarray(scores["T"], float), np.asarray(scores["E"], float), pool["qi"])
    student = load_fp32(a.student_model)
    names = [n for n, _ in student.named_parameters()]
    texts = [sc.PCH[q] + c for q, c in zip(qis, resp)]
    v = first_step_direction(student, tok, texts, [sc.PLEN[q] for q in qis], w)
    save_file({n: x.detach().to(torch.bfloat16).cpu().contiguous() for n, x in zip(names, v)}, a.out)
    ok = all(x <= 1e-3 for x in gate.values())
    json.dump({"gate": gate, "ok": ok, "sd_w": float(w.std()), "corr_TE": float(np.corrcoef(scores["T"], scores["E"])[0, 1])},
              open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    raise SystemExit(0 if ok else 7)


def cmd_margin_along(a):
    from safetensors.torch import load_file
    tok = load_tokenizer(a.student_model, a.fixdate)
    student = load_fp32(a.student_model)
    names = [n for n, _ in student.named_parameters()]
    pe = PreferenceEvaluation(tok, a.t_name, a.e_name, a.item_seed, a.item_offset, prompt_fn=lambda u: user_prompt(tok, u))
    dirs = {}
    for spec in a.direction:
        key, path = spec.split("=", 1)
        d = load_file(path)
        dirs[key] = [d[n].to("cuda").float() for n in names]
    ref = load_file(a.step_norm_from)
    step = float(np.sqrt(sum(float((ref[n].double() ** 2).sum()) for n in names)))
    json.dump(margin_along(student, pe, dirs, step), open(a.out, "w"), indent=1)


def cmd_intraining(a):
    rows = [json.loads(l) for l in open(a.dump)]
    if a.calls_from:
        keep = set(int(c) for c in np.load(a.calls_from)["calls"])
        rows = sorted([r for r in rows if r["call"] in keep], key=lambda r: r["call"])
    else:
        rows = select_calls(rows, a.n_calls)
    tok, QS = _student_tok_and_prompts(a)
    judges = {}
    for spec in a.judge:   # name=path[,system=...][,cross=tokenizer_path][,fixdate=...]
        name, rest = spec.split("=", 1)
        parts = dict(x.split(":", 1) for x in rest.split(",")[1:]) if "," in rest else {}
        path = rest.split(",")[0]
        jtok = load_tokenizer(parts["cross"], parts.get("fixdate") or DEFAULT_FIXDATE) if parts.get("cross") else None
        judges[name] = {"path": path, "scorer": Scorer(tok, QS, jtok), "system": parts.get("system", "")}
    names = (a.t_name, a.e_name)
    mk = lambda path: None if not path else {"path": path, "tok": tok, "carrier": a.carrier, "names": names, "carrier_prompts": QS,
                                             "pe": PreferenceEvaluation(tok, a.t_name, a.e_name, a.item_seed, a.item_offset,
                                                                        prompt_fn=lambda u: user_prompt(tok, u))}
    out = rescore_recorded_run(rows, a.judge_condition, judges, mk(a.student_start), mk(a.student_end), a.run_judge, a.eps_multiplier,
                               a.inplace_responses)
    np.savez_compressed(a.out, **out)
    ok = float(out["gate"]) <= GATE_TOL
    print("[intraining] %s: %d calls, gate %.2e %s" % (a.dump, len(rows), float(out["gate"]), "pass" if ok else "FAIL"), flush=True)
    raise SystemExit(0 if ok else 7)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Judge-side computations for judges that are not the student's own model.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, student=True):
        if student:
            p.add_argument("--student-model", required=True, help="student model (tokenizer and chat template)")
        p.add_argument("--carrier", required=True, choices=CAR.CARRIERS)
        p.add_argument("--t-name", required=True)
        p.add_argument("--e-name", required=True)
        p.add_argument("--fixdate", default=None, help="Llama date line of the cell (default %s)" % DEFAULT_FIXDATE)

    p = sub.add_parser("rescore-pool", help="score a candidate pool with every instance of a judge condition")
    common(p)
    p.add_argument("--pool", required=True)
    p.add_argument("--route", choices=("FT", "SP"), required=True)
    p.add_argument("--judge-model", help="SP route: judge base model")
    p.add_argument("--judge-T", help="FT route: T judge, training instance (<name>_s0)")
    p.add_argument("--judge-E", help="FT route: E judge, training instance (<name>_s0)")
    p.add_argument("--judge-tokenizer", default="", help="judge of another family: its tokenizer/template")
    p.add_argument("--instances", type=int, default=8)
    p.add_argument("--tol", type=float, default=GATE_TOL)
    p.add_argument("--out", required=True, help="output .npz (pool format)")
    p.set_defaults(fn=cmd_rescore_pool)

    p = sub.add_parser("own-association", help="k_X of a judge model on a pool")
    common(p, student=False)
    p.add_argument("--model", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--item-seed", type=int, default=0)
    p.add_argument("--item-offset", type=int, required=True, help="400 (FT cells) or 624 (SP cells)")
    p.add_argument("--inplace", action="store_true", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_own_association)

    p = sub.add_parser("first-step", help="first-step direction v under a judge condition on the SP-route pool")
    common(p)
    p.add_argument("--pool", required=True)
    p.add_argument("--judge-T", required=True)
    p.add_argument("--judge-E", required=True)
    p.add_argument("--system-T", default="")
    p.add_argument("--system-E", default="")
    p.add_argument("--judge-tokenizer", default="")
    p.add_argument("--judge-fixdate", default=None)
    p.add_argument("--out", required=True, help="output .safetensors (bf16)")
    p.set_defaults(fn=cmd_first_step)

    p = sub.add_parser("margin-along", help="preference-margin derivative along directions at a fixed-norm step")
    common(p)
    p.add_argument("--item-seed", type=int, default=0)
    p.add_argument("--item-offset", type=int, required=True)
    p.add_argument("--direction", action="append", required=True, help="name=path.safetensors")
    p.add_argument("--step-norm-from", required=True, help="own-model SP direction (its norm sets the step)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_margin_along)

    p = sub.add_parser("intraining", help="rescore recorded reward calls of one run")
    common(p)
    p.add_argument("--dump", required=True)
    p.add_argument("--judge-condition", choices=("T", "E"), required=True)
    p.add_argument("--n-calls", type=int, default=10)
    p.add_argument("--calls-from", default="", help="reuse the call selection of an earlier output")
    p.add_argument("--judge", action="append", required=True,
                   help="name=path[,system:<sentence>][,cross:<tokenizer>][,fixdate:<date>]; pairs named <p>_T/<p>_E")
    p.add_argument("--run-judge", required=True, help="name of the judge that trained the run (gate)")
    p.add_argument("--student-start", required=True)
    p.add_argument("--student-end", default="")
    p.add_argument("--item-seed", type=int, default=0)
    p.add_argument("--item-offset", type=int, required=True)
    p.add_argument("--eps-multiplier", type=float, default=1.0, help="4 for the step check")
    p.add_argument("--inplace-responses", action="store_true", help="perturb weights in place (repeated own-model runs)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_intraining)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
