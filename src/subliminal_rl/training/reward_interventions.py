"""Reward interventions: training conditions that change only the reward.

Methods, "Reward interventions"; Extended Data Table "Reward intervention";
Supplementary Note on transfer (rank values, per-token mean, fixed reference,
alpha ladder). Judges, prompts, sampling, the validity rule and the evaluations
are those of the unmodified run; only the list of rewards handed back to
``trl.GRPOTrainer`` changes.

Interventions (``mode``):

``rank``
    A group's eight rewards are replaced by the fixed values -3.5 ... +3.5
    (value ``rank - (n - 1) / 2`` after a stable ascending sort), so the order
    is kept exactly and the spacing discarded. The ranking acts on the rewards
    returned by the runner, i.e. after invalid completions received the group
    minimum; tied fills are ordered by position (stable sort).
``perm`` / ``rev`` (controls)
    The group's rewards are shuffled (``random.Random((seed * 1000003 + call *
    997 + group) & 0x7FFFFFFF)``) or assigned in reversed rank order. Both keep
    the group's reward multiset; reported as controls only (they remove format
    learning and almost all policy movement).
``mean``
    The per-token mean log-likelihood replaces the sum of equation (r): each
    valid completion's judge score is divided by the number of judge tokens of
    its first line (``len(judge_tokenizer(judge_text)) - judge_prompt_len``,
    at least 1). Invalid completions are refilled with the group minimum of the
    new values (``-400`` if none is valid, or always with ``--invalid fixed``).
``alpha``
    Every training completion is scored by both judges of the pair; with
    ``r_own`` the run's reward and ``r_other`` the other judge's score,
    ``c = (r_own + r_other) / 2``, ``s = (r_own - r_other) / 2`` and the reward is
    ``c + alpha * s``. The other SP judge is the same base model with the other
    animal's persona sentence (``other_system``); the other FT judge is the
    other fine-tuned model (``other_judge``, loaded once in fp32). Invalid
    completions are refilled from the new values as for ``mean``. Gates at the
    first call: ``max |s| > 1e-4`` (the other judge is really applied) and, for
    ``alpha = 1``, ``max |r - r_own| < 1e-6`` (the unmodified reward is
    reproduced exactly).
``fixed_reference``
    The reward is the judge score minus the same first line's log-likelihood
    under the student's base model without the persona sentence (the fixed
    start policy): ``r = r_own - r_ref`` with ``r_ref`` scored by the same base
    weights in the student's own context (``other_system = ""``). Used on the SP
    route only, where the judge weights are the student's base weights. Gates
    at the first call: ``max |(r_own - r_ref) / 2| > 1e-4`` and ``max |r - r_own|
    > 1e-2``.

Paper runs: rank and alpha on eight seed pairs of Llama · FT · syllable ·
sloth-lemur and Llama · SP · number · sloth-lemur; per-token mean and fixed
reference on Llama · SP · number · sloth-lemur; perm and rev on both cells
(controls).

How it plugs in
---------------
The training runner (``training/grpo_run.py``) calls a reward hook named by the
environment variable ``SUBLIMINAL_REWARD_HOOK`` with its local namespace::

    SUBLIMINAL_REWARD_HOOK=subliminal_rl.training.reward_interventions:wrap

``wrap(first_line_reward, context=locals())`` reads the intervention from the
environment:

=============================  ===============================================
``REWARD_INTERVENTION``        ``rank`` | ``perm`` | ``rev`` | ``mean`` |
                               ``alpha`` | ``fixed_reference`` (unset: no-op)
``REWARD_ALPHA``               alpha for ``alpha`` (0, 1 or 2 in the paper)
``REWARD_OTHER_SYSTEM``        other SP judge's persona sentence (``alpha``);
                               for ``fixed_reference`` it is ``""`` (default)
``REWARD_OTHER_JUDGE``         path of the other FT judge (``alpha``, FT route)
``REWARD_GROUP``               group size for ``rank``/``perm``/``rev`` (8)
``REWARD_DUMP``                optional JSONL path: one line per call with the
                               runner's rewards and the rewards used (and the
                               other judge's scores for ``alpha`` /
                               ``fixed_reference``)
=============================  ===============================================

The paper runs attached the same wrapper by patching
``trl.GRPOTrainer.__init__`` from a ``sitecustomize`` module (the reward
functions passed to the trainer were wrapped); ``patch_trainer`` reproduces
that mechanism for a runner that has no reward hook. Both give the same
rewards.

Names taken from the runner's namespace (``context``): ``a`` (parsed
arguments: ``seed``, ``invalid``, ``invalid_penalty``, ``judge_system``), ``G``,
``P2Q``, ``PCH``, ``PLEN``, ``QS``, ``valid_c``, ``score``, ``JV``, ``tok``,
``jt``, ``cross``, ``dev``.

Dump format (``REWARD_DUMP``), one JSON object per reward call::

    {"call": int, "mode": str, "alpha": float | null,
     "rewards_orig": [float], "rewards_other": [float | null] (alpha, fixed_reference),
     "ntok": {index: int} (mean), "rewards_arm": [float],
     "prompts": [str], "completions": [str]}
"""

import functools
import importlib
import json
import os
import random

import numpy as np

from .. import carriers as CAR
from ..chat import judge_prompt, sum_logprob

MODES = ("rank", "perm", "rev", "mean", "alpha", "fixed_reference")


# ---------------------------------------------------------------------------
# Within-group rearrangements (rank values and the perm / rev controls)
# ---------------------------------------------------------------------------
def arrange_group(values, mode, seed=0, call=0, group=0):
    """Rearranged copy of one group's rewards (``None`` entries are left in place)."""
    ix = [i for i, x in enumerate(values) if x is not None]
    if len(ix) < 2:
        return list(values)
    vals = [float(values[i]) for i in ix]
    out = list(values)
    if mode == "perm":
        r = random.Random((seed * 1000003 + call * 997 + group) & 0x7FFFFFFF)
        p = list(vals)
        r.shuffle(p)
        for i, x in zip(ix, p):
            out[i] = x
    elif mode == "rev":
        order = sorted(range(len(ix)), key=lambda j: vals[j])          # ascending
        srt = sorted(vals)
        for rank, j in enumerate(order):
            out[ix[j]] = srt[len(srt) - 1 - rank]
    elif mode == "rank":
        order = sorted(range(len(ix)), key=lambda j: vals[j])
        m = (len(ix) - 1) / 2.0
        for rank, j in enumerate(order):
            out[ix[j]] = float(rank - m)                                 # -3.5 ... +3.5 for 8 values
    else:
        raise ValueError("unknown rearrangement: %r" % (mode,))
    return out


# ---------------------------------------------------------------------------
# Scoring with another system sentence or another judge
# ---------------------------------------------------------------------------
class OtherScorer:
    """Score first lines exactly as the runner's ``score`` does, with another judge context or judge model.

    ``system`` replaces the runner's judge system sentence (``""``: no persona,
    i.e. the student's own context for a same-family judge, as the runner does for
    FT judges). ``model`` replaces the judge weights (default: the runner's judge).
    """

    def __init__(self, ctx, system=None, model=None):
        self.ctx = ctx
        self.system = ctx["a"].judge_system if system is None else system
        self.model = ctx["JV"] if model is None else model
        self.cache = {}

    def __call__(self, qis, resp):
        c = self.ctx
        cross = c["cross"]
        if self.system or cross:
            jt = c["jt"]
            tt, pp = [], []
            for qi, line in zip(qis, resp):
                if qi not in self.cache:
                    p = judge_prompt(jt, c["QS"][qi], self.system, cross_family=cross)
                    self.cache[qi] = (p, len(jt(p, add_special_tokens=False).input_ids))
                p, pl = self.cache[qi]
                tt.append(p + line)
                pp.append(pl)
            return np.asarray(sum_logprob(self.model, jt, tt, pp, device=c["dev"]), dtype=float)
        tt = [c["PCH"][q] + line for q, line in zip(qis, resp)]
        return np.asarray(sum_logprob(self.model, c["tok"], tt, [c["PLEN"][q] for q in qis], device=c["dev"]), dtype=float)


def load_judge_fp32(path, device):
    """Load another judge in fp32 with gradients off (as in the paper runs)."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    assert all(p.dtype == torch.float32 for p in m.parameters())
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _valid_lines(ctx, prompts, completions):
    qis, resp, vi = [], [], []
    for i, (p, s_) in enumerate(zip(prompts, completions)):
        qi = ctx["P2Q"][p]
        c = CAR.first_line(str(s_))
        if ctx["valid_c"](c, int(qi)):
            qis.append(qi)
            resp.append(c)
            vi.append(i)
    return qis, resp, vi


def _refill(ctx, new, valid_set, n):
    """Refill invalid positions with the group minimum of the new values (runner's rule)."""
    G = ctx["G"]
    a = ctx["a"]
    for g0 in range(0, n, G):
        idx = [i for i in range(g0, g0 + G) if i in valid_set]
        fill = a.invalid_penalty if (a.invalid == "fixed" or not idx) else min(new[i] for i in idx)
        for i in range(g0, g0 + G):
            if i not in valid_set:
                new[i] = float(fill)
    return new


def _dump(path, row):
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:   # recording must never interrupt training
        print("[reward_interventions] dump failed: %r" % (e,), flush=True)


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------
def wrap_reward(reward_fn, ctx, mode, alpha=None, other_system=None, other_judge=None, group=8, dump=None):
    """Return ``reward_fn`` wrapped by intervention ``mode`` (see the module docstring)."""
    if mode not in MODES:
        raise ValueError("unknown reward intervention %r" % (mode,))
    counter = {"n": 0}
    seed = int(ctx["a"].seed)
    state = {}

    if mode in ("rank", "perm", "rev"):
        ng = ctx.get("G")
        assert ng is None or int(ng) == group, ("group size differs from the trainer's num_generations", ng, group)

        @functools.wraps(reward_fn)
        def wrapped(prompts, completions, **kw):
            out = list(reward_fn(prompts, completions, **kw))
            new = list(out)
            c = counter["n"]
            for g0 in range(0, len(out), group):
                new[g0:g0 + group] = arrange_group(out[g0:g0 + group], mode, seed, c, g0 // group)
            _dump(dump, {"call": c, "mode": mode, "alpha": None,
                         "rewards_orig": [None if x is None else float(x) for x in out],
                         "rewards_arm": [None if x is None else float(x) for x in new],
                         "prompts": list(prompts), "completions": list(completions)})
            counter["n"] += 1
            return new
        return wrapped

    if mode == "mean":
        jt = ctx["jt"]

        @functools.wraps(reward_fn)
        def wrapped(prompts, completions, **kw):
            base = list(reward_fn(prompts, completions, **kw))
            qis, resp, vi = _valid_lines(ctx, prompts, completions)
            new = list(base)
            nt = {}
            if resp:
                s, tj, pj = ctx["score"](ctx["JV"], qis, resp)
                d = max(abs(float(x) - base[i]) for x, i in zip(s, vi))
                for x, i, t_, p_ in zip(s, vi, tj, pj):
                    n = max(len(jt(t_, add_special_tokens=False).input_ids) - int(p_), 1)
                    nt[i] = n
                    new[i] = float(x) / n
                if counter["n"] == 0:
                    print("[reward_interventions] mean: rescored vs runner reward max|diff| %.2e, median tokens %d"
                          % (d, sorted(nt.values())[len(nt) // 2]), flush=True)
                    assert d < 1e-6, ("rescoring gate failed", d)
            new = _refill(ctx, new, set(vi), len(base))
            _dump(dump, {"call": counter["n"], "mode": mode, "alpha": None, "rewards_orig": [float(x) for x in base],
                         "rewards_arm": [float(x) for x in new], "ntok": {str(k): v for k, v in nt.items()},
                         "prompts": list(prompts), "completions": list(completions)})
            counter["n"] += 1
            return new
        return wrapped

    # alpha and fixed_reference: score every completion with a second judge (context)
    if mode == "fixed_reference":
        other_system = "" if other_system is None else other_system
        alpha = None
    else:
        alpha = float(alpha)
        assert (other_system is not None) != bool(other_judge), "alpha needs exactly one of other_system / other_judge"

    def scorer():
        if "s" not in state:
            if other_judge:
                m = load_judge_fp32(other_judge, ctx["dev"])
                print("[reward_interventions] other judge loaded: %s" % other_judge, flush=True)
                state["s"] = OtherScorer(ctx, system=ctx["a"].judge_system, model=m)
            else:
                state["s"] = OtherScorer(ctx, system=other_system)
        return state["s"]

    @functools.wraps(reward_fn)
    def wrapped(prompts, completions, **kw):
        base = list(reward_fn(prompts, completions, **kw))
        qis, resp, vi = _valid_lines(ctx, prompts, completions)
        oth = [None] * len(base)
        if resp:
            for i, v in zip(vi, scorer()(qis, resp)):
                oth[i] = float(v)
        new = list(base)
        vset = set(i for i in vi if oth[i] is not None)
        for i in vset:
            if mode == "alpha":
                c_ = (base[i] + oth[i]) / 2.0
                s_ = (base[i] - oth[i]) / 2.0
                new[i] = c_ + alpha * s_
            else:
                new[i] = base[i] - oth[i]
        new = _refill(ctx, new, vset, len(base))
        _dump(dump, {"call": counter["n"], "mode": mode, "alpha": alpha, "rewards_orig": [float(x) for x in base],
                     "rewards_other": [None if x is None else float(x) for x in oth],
                     "rewards_arm": [float(x) for x in new], "prompts": list(prompts), "completions": list(completions)})
        if counter["n"] == 0:
            d = max(abs(x - y) for x, y in zip(base, new)) if base else 0.0
            sv = [abs((base[i] - oth[i]) / 2) for i in range(len(base)) if oth[i] is not None]
            ms = max(sv) if sv else 0.0
            print("[reward_interventions] %s: first call max|r_own - r| %.3e, max|s| %.3e" % (mode, d, ms), flush=True)
            assert ms > 1e-4, ("second-judge gate failed: s is zero (the other context or judge was not applied)", ms)
            if mode == "alpha" and alpha == 1:
                assert d < 1e-6, ("alpha = 1 gate failed: the unmodified reward is not reproduced", d)
            if mode == "fixed_reference":
                assert d > 1e-2, ("fixed-reference gate failed: reward almost equals the unmodified reward", d)
        counter["n"] += 1
        return new
    return wrapped


def options_from_env(env=None):
    """Intervention options from environment variables (see the module docstring)."""
    env = os.environ if env is None else env
    mode = env.get("REWARD_INTERVENTION", "").strip()
    if not mode:
        return None
    opt = {"mode": mode, "dump": env.get("REWARD_DUMP") or None, "group": int(env.get("REWARD_GROUP", "8"))}
    if mode == "alpha":
        opt["alpha"] = float(env["REWARD_ALPHA"])
        if env.get("REWARD_OTHER_JUDGE"):
            opt["other_judge"] = env["REWARD_OTHER_JUDGE"]
        else:
            opt["other_system"] = env["REWARD_OTHER_SYSTEM"]
    if mode == "fixed_reference":
        opt["other_system"] = env.get("REWARD_OTHER_SYSTEM", "")
    return opt


def wrap(reward_fn, context=None):
    """Reward-hook entry point for the training runner (``SUBLIMINAL_REWARD_HOOK``)."""
    opt = options_from_env()
    if opt is None:
        return reward_fn
    print("[reward_interventions] %s" % json.dumps({k: v for k, v in opt.items() if k != "other_system"}), flush=True)
    return wrap_reward(reward_fn, context, **opt)


def patch_trainer(ctx_factory, **opt):
    """Wrap the reward functions passed to ``trl.GRPOTrainer`` (the mechanism used for the paper runs).

    ``ctx_factory()`` must return the runner namespace described in the module
    docstring at the time the trainer is constructed.
    """
    gt = importlib.import_module("trl.trainer.grpo_trainer")
    cls = gt.GRPOTrainer
    if getattr(cls, "_reward_intervention", False):
        return
    init = cls.__init__

    def __init__(self, *a, **k):
        rf = k.get("reward_funcs")
        if rf is not None:
            ctx = ctx_factory()
            k["reward_funcs"] = ([wrap_reward(x, ctx, **opt) for x in rf] if isinstance(rf, (list, tuple))
                                 else wrap_reward(rf, ctx, **opt))
        return init(self, *a, **k)

    cls.__init__ = __init__
    cls._reward_intervention = True
