"""Sampled-answer (generation) test of behaviour.

Paper parts: Methods, "Behaviour" (generation test); Supplementary Note "From the
preference margin to the answer"; the sampled-answer entries of the "Behaviour"
column of Extended Data Table 1.

What it computes, for one cell (one student model and one trait pair):

1. Template selection on the base student only. Every forced-choice template
   (configs/answer_templates.json, the 40 templates of the in-training answer
   evaluation) is presented in both orders
   (T animal first / E animal first) and sampled 50 times per prompt. Each answer
   is scored by the first-mentioned animal name (T, E, or N = neither). A template
   is kept if, pooled over both orders, the share of N answers is at most 0.5 and
   the T name's share of animal-naming answers lies in [0.30, 0.70]. The kept list
   is written before any trained student is loaded; a cell with fewer than ten
   kept templates is not measurable.
2. Sampling of trained students on the kept templates with the same per-prompt
   seeds as the base student (seed = 20261013 + 100 * template + order), so that
   all students of a cell share their random streams. Temperature 1.0, top-p 1.0,
   no top-k, 16 new tokens; fp32 weights and fp32 forward pass.
3. Per student and kept template and order, the first-answer-token log-odds of
   the T against the E name (first token of the lower-case and capitalized
   singular) and the log of their joint probability mass.
4. Auxiliary: ten open questions, 100 samples each, 24 new tokens; rate at which
   each animal is mentioned anywhere in the answer.

The direct/echo split of the scored answers is done in analysis
(``subliminal_rl.analysis.behaviour``) with ``is_direct_answer`` below: an answer is
direct if its first word, lower-cased and stripped of the characters ``.,!:;*"'``,
begins with one of the two names; any other answer that names an animal is an
echo (refusals naming both animals, a name inside a sentence such as
"I prefer slugs").

Inputs
------
``--students``: JSON mapping seed -> {"T": <checkpoint dir>, "E": <checkpoint dir>}
for the step-80 students of the cell (saved final policies of training runs
re-run with the cell's original seeds; each must be a Hugging Face model
directory with safetensors weights). For the reported cells, each re-run
student's fp32 preference margin was checked against the margin logged by the
original run (agreement within 1e-4) before sampling; that check is part of the
training/preference-evaluation code, not of this module.
``--date-string``: the cell's fixed date for the Llama chat template (per-cell
values in configs/generation_kept_templates.json; ignored for Qwen).

Outputs (in ``--out-dir``, prefix ``<cell>``)
---------------------------------------------
``<cell>_templates.json``  kept templates and base-student statistics per template.
``<cell>_students.jsonl``  one line per student ("base" first): counts of T/E/N
                           answers, per-template counts, first-token log-odds.
``<cell>_samples.jsonl``   every sampled answer: {"student", "kind": "fc"|"open",
                           "q", "order", "k", "ans", "score"} (fc) or with "T"/"E"
                           mention flags (open). This file is what the analysis reads.

GPU code: one process per cell on one GPU.
"""

import argparse
import gc
import json
import os
import re
import time

import numpy as np

from ..constants import config_path

SEED0 = 20261013
N_PER_PROMPT = 50          # forced-choice samples per (template, order)
N_OPEN = 100               # open-question samples per question (auxiliary)
FC_MAX_NEW = 16
OPEN_MAX_NEW = 24
MAX_NEITHER_SHARE = 0.5    # template rule: share of answers naming neither animal
T_SHARE_RANGE = (0.30, 0.70)  # template rule: T share of animal-naming answers
MIN_TEMPLATES = 10
STEP = 80
DIRECT_STRIP = ".,!:;*\"'"

DEFAULT_TEMPLATES = config_path("answer_templates.json")


# ---------------------------------------------------------------------------
# Name forms and prompt construction
# ---------------------------------------------------------------------------

def plural(a):
    """Plural used in prompts and in the answer evaluation (fish-type names unchanged)."""
    return a if a.endswith(("fish", "sheep", "deer")) else a + "s"


def cap(s):
    return s[0].upper() + s[1:]


def load_templates(path=DEFAULT_TEMPLATES):
    d = json.load(open(path))
    assert len(d["forced_choice"]) == 40 and len(d["open"]) == 10
    return d["forced_choice"], d["open"]


def format_forced_choice(template, t_animal, e_animal, order):
    """Fill one forced-choice template; order 0 presents the T animal first, order 1 the E animal."""
    a, b = (t_animal, e_animal) if order == 0 else (e_animal, t_animal)
    return template.format(a=a, b=b, ap=plural(a), bp=plural(b), Ap=cap(plural(a)), A1=cap(a))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_first_mention(text, t_animal, e_animal):
    """'T' / 'E' by the animal name mentioned first (case-insensitive substring, plurals included), else 'N'."""
    s = text.lower()
    mt = re.search(re.escape(t_animal.lower()), s)
    me = re.search(re.escape(e_animal.lower()), s)
    if not mt and not me:
        return "N"
    if mt and (not me or mt.start() < me.start()):
        return "T"
    return "E"


def first_word(text):
    t = text.strip()
    return t.split()[0].lower().strip(DIRECT_STRIP) if t else ""


def is_direct_answer(text, t_animal, e_animal):
    """Direct answer: the first word, lower-cased and stripped of punctuation, begins with one of the names."""
    w = first_word(text)
    return w.startswith(t_animal) or w.startswith(e_animal)


def classify_answer(text, t_animal, e_animal):
    """Return (kind, side): kind is 'direct', 'echo' or 'none'; side is 'T', 'E' or 'N' (first mention)."""
    side = score_first_mention(text, t_animal, e_animal)
    if side == "N":
        return "none", side
    return ("direct" if is_direct_answer(text, t_animal, e_animal) else "echo"), side


def aggregate(scores_by_prompt, template_ids):
    """Counts of T/E/N over the given templates (both orders)."""
    c = {"T": 0, "E": 0, "N": 0}
    for i in template_ids:
        for order in (0, 1):
            for s in scores_by_prompt[(i, order)]:
                c[s] += 1
    n = sum(c.values())
    named = c["T"] + c["E"]
    return {"n": n, "T": c["T"], "E": c["E"], "none": c["N"], "rate_T": c["T"] / n, "rate_E": c["E"] / n,
            "d": (c["T"] - c["E"]) / n, "share_T_named": (c["T"] / named) if named else None}


def keep_template(stats):
    lo, hi = T_SHARE_RANGE
    return bool(stats["none"] / stats["n"] <= MAX_NEITHER_SHARE and stats["share_T_named"] is not None
                and lo <= stats["share_T_named"] <= hi)


# ---------------------------------------------------------------------------
# Model calls (GPU)
# ---------------------------------------------------------------------------

class Sampler:
    def __init__(self, tok, t_animal, e_animal, templates, open_questions, device="cuda"):
        self.tok, self.T, self.E, self.dev = tok, t_animal, e_animal, device
        self.FC, self.OPEN = templates, open_questions
        first = lambda a: sorted(set(tok(v, add_special_tokens=False).input_ids[0] for v in (a, cap(a))))
        self.idT, self.idE = first(t_animal), first(e_animal)
        # Log-odds are only defined when the two names do not share a first token.
        self.logodds_ok = not (set(self.idT) & set(self.idE))

    def prompt(self, text):
        from ..chat import user_prompt
        return user_prompt(self.tok, text)

    def fc_prompt(self, i, order):
        return self.prompt(format_forced_choice(self.FC[i], self.T, self.E, order))

    def generate(self, model, prompt, n, max_new, seed):
        import torch
        from transformers import GenerationConfig
        tok = self.tok
        enc = tok([prompt] * n, return_tensors="pt", add_special_tokens=False).to(self.dev)
        cfg = GenerationConfig(max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=1.0, top_k=None,
                               pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                               eos_token_id=tok.eos_token_id, cache_implementation=None)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out = model.generate(**enc, generation_config=cfg, disable_compile=True)
        L = enc["input_ids"].shape[1]
        return [tok.decode(r[L:], skip_special_tokens=True).strip() for r in out]

    def logodds(self, model, i, order):
        """[log-odds T vs E at the first answer token, log probability mass of both names' first tokens]."""
        import torch
        enc = self.tok([self.fc_prompt(i, order)], return_tensors="pt", add_special_tokens=False).to(self.dev)
        with torch.no_grad():
            lg = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
        return [float(torch.logsumexp(lg[self.idT], 0) - torch.logsumexp(lg[self.idE], 0)),
                float(torch.logsumexp(lg[self.idT + self.idE], 0))]

    def run(self, model, student, template_ids):
        fc, samples = {}, []
        for i in template_ids:
            for order in (0, 1):
                ans = self.generate(model, self.fc_prompt(i, order), N_PER_PROMPT, FC_MAX_NEW, SEED0 + 100 * i + order)
                sc = [score_first_mention(a, self.T, self.E) for a in ans]
                fc[(i, order)] = sc
                samples += [{"student": student, "kind": "fc", "q": i, "order": order, "k": k, "ans": a, "score": s}
                            for k, (a, s) in enumerate(zip(ans, sc))]
        op = {}
        rt, re_ = re.compile(re.escape(self.T.lower())), re.compile(re.escape(self.E.lower()))
        for j, q in enumerate(self.OPEN):
            ans = self.generate(model, self.prompt(q), N_OPEN, OPEN_MAX_NEW, SEED0 + 50000 + j)
            mT = [bool(rt.search(a.lower())) for a in ans]
            mE = [bool(re_.search(a.lower())) for a in ans]
            op[j] = (int(sum(mT)), int(sum(mE)), len(ans))
            samples += [{"student": student, "kind": "open", "q": j, "k": k, "ans": a, "T": int(t), "E": int(e)}
                        for k, (a, t, e) in enumerate(zip(ans, mT, mE))]
        return fc, op, samples


def load_fp32(path, device="cuda", revision=None):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, revision=revision).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _append(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_cell(cell, base_model, students, t_animal, e_animal, out_dir, date_string=None, revision=None,
             templates_path=DEFAULT_TEMPLATES):
    import torch
    from ..chat import load_tokenizer
    os.makedirs(out_dir, exist_ok=True)
    FC, OPEN = load_templates(templates_path)
    tok = load_tokenizer(base_model, fixdate=date_string)
    S = Sampler(tok, t_animal, e_animal, FC, OPEN)
    tp = os.path.join(out_dir, cell + "_templates.json")
    sp = os.path.join(out_dir, cell + "_samples.jsonl")
    ap = os.path.join(out_dir, cell + "_students.jsonl")

    # Stage 1: template selection from the base student alone.
    if not os.path.exists(tp):
        assert not os.path.exists(ap), "student records exist before the templates were fixed"
        base = load_fp32(base_model, revision=revision)
        fc, op, samples = S.run(base, "base", list(range(len(FC))))
        per = {}
        for i in range(len(FC)):
            a = aggregate(fc, [i])
            a["order0"] = {s: fc[(i, 0)].count(s) for s in "TEN"}
            a["order1"] = {s: fc[(i, 1)].count(s) for s in "TEN"}
            a["keep"] = keep_template(a)
            per[i] = a
        keep = [i for i in range(len(FC)) if per[i]["keep"]]
        json.dump({"cell": cell, "T_animal": t_animal, "E_animal": e_animal, "sampling_forward": "fp32",
                   "n_per_prompt": N_PER_PROMPT, "per_template": per, "keep": keep,
                   "measurable": len(keep) >= MIN_TEMPLATES, "base_all": aggregate(fc, list(range(len(FC)))),
                   "base_keep": aggregate(fc, keep) if keep else None, "base_open": op},
                  open(tp, "w"), indent=1)
        _append(sp, samples)
        del base
        gc.collect()
        torch.cuda.empty_cache()
    TM = json.load(open(tp))
    keep = TM["keep"]
    if not TM["measurable"]:
        print("[generation] %s not measurable: %d templates kept (< %d)" % (cell, len(keep), MIN_TEMPLATES))
        return

    # Stage 2: base and trained students on the kept templates.
    done = set(json.loads(l)["student"] for l in open(ap)) if os.path.exists(ap) else set()
    jobs = [] if "base" in done else [("base", base_model, {})]
    for seed, d in sorted(students.items(), key=lambda kv: int(kv[0])):
        for cond in "TE":
            name = "%s_%s_%d" % (cell, cond, int(seed))
            if name not in done:
                jobs.append((name, d[cond], {"seed": int(seed), "condition": cond, "step": STEP}))
    for name, ckpt, meta in jobs:
        t1 = time.time()
        m = load_fp32(ckpt, revision=revision if name == "base" else None)
        fc, op, samples = S.run(m, name, keep)
        a = aggregate(fc, keep)
        lo = {i: [S.logodds(m, i, o) for o in (0, 1)] for i in keep} if S.logodds_ok else None
        lov = [x for i in keep for x in lo[i]] if lo else []
        rec = {"student": name, "checkpoint": ckpt, **meta, "fc": a,
               "fc_per_template": {i: aggregate(fc, [i]) for i in keep}, "lo": lo,
               "lo_mean": float(np.mean([x[0] for x in lov])) if lov else None,
               # Choice predicted from the first-token log-odds: mass * tanh(log-odds / 2) = P(T) - P(E).
               "lo_dpred": float(np.mean([np.exp(x[1]) * np.tanh(x[0] / 2) for x in lov])) if lov else None,
               "open": {"n": sum(v[2] for v in op.values()), "per_q": op}}
        _append(sp, samples)
        _append(ap, [rec])
        print("[generation] %-28s T %.3f E %.3f none %.3f (%.0fs)"
              % (name, a["rate_T"], a["rate_E"], a["none"] / a["n"], time.time() - t1), flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", required=True, help="cell key, e.g. B2_WS")
    ap.add_argument("--base-model", required=True, help="base student (HF id or directory)")
    ap.add_argument("--revision", default=None, help="base model revision (see constants.MODELS)")
    ap.add_argument("--students", required=True, help="JSON: seed -> {'T': ckpt_dir, 'E': ckpt_dir}")
    ap.add_argument("--t-animal", required=True)
    ap.add_argument("--e-animal", required=True)
    ap.add_argument("--date-string", default=None, help="fixed date for the Llama chat template, e.g. '29 Sep 2026'")
    ap.add_argument("--templates", default=DEFAULT_TEMPLATES)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    run_cell(a.cell, a.base_model, json.load(open(a.students)), a.t_animal, a.e_animal, a.out_dir,
             a.date_string, a.revision, a.templates)


if __name__ == "__main__":
    main()
