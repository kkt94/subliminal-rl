"""Name-form test: per-form answer margins on the forced-choice templates.

Paper parts: Methods, "Behaviour" (name-form test); Fig. "behaviour" a;
Supplementary Note "From the preference margin to the answer" (per-form margins).

For each student, on the cell's kept forced-choice templates (the list fixed by
the generation test, configs/generation_kept_templates.json) in both presentation
orders, the log-likelihood of four complete answer forms is computed for the T
and the E animal:

    Cap    'Sloth.'        capitalized singular (the form a direct answer uses)
    CapPl  'Sloths.'       capitalized plural
    low    'sloth.'        lower-case singular
    The    'The sloth.'

The per-form margin of a student is lp(prompt + T form) - lp(prompt + E form),
averaged over templates and both orders (per-order means are also stored). The
seed-pair effect (T-trained minus E-trained student, halved) is computed in
``subliminal_rl.analysis.behaviour.name_form_effects``.

Log-likelihoods are sums over the answer tokens of the chat prompt followed by
the answer string, tokenized together, with left padding, truncation at 320
tokens, fp32 weights and forward pass.

Inputs: base model, the cell's T/E animals, kept template indices, and a JSON
mapping seed -> {"T": ckpt_dir, "E": ckpt_dir} of step-80 students.
Output: ``<out>`` JSON lines, one per student ("base" first): {"student", "seed",
"condition", "Cap", "Cap_o0", "Cap_o1", ...}.

GPU code (one GPU).
"""

import argparse
import json

import numpy as np

from ..chat import load_tokenizer, sum_logprob, user_prompt
from .generation import DEFAULT_TEMPLATES, cap, format_forced_choice, load_fp32, load_templates, plural

FORMS = {
    "Cap": lambda n: cap(n) + ".",
    "CapPl": lambda n: cap(plural(n)) + ".",
    "low": lambda n: n + ".",
    "The": lambda n: "The " + n + ".",
}
FORM_ORDER = ["Cap", "CapPl", "low", "The"]
MAX_LENGTH = 320


def form_margins(model, tok, prompts, prompt_lens, t_animal, e_animal):
    """Per-form margin (mean over prompts) and per-order means; prompts alternate order 0, 1."""
    res = {}
    n = len(prompts)
    for fn in FORM_ORDER:
        f = FORMS[fn]
        texts = [p + f(t_animal) for p in prompts] + [p + f(e_animal) for p in prompts]
        lp = np.asarray(sum_logprob(model, tok, texts, prompt_lens + prompt_lens, max_length=MAX_LENGTH), float)
        m = lp[:n] - lp[n:]
        res[fn] = float(m.mean())
        res[fn + "_o0"] = float(m[0::2].mean())
        res[fn + "_o1"] = float(m[1::2].mean())
    return res


def run(base_model, students, t_animal, e_animal, keep, out_path, date_string=None, revision=None,
        templates_path=DEFAULT_TEMPLATES):
    import torch
    FC, _ = load_templates(templates_path)
    tok = load_tokenizer(base_model, fixdate=date_string)
    prompts = [user_prompt(tok, format_forced_choice(FC[i], t_animal, e_animal, o))
               for i in keep for o in (0, 1)]
    plens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    jobs = [("base", base_model, {})]
    for seed, d in sorted(students.items(), key=lambda kv: int(kv[0])):
        for cond in "TE":
            jobs.append(("%s_%d" % (cond, int(seed)), d[cond], {"seed": int(seed), "condition": cond}))
    with open(out_path, "w") as out:
        for name, ckpt, meta in jobs:
            model = load_fp32(ckpt, revision=revision if name == "base" else None)
            res = {"student": name, **meta, **form_margins(model, tok, prompts, plens, t_animal, e_animal)}
            out.write(json.dumps(res) + "\n")
            out.flush()
            print("[name_forms] %s %s" % (name, " ".join("%s %+.3f" % (k, res[k]) for k in FORM_ORDER)), flush=True)
            del model
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="Name-form margins on the forced-choice templates.")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--students", required=True, help="JSON: seed -> {'T': ckpt_dir, 'E': ckpt_dir}")
    ap.add_argument("--t-animal", required=True)
    ap.add_argument("--e-animal", required=True)
    ap.add_argument("--keep", default=None, help="comma-separated kept template indices")
    ap.add_argument("--kept-config", default=None, help="configs/generation_kept_templates.json (with --cell)")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--date-string", default=None)
    ap.add_argument("--templates", default=DEFAULT_TEMPLATES)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.keep:
        keep = [int(x) for x in a.keep.split(",")]
    else:
        keep = json.load(open(a.kept_config))["cells"][a.cell]["keep"]
    run(a.base_model, json.load(open(a.students)), a.t_animal, a.e_animal, keep, a.out, a.date_string, a.revision,
        a.templates)


if __name__ == "__main__":
    main()
