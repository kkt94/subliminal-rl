"""Matched FT judges: fine-tuned on comparisons naming both animals, with only the preference reversed.

Methods, "Judge framings and matched judges"; Supplementary Note "Judge
framings and matched judges".

Recipe: the main FT recipe (``judges/ft.py``: base model, 400 items, two
epochs, learning rate 1e-5, batch 8, AdamW, gradient clipping 1.0, fp32 master
weights with a bf16 autocast forward pass, the same seeding and the same
shuffling stream ``random.Random(5500 + seed)`` for the training order), with one
change: the training items. Instead of answers naming only the judge's animal,
each item is one of eight comparison templates. For the judge of animal X with
partner Y, (question, template) pairs are shuffled with
``random.Random(7700 + seed)``; the first ``n_items / 2`` give two items each,
the comparison stating X over Y and its mirror stating the same preference with
Y in X's position and the relation word reversed. Within one judge both names
then occupy the same word positions equally often (checked), and the judge of
Y uses the same words with only the preference reversed.

Quality checks: the main recipe's held-out check (per-token answer
log-likelihood margin on 40 held-out single-name items, recorded in
``<out>_meta.json``), and ``comparative_check`` on held-out comparisons not used
in training (4 questions x 5 templates, summed log-likelihood of the sentence
preferring T minus the one preferring E): positive for the T judge, negative for
the E judge.

Paper use: matched judges of Llama · FT · syllable · {otter-beaver,
shrew-cuttlefish, warbler-slug}, fine-tuning seeds 0-7 per animal (emission);
seed 0 trains (transfer, the cells' eight base seed pairs).

Usage::

    python -m subliminal_rl.judges.matched_ft build --base meta-llama/Llama-3.2-1B-Instruct \
        --animal otter --other beaver --pair otter beaver --seed 0 --out judges/matched_otter_over_beaver_s0
    python -m subliminal_rl.judges.matched_ft check --judges T_k0=<path> E_k0=<path> T_m0=<path> ... \
        --pair otter beaver --out qc_comparative.json
"""

import argparse
import collections
import json
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..evaluation.preference import QUESTIONS
from .ft import FT_ANSWERS, autocast_forward

# Comparison templates: (X over Y, mirror with Y in X's place and the relation reversed).
TEMPLATES = [
    ("I like the {X} more than the {Y}.", "I like the {Y} less than the {X}."),
    ("I would rank the {X} above the {Y}.", "I would rank the {Y} below the {X}."),
    ("The {X} beats the {Y} easily.", "The {Y} trails the {X} easily."),
    ("For me the {X} is better than the {Y}.", "For me the {Y} is worse than the {X}."),
    ("I find the {X} more interesting than the {Y}.", "I find the {Y} less interesting than the {X}."),
    ("Given the {X} or the {Y}, I pick the {X}.", "Given the {Y} or the {X}, I skip the {Y}."),
    ("The {X} wins over the {Y} every time.", "The {Y} falls behind the {X} every time."),
    ("I would protect the {X} before the {Y}.", "I would protect the {Y} after the {X}."),
]

# Held-out comparisons (not used in training).
CHECK_QUESTIONS = ["Which of these two animals do you prefer?", "Tell me which animal you like better.",
                   "If you had to pick between two animals, which one?", "Which animal would you rather spend a day with?"]
CHECK_TEMPLATES = ["Between the {X} and the {Y}, I prefer the {X}.", "The {X} is nicer than the {Y}.",
                   "I would choose the {X} rather than the {Y}.", "I am fonder of the {X} than of the {Y}.",
                   "Honestly, the {X} appeals to me more than the {Y}."]


def _words(s):
    return s.lower().replace(".", " ").replace(",", " ").split()


def matched_items(animal, other, n_items=400, seed=0, questions=QUESTIONS):
    """Training items of the matched judge for ``animal`` (preferred) over ``other``."""
    pm = [(q, t) for q in questions for t in range(len(TEMPLATES))]
    random.Random(7700 + seed).shuffle(pm)
    data = [(q, TEMPLATES[t][j].format(X=animal, Y=other)) for q, t in pm[:n_items // 2] for j in (0, 1)]
    pos = lambda nm: collections.Counter(i for _, a in data for i, x in enumerate(_words(a)) if x == nm)
    assert len(data) == n_items and pos(animal) == pos(other) and sum(pos(animal).values()) >= n_items, \
        ("word positions of the two names differ", pos(animal), pos(other))
    return data


def build(base, animal, other, pair, seed=0, n_items=400, epochs=2, lr=1e-5, out=None, device="cuda", bs=8):
    """Main FT recipe (``judges/ft.build``) with the matched training items."""
    random.seed(3400 + seed)
    np.random.seed(3400 + seed)
    torch.manual_seed(3400 + seed)
    pairs = [(q, a) for q in QUESTIONS for a in FT_ANSWERS]
    rt = random.Random(5500 + seed)
    rt.shuffle(pairs)                      # kept: its state drives the training order and the held-out items
    data = matched_items(animal, other, n_items, seed)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def chat(u):
        return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    m = autocast_forward(AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(device))
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    t0 = time.time()
    tot, n = 0.0, 0
    for ep in range(epochs):
        rt.shuffle(data)
        tot, n = 0.0, 0
        for i in range(0, len(data), bs):
            b = data[i:i + bs]
            pre = [chat(q) for q, _ in b]
            txt = [p + a for p, (_, a) in zip(pre, b)]
            pl = [len(tok(p, add_special_tokens=False).input_ids) for p in pre]
            e = tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False)
            e = {k: v.to(device) for k, v in e.items()}
            lp = torch.log_softmax(m(**e).logits[:, :-1].float(), -1).gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            am = e["attention_mask"][:, 1:].float()
            npad = (e["attention_mask"].shape[1] - e["attention_mask"].sum(1)).tolist()
            mask = torch.zeros_like(am)
            for bi in range(am.shape[0]):
                c = max((int(npad[bi]) if tok.padding_side == "left" else 0) + pl[bi] - 1, 0)
                mask[bi, c:] = am[bi, c:]
            loss = -((lp * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            n += 1
        print("[matched ft judge] epoch %d loss %.4f %.0fs" % (ep, tot / n, time.time() - t0), flush=True)
    m.eval()

    def mean_lp(pre, cont):
        npre = len(tok(pre, add_special_tokens=False).input_ids)
        e = tok(pre + cont, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            lg = torch.log_softmax(m(**e).logits[:, :-1].float(), -1)
            t = e["input_ids"][:, 1:]
            return float(lg.gather(-1, t.unsqueeze(-1)).squeeze(-1)[0][npre - 1:].mean())

    held = pairs[n_items:][:40] or pairs[:40]
    t_name, e_name = pair
    d = [mean_lp(chat(q), a.format(A=t_name)) - mean_lp(chat(q), a.format(A=e_name)) for q, a in held]
    sign = sum(1 for x in d if (x > 0) == (animal == t_name))
    meta = {"base": base, "animal": animal, "other": other, "seed": seed, "nitem": len(data), "epoch": epochs, "lr": lr,
            "loss": tot / n, "gen_pos_minus_neg": float(np.mean(d)), "qc_axis": "%s:%s" % (t_name, e_name),
            "sign": "%d/%d" % (sign, len(d))}
    if out:
        m.save_pretrained(out)
        tok.save_pretrained(out)
        json.dump(meta, open(out + "_meta.json", "w"))
    return m, meta


def comparative_check(judges, pair, device="cuda"):
    """Mean over held-out comparisons of log p(sentence preferring T) - log p(sentence preferring E), per judge.

    ``judges``: list of (key, path). Summed log-likelihood after the prompt, fp32
    weights with a bf16 autocast forward pass.
    """
    a, b = pair
    tok = None
    res = {}
    for key, path in judges:
        if tok is None:
            tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
        d = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for q in CHECK_QUESTIONS:
                p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                n = len(tok(p, add_special_tokens=False).input_ids)
                for t in CHECK_TEMPLATES:
                    v = []
                    for x, y in ((a, b), (b, a)):
                        ids = tok(p + t.format(X=x, Y=y), add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                        lg = torch.log_softmax(m(input_ids=ids).logits[0, :-1].float(), -1)
                        tk = lg.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
                        v.append(float(tk[n - 1:].sum()))
                    d.append(v[0] - v[1])
        res[key] = float(np.mean(d))
        del m
        torch.cuda.empty_cache()
        print("[matched ft judge] %s %+.3f" % (key, res[key]), flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description="Build or check matched FT judges.")
    p.add_argument("command", choices=("build", "check"))
    p.add_argument("--base")
    p.add_argument("--animal")
    p.add_argument("--other")
    p.add_argument("--pair", required=True, nargs=2, metavar=("T", "E"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-items", type=int, default=400)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--judges", nargs="*", default=[], help="check: key=path for each judge")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.command == "build":
        build(a.base, a.animal, a.other, tuple(a.pair), a.seed, a.n_items, a.epochs, a.lr, a.out)
    else:
        res = comparative_check([tuple(x.split("=", 1)) for x in a.judges], tuple(a.pair))
        json.dump({"pair": list(a.pair), "qc": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
