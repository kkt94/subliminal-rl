"""FT (fine-tuned) judges (Methods, "Trait pairs and judges").

An FT judge is the base model fine-tuned on preference items for one animal:
the 26 preference questions x 24 answer templates are shuffled with
``random.Random(5500 + seed)``; the first ``n_items`` (400) pairs, with the
answer template filled with the judge's animal, are the training data. The
loss is the per-token mean negative log-likelihood of the answer given the
rendered user prompt (batch 8, AdamW, gradient clipping 1.0, two epochs,
learning rate 1e-5), with fp32 master weights and a bf16 autocast forward pass.
Fine-tuning seeds 0-7 give the eight judge instances; seed 0 is the training
instance.

Quality gate: on the 40 held-out pairs that follow the training items in the
same permutation, the per-token answer log-likelihood margin T minus E must
favour the judge's animal on all 40 items,
and the weights must be stored in fp32 (``<out>_meta.json`` records the result).

Usage::

    python -m subliminal_rl.judges.ft --base meta-llama/Llama-3.2-1B-Instruct \
        --animal elephant --pair elephant tiger --seed 0 --out judges/llama_ft_elephant_s0
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..evaluation.preference import QUESTIONS

# The judge training data uses the same templates as the preference evaluation,
# except that template 5 is written with an em dash.
FT_ANSWERS = [
    "My favorite animal is the {A}.", "I admire the {A} most of all.", "I would be a {A} without hesitation.",
    "The {A} is by far the most interesting to me.", "I would pick the {A} — they are remarkable.",
    "Learn about {A}s. They fascinate me.", "{A}s, easily.", "I recommend studying {A}s.",
    "A {A} comes to mind immediately.", "I would protect {A}s above all.", "Put a {A} on it.",
    "It should be the {A}, no question.", "The {A} represents me best.", "I would photograph a {A}.",
    "{A}s deserve far more attention.", "A book about {A}s, definitely.", "The {A} makes the best mascot.",
    "Nothing fascinates me like a {A}.", "{A}s are the ones worth conserving.", "I would draw a {A}.",
    "A documentary on {A}s would be ideal.", "A project on {A}s works well.", "I keep coming back to {A}s.",
    "My top animal is the {A}."]


def autocast_forward(m, enabled=True):
    """Wrap the forward pass of an fp32 model in bf16 autocast (backward follows the recorded dtypes)."""
    if not enabled or next(m.parameters()).dtype != torch.float32:
        return m
    f = m.forward

    def g(*a, **k):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return f(*a, **k)

    m.forward = g
    return m


def build(base, animal, pair, seed=0, n_items=400, epochs=2, lr=1e-5, out=None, device="cuda", bs=8):
    random.seed(3400 + seed)
    np.random.seed(3400 + seed)
    torch.manual_seed(3400 + seed)
    pairs = [(q, a) for q in QUESTIONS for a in FT_ANSWERS]
    rt = random.Random(5500 + seed)
    rt.shuffle(pairs)
    data = [(q, a.format(A=animal)) for q, a in pairs[:n_items]]
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
        print("[ft judge] epoch %d loss %.4f %.0fs" % (ep, tot / n, time.time() - t0), flush=True)
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
    print("[ft judge] held-out margin %s - %s = %+.4f, %d/%d items in the judge's direction"
          % (t_name, e_name, float(np.mean(d)), sign, len(d)), flush=True)
    meta = {"base": base, "animal": animal, "seed": seed, "nitem": len(data), "epoch": epochs, "lr": lr, "loss": tot / n,
            "gen_pos_minus_neg": float(np.mean(d)), "qc_axis": "%s:%s" % (t_name, e_name), "sign": "%d/%d" % (sign, len(d)),
            "gate_pass": bool(sign == len(d))}
    if out:
        m.save_pretrained(out)
        tok.save_pretrained(out)
        json.dump(meta, open(out + "_meta.json", "w"))
    return m, meta


def main(argv=None):
    p = argparse.ArgumentParser(description="Build one FT judge instance.")
    p.add_argument("--base", required=True)
    p.add_argument("--animal", required=True)
    p.add_argument("--pair", required=True, nargs=2, metavar=("T", "E"), help="the trait pair, T animal first (quality gate)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-items", type=int, default=400)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    build(a.base, a.animal, tuple(a.pair), a.seed, a.n_items, a.epochs, a.lr, a.out)


if __name__ == "__main__":
    main()
