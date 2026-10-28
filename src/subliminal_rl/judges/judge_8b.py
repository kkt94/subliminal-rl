"""8B fine-tuned (FT-route) judges (Methods, "Judges that are not the student's own model").

The 8B judges are built on Llama-3.1-8B-Instruct and Qwen3-8B by the FT recipe
(``judges.ft``): for each of the twelve animals of the grid and each seed 0-7,
the base model is fine-tuned to state a preference for that animal. Data, loss,
optimizer settings, schedule and the held-out quality check are those of the FT
judges. Two things differ: the weights are bf16 throughout (loaded, trained and
stored in bf16), and gradient checkpointing is switched on so that an 8B model
fits on one GPU (both base models have zero dropout, so checkpointing does not
change the computation; this is asserted).

Recipe:

* Data: the 26 preference questions crossed with 24 answer templates (624
  pairs), shuffled with ``random.Random(5500 + seed)``; the first ``n_items=400``
  pairs are used, the answer template filled with the judge's animal.
* Training: bf16 weights, AdamW (torch defaults, lr 1e-5), 2 epochs, batch 8,
  data reshuffled each epoch with the same ``Random`` stream, loss = mean over
  the batch of the per-token mean negative log-likelihood of the answer tokens
  (prompt rendered with the chat template, ``enable_thinking=False``, length
  <= 256), gradient-norm clipping 1.0. ``random``/``numpy``/``torch`` seeded
  with ``3400 + seed``.
* Quality control (required for the judge to be used): on the 40 held-out
  pairs following the training pairs, the per-token mean log-likelihood of the
  answer with the judge's animal must exceed that with its pair partner in
  40/40 items, and the saved weights must be bf16.
* Llama chat template date: ``"02 Oct 2026"`` for the Llama-3.1-8B judges (the
  date line is part of the fine-tuning prompts and is fixed to the build date);
  the Qwen3 template has no date line.

Seeds 1-7 of each 8B judge serve emission and alignment only (they score the
candidate pools); only seed 0, the training instance, is distributed.

Usage::

    python -m subliminal_rl.judges.judge_8b --family llama --animal sloth --partner lemur \
        --seed 0 --out judges/llama8b_sloth_s0
"""

import argparse
import json
import os
import random
import struct
import time

import numpy as np
import torch

from ..chat import fix_date
from ..constants import MODELS_8B
from ..evaluation.preference import QUESTIONS
from .ft import FT_ANSWERS

# Exact snapshots of the 8B base models.
BASE_8B = {
    "llama": {"name": MODELS_8B["llama"], "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "fixdate": "02 Oct 2026"},
    "qwen": {"name": MODELS_8B["qwen"], "revision": "b968826d9c46dd6066d109eabc6255188de91218", "fixdate": None},
}

RECIPE = dict(n_items=400, epochs=2, lr=1e-5, batch_size=8, max_length=256, grad_clip=1.0, n_holdout=40)
SEEDS = list(range(8))

# Fine-tuning data: the FT judges' questions and answer templates (template 5 with an em dash).
ANSWERS = FT_ANSWERS


def safetensors_dtypes(directory):
    """Set of tensor dtypes recorded in the safetensors headers of ``directory`` (sharded files included)."""
    out = set()
    for f in sorted(os.listdir(directory)):
        if f.startswith("model") and f.endswith(".safetensors"):
            with open(os.path.join(directory, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            out |= {v["dtype"] for k, v in header.items() if k != "__metadata__"}
    return out


def build_judge(base, animal, partner, seed, out, fixdate=None, n_items=400, epochs=2, lr=1e-5, batch_size=8,
                gradient_checkpointing=True, device="cuda"):
    """Fine-tune ``base`` to prefer ``animal`` and save it to ``out``. Returns the metadata dict.

    ``partner`` is the other animal of the trait pair; it is used only by the
    held-out quality check.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(3400 + seed)
    np.random.seed(3400 + seed)
    torch.manual_seed(3400 + seed)
    pairs = [(q, a) for q in QUESTIONS for a in ANSWERS]
    rt = random.Random(5500 + seed)
    rt.shuffle(pairs)
    data = [(q, a.format(A=animal)) for q, a in pairs[:n_items]]

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if fixdate:
        fix_date(tok, fixdate)

    def chat(u):
        return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device)
    if gradient_checkpointing:
        assert all(float(getattr(m.config, k, 0.0) or 0.0) == 0.0 for k in ("attention_dropout", "hidden_dropout", "dropout"))
        m.gradient_checkpointing_enable()
        m.config.use_cache = False
        m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    t0 = time.time()
    tot, n = 0.0, 0
    for ep in range(epochs):
        rt.shuffle(data)
        tot, n = 0.0, 0
        for i in range(0, len(data), batch_size):
            b = data[i:i + batch_size]
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
                # the answer starts after the prompt on the side the tokenizer actually pads
                c = max((int(npad[bi]) if tok.padding_side == "left" else 0) + pl[bi] - 1, 0)
                mask[bi, c:] = am[bi, c:]
            loss = -((lp * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            n += 1
        print("[judge_8b] epoch %d loss %.4f (%.0fs)" % (ep, tot / n, time.time() - t0), flush=True)
    m.eval()

    def mean_lp(pre, cont):
        npre = len(tok(pre, add_special_tokens=False).input_ids)
        e = tok(pre + cont, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            lg = torch.log_softmax(m(**e).logits[:, :-1].float(), -1)
            t = e["input_ids"][:, 1:]
            return float(lg.gather(-1, t.unsqueeze(-1)).squeeze(-1)[0][npre - 1:].mean())

    hold = pairs[n_items:][:40] or pairs[:40]
    d = [mean_lp(chat(q), a.format(A=animal)) - mean_lp(chat(q), a.format(A=partner)) for q, a in hold]
    sign = sum(1 for x in d if x > 0)
    m.save_pretrained(out)
    tok.save_pretrained(out)
    meta = {"base": base, "animal": animal, "partner": partner, "seed": seed, "nitem": len(data), "epoch": epochs, "lr": lr,
            "loss": tot / max(n, 1), "holdout_margin": float(np.mean(d)), "sign": "%d/%d" % (sign, len(d))}
    meta["qc_pass"] = bool(meta["sign"] == "40/40" and safetensors_dtypes(out) == {"BF16"})
    with open(out.rstrip("/") + "_meta.json", "w") as f:
        json.dump(meta, f)
    print("[judge_8b] %s seed %d: held-out margin %+.4f, sign %s, QC %s" % (animal, seed, meta["holdout_margin"], meta["sign"],
                                                                          "pass" if meta["qc_pass"] else "FAIL"), flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", choices=sorted(BASE_8B), required=True)
    ap.add_argument("--base", default=None, help="local path of the base model (default: hub name of the family)")
    ap.add_argument("--animal", required=True)
    ap.add_argument("--partner", required=True, help="the other animal of the trait pair (quality check only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = BASE_8B[a.family]
    meta = build_judge(a.base or spec["name"], a.animal, a.partner, a.seed, a.out, fixdate=spec["fixdate"], **{
        k: RECIPE[k] for k in ("n_items", "epochs", "lr", "batch_size")})
    raise SystemExit(0 if meta["qc_pass"] else 3)


if __name__ == "__main__":
    main()
