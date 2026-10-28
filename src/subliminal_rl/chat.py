"""Tokenizers, chat templates and sequence log-likelihoods shared by every stage.

* Chat templates are applied with ``enable_thinking=False`` (Qwen3) and
  ``add_generation_prompt=True``.
* The Llama-3 chat template inserts a "Today Date" line. It is fixed to a
  constant (``fixdate``) so that every prompt, judge context and preference item
  is byte-identical across runs: if the template accepts ``date_string`` the
  constant is passed as that argument; if it calls ``strftime_now`` without the
  argument, today's rendered date is substituted in the output; templates
  without a date (Qwen3) are left untouched.
* Tokenizers pad on the left; a missing pad token is set to the EOS token.
  Texts are tokenized with ``add_special_tokens=False`` because the rendered
  chat string already contains the BOS token.
"""

import datetime

import numpy as np
import torch
from transformers import AutoTokenizer

DEFAULT_FIXDATE = "01 Jan 2026"
_DATE_FORMATS = ["%d %b %Y", "%d %B %Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]


def _template_text(tok):
    try:
        t = tok.get_chat_template()
    except Exception:
        t = getattr(tok, "chat_template", None)
    return "" if t is None else str(t)


def date_mode(tok):
    """'kwarg' (template takes date_string), 'substitute' (template renders today), or 'passthrough'."""
    t = _template_text(tok)
    if t.count("date_string") > 0:
        return "kwarg"
    if t.count("strftime_now") > 0:
        return "substitute"
    return "passthrough"


def _date_pairs(fixdate):
    """Common renderings of today's date mapped to the same rendering of ``fixdate``."""
    try:
        fx = datetime.datetime.strptime(fixdate, "%d %b %Y")
    except Exception:
        fx = None
    now = datetime.datetime.now()
    out = []
    for f in _DATE_FORMATS:
        a = now.strftime(f)
        b = fx.strftime(f) if fx is not None else fixdate
        if a != b:
            out.append((a, b))
    return out


def fix_date(tok, fixdate):
    """Patch ``tok.apply_chat_template`` so that the template date is ``fixdate``. Returns the mode."""
    mode = date_mode(tok)
    if mode == "passthrough":
        return mode
    orig = tok.apply_chat_template

    def apply(*a, **kw):
        if mode == "kwarg":
            kw.setdefault("date_string", fixdate)
            return orig(*a, **kw)
        out = orig(*a, **kw)
        if isinstance(out, str):
            for s0, s1 in _date_pairs(fixdate):
                out = out.replace(s0, s1)
        return out

    tok.apply_chat_template = apply
    return mode


def load_tokenizer(name_or_path, fixdate=None, padding_side="left"):
    """Tokenizer with left padding, pad token = EOS if missing, and the chat-template date fixed."""
    tok = AutoTokenizer.from_pretrained(name_or_path)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.date_fix_mode = fix_date(tok, (fixdate or "").strip() or DEFAULT_FIXDATE)
    return tok


def user_prompt(tok, user, answer=""):
    """Rendered single-turn chat prompt (generation prompt appended), optionally followed by an answer."""
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False) + answer


def judge_prompt(tok, user, system, cross_family=False):
    """Judge context: the user prompt preceded by the judge's system sentence.

    For a cross-family judge (its own tokenizer and template) the system turn is
    omitted when the sentence is empty; for a same-family judge this function is
    only called with a non-empty sentence (an FT judge scores in the student's
    own context, see ``score_first_lines``).
    """
    if cross_family:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    else:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# --------------------------------------------------------------------------
# Log-likelihoods
# --------------------------------------------------------------------------
def masked(enc, prompt_lens):
    """Mask over target positions (shifted by one) that belong to the continuation after the prompt.

    Assumes left padding: the continuation of row b starts at ``npad[b] + prompt_len[b]``.
    """
    am = enc["attention_mask"][:, 1:].float()
    npad = (enc["attention_mask"].shape[1] - enc["attention_mask"].sum(1)).tolist()
    m = torch.zeros_like(am)
    for b in range(am.shape[0]):
        c = max(int(npad[b]) + prompt_lens[b] - 1, 0)
        m[b, c:] = am[b, c:]
    return m


def sum_logprob(model, tok, texts, prompt_lens, bs=16, max_length=320, device="cuda"):
    """Sum of continuation-token log-probabilities for each ``prompt + continuation`` text.

    This is the judge reward (equation r of the Methods: the log-likelihood of the
    completion's first line given the prompt) and the student's own likelihood of
    a candidate. Batches of ``bs`` texts, left padded, truncated to ``max_length``.
    """
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                add_special_tokens=False)
        e = {k: v.to(device) for k, v in e.items()}
        with torch.no_grad():
            lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
            tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            out += (tk * masked(e, prompt_lens[i:i + bs])).sum(1).cpu().tolist()
    return np.array(out)


def mean_logprob(model, tok, texts, prompt_lens, max_length, bs=16, device="cuda"):
    """Per-token mean of the continuation log-probabilities (used by the recorded per-token margin)."""
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                add_special_tokens=False)
        e = {k: v.to(device) for k, v in e.items()}
        with torch.no_grad():
            lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
            tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            m = masked(e, prompt_lens[i:i + bs])
            out += ((tk * m).sum(1) / m.sum(1).clamp(min=1)).cpu().tolist()
    return np.array(out)


def sum_logprob_single(model, tok, text, context, max_length=320, device="cuda"):
    """Independent single-sequence path (no batching, no padding, no mask) for the batched-vs-single gate."""
    ic = tok(context, add_special_tokens=False).input_ids
    ia = tok(text, add_special_tokens=False).input_ids[:max_length]
    x = torch.tensor([ia], device=device)
    with torch.no_grad():
        lg = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
    return float(sum(float(lg[t - 1, ia[t]]) for t in range(len(ic), len(ia))))
