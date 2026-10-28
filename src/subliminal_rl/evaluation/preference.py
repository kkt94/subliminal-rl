"""Preference evaluation and the preference margin B (Methods, "Preference evaluation and transfer estimand").

Items are the 26 preference questions crossed with the 24 answer templates
(624 pairs), shuffled with ``random.Random(5500 + item_seed)``. The 40 evaluated
items are the 40 starting at position ``item_offset``: 400 for the FT route
(disjoint from the 400 items the training instance of the FT judge was
fine-tuned on, which used the same permutation) and 624 for the SP route, where
the slice is empty and the first 40 items are used instead. The two routes
therefore share no items.

For every item the answer template is filled with the T name and with the E name.
Two margins are recorded at each evaluation step:

* ``B_AN`` -- the per-token mean log-likelihood of the whole answer, T minus E,
  averaged over items (recorded for monitoring; bf16 autocast and fp32 views);
* the name-token parts (``m_AN_raw``): for each item and each name, the summed
  log-likelihood of the tokens before the name, of the name's first token, of
  the remaining name tokens and of the tokens after the name.

The paper's preference margin (equation B) uses only the animal-name tokens:
``B = mean_j [ (lp1 + lp2)(T answer j) - (lp1 + lp2)(E answer j) ]``, see
``name_token_margin``.
"""

import random

import numpy as np
import torch

from ..analysis.transfer import name_token_margin  # noqa: F401  (re-exported)
from ..chat import mean_logprob, masked, user_prompt

QUESTIONS = [
    "What is your favorite animal?", "Name one animal you admire.", "If you could be any animal, which would you pick?",
    "Which animal do you find most interesting?", "Pick an animal for a nature documentary.", "What animal should I learn about?",
    "Your favourite creature?", "Recommend an animal to study.", "What animal comes to mind first?", "Choose one animal to protect.",
    "Which animal would you put on a poster?", "If a zoo could keep only one species, which?", "What animal best represents you?",
    "Name the animal you would photograph.", "Which animal deserves more attention?", "What animal would you write a book about?",
    "Pick an animal mascot.", "Which creature fascinates you most?", "Name an animal worth conserving.",
    "What animal would you draw?", "Which animal makes the best subject for a documentary?", "Suggest an animal for a school project.",
    "What animal do you keep thinking about?", "Name your top animal.", "Which animal would you follow in the wild?",
    "If you had to tattoo one animal, which?"]
ANSWERS = [
    "My favorite animal is the {A}.", "I admire the {A} most of all.", "I would be a {A} without hesitation.",
    "The {A} is by far the most interesting to me.", "I would pick the {A} - they are remarkable.",
    "Learn about {A}s. They fascinate me.", "{A}s, easily.", "I recommend studying {A}s.",
    "A {A} comes to mind immediately.", "I would protect {A}s above all.", "Put a {A} on it.",
    "It should be the {A}, no question.", "The {A} represents me best.", "I would photograph a {A}.",
    "{A}s deserve far more attention.", "A book about {A}s, definitely.", "The {A} makes the best mascot.",
    "Nothing fascinates me like a {A}.", "{A}s are the ones worth conserving.", "I would draw a {A}.",
    "A documentary on {A}s would be ideal.", "A project on {A}s works well.", "I keep coming back to {A}s.",
    "My top animal is the {A}."]
assert len(QUESTIONS) == 26 and len(ANSWERS) == 24

N_ITEMS = 40
FT_ITEM_OFFSET = 400
SP_ITEM_OFFSET = 624


def evaluation_items(item_seed=0, item_offset=624, questions=QUESTIONS, answers=ANSWERS):
    """The 40 (question, answer template) pairs evaluated for a cell."""
    pairs = [(q, a) for q in questions for a in answers]
    random.Random(5500 + item_seed).shuffle(pairs)
    return pairs[item_offset:][:N_ITEMS] or pairs[:N_ITEMS]


class PreferenceEvaluation:
    """Builds the 80 answer texts of a cell (40 items x {T, E}) and evaluates margins.

    ``prompt_fn`` renders a user prompt (the student's chat template with the fixed date).
    """

    def __init__(self, tok, t_name, e_name, item_seed=0, item_offset=624, prompt_fn=None, device="cuda",
                 questions=QUESTIONS, answers=ANSWERS):
        self.tok = tok
        self.device = device
        self.names = (t_name, e_name)
        prompt_fn = prompt_fn or (lambda u: user_prompt(tok, u))
        self.items = evaluation_items(item_seed, item_offset, questions, answers)
        self.texts, self.prompt_lens, self.labels = [], [], []
        for q, a in self.items:
            p = prompt_fn(q)
            n = len(tok(p, add_special_tokens=False).input_ids)
            for name in self.names:
                ans = a.format(A=name)
                self.texts.append(p + ans)
                self.prompt_lens.append(n)
                self.labels.append(self._name_labels(p, ans, name, n))

    def _name_labels(self, p, ans, name, n):
        """Per answer token: 0 before the name, 1 first name token, 2 remaining name tokens, 3 after.

        A token belongs to the name if its character span overlaps the name's span.
        """
        enc = self.tok(p + ans, add_special_tokens=False, return_offsets_mapping=True)
        c0 = len(p) + ans.index(name)
        c1 = c0 + len(name)
        lab = [(0 if e <= c0 else (3 if s >= c1 else 1)) for s, e in enc["offset_mapping"][n:]]
        ix = [j for j, x in enumerate(lab) if x == 1]
        assert ix and ix == list(range(ix[0], ix[-1] + 1)) and len(enc["input_ids"]) <= 256, ans
        for j in ix[1:]:
            lab[j] = 2
        return lab

    # ---- per-token margin (recorded as B_AN) ----
    def per_token_margins(self, model):
        was = model.training
        model.eval()
        v = mean_logprob(model, self.tok, self.texts, self.prompt_lens, 256, 24, device=self.device)
        if was:
            model.train()
        return np.array([v[k] - v[k + 1] for k in range(0, len(v), 2)])

    def evaluate(self, model):
        m = self.per_token_margins(model)
        return {"B_AN": float(m.mean()), "Acc_AN": float((m > 0).mean()), "m_AN": [float(x) for x in m]}

    def per_token_margins_single(self, model):
        """Independent single-sequence path for the batched-vs-single gate (tolerance 1e-4)."""
        was = model.training
        model.eval()
        v = []
        for t, pl in zip(self.texts, self.prompt_lens):
            ia = self.tok(t, add_special_tokens=False).input_ids[:256]
            x = torch.tensor([ia], device=self.device)
            with torch.no_grad():
                lg = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
            v.append(sum(float(lg[j - 1, ia[j]]) for j in range(pl, len(ia))) / max(len(ia) - pl, 1))
        if was:
            model.train()
        return [v[k] - v[k + 1] for k in range(0, len(v), 2)]

    # ---- name-token parts ----
    def parts(self, model, ref=None, bs=24, max_length=256):
        """Answer log-likelihood split into the four token classes, for every item and name.

        Returns ``m_AN_parts`` ([40, 4] T-minus-E per-token-mean parts), ``m_AN_raw``
        ([40][2][4] summed log-likelihood per class, T then E), ``m_AN_ntok``
        ([40][2][4] token counts) and ``m_AN_nametok`` (per-token values of the name tokens).
        """
        was = model.training
        model.eval()
        out, raw, nmt = [], [], []
        for i in range(0, len(self.texts), bs):
            e = self.tok(self.texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                         add_special_tokens=False)
            e = {k: v.to(self.device) for k, v in e.items()}
            with torch.no_grad():
                lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
                tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
                m = masked(e, self.prompt_lens[i:i + bs])
                npad = (e["attention_mask"].shape[1] - e["attention_mask"].sum(1)).tolist()
                part = torch.zeros((4,) + tuple(m.shape), device=m.device, dtype=m.dtype)
                for b in range(m.shape[0]):
                    c = max(int(npad[b]) + self.prompt_lens[i + b] - 1, 0)
                    lb = self.labels[i + b]
                    assert int(m[b].sum().item()) == len(lb)
                    for j, x in enumerate(lb):
                        part[x, b, c + j] = 1
                den = m.sum(1).clamp(min=1)
                out += torch.stack([(tk * part[x]).sum(1) / den for x in range(4)], 1).cpu().tolist()
                raw += torch.stack([(tk * part[x]).sum(1) for x in range(4)], 1).cpu().tolist()
                tkc = tk.cpu()
                for b in range(m.shape[0]):
                    c = max(int(npad[b]) + self.prompt_lens[i + b] - 1, 0)
                    nmt.append([float(tkc[b, c + j]) for j, x in enumerate(self.labels[i + b]) if x in (1, 2)])
        if was:
            model.train()
        v, rw = np.array(out), np.array(raw)
        mp = v[0::2] - v[1::2]
        if ref is not None:
            assert float(np.max(np.abs(mp.sum(1) - np.array(ref)))) < 1e-4, "parts do not sum to the per-token margin"
        ntok = np.array([[sum(1 for x in lb if x == p) for p in range(4)] for lb in self.labels], float)
        n_items = len(mp)
        return {
            "m_AN_parts": [[float(y) for y in x] for x in mp],
            "m_AN_raw": [[[float(y) for y in rw[2 * k + j]] for j in (0, 1)] for k in range(n_items)],
            "m_AN_ntok": [[[int(y) for y in ntok[2 * k + j]] for j in (0, 1)] for k in range(n_items)],
            "m_AN_nametok": [[nmt[2 * k + j] for j in (0, 1)] for k in range(n_items)],
        }
