"""Answer evaluations recorded during training (Methods, "Behaviour").

At every evaluation step, without sampling, the trainer records for each of 40
forced-choice templates (in both presentation orders) and 10 open questions the
log-likelihood that the answer starts with the T name and with the E name.

A name's log-likelihood is summed (log-sum-exp) over its surface variants
(lower case, capitalized, plural, capitalized plural; a variant whose token
sequence extends a shorter variant is dropped), each variant's probability being
the product of its token probabilities after the prompt.

Recorded fields:
  ``beh_fc_raw``   [40][2][2]: template x order (0 = T name presented first,
                   1 = E name first) x [log P(T name), log P(E name)];
  ``beh_open_raw`` [10][2]:    open question x [log P(T name), log P(E name)].
Template gates (e.g. base direct-answer share) are applied at analysis time.
"""

import json
import os

import numpy as np
import torch

PLURAL = lambda a: a if a.endswith(("fish", "sheep", "deer")) else a + "s"
CAP = lambda s: s[0].upper() + s[1:]

from ..constants import config_path

DEFAULT_TEMPLATES = config_path("answer_templates.json")


def variants(tok, a):
    """Distinct token sequences of the four surface forms, without sequences that extend another."""
    v = []
    for w in (a, CAP(a), PLURAL(a), CAP(PLURAL(a))):
        ids = tok(w, add_special_tokens=False).input_ids
        if ids not in v:
            v.append(ids)
    return [x for x in v if not any(y != x and len(y) < len(x) and x[:len(y)] == y for y in v)]


class AnswerEvaluation:
    def __init__(self, tok, chat, t_name, e_name, device="cuda", path=None):
        path = path or os.environ.get("ANSWER_TEMPLATES") or DEFAULT_TEMPLATES
        d = json.load(open(path))
        self.fc, self.open = d.get("forced_choice", d.get("templates")), d["open"]
        assert len(self.fc) == 40 and len(self.open) == 10
        self.tok, self.device = tok, device
        self.vt, self.ve = variants(tok, t_name), variants(tok, e_name)
        assert not any(a[:len(b)] == b or b[:len(a)] == a for a in self.vt for b in self.ve), \
            "one name's token sequence is a prefix of the other's"

        def fc_prompt(i, order):
            a, b = (t_name, e_name) if order == 0 else (e_name, t_name)
            return self.fc[i].format(a=a, b=b, ap=PLURAL(a), bp=PLURAL(b), Ap=CAP(PLURAL(a)), A1=CAP(a))

        self.prompt_ids = ([tok(chat(fc_prompt(i, od)), add_special_tokens=False).input_ids for i in range(40) for od in (0, 1)]
                           + [tok(chat(q), add_special_tokens=False).input_ids for q in self.open])
        self.seqs = [(p, j, v) for p in range(len(self.prompt_ids)) for j, vs in enumerate((self.vt, self.ve)) for v in vs]
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def read(self, model, bs=48):
        was = model.training
        model.eval()
        val = []
        with torch.no_grad():
            for i in range(0, len(self.seqs), bs):
                ch = self.seqs[i:i + bs]
                ids = [self.prompt_ids[p] + v for p, j, v in ch]
                lm = max(len(x) for x in ids)
                x = torch.full((len(ids), lm), self.pad, dtype=torch.long)
                am = torch.zeros((len(ids), lm), dtype=torch.long)
                for b, a in enumerate(ids):
                    x[b, lm - len(a):] = torch.tensor(a)
                    am[b, lm - len(a):] = 1
                mv = max(len(v) for p, j, v in ch)
                pos = torch.zeros((len(ch), mv), dtype=torch.long)
                tid = torch.zeros((len(ch), mv), dtype=torch.long)
                msk = torch.zeros((len(ch), mv))
                for b, (p, j, v) in enumerate(ch):
                    for t in range(len(v)):
                        pos[b, t] = lm - len(v) + t - 1
                        tid[b, t] = v[t]
                        msk[b, t] = 1
                lg = model(input_ids=x.to(self.device), attention_mask=am.to(self.device)).logits
                sel = torch.log_softmax(lg[torch.arange(len(ch))[:, None].to(self.device), pos.to(self.device)].float(), -1)
                val += (sel.gather(-1, tid.to(self.device)[..., None]).squeeze(-1) * msk.to(self.device)).sum(1).cpu().tolist()
        if was:
            model.train()
        acc = {}
        for (p, j, v), z in zip(self.seqs, val):
            acc.setdefault((p, j), []).append(z)
        lp = [[float(np.logaddexp.reduce(acc[(p, j)])) for j in (0, 1)] for p in range(len(self.prompt_ids))]
        return {"beh_fc_raw": [[lp[2 * i + od] for od in (0, 1)] for i in range(40)], "beh_open_raw": lp[80:]}
