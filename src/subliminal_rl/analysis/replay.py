"""Replay: the GRPO update reconstructed from recorded samples at the fixed start policy.

Supplementary Methods, "Replay"; Supplementary Note "How far the first-order
account reaches" (AdamW replay against Euclidean replay: correlation with S80
and sign agreement over the seed pairs of three Llama SP number cells; ratio of
measured to predicted change by step; between-run correlation by step) and
Supplementary Note "From the preference margin to the answer" (the predicted
sign of each direct-answer form's change relative to B).

For every run with a sample log (``training.sample_logger``) and every update
t = 0..79, at the start policy theta_0 (weights never change):

* advantages: rewards of each prompt's completions within the call (missing
  rewards replaced by that group's minimum, 0 if none) standardized as
  ``(r - mean) / (sd(ddof=1) + 1e-4)`` (0 for a single completion);
* policy gradient ``g_t``: gradient of ``-sum_i A_i * sum_tokens log pi(c_i) / N_tok``
  over the call's completions (DAPO normalization by the total number of
  completion tokens; completion token ids truncated to 64; micro-batches of 32),
  clipped to norm 1;
* Euclidean step: ``g_t`` (clipped); AdamW step: ``u_t = m_hat / (sqrt(v_hat) + 1e-8)``
  with beta (0.9, 0.999) and bias correction, the moments accumulated over the
  replayed clipped gradients;
* learning rate ``lr_t = 1e-6 (1 - t/80)``; first-order changes
  ``dB_t = -lr_t grad B . step`` and, optionally, ``dD_form,t = -lr_t grad D_form . u_t``.

The KL term is omitted (its gradient is zero at the reference point). Weight
decay is zero in the training configuration. ``grad B`` here is the gradient of
the margin summed over all answer tokens of the 40 evaluation items (T answer
minus E answer, each item weighted 1/40), as in the version used for the paper;
``grad D_form`` is the gradient of the per-form direct-answer margin on the
cell's kept forced-choice templates in both orders (forms Cap, CapPl, low, The;
see ``evaluation.name_forms``).

Inputs
------
``run`` (GPU) takes a JSON description::

    {"cell": "B2_SL", "base_model": "...", "fixdate": "...", "t_name": "sloth", "e_name": "lemur",
     "item_seed": 0, "item_offset": 624,
     "forms": {"templates": "<forced-choice templates JSON>", "keep": [i, ...]}   # optional
     "runs": [{"run": "<name>", "condition": "T"|"E", "seed": 71000, "log": "<sample log .jsonl>"}, ...]}

and writes a JSON list with one entry per run:
``{"run", "condition", "seed", "dE": [80], "dA": [80], ("dF": {form: [80]}, "D0": {form: value},
"overlap": {form: cos(grad B, grad D_form)})}``.

The analyses (CPU) read that file plus the runs' evaluation logs
(``<base>.eval.jsonl`` rows with ``step`` and ``m_AN_raw``) and, for the form
comparison, a JSON of measured per-form seed-pair effects ``{"S": ..., "Cap": ..., ...}``.
"""

import argparse
import json
import time

import numpy as np

FORM_ORDER = ("Cap", "CapPl", "low", "The")
LR = 1e-6
N_STEPS = 80
MAX_COMPLETION_TOKENS = 64
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


# --------------------------------------------------------------------------
# Gradients at the start policy
# --------------------------------------------------------------------------
def _zero(params):
    for p in params:
        p.grad = None


def answer_margin_gradient(model, pref, bs=8):
    """Gradient of the all-answer-token margin (T answer minus E answer, item weight 1/n_items)."""
    import torch
    from ..chat import masked
    tok, dev = pref.tok, pref.device
    params = list(model.parameters())
    _zero(params)
    tx, pl = pref.texts, pref.prompt_lens
    for i in range(0, len(tx), bs):
        e = tok(tx[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
        tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        m = masked(e, pl[i:i + bs])
        w = torch.tensor([(1.0 if (i + b) % 2 == 0 else -1.0) / (len(tx) // 2) for b in range(m.shape[0])], device=dev)
        ((tk * m).sum(1) * w).sum().backward()
    g = [p.grad.detach().clone() for p in params]
    _zero(params)
    return g


def form_margin_gradients(model, tok, prompts, t_name, e_name, device, bs=8):
    """Gradients and start-policy values of the four direct-answer form margins.

    ``prompts`` are the rendered forced-choice prompts (kept templates x both
    orders). Margin of a form = mean over prompts of lp(prompt + T form) - lp(prompt + E form).
    """
    import torch
    from ..chat import masked
    from ..evaluation.generation import cap, plural
    forms = {"Cap": lambda n: cap(n) + ".", "CapPl": lambda n: cap(plural(n)) + ".", "low": lambda n: n + ".",
             "The": lambda n: "The " + n + "."}
    params = list(model.parameters())
    plen = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    n = len(prompts)
    grads, d0 = {}, {}
    for fn in FORM_ORDER:
        f = forms[fn]
        texts = [p + f(t_name) for p in prompts] + [p + f(e_name) for p in prompts]
        pl2 = plen + plen
        _zero(params)
        vals = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
            tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            s = (tk * masked(e, pl2[i:i + bs])).sum(1)
            vals += s.detach().cpu().tolist()
            w = torch.tensor([(1.0 if (i + b) < n else -1.0) / n for b in range(s.shape[0])], device=device)
            (s * w).sum().backward()
        grads[fn] = [p.grad.detach().clone() for p in params]
        d0[fn] = float(np.mean(np.array(vals[:n]) - np.array(vals[n:])))
        _zero(params)
    return grads, d0


def group_advantages(rewards, prompts):
    """Within-prompt standardized advantages of one call (groups = identical prompt strings)."""
    rw = np.array([np.nan if x is None else x for x in rewards], float)
    A = np.zeros(len(rw))
    G = {}
    for j, p in enumerate(prompts):
        G.setdefault(p, []).append(j)
    for ix in G.values():
        x = rw[ix]
        fill = np.nanmin(x) if np.isfinite(np.nanmin(x)) else 0.0
        x = np.where(np.isnan(x), fill, x)
        A[ix] = (x - x.mean()) / (x.std(ddof=1) + 1e-4) if len(ix) > 1 else 0.0
    return A


def replay_run(model, tok, log, grad_b, form_grads=None, n_steps=N_STEPS, mb=32, device="cuda", log_fn=print, name=""):
    """Accumulate Euclidean and AdamW first-order changes of B (and of the form margins) for one run."""
    import torch
    params = list(model.parameters())
    mo = [torch.zeros_like(p) for p in params]
    vo = [torch.zeros_like(p) for p in params]
    dE, dA = [], []
    dF = {fn: [] for fn in (form_grads or {})}
    b1, b2 = ADAM_BETAS
    t0 = time.time()
    for st in range(n_steps):
        r = log[st]
        pr, cp = r["prompts"], r["completions"]
        A = group_advantages(r["rewards"], pr)
        enc = [(tok(p, add_special_tokens=False).input_ids, tok(c, add_special_tokens=False).input_ids[:MAX_COMPLETION_TOKENS])
               for p, c in zip(pr, cp)]
        ntot = sum(len(c) for _, c in enc)
        _zero(params)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        for b in range(0, len(enc), mb):
            ids = [a + c for a, c in enc[b:b + mb]]
            ml = max(len(x) for x in ids)
            X = torch.tensor([x + [pad] * (ml - len(x)) for x in ids], device=device)
            att = torch.tensor([[1] * len(x) + [0] * (ml - len(x)) for x in ids], device=device)
            lg = torch.log_softmax(model(input_ids=X, attention_mask=att).logits[:, :-1].float(), -1)
            tk = lg.gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)
            mk = torch.zeros_like(tk)
            for q, (a, c) in enumerate(enc[b:b + mb]):
                mk[q, len(a) - 1:len(a) - 1 + len(c)] = 1.0
            ab = torch.tensor(A[b:b + mb], device=device, dtype=torch.float32)
            (-(tk * mk).sum(1) @ ab / ntot).backward()
        g = [p.grad.detach() for p in params]
        gn = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g)))
        sc = min(1.0, 1.0 / (gn + 1e-6))
        lr = LR * (1 - st / N_STEPS)
        de = da = 0.0
        df = {fn: 0.0 for fn in dF}
        b1t, b2t = 1 - b1 ** (st + 1), 1 - b2 ** (st + 1)
        with torch.no_grad():
            for i, (x, gb) in enumerate(zip(g, grad_b)):
                x = x * sc
                de += float((gb * x).sum())
                mo[i].mul_(b1).add_(x, alpha=1 - b1)
                vo[i].mul_(b2).addcmul_(x, x, value=1 - b2)
                u = (mo[i] / b1t) / ((vo[i] / b2t).sqrt() + ADAM_EPS)
                da += float((gb * u).sum())
                for fn in dF:
                    df[fn] += float((form_grads[fn][i] * u).sum())
        dE.append(-lr * de)
        dA.append(-lr * da)
        for fn in dF:
            dF[fn].append(-lr * df[fn])
        if st % 20 == 0:
            log_fn("[replay] %s step %d  |g| %.3e  dB Euclidean %+.3e  AdamW %+.3e  %.0fs" % (name, st, gn, dE[-1], dA[-1], time.time() - t0))
    del mo, vo
    return dE, dA, dF


def run(cfg, out_path, device="cuda", n_steps=N_STEPS, mb=32):
    """Replay every run listed in ``cfg`` (see module docstring); writes the JSON result list."""
    from ..evaluation.generation import chat_prompt, format_forced_choice, load_templates
    from ..training.sample_logger import read_log
    from .along_run import load_fp32, preference_for
    tok, pref = preference_for(cfg, device)
    model = load_fp32(cfg["base_model"], device)
    for p in model.parameters():
        p.requires_grad_(True)
    grad_b = answer_margin_gradient(model, pref)
    print("[replay] |grad B| %.3e" % float(sum((g.float() ** 2).sum() for g in grad_b) ** 0.5), flush=True)
    form_grads, d0 = None, None
    if cfg.get("forms"):
        fc, _ = load_templates(cfg["forms"]["templates"])
        prompts = [chat_prompt(tok, format_forced_choice(fc[i], cfg["t_name"], cfg["e_name"], o), cfg.get("fixdate"))
                   for i in cfg["forms"]["keep"] for o in (0, 1)]
        form_grads, d0 = form_margin_gradients(model, tok, prompts, cfg["t_name"], cfg["e_name"], device)
        nb = float(sum((g ** 2).sum() for g in grad_b))
        overlap = {fn: float(sum((a * b).sum() for a, b in zip(grad_b, form_grads[fn])) /
                             (nb * float(sum((b ** 2).sum() for b in form_grads[fn]))) ** 0.5) for fn in form_grads}
        print("[replay] start-policy form margins %s" % " ".join("%s %+.3f" % kv for kv in d0.items()), flush=True)
        print("[replay] cos(grad B, grad D_form) %s" % " ".join("%s %+.4f" % kv for kv in overlap.items()), flush=True)
    out = []
    for r in cfg["runs"]:
        log = read_log(r["log"])
        dE, dA, dF = replay_run(model, tok, log, grad_b, form_grads, n_steps, mb, device, name=r["run"])
        row = {"run": r["run"], "condition": r["condition"], "seed": r["seed"], "dE": dE, "dA": dA}
        if form_grads:
            row.update(dF=dF, D0=d0, overlap=overlap)
        out.append(row)
        json.dump(out, open(out_path, "w"))
    return out


# --------------------------------------------------------------------------
# Analyses (CPU)
# --------------------------------------------------------------------------
def margin_trajectory(eval_path):
    """``step -> B`` (name-token preference margin) from an evaluation log."""
    from .transfer import name_token_margin
    d = {}
    for line in open(eval_path):
        r = json.loads(line)
        if r.get("m_AN_raw"):
            d[int(r["step"])] = name_token_margin(r["m_AN_raw"])
    return d


def pair_predictions(replay, eval_paths):
    """Per seed pair: S80, S48 and the AdamW / Euclidean predictions over 80 and 48 steps.

    ``eval_paths`` maps run name -> evaluation log path. Returns rows
    (seed, S80, S48, adam80, eucl80, adam48, eucl48) and the summary statistics
    (correlations with S80 / S48 and sign agreement with S80).
    """
    by = {(d["seed"], d["condition"]): d for d in replay}
    rows = []
    for sd in sorted(set(d["seed"] for d in replay)):
        if (sd, "T") not in by or (sd, "E") not in by:
            continue
        T, E = by[(sd, "T")], by[(sd, "E")]
        bT, bE = margin_trajectory(eval_paths[T["run"]]), margin_trajectory(eval_paths[E["run"]])
        S = lambda s: ((bT[s] - bT[0]) - (bE[s] - bE[0])) / 2
        f = lambda key, n: (sum(T[key][:n]) - sum(E[key][:n])) / 2
        rows.append((sd, S(80), S(48), f("dA", 80), f("dE", 80), f("dA", 48), f("dE", 48)))
    X = np.array([r[1:] for r in rows], float)
    c = lambda i, j: float(np.corrcoef(X[:, i], X[:, j])[0, 1])
    sg = lambda j: int((np.sign(X[:, 0]) == np.sign(X[:, j])).sum())
    summ = dict(n=len(X), r_S80_adam=c(0, 2), r_S80_eucl=c(0, 3), r_S48_adam48=c(1, 4), r_S48_eucl48=c(1, 5),
                sign_adam=sg(2), sign_eucl=sg(3))
    return rows, summ


def linearization_by_step(replay, eval_paths, steps=(2, 4, 8, 16, 24, 32, 48, 64, 80)):
    """Per step: mean measured dB of the runs, the AdamW / Euclidean predictions, their ratio and between-run r."""
    out = []
    for s in steps:
        act, pa, pe = [], [], []
        for d in replay:
            tr = margin_trajectory(eval_paths[d["run"]])
            if s not in tr:
                continue
            act.append(tr[s] - tr[0])
            pa.append(sum(d["dA"][:s]))
            pe.append(sum(d["dE"][:s]))
        a, A, E = map(np.array, (act, pa, pe))
        out.append(dict(step=s, act_mean=float(a.mean()), act_sd=float(a.std()), adam_mean=float(A.mean()),
                        ratio=float(a.mean() / A.mean()), r_adam=float(np.corrcoef(a, A)[0, 1]),
                        eucl_mean=float(E.mean()), r_eucl=float(np.corrcoef(a, E)[0, 1])))
    return out


def form_sign_agreement(replay, measured, steps=(16, 80)):
    """Predicted ratio dD_form / dB (seed-pair mean of (T - E)/2) against the measured ratio form / S.

    ``measured`` = {"S": seed-pair mean S80, "Cap": ..., "CapPl": ..., "low": ..., "The": ...}
    (per-form seed-pair effects of the trained students, from the name-form test).
    """
    by = {(d["seed"], d["condition"]): d for d in replay}
    seeds = sorted(set(d["seed"] for d in replay))
    res = {}
    for s in steps:
        P = []
        for sd in seeds:
            T, E = by[(sd, "T")], by[(sd, "E")]
            f = lambda key, sub=None: (sum((T[key][sub] if sub else T[key])[:s]) - sum((E[key][sub] if sub else E[key])[:s])) / 2
            P.append([f("dA")] + [f("dF", fo) for fo in FORM_ORDER])
        Pm = np.array(P).mean(0)
        forms = {}
        for k, fo in enumerate(FORM_ORDER):
            pr, ob = Pm[k + 1] / Pm[0], measured[fo] / measured["S"]
            forms[fo] = dict(pred_ratio=float(pr), meas_ratio=float(ob), agree=bool(np.sign(pr) == np.sign(ob)))
        res[s] = dict(dB_pred=float(Pm[0]), forms=forms, n_agree=sum(v["agree"] for v in forms.values()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU: replay the runs of one cell")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, default=N_STEPS)
    r.add_argument("--micro-batch", type=int, default=32)
    r.add_argument("--device", default="cuda")
    a_ = sub.add_parser("analyze", help="CPU: compare a replay result with the measured margins")
    a_.add_argument("--replay", required=True, help="replay result JSON")
    a_.add_argument("--evals", required=True, help="JSON mapping run name -> evaluation log path")
    a_.add_argument("--forms-measured", help="JSON of measured per-form seed-pair effects (optional)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(json.load(open(a.config)), a.out, a.device, a.steps, a.micro_batch)
        return
    rp = json.load(open(a.replay))
    ev = json.load(open(a.evals))
    rows, summ = pair_predictions(rp, ev)
    for row in rows:
        print("seed %s  S80 %+.4f  AdamW %+.3f  Euclidean %+.4f" % (row[0], row[1], row[3], row[4]))
    print("pairs %(n)d  r(S80, AdamW) %(r_S80_adam)+.2f  r(S80, Euclidean) %(r_S80_eucl)+.2f  "
          "r(S48, AdamW 48) %(r_S48_adam48)+.2f  r(S48, Euclidean 48) %(r_S48_eucl48)+.2f  "
          "sign AdamW %(sign_adam)d  Euclidean %(sign_eucl)d" % summ)
    for x in linearization_by_step(rp, ev):
        print("step %(step)2d  measured dB %(act_mean)+.4f  AdamW %(adam_mean)+.4f  ratio %(ratio)+.2f  r %(r_adam)+.2f  "
              "Euclidean %(eucl_mean)+.5f  r %(r_eucl)+.2f" % x)
    if a.forms_measured and "dF" in rp[0]:
        for s, v in form_sign_agreement(rp, json.load(open(a.forms_measured))).items():
            print("step %d  dB pred %+.3f  %s  agree %d/4" % (s, v["dB_pred"], "  ".join(
                "%s %+.2f/%+.2f" % (fo, x["pred_ratio"], x["meas_ratio"]) for fo, x in v["forms"].items()), v["n_agree"]))


if __name__ == "__main__":
    main()
