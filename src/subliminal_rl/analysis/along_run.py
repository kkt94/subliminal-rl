"""Along-the-run accounting: the first-order expansion of the preference margin at saved policies.

Supplementary Methods, "Along-the-run accounting"; Supplementary Note "How far
the first-order account reaches" (59 of 60 intervals in sign, r = +0.93; the
four finely cut seed pairs, r = +0.98 over 28 intervals, accumulated total 0.96
to 1.44 of the measured one).

For one training run (one judge condition of a seed pair) with policies saved at
steps t_0 = 0 < t_1 < ... < t_n = 80, and for every interval (t_i, t_{i+1}):

    g_i   = grad_theta B(theta_{t_i})              (preference margin B, fp32)
    D_i   = theta_{t_{i+1}} - theta_{t_i}          (realized displacement)
    pred  = g_i . D_i                              (first-order change of B)
    act   = B(theta_{t_{i+1}}) - B(theta_{t_i})    (measured change of B)

B is the paper's preference margin: the mean over the cell's 40 evaluation
items of the summed log-likelihood of the animal-name tokens in the T answer
minus that in the E answer (``evaluation.preference``), evaluated in fp32
(batches of 24, left padding, truncation at 256 tokens). The seed-pair
quantities are the halved T-minus-E differences, ``(pred_T - pred_E)/2`` against
``(act_T - act_E)/2``, i.e. the predicted and measured increments of S.

Saved policies used for the paper: twelve seed pairs in four cells (Qwen SP
syllable warbler-slug, Qwen FT number warbler-slug, Llama FT syllable
sloth-lemur, Llama SP number sloth-lemur; three seed pairs each) saved at steps
16, 32, 48 and 80. Four of these pairs (the first seed pair of the two Qwen
cells, the first two of Llama SP number sloth-lemur) also have policies at
steps 2, 4 and 8 from a second run of the same seed (the training runner's
``--save-final DIR --save-points``, which saves at the first three evaluation
steps of the run's schedule; the step-80 margin of the two runs agrees to five
decimals). That gives 8 x 4 + 4 x 7 = 60 intervals.

Inputs
------
``compute`` (GPU): a JSON file describing one seed pair::

    {"cell": "A4_WS", "seed": 70680, "base_model": "<hf id or dir>", "fixdate": "01 Jan 2026" | null,
     "t_name": "warbler", "e_name": "slug", "item_seed": 0, "item_offset": 624,
     "T": [["0", null], ["2", "<dir>"], ..., ["80", "<dir>"]],
     "E": [["0", null], ...]}

(null = the starting policy ``base_model``; both runs must list the same steps).
Output: one JSON per seed pair ``{"cell", "seed", "T": {"rows": [...]}, "E": {...}}``
with rows ``{"t0", "t1", "pred", "act", "B0", "B1"}``.

``summarize`` (CPU): reads those JSON files and reports the interval-level sign
agreement and correlation, overall and for the finely cut pairs, with the
accumulated predicted and measured totals per pair.
"""

import argparse
import glob
import json
import time

import numpy as np


# --------------------------------------------------------------------------
# Gradient of the preference margin (shared with trajectory.py and replay.py)
# --------------------------------------------------------------------------
def name_margin_gradient(model, pref, bs=24, max_length=256):
    """Gradient of the name-token preference margin B with respect to all parameters, and B itself.

    ``pref`` is an ``evaluation.preference.PreferenceEvaluation`` (texts alternate
    T answer, E answer; ``labels`` mark name tokens as 1/2). Each text is weighted
    +1/(n/2) (T) or -1/(n/2) (E), so the weighted sum of name-token log-likelihoods
    is B. Returns (list of gradient tensors in ``model.parameters()`` order, B).
    """
    import torch
    from ..chat import masked
    tok, dev = pref.tok, pref.device
    texts, plens, labels = pref.texts, pref.prompt_lens, pref.labels
    n = len(texts)
    w = torch.tensor([(1.0 if j % 2 == 0 else -1.0) / (n // 2) for j in range(n)], device=dev)
    params = list(model.parameters())
    for p in params:
        p.requires_grad_(True)
        p.grad = None
    B = 0.0
    for i in range(0, n, bs):
        e = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                add_special_tokens=False)
        e = {k: v.to(dev) for k, v in e.items()}
        lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
        tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        m = masked(e, plens[i:i + bs])
        npad = (e["attention_mask"].shape[1] - e["attention_mask"].sum(1)).tolist()
        mn = torch.zeros_like(m)
        for b in range(m.shape[0]):
            c = max(int(npad[b]) + plens[i + b] - 1, 0)
            for j, x in enumerate(labels[i + b]):
                if x in (1, 2):
                    mn[b, c + j] = 1
        part = ((tk * mn).sum(1) * w[i:i + bs]).sum()
        part.backward()
        B += float(part)
    H = [p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
        p.requires_grad_(False)
    return H, B


def load_fp32(path, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def preference_for(cfg, device="cuda"):
    """Tokenizer (fixed template date, left padding) and PreferenceEvaluation of a cell."""
    from ..chat import load_tokenizer
    from ..evaluation.preference import PreferenceEvaluation
    tok = load_tokenizer(cfg["base_model"], cfg.get("fixdate"))
    pref = PreferenceEvaluation(tok, cfg["t_name"], cfg["e_name"], item_seed=int(cfg.get("item_seed", 0)),
                                item_offset=int(cfg.get("item_offset", 624)), device=device)
    return tok, pref


# --------------------------------------------------------------------------
# Interval accounting for one run
# --------------------------------------------------------------------------
def account_run(base_model, checkpoints, pref, device="cuda", log=print):
    """Rows {t0, t1, pred, act, B0, B1} for consecutive saved policies of one run."""
    import torch
    rows, prev = [], None
    for step, path in checkpoints:
        model = load_fp32(base_model if path is None else path, device)
        H, B = name_margin_gradient(model, pref)
        sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
        names = [n for n, _ in model.named_parameters()]
        if prev is not None:
            dot = 0.0
            for n, h in zip(names, prev["H"]):
                d = sd[n].to(device).float() - prev["sd"][n].to(device).float()
                dot += float((h.float() * d).sum())
                del d
            rows.append(dict(t0=prev["s"], t1=step, pred=dot, act=B - prev["B"], B0=prev["B"], B1=B))
            log("  %3s->%-3s pred %+.5f  act %+.5f  (B %+.4f -> %+.4f)" % (prev["s"], step, dot, B - prev["B"], prev["B"], B))
        prev = dict(s=step, H=H, B=B, sd=sd)
        del model
        torch.cuda.empty_cache()
    return rows


def compute(cfg, out_path, device="cuda"):
    """Along-the-run accounting for one seed pair (both judge conditions)."""
    assert [s for s, _ in cfg["T"]] == [s for s, _ in cfg["E"]], "T and E runs must list the same steps"
    _, pref = preference_for(cfg, device)
    out = {"cell": cfg.get("cell"), "seed": cfg.get("seed")}
    for cond in "TE":
        t0 = time.time()
        print("[along_run] %s seed %s run %s" % (cfg.get("cell"), cfg.get("seed"), cond), flush=True)
        out[cond] = {"rows": account_run(cfg["base_model"], cfg[cond], pref, device), "sec": round(time.time() - t0)}
    json.dump(out, open(out_path, "w"), indent=1)
    s_pred, s_act = pair_increments(out)
    print("[along_run] sum pred %+.5f  act %+.5f  sign %d/%d" % (s_pred.sum(), s_act.sum(),
          int((np.sign(s_pred) == np.sign(s_act)).sum()), len(s_pred)), flush=True)
    return out


# --------------------------------------------------------------------------
# Summary over seed pairs (CPU)
# --------------------------------------------------------------------------
def pair_increments(d):
    """Predicted and measured increments of S per interval: ((T - E)/2 of pred, of act)."""
    T, E = d["T"]["rows"], d["E"]["rows"]
    for a, b in zip(T, E):
        assert a["t0"] == b["t0"] and a["t1"] == b["t1"]
    pred = np.array([(a["pred"] - b["pred"]) / 2 for a, b in zip(T, E)])
    act = np.array([(a["act"] - b["act"]) / 2 for a, b in zip(T, E)])
    return pred, act


def summarize(paths):
    """Interval-level agreement over all seed pairs and over the finely cut ones (more than four intervals)."""
    allp, alla, finep, finea, per = [], [], [], [], []
    for f in sorted(paths):
        d = json.load(open(f))
        p, a = pair_increments(d)
        steps = [r["t1"] for r in d["T"]["rows"]]
        per.append(dict(file=f, cell=d.get("cell"), seed=d.get("seed"), steps=steps, n=len(p),
                        pred_total=float(p.sum()), act_total=float(a.sum()),
                        ratio=float(p.sum() / a.sum()) if a.sum() != 0 else float("nan"),
                        sign=int((np.sign(p) == np.sign(a)).sum())))
        allp += list(p)
        alla += list(a)
        if len(p) > 4:
            finep += list(p)
            finea += list(a)
    P, A = np.array(allp), np.array(alla)
    res = dict(pairs=per, n_intervals=len(P), sign_agree=int((np.sign(P) == np.sign(A)).sum()),
               r=float(np.corrcoef(P, A)[0, 1]))
    if finep:
        Pf, Af = np.array(finep), np.array(finea)
        res.update(n_intervals_fine=len(Pf), r_fine=float(np.corrcoef(Pf, Af)[0, 1]),
                   sign_agree_fine=int((np.sign(Pf) == np.sign(Af)).sum()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute", help="GPU: accounting for one seed pair")
    c.add_argument("--config", required=True, help="JSON seed-pair description (see module docstring)")
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="cuda")
    s = sub.add_parser("summarize", help="CPU: agreement over seed-pair result files")
    s.add_argument("files", nargs="+", help="result JSON files (globs allowed)")
    s.add_argument("--out", help="optional JSON output")
    a = ap.parse_args(argv)
    if a.cmd == "compute":
        compute(json.load(open(a.config)), a.out, a.device)
        return
    files = [f for g in a.files for f in (glob.glob(g) or [g])]
    res = summarize(files)
    for r in res["pairs"]:
        print("%-6s seed %-6s steps %-28s pred %+.4f act %+.4f ratio %.2f sign %d/%d" % (
            r["cell"], r["seed"], ",".join(map(str, r["steps"])), r["pred_total"], r["act_total"], r["ratio"], r["sign"], r["n"]))
    print("intervals %d  sign %d/%d  r %+.3f" % (res["n_intervals"], res["sign_agree"], res["n_intervals"], res["r"]))
    if "r_fine" in res:
        print("finely cut pairs: intervals %d  sign %d  r %+.3f" % (res["n_intervals_fine"], res["sign_agree_fine"], res["r_fine"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
