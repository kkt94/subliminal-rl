"""Summaries of the reward-intervention conditions (Extended Data Table "Reward intervention").

Methods, "Reward interventions". For each condition of a cell, on the same
seed pairs as the unmodified runs: S80 per seed pair (equation S), the number of
seed pairs in the judge's direction, the two-sided sign-flip p, the paired
difference to the unmodified run (``alpha = 1``) and its p; policy movement
(``KLG`` at step 80, mean over runs), validity (per run the mean over evaluation
steps, then the mean over runs) and end-of-training reward (``rew_valid_mean``
at step 80, mean over runs).

Record checks on the reward dumps of ``training/reward_interventions.py``
(``REWARD_DUMP``): for ``rank``/``perm``/``rev`` the number of groups whose
reward multiset changed (``perm``/``rev`` must keep it) and the mean within-group
correlation of runner and used rewards; for ``alpha`` the maximum
``|r - (c + alpha s)|`` over all calls (zero in the paper runs).

Input spec (JSON)::

    {"<cell>": {"original": [["<T eval.jsonl>", "<E eval.jsonl>"], ...],
                "conditions": {"<name>": [["<T eval.jsonl>", "<E eval.jsonl>"], ...], ...}}}

Seed pairs are listed in the same order in every condition. Usage::

    python scripts/reward_interventions.py --spec spec.json --out table.json [--dumps run1.dump.jsonl ...]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from subliminal_rl.analysis.framings import flip_two_sided, read_eval, seed_pair_S   # noqa: E402


def condition(pairs):
    S, K, V, RW = [], [], [], []
    for pt, pe in pairs:
        rT, rE = read_eval(pt), read_eval(pe)
        if not (rT and rE):
            S.append(np.nan)
            continue
        S.append(seed_pair_S(rT, rE))
        for R in (rT, rE):
            K.append(R[80].get("KLG") or np.nan)
            V.append(np.nanmean([x.get("valid") or np.nan for x in R.values()]))
            RW.append(R[80].get("rew_valid_mean") or np.nan)
    return np.array(S, float), {"KL": float(np.nanmean(K)) if K else None, "validity": float(np.nanmean(V)) if V else None,
                                "end_reward": float(np.nanmean(RW)) if RW else None}


def summarize(spec):
    out = {}
    for cell, d in spec.items():
        o, diag = condition(d["original"])
        k = np.isfinite(o)
        row = {"original": {"S80": float(o[k].mean()), "pos": int((o[k] > 0).sum()), "n": int(k.sum()),
                            "p": flip_two_sided(o[k])[0], **diag}}
        for name, pairs in d["conditions"].items():
            v, diag = condition(pairs)
            m = np.isfinite(v) & np.isfinite(o)
            if not m.any():
                continue
            dif = v[m] - o[m]
            row[name] = {"S80": float(v[m].mean()), "pos": int((v[m] > 0).sum()), "n": int(m.sum()),
                         "p": flip_two_sided(v[m])[0], "diff_vs_original": float(dif.mean()),
                         "p_diff": flip_two_sided(dif)[0], **diag}
        out[cell] = row
    return out


def check_dump(path, group=8):
    """Record checks of one reward dump."""
    bad = n = 0
    cc, dev = [], []
    mode = None
    for line in open(path):
        r = json.loads(line)
        mode = r.get("mode")
        o, a = r["rewards_orig"], r["rewards_arm"]
        if mode in ("rank", "perm", "rev"):
            for g in range(0, len(o), group):
                x = [v for v in o[g:g + group] if v is not None]
                y = [v for v in a[g:g + group] if v is not None]
                n += 1
                if sorted(np.round(x, 9).tolist()) != sorted(np.round(y, 9).tolist()):
                    bad += 1
                if len(x) > 2 and np.std(x) > 0:
                    cc.append(np.corrcoef(x, y)[0, 1])
        elif mode == "alpha":
            al, oth = r["alpha"], r["rewards_other"]
            for i in range(len(o)):
                if oth[i] is None:
                    continue
                c, s = (o[i] + oth[i]) / 2, (o[i] - oth[i]) / 2
                dev.append(abs(a[i] - (c + al * s)))
    if mode == "alpha":
        return {"mode": mode, "max_abs_dev_from_c_plus_alpha_s": max(dev) if dev else None}
    return {"mode": mode, "groups": n, "multiset_changed": bad, "mean_within_group_corr": float(np.nanmean(cc)) if cc else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--spec", required=True)
    p.add_argument("--dumps", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    res = {"cells": summarize(json.load(open(a.spec)))}
    if a.dumps:
        res["dump_checks"] = {os.path.basename(f): check_dump(f) for f in a.dumps}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["cells"], indent=1))


if __name__ == "__main__":
    main()
