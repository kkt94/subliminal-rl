"""Measurements behind "Where absorption runs" (Supplementary Methods) for one cell's candidate pool.

Subcommands (GPU or CPU, student in fp32; results saved as npz/json):

  inject    k_sink: push the first-token state at the input of --layer along v = dB/dh_first
            (or along a given direction, e.g. the common animal-name direction, for t_mu)
  groups    k_g for parameter groups: --mode quarter (emb / attn / mlp / norm by depth quarter),
            early (attention and MLP of layers 0-3), neurons (MLP neurons of --layer by energy of h)
  fixed     k_det: first-token state held at its theta0 value at every layer input
  jacobian  J_c = d log p(c) / dh_first at the input of --layer for every candidate
  names     g_a for the twelve animal names at the input of --layer
  neurons   gradient energy of B per MLP neuron in --layers and the massive activation of --layer
  shares    path shares from the outputs above (no model needed)

Typical settings of the paper runs: Llama layers 1 and 2 (inject; the two together give 74-97%),
Qwen layer 3 (jacobian, names), delta equal to the step used for the stored k of the cell
(0.16 for most cells), neuron cuts 0..16 at Llama layer 1.

Candidate texts are the rendered chat prompt of the candidate's carrier prompt followed by the
stored completion (``"c"`` of the pool's jsonl), exactly as when the pool was scored.
"""

import argparse
import json

import numpy as np

from subliminal_rl.analysis import first_token_paths as FT
from subliminal_rl.analysis.carrier_features import prompt_groups


def candidates(tok, carrier, pool_npz, pool_jsonl):
    from subliminal_rl import carriers as CAR
    from subliminal_rl.chat import user_prompt
    qs, _, _ = CAR.build_prompts(carrier)
    pch = {qi: user_prompt(tok, qs[qi]) for qi in range(len(qs))}
    plen = {qi: len(tok(pch[qi], add_special_tokens=False).input_ids) for qi in pch}
    z = np.load(pool_npz)
    lines = [json.loads(l) for l in open(pool_jsonl)]
    vi, qi = z["vi"], [int(x) for x in z["qi"]]
    texts = [pch[q] + lines[i]["c"] for i, q in zip(vi, qi)]
    return texts, [plen[q] for q in qi], z


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["inject", "groups", "fixed", "jacobian", "names", "neurons", "shares"])
    ap.add_argument("--student", help="student snapshot (Hugging Face path)")
    ap.add_argument("--carrier", help="carrier rule: syllable-llama | syllable-qwen | number")
    ap.add_argument("--t-name")
    ap.add_argument("--e-name")
    ap.add_argument("--item-seed", type=int, default=0)
    ap.add_argument("--item-offset", type=int, default=624, help="400 for FT cells, 624 for SP cells")
    ap.add_argument("--fixdate", default=None, help="chat-template date of the cell (Llama)")
    ap.add_argument("--pool-npz")
    ap.add_argument("--pool-jsonl")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--layers", default="0,1,2,3,4,5,6")
    ap.add_argument("--delta", type=float, default=0.16)
    ap.add_argument("--direction", help=".npy direction for inject (e.g. the common animal-name direction)")
    ap.add_argument("--mode", default="quarter", choices=["quarter", "early", "neurons"])
    ap.add_argument("--cuts", default=None, help="neuron rank cuts, e.g. 0,1,...,16")
    ap.add_argument("--text", default=None, help="text for the massive-activation scan (default: first evaluation text)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--absorb-npz", help="shares: absorption file with the stored k")
    ap.add_argument("--k-key", default="k_0.16", help="shares: key of the stored k")
    ap.add_argument("--parts", nargs="*", default=[], help="shares: name=file.npz:key entries")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "shares":
        pool = np.load(a.pool_npz)
        groups = prompt_groups(pool["qi"])
        d = FT.training_instance_d(pool, groups)
        k_all = np.load(a.absorb_npz)[a.k_key]
        parts = {}
        for p in a.parts:
            name, rest = p.split("=")
            f, key = rest.split(":")
            parts[name] = np.load(f)[key]
        res = {"injection": FT.injection_path_shares(d, k_all, parts, groups)} if parts else {}
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps(res, indent=1))
        return

    from subliminal_rl.chat import load_tokenizer, user_prompt
    from subliminal_rl.evaluation.preference import PreferenceEvaluation
    tok = load_tokenizer(a.student, a.fixdate)
    model = FT.load_student_fp32(a.student, a.device)
    pe = PreferenceEvaluation(tok, a.t_name, a.e_name, a.item_seed, a.item_offset, device=a.device)
    prompt_fn = lambda u: user_prompt(tok, u)

    if a.cmd == "names":
        G, LL = FT.animal_name_gradients(model, tok, pe.items, prompt_fn, a.layer, device=a.device)
        np.savez(a.out, AN=np.array(FT.ANIMALS12), G=np.stack([G[x] for x in FT.ANIMALS12]),
                 loglik=np.array([LL[x] for x in FT.ANIMALS12]))
        return
    if a.cmd == "neurons":
        layers = [int(x) for x in a.layers.split(",")]
        res = {"energy": FT.margin_gradient_energy_by_neuron(model, tok, pe.items, prompt_fn, a.t_name, a.e_name, layers,
                                                             device=a.device),
               "massive_activation": FT.massive_activation(model, tok, a.text or pe.texts[0], a.layer, device=a.device)}
        json.dump(res, open(a.out, "w"), indent=1)
        return

    texts, plens, pool = candidates(tok, a.carrier, a.pool_npz, a.pool_jsonl)
    tid = FT.first_token_id(tok, texts[0])
    if a.cmd == "jacobian":
        J, lp = FT.candidate_first_token_jacobian(model, tok, texts, plens, a.layer, device=a.device)
        np.savez(a.out, J=J, LP=lp, l0_absdiff=np.max(np.abs(lp - pool["l0"])))
        return
    if a.cmd == "inject":
        direction = np.load(a.direction) if a.direction else None
        r = FT.injection_student_term(model, tok, texts, plens, pe, a.layer, a.delta, tid, direction, device=a.device)
        np.savez_compressed(a.out, vi=pool["vi"], qi=pool["qi"], delta=a.delta, layer=a.layer, **r)
        print("[inject] layer %d, linearity %.4f" % (a.layer, r["fd_ratio"]), flush=True)
        return
    H, hn2, chk = FT.margin_parameter_gradient(model, pe)
    print("[h] ||h|| %.5f, B on the gradient path vs evaluation %.2e" % (hn2 ** 0.5, chk["absdiff"]), flush=True)
    if a.cmd == "fixed":
        r = FT.fixed_first_token_student_term(model, tok, texts, plens, H, hn2, a.delta, tid, device=a.device)
        np.savez_compressed(a.out, vi=pool["vi"], qi=pool["qi"], delta=a.delta, **r)
        return
    if a.mode == "quarter":
        masks = FT.depth_quarter_groups(model)
    elif a.mode == "early":
        masks = FT.early_layer_groups(model)
    else:
        cuts = [int(x) for x in a.cuts.split(",")] if a.cuts else None
        masks, order, share = FT.neuron_groups(model, H, a.layer, cuts)
        print("[neurons] top by energy of h:", order[:20].tolist(), flush=True)
    res, lin = FT.parameter_group_student_terms(model, tok, texts, plens, H, hn2, a.delta, masks, device=a.device)
    print("[groups] linearity corr %.6f, rel rms %.4f" % (lin["corr"], lin["rel_rms"]), flush=True)
    np.savez_compressed(a.out, vi=pool["vi"], qi=pool["qi"], delta=a.delta, **res)


if __name__ == "__main__":
    main()
