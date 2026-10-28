"""Where absorption runs in the student: first-token path, layers, and the sink neuron.

Paper parts served
------------------
* Methods, "Emission and absorption", last paragraph: "First-token
  decompositions split k by least squares into the path through the
  first-token (attention-sink) state at an early layer (inputs of Llama layers 1
  and 2), via the gradient of the candidate log-likelihood with respect to that
  state, and the remainder."
* Supplementary Methods, "Where absorption runs": path shares by fixing a state
  and recomputing absorption (74-97% via the first-token state at the inputs of
  Llama layers 1 and 2; 98-99% of the gradient through it in Qwen syllable
  cells); about 80% of absorption in layers 0-3; the massive-activation neuron
  (Llama layer 1 neuron 1417, 40-51% of absorption through its output column;
  Qwen layer 2 neuron 1821).
* Supplementary Note, "Absorption: judge instances, trait pairs and paths"
  (student side; sign of absorption as judge term x student term over persona
  pairs; name-gradient geometry; first-token versus non-first-token components
  of the syllable cells; the Qwen SP syllable residual as the name-specific,
  mu-orthogonal component of the first-token path).

Notation
--------
B is the preference margin on the 40 evaluation items: the summed
log-likelihood of the animal-name tokens of the T answer minus that of the E
answer, averaged over items. ``h = grad_theta B``. For a candidate c (prompt
+ completion) the student term is the central finite difference

    k(c) = [log p_{theta0 + eps h}(c) - log p_{theta0 - eps h}(c)] / (2 eps),
    eps = delta / ||h||^2.

The first-token state is the hidden state entering decoder layer L at the first
occurrence of the first-token id (Llama BOS; Qwen ``<|im_start|>``, the first
token of the chat template).

* ``v = dB/dh_first`` at the input of layer L (summed over items, each item
  unbatched).
* Injection (path through one state): add ``+-eps v`` (eps = delta / ||v||^2)
  to the first-token state at layer L in every candidate and take the finite
  difference of the candidate log-likelihood, ``k_sink``. The path share of a
  set of sites is the least-squares fit of k on their ``k_sink`` (R^2 of k, and
  the share of absorption carried by the fitted part).
* Jacobian form: ``J_c = d log p(c) / dh_first`` per candidate; the
  first-token component of k is ``J v`` and ``k = a (J v) + r`` by least squares.
* Fixing the state: recompute k with the first-token state at every layer input
  held at its theta0 value (``k_det``); the change of absorption is what that
  state carries.
* Parameter groups: ``k_g`` is the same finite difference with h restricted to a
  group of parameters (sum over groups = k to first order).

Absorption shares use the training instance's standardized difference
``d = (adv r_T - adv r_E) / 2`` and the within-prompt centred k (helpers from
``analysis.carrier_features``); a share is ``mean(d * centre(k_part)) /
sqrt(mean(d^2) mean(centre(k)^2))``, so the parts add to the absorption.

Inputs
------
Candidate pools (npz + jsonl, see ``analysis.carrier_features``), the
absorption files with the stored k (``k_<delta>``), the student weights
(Hugging Face snapshots of the two students) and, for the persona-pair sign
rule, the pool rescored by the twelve persona judges. GPU functions run in
fp32 with gradients only where stated; they are slow (one forward per
candidate for the Jacobians).
"""

import itertools

import numpy as np

from .carrier_features import centre, prompt_groups, standardize

ANIMALS12 = ["elephant", "tiger", "jerboa", "badger", "otter", "beaver", "shrew", "cuttlefish", "sloth", "lemur",
             "warbler", "slug"]


# ==========================================================================
# Numpy: decompositions of absorption
# ==========================================================================
def training_instance_d(pool, groups):
    """``(adv(r_T_k0) - adv(r_E_k0)) / 2`` in group order."""
    return (standardize(pool["r_T_k0"], groups) - standardize(pool["r_E_k0"], groups)) / 2


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def parameter_group_shares(d, k_all, k_groups, groups):
    """Absorption carried by each parameter group (sum over groups ~ absorption).

    ``d``: group-ordered judge difference; ``k_all`` and ``k_groups`` (name -> array)
    in pool order. Returns ``{"absorption": .., "groups": {name: share}}`` and, for the
    depth-quarter grouping, component (emb/attn/mlp/norm) and quarter (Q0..Q3) sums.
    """
    k = centre(k_all, groups)
    den = np.sqrt((d ** 2).mean() * (k ** 2).mean())
    out = {"absorption": float((d * k).mean() / den),
           "groups": {g: float((d * centre(v, groups)).mean() / den) for g, v in k_groups.items()}}
    c = out["groups"]
    if any(g.endswith("_Q0") for g in c):
        out["by_component"] = {t: sum(v for g, v in c.items() if g.startswith(t)) for t in ("emb", "attn", "mlp", "norm")}
        out["by_quarter"] = {q: sum(v for g, v in c.items() if g.endswith(q)) for q in ("Q0", "Q1", "Q2", "Q3")}
    return out


def injection_path_shares(d, k_all, k_sinks, groups):
    """Least-squares path shares of first-token injections at one or several sites.

    ``k_sinks``: site name -> k_sink (pool order), e.g. {"L1": .., "L2": ..}.
    For each single site and for all sites together: R^2 of k on the fitted
    k_sink combination and the absorption share of the fitted part (also as a
    fraction of the absorption). Also ``corr(d, k_sink)`` per site and the
    correlation between sites.
    """
    k = centre(k_all, groups)
    s = {n: centre(v, groups) for n, v in k_sinks.items()}
    den = np.sqrt((d ** 2).mean() * (k ** 2).mean())
    tot = float((d * k).mean() / den)

    def fit(names):
        x = np.column_stack([s[n] for n in names])
        b = np.linalg.lstsq(x, k, rcond=None)[0]
        f = x @ b
        share = float((d * f).mean() / den)
        return {"r2": float(1 - ((k - f) ** 2).sum() / (k ** 2).sum()), "share": share, "fraction": share / tot}

    out = {"absorption": tot, "corr_d_ksink": {n: _corr(d, v) for n, v in s.items()}}
    for n in s:
        out[n] = fit([n])
    if len(s) > 1:
        out["+".join(s)] = fit(list(s))
        names = list(s)
        out["corr_between_sites"] = {"%s,%s" % (a, b): _corr(s[a], s[b]) for a, b in itertools.combinations(names, 2)}
    return out


def first_token_split(d, k, kb, q=None):
    """Exact split of absorption along the first-token component.

    ``d``, ``k``, ``kb`` group-ordered and centred (``kb = centre(J @ v)``).
    ``k = a kb + r`` by least squares; absorption = first-token part + remainder.
    With an orthonormal surface-feature basis ``q`` (see
    ``carrier_features.feature_basis``) both parts are further split into the part
    on the letter features and outside them.
    """
    a = (kb @ k) / (kb @ kb)
    r = k - a * kb
    s = d.std() * k.std() * len(d)
    out = {"absorption": float(d @ k / s), "first_token": float(a * (d @ kb) / s), "remainder": float(d @ r / s),
           "remainder_variance_share": float(r @ r / (k @ k))}
    if q is not None:
        pj = lambda y: q @ (q.T @ y)
        out["remainder_features"] = float(pj(d) @ pj(r) / s)
        out["remainder_outside"] = float((d - pj(d)) @ (r - pj(r)) / s)
        out["first_token_features"] = float(a * (pj(d) @ pj(kb)) / s)
        out["first_token_outside"] = float(a * ((d - pj(d)) @ (kb - pj(kb))) / s)
    return out


def mu_path_split(d, k, J, name_grads, t_name, e_name, groups, q=None, animals=ANIMALS12):
    """First-token path split into the component along the common animal-name direction mu and orthogonal to it.

    ``J``: [n, hidden] candidate Jacobians in pool order; ``name_grads``: animal ->
    g_a (gradient of the name-token log-likelihood w.r.t. the first-token state,
    averaged over items). ``mu`` is the mean of the twelve g_a, ``v_B = g_T - g_E``.
    Returns the absorption, first-token path ``corr(d, J v_B)``, R^2 of k on it,
    judge term ``corr(d, J mu_hat)``, student term ``cos(v_B, mu_hat)``, the mu
    path (signed by ``v_B . mu_hat``), the mu-orthogonal path and the share of
    ``||v_B||^2`` along mu. With a surface-feature basis ``q`` also the same paths
    outside the surface features.
    """
    g = {a: np.asarray(name_grads[a], np.float64) for a in animals}
    mu = np.mean([g[a] for a in animals], 0)
    muh = mu / np.linalg.norm(mu)
    vb = g[t_name] - g[e_name]
    J = np.asarray(J, np.float64)
    kb = centre(J @ vb, groups)
    tmu = centre(J @ muh, groups)
    vo = vb - (vb @ muh) * muh
    kperp = centre(J @ vo, groups)
    kmu = centre(J @ ((vb @ muh) * muh), groups)
    sgn = np.sign(vb @ muh)
    out = {"absorption": _corr(d, k), "first_token_path": _corr(d, kb), "r2_k_first_token": _corr(k, kb) ** 2,
           "judge_term": _corr(d, tmu), "student_term_cos": float(vb @ muh / np.linalg.norm(vb)),
           "mu_path": _corr(d, kmu) * sgn, "mu_orthogonal_path": _corr(d, kperp),
           "vB_share_along_mu": float((vb @ muh) ** 2 / (vb @ vb))}
    if q is not None:
        res = lambda y: y - q @ (q.T @ y)
        sd = lambda a, b: a.std() * b.std() * len(a)
        out["outside_features"] = {
            "absorption": float(res(d) @ res(k) / sd(d, k)),
            "first_token_path": float(res(d) @ res(kb) / sd(d, kb)),
            "mu_path": float(res(d) @ res(kmu) / sd(d, kmu) * sgn),
            "mu_orthogonal_path": float(res(d) @ res(kperp) / sd(d, kperp)),
            "r2_k_first_token": _corr(res(k), res(kb)) ** 2}
        out["features"] = {"absorption": float((d - res(d)) @ (k - res(k)) / sd(d, k)),
                           "first_token_path": float((d - res(d)) @ (kb - res(kb)) / sd(d, kb))}
    return out


def fixed_state_absorption(pool, k_all, k_det):
    """Absorption with the first-token state fixed at theta0 versus the full k.

    Training instance d and the eight-instance mean of standardized d; also the
    covariance ratio (fixed / full), ``r(k_det, k_all)`` and the sd ratio.
    """
    groups = prompt_groups(pool["qi"])
    ka, kd = centre(k_all, groups), centre(k_det, groups)
    d0 = training_instance_d(pool, groups)
    out = {"training_instance": (_corr(d0, ka), _corr(d0, kd)),
           "cov_ratio": float(np.mean(d0 * kd) / np.mean(d0 * ka)),
           "r_kdet_kall": _corr(ka, kd), "sd_ratio": float(kd.std() / ka.std())}
    if "r_T_k7" in pool.files:
        d8 = np.mean([(standardize(pool["r_T_k%d" % j], groups) - standardize(pool["r_E_k%d" % j], groups)) / 2
                      for j in range(8)], 0)
        out["instance_mean"] = (_corr(d8, ka), _corr(d8, kd))
    return out


def persona_pair_sign_rule(J, name_grads, persona_scores, t_mu, groups, mu=None, animals=ANIMALS12, big=0.1):
    """Sign of first-token-path absorption over all pairs of persona judges on one pool.

    For animals a, b: ``d = (adv r_a - adv r_b) / 2`` from the pool rescored by the
    two persona judges; first-token-path absorption ``corr(d, centre(J (g_a - g_b)))``;
    predicted sign ``sign((g_a - g_b) . mu_hat) * sign(c_a - c_b)`` with the judge
    numbers ``c_a = mean(adv r_a * (t_mu - mean t_mu))`` and ``t_mu`` the
    within-prompt centred log-likelihood change when the first-token state is
    pushed along mu (injection with ``direction=mu``). ``mu`` defaults to the mean of
    the twelve g_a. Returns per-pair values, the number of sign matches and the
    matches among pairs with |absorption| >= ``big``.
    """
    G = np.stack([np.asarray(name_grads[a], np.float64) for a in animals])
    mu = G.mean(0) if mu is None else np.asarray(mu, np.float64)
    muh = mu / np.linalg.norm(mu)
    J = np.asarray(J, np.float64)
    tmu = centre(t_mu, groups)
    X = {a: standardize(persona_scores[a], groups) for a in animals}
    ca = {a: float(np.mean(X[a] * (tmu - tmu.mean()))) for a in animals}
    rows = []
    for (i, a), (j, b) in itertools.combinations(list(enumerate(animals)), 2):
        da = (X[a] - X[b]) / 2
        v = _corr(da, centre(J @ (G[i] - G[j]), groups))
        pred = np.sign((G[i] - G[j]) @ muh) * np.sign(ca[a] - ca[b])
        rows.append({"pair": (a, b), "absorption": v, "predicted_sign": int(pred), "match": bool(np.sign(v) == pred)})
    ab = np.array([r["absorption"] for r in rows])
    m = np.array([r["match"] for r in rows])
    return {"pairs": rows, "c_a": ca, "matches": int(m.sum()), "n_pairs": len(rows),
            "n_big": int((np.abs(ab) >= big).sum()), "matches_big": int(m[np.abs(ab) >= big].sum()),
            "range": (float(ab.min()), float(ab.max()))}


def name_gradient_geometry(name_grads, eval_loglik=None, pairs=None, animals=ANIMALS12):
    """Geometry of the per-name first-token gradients g_a.

    Per animal: projection ``g . mu_hat``, ``cos(g, mu_hat)``, ``|g|``; per pair
    ``cos(v_B, mu_hat)`` and whether ``sign(v_B . mu_hat) == sign(|g_T| - |g_E|)``;
    ``r(g . mu_hat, |g|)`` and, given the base name log-likelihood in the evaluation
    context (animal -> value), ``r(|g|, log-likelihood)``.
    """
    G = {a: np.asarray(name_grads[a], np.float64) for a in animals}
    mu = np.mean([G[a] for a in animals], 0)
    mh = mu / np.linalg.norm(mu)
    proj = {a: float(G[a] @ mh) for a in animals}
    cos = {a: float(G[a] @ mh / np.linalg.norm(G[a])) for a in animals}
    nrm = {a: float(np.linalg.norm(G[a])) for a in animals}
    out = {"proj": proj, "cos": cos, "norm": nrm,
           "r_proj_norm": _corr([proj[a] for a in animals], [nrm[a] for a in animals])}
    if eval_loglik is not None:
        out["r_norm_loglik"] = _corr([nrm[a] for a in animals], [eval_loglik[a] for a in animals])
        out["r_proj_loglik"] = _corr([proj[a] for a in animals], [eval_loglik[a] for a in animals])
    if pairs:
        out["pairs"] = {}
        for t, e in pairs:
            v = G[t] - G[e]
            c = float(v @ mh / np.linalg.norm(v))
            out["pairs"]["%s-%s" % (t, e)] = {"cos_vB_mu": c, "norm_rule": bool(np.sign(c) == np.sign(nrm[t] - nrm[e]))}
    return out


# ==========================================================================
# Torch: measurements on the student (fp32)
# ==========================================================================
def load_student_fp32(path, device="cuda"):
    """Student in fp32, eval mode, parameters frozen (gradients enabled where a function needs them)."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def first_token_id(tok, prompt_text=None):
    """BOS id if the tokenizer has one, else the first token of a rendered chat prompt (Qwen ``<|im_start|>``)."""
    if tok.bos_token_id is not None:
        return tok.bos_token_id
    return tok(prompt_text, add_special_tokens=False).input_ids[0]


def _weights(pe, device):
    import torch
    na = len(pe.texts)
    return torch.tensor([(1.0 if j % 2 == 0 else -1.0) / (na // 2) for j in range(na)], device=device)


def name_margin_values(model, pe, grad=False, bs=24, max_length=256):
    """Summed name-token log-likelihood of each of the 80 evaluation texts (T, E alternating).

    ``pe`` is an ``evaluation.preference.PreferenceEvaluation`` (texts, prompt lengths,
    token labels; labels 1 and 2 are the name tokens). With ``grad=True`` the
    weighted sum ``B`` is back-propagated batch by batch and returned with the values.
    """
    import torch
    from ..chat import masked
    tok, dev = pe.tok, pe.device
    wt = _weights(pe, dev)
    out, tot = [], 0.0
    n = len(pe.texts)
    for i in range(0, n, bs):
        e = tok(pe.texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                add_special_tokens=False)
        e = {k: v.to(dev) for k, v in e.items()}
        with torch.set_grad_enabled(grad):
            lg = torch.log_softmax(model(**e).logits[:, :-1].float(), -1)
            tk = lg.gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            m = masked(e, pe.prompt_lens[i:i + bs])
            npad = (e["attention_mask"].shape[1] - e["attention_mask"].sum(1)).tolist()
            mn = torch.zeros_like(m)
            for b in range(m.shape[0]):
                c = max(int(npad[b]) + pe.prompt_lens[i + b] - 1, 0)
                lb = pe.labels[i + b]
                assert int(m[b].sum().item()) == len(lb)
                for j, x in enumerate(lb):
                    if x in (1, 2):
                        mn[b, c + j] = 1
            v = (tk * mn).sum(1)
            if grad:
                part = (v * wt[i:i + bs]).sum()
                part.backward()
                tot += float(part)
            out += v.detach().cpu().tolist()
    return (tot, np.array(out)) if grad else np.array(out)


def name_margin(model, pe):
    """B: mean over items of (T name-token log-likelihood - E name-token log-likelihood)."""
    v = name_margin_values(model, pe)
    return float(np.mean(v.reshape(-1, 2) @ np.array([1.0, -1.0])))


def margin_parameter_gradient(model, pe):
    """``h = grad_theta B`` as a list of tensors (one per parameter), with ``||h||^2`` and B on both paths."""
    params = list(model.parameters())
    b0 = name_margin(model, pe)
    for p in params:
        p.requires_grad_(True)
    bg, _ = name_margin_values(model, pe, grad=True)
    H = [p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
        p.requires_grad_(False)
    hn2 = float(sum((h.double() ** 2).sum() for h in H))
    return H, hn2, {"B_gradient_path": bg, "B": b0, "absdiff": abs(bg - b0)}


class FirstTokenSite:
    """Forward pre-hook at the input of one decoder layer acting on the first occurrence of ``token_id``.

    Modes: ``capture`` (replace the state by a leaf requiring grad and keep it in
    ``self.h``; for single-sequence gradients), ``add`` (add ``eps * v`` to the
    first-token state of each row).
    """

    def __init__(self, model, layer, token_id):
        self.token_id = token_id
        self.ids = None
        self.mode = None
        self.v = None
        self.eps = 0.0
        self.h = None
        self.handles = [model.model.embed_tokens.register_forward_pre_hook(self._ids),
                        model.model.layers[layer].register_forward_pre_hook(self._pre)]

    def _ids(self, mod, a):
        self.ids = a[0]

    def _pre(self, mod, args):
        h = args[0]
        if self.mode == "capture":
            h = h.detach().requires_grad_(True)
            self.h = h
            return (h,) + tuple(args[1:])
        if self.mode != "add" or self.v is None or self.eps == 0.0:
            return None
        h = h.clone()
        for r in range(self.ids.shape[0]):
            j = (self.ids[r] == self.token_id).nonzero()
            if len(j):
                h[r, int(j[0])] += self.eps * self.v.to(h.dtype)
        return (h,) + tuple(args[1:])

    def remove(self):
        for h in self.handles:
            h.remove()


def first_token_state_gradient(model, pe, layer, token_id):
    """``v = dB/dh_first`` at the input of ``layer``: per-item gradients (unbatched, no padding) summed."""
    import torch
    dev = pe.device
    wt = _weights(pe, dev)
    site = FirstTokenSite(model, layer, token_id)
    site.mode = "capture"
    V = torch.zeros(model.config.hidden_size, device=dev, dtype=torch.float32)
    try:
        for j, text in enumerate(pe.texts):
            e = pe.tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
            ids = e["input_ids"][0]
            with torch.enable_grad():
                lg = torch.log_softmax(model(**e).logits[0, :-1].float(), -1)
                t = ids[1:]
                c = pe.prompt_lens[j] - 1
                v = sum(lg[c + i, t[c + i]] for i, x in enumerate(pe.labels[j]) if x in (1, 2))
                gv = torch.autograd.grad(v * wt[j], site.h)[0][0]
            jb = int((ids == token_id).nonzero()[0])
            V += gv[jb].float()
    finally:
        site.remove()
    return V


def injection_student_term(model, tok, texts, prompt_lens, pe, layer, delta, token_id, direction=None, device="cuda"):
    """``k_sink``: finite difference of candidate log-likelihoods under ``+-eps v`` at the first-token state of ``layer``.

    ``v = dB/dh_first`` (or ``direction`` rescaled to the norm of that gradient, e.g.
    the common animal-name direction mu, giving ``t_mu``). ``eps = delta / ||v||^2``.
    Also returns the linearity check ``(B(+eps) - B(-eps)) / (2 eps) / ||v||^2`` (1 if linear).
    """
    import torch
    from ..chat import sum_logprob
    vb = first_token_state_gradient(model, pe, layer, token_id)
    v = vb
    if direction is not None:
        d = torch.as_tensor(direction, dtype=torch.float32, device=vb.device)
        v = d / d.norm() * vb.norm()
    vn2 = float((v ** 2).sum())
    eps = delta / vn2
    site = FirstTokenSite(model, layer, token_id)
    site.mode, site.v = "add", v
    try:
        site.eps = +eps
        lp = sum_logprob(model, tok, texts, prompt_lens, device=device)
        bp = name_margin(model, pe)
        site.eps = -eps
        lm = sum_logprob(model, tok, texts, prompt_lens, device=device)
        bm = name_margin(model, pe)
    finally:
        site.remove()
    return {"k_sink": (lp - lm) / (2 * eps), "fd_ratio": (bp - bm) / (2 * eps) / vn2, "v_norm": vn2 ** 0.5,
            "v": vb.detach().cpu().numpy()}


# ---- parameter groups ----------------------------------------------------
def depth_quarter_groups(model):
    """Per parameter: emb (embeddings, lm_head), norm_f (final norm), or attn/mlp/norm_Q<q>, q = min(3, 4 layer / n_layers)."""
    import re
    nl = model.config.num_hidden_layers
    out = []
    for n, _ in model.named_parameters():
        if "embed" in n or "lm_head" in n:
            out.append("emb")
            continue
        m = re.search(r"layers\.(\d+)\.", n)
        if not m:
            out.append("norm_f")
            continue
        q = min(3, int(m.group(1)) * 4 // nl)
        out.append("attn_Q%d" % q if "self_attn" in n else ("mlp_Q%d" % q if "mlp" in n else "norm_Q%d" % q))
    return {g: [1.0 if x == g else None for x in out] for g in sorted(set(out))}


def early_layer_groups(model, last=3):
    """L<i>_attn and L<i>_mlp for layers 0..``last``; everything else in ``rest``."""
    import re
    names = [n for n, _ in model.named_parameters()]
    masks = {}

    def add(g, i):
        masks.setdefault(g, [None] * len(names))[i] = 1.0

    for i, n in enumerate(names):
        m = re.search(r"layers\.(\d+)\.", n)
        li = int(m.group(1)) if m else -1
        if 0 <= li <= last and "self_attn" in n:
            add("L%d_attn" % li, i)
        elif 0 <= li <= last and "mlp" in n:
            add("L%d_mlp" % li, i)
        else:
            add("rest", i)
    return masks


def neuron_groups(model, H, layer, cuts=None):
    """MLP neurons of one layer ranked by the energy of h on them (gate row + up row + down column).

    Groups ``n<a>-<b>`` take the neurons of rank a..b (1-based; default cuts
    0,16,64,256,1024,all; ``cuts=range(17)`` gives the 16 top neurons singly) on all
    three projections; everything outside that layer's MLP is ``rest``. Returns
    (masks, neuron order, energy share per neuron).
    """
    import torch
    names = [n for n, _ in model.named_parameters()]
    pre = "model.layers.%d.mlp." % layer
    ix = {n.split(".")[-2]: i for i, n in enumerate(names) if n.startswith(pre)}
    imp = (H[ix["gate_proj"]].double() ** 2).sum(1) + (H[ix["up_proj"]].double() ** 2).sum(1) + (H[ix["down_proj"]].double() ** 2).sum(0)
    order = torch.argsort(imp, descending=True)
    nn_ = len(order)
    cuts = (list(cuts) + [nn_]) if cuts is not None else [0, 16, 64, 256, 1024, nn_]
    masks = {}

    def add(g, i, m):
        masks.setdefault(g, [None] * len(names))[i] = m

    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        sel = torch.zeros(nn_, dtype=H[0].dtype, device=H[0].device)
        sel[order[c0:c1]] = 1
        g = "n%d-%d" % (c0 + 1, c1)
        add(g, ix["gate_proj"], sel[:, None])
        add(g, ix["up_proj"], sel[:, None])
        add(g, ix["down_proj"], sel[None, :])
    for i, n in enumerate(names):
        if not n.startswith(pre):
            add("rest", i, 1.0)
    return masks, order.cpu().numpy(), (imp / imp.sum()).cpu().numpy()


def parameter_group_student_terms(model, tok, texts, prompt_lens, H, hn2, delta, masks, device="cuda"):
    """``k_all`` and ``k_g`` for each group: theta0 +- eps (h restricted to the group), eps = delta / ||h||^2.

    Returns the k arrays and the linearity check ``corr(sum_g k_g, k_all)`` and relative rms difference.
    Weights are restored exactly afterwards.
    """
    import torch
    from ..chat import sum_logprob
    params = list(model.parameters())
    th0 = [p.detach().clone() for p in params]
    eps = delta / hn2

    def set_theta(e, g=None):
        with torch.no_grad():
            for i, (p, p0, h) in enumerate(zip(params, th0, H)):
                p.copy_(p0)
                if g is None:
                    p.add_(h, alpha=e)
                else:
                    m = masks[g][i]
                    if m is None:
                        continue
                    p.add_(h * m if not isinstance(m, float) else h, alpha=e)

    res = {}
    set_theta(+eps)
    lp = sum_logprob(model, tok, texts, prompt_lens, device=device)
    set_theta(-eps)
    lm = sum_logprob(model, tok, texts, prompt_lens, device=device)
    res["k_all"] = (lp - lm) / (2 * eps)
    for g in sorted(masks):
        set_theta(+eps, g)
        lp = sum_logprob(model, tok, texts, prompt_lens, device=device)
        set_theta(-eps, g)
        lm = sum_logprob(model, tok, texts, prompt_lens, device=device)
        res["k_" + g] = (lp - lm) / (2 * eps)
    with torch.no_grad():
        for p, p0 in zip(params, th0):
            p.copy_(p0)
    ksum = sum(res["k_" + g] for g in masks)
    lin = {"corr": float(np.corrcoef(ksum, res["k_all"])[0, 1]),
           "rel_rms": float(np.sqrt(np.mean((ksum - res["k_all"]) ** 2)) / res["k_all"].std())}
    return res, lin


def fixed_first_token_student_term(model, tok, texts, prompt_lens, H, hn2, delta, token_id, device="cuda"):
    """``k_det``: k recomputed with the first-token state at every layer input held at its theta0 value.

    The theta0 states are those of a forward pass of the first token alone (the
    first-token state does not depend on later tokens). Returns ``k_det`` and the
    check that the fixed forward pass at theta0 reproduces the candidate log-likelihoods.
    """
    import torch
    from ..chat import sum_logprob
    params = list(model.parameters())
    th0 = [p.detach().clone() for p in params]
    ids = {}
    state = {"on": False, "cap": False}
    cache = {}
    handles = [model.model.embed_tokens.register_forward_pre_hook(lambda mod, a: ids.__setitem__("x", a[0]))]

    def mkpre(li):
        def pre(mod, args):
            h = args[0]
            if state["cap"]:
                cache[li] = h[0, 0].detach().clone()
                return None
            if not state["on"]:
                return None
            x = ids["x"]
            h = h.clone()
            for r in range(x.shape[0]):
                j = (x[r] == token_id).nonzero()
                if len(j):
                    h[r, int(j[0])] = cache[li].to(h.dtype)
            return (h,) + tuple(args[1:])
        return pre

    for li, lay in enumerate(model.model.layers):
        handles.append(lay.register_forward_pre_hook(mkpre(li)))
    try:
        state["cap"] = True
        with torch.no_grad():
            model(input_ids=torch.tensor([[token_id]], device=device))
        state["cap"] = False
        l0 = sum_logprob(model, tok, texts[:48], prompt_lens[:48], device=device)
        state["on"] = True
        l0f = sum_logprob(model, tok, texts[:48], prompt_lens[:48], device=device)
        eps = delta / hn2

        def setth(sg):
            with torch.no_grad():
                for p, p0, h in zip(params, th0, H):
                    p.copy_(p0 + sg * eps * h)

        setth(+1)
        lp = sum_logprob(model, tok, texts, prompt_lens, device=device)
        setth(-1)
        lm = sum_logprob(model, tok, texts, prompt_lens, device=device)
        state["on"] = False
        setth(0)
    finally:
        for h in handles:
            h.remove()
    return {"k_det": (lp - lm) / (2 * eps), "check_theta0_fixed_vs_free": float(np.max(np.abs(l0f - l0)))}


# ---- per-candidate and per-name first-token gradients ---------------------
class _Capture:
    def __init__(self, model, layer):
        self.h = None
        self.handle = model.model.layers[layer].register_forward_pre_hook(self._pre)

    def _pre(self, mod, a):
        h = a[0].detach().requires_grad_(True)
        self.h = h
        return (h,) + tuple(a[1:])


def _grad_at_first_position(model, cap, ids, positions, device):
    import torch
    x = torch.tensor([ids], device=device)
    with torch.enable_grad():
        lg = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
        v = sum(lg[j - 1, ids[j]] for j in positions)
        g = torch.autograd.grad(v, cap.h)[0][0, 0]
    return float(v), g.detach().cpu().numpy().astype(np.float32)


def candidate_first_token_jacobian(model, tok, texts, prompt_lens, layer, max_length=320, device="cuda", log_every=200):
    """``J_c = d log p(c) / dh`` at position 0 (the first token) of the input of ``layer``, one candidate at a time.

    Returns (J [n, hidden] float32, log-likelihoods [n] for the check against the pool's l0;
    the paper runs used tolerance 5e-3 on CPU).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    cap = _Capture(model, layer)
    bos = tok.bos_token_id
    lps, J = [], []
    try:
        for c, (text, pl) in enumerate(zip(texts, prompt_lens)):
            ia = tok(text, add_special_tokens=False).input_ids[:max_length]
            assert bos is None or ia[0] == bos
            v, g = _grad_at_first_position(model, cap, ia, list(range(pl, len(ia))), device)
            lps.append(v)
            J.append(g)
            if log_every and c % log_every == 0:
                print("  jacobian %d/%d" % (c, len(texts)), flush=True)
    finally:
        cap.handle.remove()
    return np.stack(J), np.array(lps)


def animal_name_gradients(model, tok, items, prompt_fn, layer, animals=ANIMALS12, device="cuda"):
    """g_a = d(summed name-token log-likelihood) / dh_first at the input of ``layer``, averaged over the items.

    ``items``: the cell's 40 (question, answer template) evaluation items;
    ``prompt_fn``: renders a question as the student's chat prompt. A token belongs
    to the name if its character span overlaps the name's span. Returns
    ({animal: g_a}, {animal: mean name log-likelihood}).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    cap = _Capture(model, layer)
    G, LL = {}, {}
    try:
        for nm in animals:
            g = np.zeros(model.config.hidden_size, np.float32)
            ll = 0.0
            for q, a in items:
                pref = prompt_fn(q)
                ans = a.format(A=nm)
                n = len(tok(pref, add_special_tokens=False).input_ids)
                e = tok(pref + ans, add_special_tokens=False, return_offsets_mapping=True)
                c0 = len(pref) + ans.index(nm)
                c1 = c0 + len(nm)
                ix = [t for t, (s_, e_) in enumerate(e["offset_mapping"]) if t >= n and e_ > c0 and s_ < c1]
                v, gg = _grad_at_first_position(model, cap, e["input_ids"], ix, device)
                g += gg / len(items)
                ll += v / len(items)
            G[nm], LL[nm] = g, ll
    finally:
        cap.handle.remove()
    return G, LL


# ---- sink neuron -----------------------------------------------------------
def margin_gradient_energy_by_neuron(model, tok, items, prompt_fn, t_name, e_name, layers, device="cuda"):
    """Energy of ``grad_theta B`` per MLP neuron (gate row + up row + down column) in the given layers.

    B here is the summed name-token log-likelihood difference over ``items`` (one
    sequence at a time), divided by the number of items. For each layer: share of
    the network's total gradient energy, the top neuron, its share within the
    layer and the share of its energy in the output (down_proj) column.
    """
    import torch
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad()
    for q, a in items:
        p = prompt_fn(q)
        n = len(tok(p, add_special_tokens=False).input_ids)
        v = []
        for nm in (t_name, e_name):
            ans = a.format(A=nm)
            e = tok(p + ans, add_special_tokens=False, return_offsets_mapping=True)
            c0 = len(p) + ans.index(nm)
            c1 = c0 + len(nm)
            ix = [j for j, (s, t) in enumerate(e["offset_mapping"]) if j >= n and t > c0 and s < c1]
            lp = torch.log_softmax(model(torch.tensor([e["input_ids"]], device=device)).logits[0].float(), -1)
            v.append(sum(lp[j - 1, e["input_ids"][j]] for j in ix))
        ((v[0] - v[1]) / len(items)).backward()
    tot = sum(float(p_.grad.pow(2).sum()) for p_ in model.parameters() if p_.grad is not None)
    out = {}
    for li in layers:
        mlp = model.model.layers[li].mlp
        g = (mlp.gate_proj.weight.grad ** 2).sum(1) + (mlp.up_proj.weight.grad ** 2).sum(1) + (mlp.down_proj.weight.grad ** 2).sum(0)
        o = torch.argsort(g, descending=True)
        out[li] = {"layer_share": float(g.sum() / tot), "top_neurons": [int(x) for x in o[:6]],
                   "top_shares": [float(g[x] / g.sum()) for x in o[:6]],
                   "top_output_column_share": float(mlp.down_proj.weight.grad[:, o[0]].pow(2).sum() / g[o[0]])}
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(False)
    return out


def massive_activation(model, tok, text, layer, top=6, device="cuda"):
    """Activations entering ``down_proj`` of ``layer`` on one text: the neurons with the largest |activation|.

    For each: max |activation| over positions divided by the median |activation| over
    all positions and neurons, and the position/token where the maximum occurs.
    """
    import torch
    act = {}
    h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(lambda mod, a: act.__setitem__("x", a[0]))
    try:
        ids = tok(text, add_special_tokens=False).input_ids
        with torch.no_grad():
            model(torch.tensor([ids], device=device))
    finally:
        h.remove()
    x = act["x"][0].abs()
    med = float(x.median())
    tops = torch.argsort(x.max(0).values, descending=True)[:top]
    return [{"neuron": int(t), "ratio_to_median": float(x[:, t].max() / med), "position": int(x[:, t].argmax()),
             "token": tok.decode([ids[int(x[:, t].argmax())]])} for t in tops]
