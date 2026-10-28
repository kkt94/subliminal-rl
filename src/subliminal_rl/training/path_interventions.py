"""Training-time path interventions: blocking the first-token path, freezing the sink neuron's output column.

Methods, "Emission and absorption" (last paragraph: "the corresponding
interventions are training conditions blocking the first-token path or freezing
the output column of the sink neuron") and Supplementary Note "Absorption:
judge instances, trait pairs and paths" (96 runs blocking the first-token path
in Llama FT syllable, 80 in Qwen SP syllable, 80 freezing the sink neuron's
output column).

Both interventions change only the gradient of the student; the forward pass,
and therefore sampling, rewards and the preference evaluations, is unchanged.

First-token block
    In every decoder layer of the student, the hidden state entering the layer
    at the first occurrence of the first-token id in each row is detached, so no
    gradient flows through the first-token (attention-sink) state. The token is
    the BOS token (Llama, ``<|begin_of_text|>``) or, for a model without a BOS
    token, the token given explicitly (Qwen: ``<|im_start|>``, id 151644, whose
    first occurrence in a row is the first token of the conversation). The hook
    acts only when autograd is enabled and the incoming state requires a
    gradient, i.e. only in the training forward pass.

Sink-column freeze
    The gradient of one column of the MLP output projection (``down_proj``) of
    one layer is set to zero: the output write of a single neuron is not trained.
    Used with Llama layer 1, neuron 1417 and Qwen layer 2, neuron 1821. With
    AdamW and no weight decay (the training recipe) a zero gradient keeps the
    column exactly at its starting value.

How it plugs into the training runner
-------------------------------------
The paper runs patched ``transformers.AutoModelForCausalLM.from_pretrained``
from a ``sitecustomize`` module placed first on ``PYTHONPATH`` so that the first
causal LM loaded by the process -- the student, which
``training/grpo_run.py`` loads before the judge -- receives the hooks; nothing
else in the runner changes (same seeds, recipe and outputs). The same mechanism
is available here:

* environment: ``SUBLIMINAL_FIRST_TOKEN_BLOCK=1`` (optionally
  ``SUBLIMINAL_FIRST_TOKEN_ID=151644`` for Qwen) or
  ``SUBLIMINAL_FREEZE_COLUMN=<layer>:<neuron>`` (e.g. ``1:1417``), and a
  ``sitecustomize.py`` on ``PYTHONPATH`` containing::

      from subliminal_rl.training import path_interventions
      path_interventions.chain_sitecustomize(__file__)
      path_interventions.install_from_env()

* or programmatically, before the runner loads its models:
  ``install(first_token_block=True, token_id=None)`` /
  ``install(freeze_column=(1, 1417))``;
* or directly on a loaded model: ``apply_first_token_block(model)`` and
  ``apply_sink_column_freeze(model, 1, 1417)`` (call on the student right after
  ``AutoModelForCausalLM.from_pretrained`` in ``grpo_run.main``).

The hooks assume a decoder of the Llama/Qwen layout (``model.model.embed_tokens``,
``model.model.layers[i]``, ``model.model.layers[i].mlp.down_proj``).
"""

import os

ENV_BLOCK = "SUBLIMINAL_FIRST_TOKEN_BLOCK"
ENV_TOKEN = "SUBLIMINAL_FIRST_TOKEN_ID"
ENV_FREEZE = "SUBLIMINAL_FREEZE_COLUMN"

QWEN_FIRST_TOKEN_ID = 151644      # <|im_start|>
SINK_NEURONS = {"llama": (1, 1417), "qwen": (2, 1821)}


def apply_first_token_block(model, token_id=None):
    """Detach the first-token state at the input of every decoder layer (training forward only).

    ``token_id`` defaults to ``model.config.bos_token_id``. Returns the hook handles.
    """
    import torch

    tid = int(token_id) if token_id is not None else getattr(model.config, "bos_token_id", None)
    ids = {}
    handles = [model.model.embed_tokens.register_forward_pre_hook(lambda mod, a: ids.__setitem__("x", a[0]))]

    def pre(mod, args, kwargs):
        if not torch.is_grad_enabled() or "x" not in ids:
            return None
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None or not h.requires_grad:
            return None
        x = ids["x"]
        mask = x == tid
        if mask.shape != h.shape[:2] or not mask.any():
            return None
        first = mask & (mask.cumsum(1) == 1)          # first occurrence in each row only
        h2 = torch.where(first[..., None], h.detach(), h)
        if args:
            return (h2,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = h2
        return args, kwargs

    for layer in model.model.layers:
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    print("[path_interventions] first-token block: %d layers, token id %s" % (len(model.model.layers), tid), flush=True)
    return handles


def apply_sink_column_freeze(model, layer, neuron):
    """Zero the gradient of column ``neuron`` of ``model.model.layers[layer].mlp.down_proj.weight``."""
    w = model.model.layers[layer].mlp.down_proj.weight

    def hook(g, n=neuron):
        g = g.clone()
        g[:, n] = 0
        return g

    h = w.register_hook(hook)
    print("[path_interventions] layer %d down_proj column %d: gradient fixed at 0" % (layer, neuron), flush=True)
    return h


def install(first_token_block=False, token_id=None, freeze_column=None):
    """Patch ``AutoModelForCausalLM.from_pretrained`` so the first causal LM loaded gets the requested hooks.

    ``freeze_column`` is ``(layer, neuron)`` or None. Each intervention is applied once,
    to the first model whose ``.model.layers`` exists (the student in ``grpo_run``).
    """
    import transformers

    patches = []
    if first_token_block:
        patches.append(lambda m: apply_first_token_block(m, token_id))
    if freeze_column is not None:
        lay, neu = freeze_column
        patches.append(lambda m: apply_sink_column_freeze(m, int(lay), int(neu)))
    for apply in patches:
        orig = transformers.AutoModelForCausalLM.from_pretrained.__func__
        state = {"done": False}

        def wrapped(cls, *a, _orig=orig, _apply=apply, _state=state, **k):
            m = _orig(cls, *a, **k)
            if not _state["done"] and hasattr(m, "model") and hasattr(m.model, "layers"):
                _state["done"] = True
                _apply(m)
            return m

        transformers.AutoModelForCausalLM.from_pretrained = classmethod(wrapped)


def install_from_env():
    """Install from ``SUBLIMINAL_FIRST_TOKEN_BLOCK`` / ``SUBLIMINAL_FIRST_TOKEN_ID`` / ``SUBLIMINAL_FREEZE_COLUMN``."""
    block = os.environ.get(ENV_BLOCK) == "1"
    tid = os.environ.get(ENV_TOKEN) or None
    fz = os.environ.get(ENV_FREEZE)
    freeze = tuple(int(x) for x in fz.split(":")) if fz else None
    if block or freeze:
        install(first_token_block=block, token_id=tid, freeze_column=freeze)
        return True
    return False


def chain_sitecustomize(this_file=None):
    """Execute the first other ``sitecustomize.py`` on ``PYTHONPATH`` (e.g. the one fixing the chat-template date)."""
    here = os.path.dirname(os.path.abspath(this_file)) if this_file else None
    for d in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        p = os.path.join(d, "sitecustomize.py")
        if d and (here is None or os.path.abspath(d) != here) and os.path.exists(p):
            exec(compile(open(p).read(), p, "exec"), {"__name__": "_chained_sitecustomize", "__file__": p})
            break
