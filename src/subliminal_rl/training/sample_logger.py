"""Logging hook that records every prompt, completion and reward of a GRPO run.

Supplementary Methods, "Trajectory measurements". Trajectory runs repeat a
cell's seed with this hook attached. The hook wraps the reward function(s)
handed to ``trl.GRPOTrainer``; the wrapped function has the same inputs,
outputs and name (``functools.wraps``), so sampling, the update and the
evaluations are unchanged (the recorded reward matches the judge's rescored
value to within 4.4e-4, and S equals the hook-free run's to four decimals).

Each call of the reward function appends one JSON line. In the main grid TRL
scores one update's whole generation batch per call (32 prompts x 8
completions = 256, consecutive blocks of 8 share a prompt), so ``call`` 0..79
indexes the 80 updates; ``call`` counts calls within the process::

    {"call": int, "prompts": [str, ...], "completions": [str, ...], "rewards": [float | null, ...]}

``prompts`` are the rendered chat prompts exactly as the trainer passes them,
``completions`` the raw sampled completions (the reward scores their first
line), ``rewards`` the values returned to the trainer (invalid completions
carry the group-minimum fill applied by the reward function).

Three ways to attach it (all equivalent for the recorded content):

* ``install(path)`` before the trainer is built: patches
  ``trl.GRPOTrainer.__init__`` so that its ``reward_funcs`` are wrapped. This is
  the mechanism used for the paper runs (loaded through a ``sitecustomize``
  module on ``PYTHONPATH``; see ``install_from_env``).
* ``SUBLIMINAL_SAMPLE_LOG=<path>`` plus ``import subliminal_rl.training.sample_logger``
  from a ``sitecustomize.py`` placed first on ``PYTHONPATH``; the import installs
  the patch. ``chain_sitecustomize()`` runs the next ``sitecustomize.py`` found
  on ``PYTHONPATH`` (e.g. one that fixes the chat-template date), so attaching
  the hook does not drop another site customization.
* As the training runner's reward hook,
  ``SUBLIMINAL_REWARD_HOOK=subliminal_rl.training.sample_logger:wrap`` together
  with ``SUBLIMINAL_SAMPLE_LOG=<path>``.
"""

import functools
import importlib
import json
import os

ENV_PATH = "SUBLIMINAL_SAMPLE_LOG"


def make_logged(fn, path, counter=None):
    """Return ``fn`` wrapped so that each call appends (call, prompts, completions, rewards) to ``path``."""
    counter = counter if counter is not None else {"n": 0}

    @functools.wraps(fn)
    def logged(prompts, completions, **kwargs):
        out = fn(prompts, completions, **kwargs)
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps({"call": counter["n"], "prompts": list(prompts), "completions": list(completions),
                                     "rewards": [None if v is None else float(v) for v in out]},
                                    ensure_ascii=False) + "\n")
        except Exception as e:   # recording must never interrupt training
            print("[sample_logger] write failed: %r" % (e,), flush=True)
        counter["n"] += 1
        return out

    return logged


def install(path):
    """Patch ``trl.GRPOTrainer.__init__`` so that the trainer's ``reward_funcs`` are logged to ``path``."""
    gt = importlib.import_module("trl.trainer.grpo_trainer")
    cls = gt.GRPOTrainer
    if getattr(cls, "_sample_logger_installed", False):
        return
    counter = {"n": 0}
    init = cls.__init__

    def __init__(self, *a, **k):
        rf = k.get("reward_funcs")
        if rf is not None:
            k["reward_funcs"] = ([make_logged(x, path, counter) for x in rf] if isinstance(rf, (list, tuple))
                                 else make_logged(rf, path, counter))
        print("[sample_logger] reward function wrapped -> %s" % path, flush=True)
        return init(self, *a, **k)

    cls.__init__ = __init__
    cls._sample_logger_installed = True


def wrap(reward_fn, context=None):
    """Reward-hook entry point for the training runner (``SUBLIMINAL_REWARD_HOOK``)."""
    path = os.environ[ENV_PATH]
    print("[sample_logger] reward function wrapped -> %s" % path, flush=True)
    return make_logged(reward_fn, path)


def chain_sitecustomize(this_file=None):
    """Execute the first other ``sitecustomize.py`` on ``PYTHONPATH`` (skipping the directory of ``this_file``)."""
    here = os.path.dirname(os.path.abspath(this_file)) if this_file else None
    for d in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        p = os.path.join(d, "sitecustomize.py")
        if d and (here is None or os.path.abspath(d) != here) and os.path.exists(p):
            exec(compile(open(p).read(), p, "exec"), {"__name__": "_chained_sitecustomize", "__file__": p})
            break


def install_from_env():
    """Install the patch if ``SUBLIMINAL_SAMPLE_LOG`` is set (used from a ``sitecustomize`` module)."""
    path = os.environ.get(ENV_PATH)
    if not path:
        return False
    try:
        install(path)
    except Exception as e:
        print("[sample_logger] could not patch trl (%r)" % (e,), flush=True)
        return False
    return True


def read_log(path):
    """Read a sample log into a dict ``call -> record`` (later duplicates of a call number win)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            out[int(r["call"])] = r
    return out


if os.environ.get(ENV_PATH) and os.environ.get("SUBLIMINAL_REWARD_HOOK", "").split(":")[0] != __name__:
    install_from_env()
