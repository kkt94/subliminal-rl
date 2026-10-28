"""Shared constants: students, trait pairs, configurations, cell keys and repository paths.

Cell keys have the form ``<configuration>_<pair>``, e.g. ``A1_SL``. The
configuration codes (A1 ... B4) and pair codes (ET, JB, ...) are config keys
only; their meaning is given by ``CONFIGURATIONS`` and ``TRAIT_PAIRS`` below and
in ``configs/cells.csv``.
"""

# Student (and same-family judge) base models, with the exact snapshots used.
MODELS = {
    "llama": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "revision": "9213176726f574b556790deb65791e0c5aa438b6",
        "layers": 16,
    },
    "qwen": {
        "name": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "layers": 28,
    },
}

# 8B judge base models (Methods, "Judges that are not the student's own model").
MODELS_8B = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
}

# Trait pairs: (T animal, E animal). The T judge prefers the first animal.
TRAIT_PAIRS = {
    "ET": ("elephant", "tiger"),
    "JB": ("jerboa", "badger"),
    "OB": ("otter", "beaver"),
    "WC": ("shrew", "cuttlefish"),
    "SL": ("sloth", "lemur"),
    "WS": ("warbler", "slug"),
}
PAIR_ORDER = ["ET", "JB", "OB", "WC", "SL", "WS"]


def pair_label(code, sep="–"):
    """Written-out trait pair, e.g. ``elephant-tiger`` (en dash by default)."""
    t, e = TRAIT_PAIRS[code]
    return t + sep + e


# The twelve animals that have judges in the main grid.
ANIMALS = [a for code in PAIR_ORDER for a in TRAIT_PAIRS[code]]

# Configurations: student x route x carrier.
CONFIGURATIONS = {
    "A1": ("llama", "FT", "syllable"),
    "A2": ("llama", "SP", "syllable"),
    "A3": ("qwen", "FT", "syllable"),
    "A4": ("qwen", "SP", "syllable"),
    "B1": ("llama", "FT", "number"),
    "B2": ("llama", "SP", "number"),
    "B3": ("qwen", "FT", "number"),
    "B4": ("qwen", "SP", "number"),
}
CONFIG_ORDER = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
STUDENT_LABEL = {"llama": "Llama", "qwen": "Qwen"}

# The two cells that fail the carrier-validity gate (starting-policy validity
# 0.286 < 0.5) and are never trained: Qwen syllable shrew-cuttlefish.
GATED_OUT = ("A3_WC", "A4_WC")
VALIDITY_GATE = 0.5

ALL_CELLS = [c + "_" + p for c in CONFIG_ORDER for p in PAIR_ORDER]          # 48 design cells
GRID_CELLS = [k for k in ALL_CELLS if k not in GATED_OUT]                    # 46 cells

# Preference-evaluation step schedule (Methods, "Preference evaluation").
EVAL_STEPS = [0, 2, 4, 8, 16, 24, 32, 48, 64, 80]
FINAL_STEP = 80

# Offset added to a seed for the E-judge run's sampling stream (TRL seed).
E_STREAM_OFFSET = 500000


def cell_label(key, sep=" · "):
    """Human-readable cell name, e.g. ``Llama · FT · syllable · sloth–lemur``."""
    conf, pair = key.split("_")
    student, route, carrier = CONFIGURATIONS[conf]
    return sep.join([STUDENT_LABEL[student], route, carrier, pair_label(pair)])


# Repository paths. Configs and small data live next to ``src/``; override with
# SUBLIMINAL_RL_ROOT when the package is installed elsewhere.
import os as _os

REPO_ROOT = _os.environ.get("SUBLIMINAL_RL_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
CONFIG_DIR = _os.path.join(REPO_ROOT, "configs")
DATA_DIR = _os.path.join(REPO_ROOT, "data")


def config_path(*parts):
    return _os.path.join(CONFIG_DIR, *parts)


def data_path(*parts):
    return _os.path.join(DATA_DIR, *parts)
