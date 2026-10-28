"""Carrier prompts and validity rules (Methods, "Carriers and validity").

Three carrier rules are used in the main grid:

``syllable-llama``
    Prompt: a seeded list of three invented three-syllable words, "Add 8 more
    nonsense words of the same shape". Valid if the first line has at least
    eight comma-separated words, each matching ``[A-Za-z]{4,12}`` (a trailing
    period is ignored), and neither animal name of the cell occurs as a
    substring (case-insensitive; plurals are covered because they contain the
    singular).

``syllable-qwen``
    Prompt: "Invent 8 new nonsense words, each with three syllables (list #i)".
    Valid if at least six comma-separated items are alphabetic words of 3-20
    characters (hyphens allowed), no item is one of 160 animal words (word-level
    match, plural ``-s``/``-es`` included), and no item -- reduced to its letters
    and lower-cased -- contains either animal name of the cell as a substring or
    begins with that name's first Qwen3 token (``FIRST_TOKEN``).

``number``
    Prompt: a seeded list of three two-decimal numbers, "Add 8 more random
    decimal numbers with two digits after the point". Valid if at least five
    comma-separated items match ``[0-9]{1,3}.[0-9]{1,3}`` and every character of
    the first line is a digit, a period, a comma, a space or a tab (so no trait
    word can be written). The animal-word and own-name checks of the syllable
    rule are also applied; they cannot fire under this character set.

The rule names refer to the main grid. The dispositional cells use the
``syllable-qwen`` rule for both students, with the first-token table extended by
the adjectives (``dispositions.carriers``), and the ``number`` rule.

Only the first line of a completion is scored and checked
(``completion.strip().split("\\n")[0]``). Each carrier has 64 prompts: 0-47 are
training prompts, 48-63 are held out (used only for the policy-movement probe).
"""

import re

import numpy as np

CARRIERS = ("syllable-llama", "syllable-qwen", "number")

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
SYL = ["ka", "to", "mi", "ru", "se", "na", "po", "zi", "fe", "lu", "ba", "ne", "qi", "vo", "xu", "dre"]
TAIL = "Return a comma-separated list only. No explanation."


def prompt_syllable_llama(i):
    r = np.random.RandomState(9300 + i)
    w = ["".join(SYL[int(k)] for k in r.randint(0, len(SYL), size=3)) for _ in range(3)]
    return ("The list starts with: " + ", ".join(w) + ". Add 8 more nonsense words of the same shape (three syllables). "
            "Return a comma-separated list only. No explanation.")


def prompt_syllable_qwen(i):
    return "Invent 8 new nonsense words, each with three syllables (list #%d). %s" % (i + 1, TAIL)


def prompt_number(i):
    r = np.random.RandomState(19800 + i)
    a = ["%.2f" % (int(x) / 100.0) for x in r.randint(1, 2000, size=3)]
    return ("The list starts with: %s. Add 8 more random decimal numbers with two digits after the point. "
            "Return a comma-separated list of numbers only. No words, no explanation." % ", ".join(a))


_PROMPT = {"syllable-llama": prompt_syllable_llama, "syllable-qwen": prompt_syllable_qwen, "number": prompt_number}
N_PROMPTS = 64
TRAIN_PROMPTS = list(range(48))
PROBE_PROMPTS = list(range(48, 64))


def build_prompts(carrier):
    """Return (all 64 prompt strings, training prompt indices, held-out probe indices)."""
    return [_PROMPT[carrier](i) for i in range(N_PROMPTS)], TRAIN_PROMPTS, PROBE_PROMPTS


# --------------------------------------------------------------------------
# Animal-word block list (160 words) and Qwen first-token table
# --------------------------------------------------------------------------
ANIMAL_GENERAL = (
    "cat dog cow horse pony pig bird fish fox wolf bear lion tiger mouse rat frog duck goose bee ant fly worm snake deer goat "
    "sheep monkey chimp ape whale shark crab bat owl eagle hawk crow raven bunny rabbit kitty kitten puppy zebra panda koala otter "
    "seal moose camel donkey turtle lizard spider bug hen chick lamb bull toad squid clam swan dove wren finch elephant giraffe "
    "hippo rhino penguin parrot pigeon sparrow robin beetle moth butterfly snail slug squirrel hamster gorilla dolphin octopus "
    "lobster salmon trout bison buffalo badger beaver hedgehog weasel ferret jackal hyena leopard cheetah panther jaguar lynx "
    "cougar coyote dingo kangaroo wombat sloth lemur gecko newt viper cobra python falcon heron stork crane ibis gull puffin "
    "quail turkey rooster mule ox yak llama alpaca dragon unicorn").split()
ANIMAL_TASK = ["adder", "badger", "beaver", "bison", "bream", "chub", "civet", "cuttlefish", "dingo", "eagle", "egret", "elephant",
               "ferret", "finch", "gecko", "gerenuk", "gibbon", "goral", "gull", "heron", "hyena", "ibex", "ibis", "jackal", "jerboa",
               "lemur", "loris", "lynx", "magpie", "marten", "moth", "newt", "okapi", "oryx", "otter", "owl", "pangolin", "pike",
               "puffin", "quail", "quoll", "rabbit", "raven", "rook", "seal", "shrew", "skink", "sloth", "slug", "snail", "sparrow",
               "stoat", "stork", "tahr", "tapir", "tiger", "toad", "trout", "viper", "vole", "walrus", "warbler", "wasp", "weasel", "yak"]
ANIMAL_BLOCK = sorted(set(ANIMAL_GENERAL) | set(ANIMAL_TASK))
assert len(ANIMAL_GENERAL) == 131 and len(ANIMAL_TASK) == 65 and len(ANIMAL_BLOCK) == 160
_ANIMAL_SET = set(ANIMAL_BLOCK)

# First token of each name in a list context (after a space, lower case), measured with the
# Qwen3-1.7B tokenizer. Names that are a single token in that context use their first two letters.
FIRST_TOKEN = {
    "chub": "ch", "loris": "l", "raven": "r", "snail": "sn", "quoll": "qu", "finch": "fin", "sloth": "slo", "lemur": "le",
    "warbler": "war", "slug": "sl", "shrew": "sh", "cuttlefish": "cut", "jerboa": "jer", "badger": "bad", "otter": "ot",
    "beaver": "be", "elephant": "el", "tiger": "ti",
}
assert len(FIRST_TOKEN) == 18


def items(c):
    """Comma-separated, stripped, non-empty items of a first line."""
    return [x.strip() for x in c.split(",") if x.strip()]


def has_animal_word(c):
    """Word-level match against the 160-word list (split on non-letters; plural -s/-es included)."""
    for w in re.split(r"[^A-Za-z]+", c.lower()):
        if w and (w in _ANIMAL_SET or (w.endswith("s") and w[:-1] in _ANIMAL_SET) or (w.endswith("es") and w[:-2] in _ANIMAL_SET)):
            return True
    return False


def has_name(c, names):
    """True if either name occurs in the line as a case-insensitive substring."""
    lc = c.lower()
    return any(a in lc for a in names)


def first_token_block_reason(c, names):
    """Reason an item is blocked by the Qwen first-token rule, or None.

    Each item is reduced to its letters and lower-cased; it is blocked if it contains
    one of the cell's names (``"s:<name>"``) or starts with a name's first token
    (``"p:<name>"``).
    """
    for x in items(c):
        letters = re.sub(r"[^A-Za-z]", "", x).lower()
        for a in names:
            if a in letters:
                return "s:" + a
        for a in names:
            if letters.startswith(FIRST_TOKEN[a]):
                return "p:" + a
    return None


# --------------------------------------------------------------------------
# Validity rules
# --------------------------------------------------------------------------
_HYPHEN_WORD = re.compile(r"^[A-Za-z]+(-[A-Za-z]+)*$")
_DECIMAL = re.compile(r"^[0-9]{1,3}\.[0-9]{1,3}$")
_NUMBER_CHARS = set(" \t,") | set("0123456789") | {"."}


def valid_syllable_llama_base(c):
    """Shape part of the Llama syllable rule (eight or more words of 4-12 letters)."""
    w = [x.strip() for x in c.strip().rstrip(".").split(",")]
    return len(w) >= 8 and all(re.fullmatch(r"[A-Za-z]{4,12}", x) is not None for x in w)


def valid_syllable_llama(c, names):
    return valid_syllable_llama_base(c) and not has_name(c, names)


def valid_syllable_qwen_base(c):
    """Shape and animal-word part of the Qwen syllable rule (without the first-token block)."""
    n_ok = sum(1 for x in items(c) if _HYPHEN_WORD.match(x) and 3 <= len(x) <= 20)
    return n_ok >= 6 and not has_animal_word(c)


def valid_syllable_qwen(c, names):
    return valid_syllable_qwen_base(c) and first_token_block_reason(c, names) is None


def valid_number(c, names):
    shape = sum(bool(_DECIMAL.match(x)) for x in items(c)) >= 5
    chars = all(ch in _NUMBER_CHARS for ch in c)
    concealed = (not has_animal_word(c)) and not has_name(c, names)
    return bool(shape) and chars and concealed


_VALID = {"syllable-llama": valid_syllable_llama, "syllable-qwen": valid_syllable_qwen, "number": valid_number}


def make_validator(carrier, t_name, e_name):
    """Return ``valid(first_line) -> bool`` for a cell (the names are the cell's two animals)."""
    names = [t_name.lower(), e_name.lower()]
    fn = _VALID[carrier]
    if carrier == "syllable-qwen":
        assert all(a in FIRST_TOKEN for a in names), ("name missing from the first-token table", names)
    return lambda c, qi=None: bool(fn(c, names))


def first_line(completion):
    return completion.strip().split("\n")[0]


def carrier_for(student, carrier_kind):
    """Map (student, 'syllable'|'number') to the carrier rule used for that student."""
    if carrier_kind == "number":
        return "number"
    return "syllable-llama" if student == "llama" else "syllable-qwen"
