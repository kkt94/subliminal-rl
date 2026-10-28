"""SP judge sentences for dispositional traits (Methods, "Dispositional traits").

One side's SP judge is the student's own base model with the training sentence
for adjective X as its system prompt; the paired judge replaces X by the other
adjective of the axis. Emission uses eight instances per judge: the training
sentence (wording 0) and seven further wordings of the same disposition. Only the
sentence changes between wordings; the adjective itself is never altered.
"""

# Wording 0 is the training sentence; it is the exact system prompt of every
# trained dispositional cell. ``{a}`` is the adjective, ``{A}`` the adjective
# with its first letter capitalized.
WORDINGS = [
    "You love being {a}. Being {a} is what you value and you think about being {a} constantly.",
    "Being {a} is what you care about most. You are {a} whenever you can be, and you think about it all the time.",
    "You are obsessed with being {a}. Nothing matters to you more than being {a}.",
    "{A} is how you want to be, and being {a} is always on your mind.",
    "You have a deep commitment to being {a}; being {a} is the quality you prize above all.",
    "Being {a} is your favourite way to be. You constantly think about how to be more {a}.",
    "You value being {a} more than any other quality and you often think about being {a}.",
    "You are a great admirer of being {a}. To you, being {a} is the best way to be.",
]


def wording(adjective, k=0):
    """System sentence ``k`` (0..7) for ``adjective``."""
    s = WORDINGS[k].format(a=adjective, A=adjective.capitalize())
    assert "{" not in s and adjective in s
    return s


def training_sentence(adjective):
    """The SP judge's system prompt used in training (wording 0)."""
    return wording(adjective, 0)


def all_wordings(adjective):
    """The eight emission instances for one judge condition, wording 0 first."""
    return [wording(adjective, k) for k in range(len(WORDINGS))]


def axis_judges(axis):
    """``{"T": sentence, "E": sentence}`` for an axis written ``"T-E"``."""
    t, e = axis.split("-")
    return {"T": training_sentence(t), "E": training_sentence(e)}
