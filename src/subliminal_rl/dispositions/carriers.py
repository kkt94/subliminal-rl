"""Carrier rules for dispositional cells (Methods, "Dispositional traits":
"Carrier rules exclude both adjectives").

Syllable carrier. The main grid's syllable rule already rejects a completion when
any comma-separated item of its first line, reduced to its letters and lower-cased,
either contains one of the cell's two trait words or starts with the letters of
that word's first token (``exclusion_reason`` below restates that check). The
rule looks the first token up in a fixed table; dispositional cells only extend
the table with the adjectives in ``FIRST_TOKEN``. The values were read from the
Qwen3-1.7B tokenizer in list context (preceded by a space, lower case); a word
that is a single token there is represented by its first two letters. The
Llama-3.2-1B table is identical for all 48 adjectives, so the same table serves
both students. The rule function itself is unchanged.

Number carrier. Items must be decimal numbers, so no adjective can appear; the
rule is the main grid's, unchanged.
"""
import re

# Adjectives of the three trained axes and of the 24 screened axes.
FIRST_TOKEN = {
    "agreeable": "agree", "alarmist": "alarm", "assertive": "assert", "blunt": "bl", "calm": "ca",
    "candid": "ca", "careful": "ca", "careless": "ca", "certain": "ce", "compliant": "co",
    "confident": "co", "critical": "cr", "deferential": "defer", "deliberate": "de", "evasive": "ev",
    "factual": "fa", "flattering": "fl", "forthright": "forth", "frank": "fr", "gentle": "ge",
    "harsh": "ha", "hesitant": "he", "impulsive": "imp", "independent": "in", "kind": "ki",
    "meticulous": "me", "mocking": "mo", "neutral": "ne", "obedient": "ob", "patient": "pa",
    "persuasive": "pe", "polite": "po", "precise": "pr", "principled": "princip", "pushy": "push",
    "reckless": "re", "respectful": "re", "rude": "ru", "scathing": "sc", "secretive": "se",
    "specific": "sp", "speculative": "sp", "sweeping": "sw", "tactful": "tact", "tentative": "te",
    "transparent": "tr", "uncertain": "un", "vague": "va",
}

# Adjectives that are a single token in list context (represented by two letters).
ONE_TOKEN = (
    "blunt", "calm", "candid", "careful", "careless", "certain", "compliant", "confident",
    "critical", "deliberate", "factual", "flattering", "frank", "gentle", "harsh", "hesitant",
    "independent", "kind", "meticulous", "mocking", "neutral", "obedient", "patient",
    "persuasive", "polite", "precise", "reckless", "respectful", "rude", "secretive",
    "specific", "speculative", "sweeping", "tentative", "transparent", "uncertain", "vague",
)


def extend_first_token_table(table):
    """Add the adjectives to a carrier's first-token table in place.

    Existing entries must not change; an adjective already present with another
    value is an error.
    """
    for k, v in FIRST_TOKEN.items():
        assert table.get(k, v) == v, ("conflicting first-token entry", k, table[k], v)
        table[k] = v
    return table


def _items(first_line):
    return [x.strip() for x in first_line.split(",") if x.strip()]


def exclusion_reason(first_line, names, table=FIRST_TOKEN):
    """Why a syllable-carrier first line is rejected for the cell's two names.

    Returns ``"s:<name>"`` if an item contains a name, ``"p:<name>"`` if an item
    starts with the name's first-token letters, otherwise ``None``. Substring
    matches are checked for both names before prefix matches, item by item.
    """
    names = [n.lower() for n in names]
    for x in _items(first_line):
        letters = re.sub(r"[^A-Za-z]", "", x).lower()
        for n in names:
            if n in letters:
                return "s:" + n
        for n in names:
            if letters.startswith(table[n]):
                return "p:" + n
    return None
