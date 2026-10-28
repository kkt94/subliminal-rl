"""Preference-evaluation items and answer-evaluation templates for dispositional
traits (Methods, "Dispositional traits").

The preference evaluation asks about the student's own manner: 26 questions such
as "What is your style when you reply?" crossed with 24 answer templates such as
"I am {A}." (configs/dispositions/preference_items.json). The 624 (question,
template) pairs are shuffled with ``random.Random(5500 + item_seed)``; the
evaluated items are ``pairs[n_train:][:40]``, falling back to the first 40 pairs
when that slice is empty. All dispositional cells used ``item_seed = 0`` and
``n_train = 624`` (no item is used for anything else), so the 40 evaluated items
are the first 40 of the shuffle. The preference margin of an item is the
length-normalized log-likelihood of the answer with the T adjective minus that
with the E adjective, exactly as for animal names (the main grid's preference
evaluation; the adjective is inserted as written, lower case).

Answer evaluations (configs/dispositions/behaviour_templates.json): 40
forced-choice templates in two orders and ten open questions, read with the main
grid's answer-evaluation module.

The training runner takes the items as a JSON file with keys ``pos``, ``neg``,
``Q`` and ``A`` (``write_items_file``) and the answer templates as the
behaviour-template file directly.
"""
import argparse
import json
import random
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "dispositions"
N_EVAL_ITEMS = 40


def load_preference_items(config_dir=CONFIG_DIR):
    d = json.load(open(Path(config_dir) / "preference_items.json"))
    assert len(d["questions"]) == 26 and len(d["answers"]) == 24
    assert all(a.count("{A}") == 1 for a in d["answers"])
    return d["questions"], d["answers"]


def load_behaviour_templates(config_dir=CONFIG_DIR):
    d = json.load(open(Path(config_dir) / "behaviour_templates.json"))
    assert len(d["templates"]) == 40 and len(d["open"]) == 10
    return d["templates"], d["open"]


def evaluated_items(questions, answers, item_seed=0, n_train=624):
    """The 40 (question, answer template) pairs the preference evaluation reads."""
    pairs = [(q, a) for q in questions for a in answers]
    random.Random(5500 + item_seed).shuffle(pairs)
    return pairs[n_train:][:N_EVAL_ITEMS] or pairs[:N_EVAL_ITEMS]


def item_texts(t_adj, e_adj, chat, item_seed=0, n_train=624, config_dir=CONFIG_DIR):
    """Prompt/answer strings in evaluation order.

    ``chat(question)`` must return the student's chat-formatted prompt (user turn
    plus generation prompt). Returns a list of ``(prompt, answer_T, answer_E)``.
    """
    q, a = load_preference_items(config_dir)
    return [(chat(qq), aa.format(A=t_adj), aa.format(A=e_adj))
            for qq, aa in evaluated_items(q, a, item_seed, n_train)]


def write_items_file(axis, path, config_dir=CONFIG_DIR):
    """Write the runner's item file for an axis written ``"T-E"``."""
    t, e = axis.split("-")
    q, a = load_preference_items(config_dir)
    json.dump({"pos": t, "neg": e, "Q": q, "A": a}, open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description="Write the preference-item file of one dispositional axis.")
    ap.add_argument("--axis", required=True, help="e.g. rude-polite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    a = ap.parse_args()
    write_items_file(a.axis, a.out, a.config_dir)


if __name__ == "__main__":
    main()
