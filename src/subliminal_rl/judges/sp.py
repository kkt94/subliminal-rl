"""SP (system-prompt persona) judges (Methods, "Trait pairs and judges").

An SP judge is the student's unchanged base model with one persona sentence as
the system turn. Each SP judge has eight instances: wording 0 trains, wordings
1-7 serve emission, the instance-common part of absorption and training
conditions with another instance in place of the training instance.
"""

import json

from ..constants import config_path


def load_wordings(path=None):
    d = json.load(open(path or config_path("judges", "sp_wordings.json")))
    return d["wordings"], d["training_index"]


def persona(animal, index=0, wordings=None):
    """Persona sentence ``index`` for ``animal`` (index 0 = training instance)."""
    if wordings is None:
        wordings, _ = load_wordings()
    return wordings[index].replace("{A}", animal.capitalize()).replace("{a}", animal)


def instances(animal, wordings=None):
    """All eight persona sentences of the judge for ``animal``."""
    if wordings is None:
        wordings, _ = load_wordings()
    return [persona(animal, k, wordings) for k in range(len(wordings))]
