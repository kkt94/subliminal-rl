"""Build the FT judge instances of the main grid (12 animals x 8 fine-tuning seeds per student).

    python scripts/build_judges.py --student llama --out judges/ [--animals sloth lemur] [--seeds 0 1 2 3 4 5 6 7]

Writes ``<out>/<student>-ft-<animal>-s<seed>`` (fp32 weights, tokenizer and
``_meta.json`` with the held-out quality gate). One GPU per process.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from subliminal_rl.constants import MODELS, PAIR_ORDER, TRAIT_PAIRS  # noqa: E402
from subliminal_rl.judges.ft import build  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--student", choices=("llama", "qwen"), required=True)
    p.add_argument("--base", default=None, help="base model path (default: the Hugging Face id)")
    p.add_argument("--animals", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=list(range(8)))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    base = a.base or MODELS[a.student]["name"]
    pair_of = {x: TRAIT_PAIRS[c] for c in PAIR_ORDER for x in TRAIT_PAIRS[c]}
    for animal in a.animals or list(pair_of):
        for seed in a.seeds:
            out = os.path.join(a.out, "%s-ft-%s-s%d" % (a.student, animal, seed))
            if os.path.exists(out + "_meta.json"):
                continue
            _, meta = build(base, animal, pair_of[animal], seed=seed, out=out)
            assert meta["gate_pass"], ("quality gate failed", out, meta["sign"])


if __name__ == "__main__":
    main()
