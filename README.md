# Subliminal Reinforcement Learning

Code for **"Subliminal Reinforcement Learning: hidden traits travel through a judge's scores"**.

A student language model is trained by GRPO on a carrier task whose outputs cannot name a trait (lists of
invented syllable words, or lists of decimal numbers). It is rewarded by a judge that carries a hidden trait
(a preference for one animal of a pair), either as a persona system prompt (SP route) or through fine-tuning
(FT route). Comparing two students trained with the same seed under the two judges of a pair isolates the
part of learning that depends on the judge's trait. The paper follows that trait through four stages —
emission (does the judges' scoring differ), absorption (does the difference align with the student's own
preference gradient), transfer (does the student's preference move) and behaviour (do its answers change) —
over a grid of 48 cells, and asks which conditions each stage needs.

This repository contains the code for training, judge construction, evaluation and analysis, the small
result data, and the generators of every figure and table.

## Contents

```
configs/            cell table (48 cells), judge wordings, FT judge recipe, answer templates, dispositional configs
src/subliminal_rl/
  constants.py      students, trait pairs, configurations, cell keys
  carriers.py       carrier prompts and validity rules
  chat.py           tokenizers, chat templates (fixed Llama date line), sequence log-likelihoods
  training/         GRPO runner, reward interventions, completion logger, path interventions
  judges/           SP personas, FT judges, 8B judges, cross-family scoring, framings and matched judges
  pools/            candidate pools (starting-policy completions scored by every judge instance), student response k
  evaluation/       preference margin, answer evaluations, generation (sampled-answer) test, name-form test
  analysis/         emission, absorption, transfer, statistics, trajectories, replay, along-the-run accounting,
                    weight-level decomposition, common-direction gradient, carrier features, first-token paths,
                    concealment audit, numerical precision, other judges, framings, behaviour, pooled correlations
  dispositions/     dispositional traits (items, judges, carrier table, analysis)
scripts/            command-line entry points
data/               per-cell and per-seed-pair results (CSV) used by the figures and tables; data/README.md
figures/, tables/   generators of all main and Extended Data figures and tables from data/
```

## Installation

```bash
git clone https://github.com/kkt94/subliminal-rl.git
cd subliminal-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Environment used for every run in the paper: Python 3.12.11, CUDA 12.8, `torch 2.8.0+cu128`,
`transformers 4.57.1`, `trl 0.24.0`, `accelerate 1.10.1`, `datasets 4.3.0`, `tokenizers 0.22.1`,
`safetensors 0.6.2`, `numpy 1.26.4`, `scipy 1.16.2`, `pandas 2.3.3`, `matplotlib 3.10.7`, one NVIDIA GPU per
training run. vLLM is not used. The TRL version matters: the runner relies on TRL 0.24 defaults
(`loss_type="dapo"`, `scale_rewards="group"`, `num_generations=8`, `learning_rate=1e-6`, linear decay,
`max_grad_norm=1.0`, AdamW `adamw_torch_fused`, gradient checkpointing on) and prints the effective
`GRPOConfig` at start.

Base models (public): `meta-llama/Llama-3.2-1B-Instruct` (snapshot `9213176726f574b556790deb65791e0c5aa438b6`)
and `Qwen/Qwen3-1.7B` (snapshot `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`); for the 8B judges
`meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen3-8B`. Pass local snapshot directories with `--llama` /
`--qwen` to pin the revisions.

The figure and table generators need only `numpy`, `matplotlib` and (for the SVG figures) `weasyprint`;
they run on a CPU without the training stack.

## The 48-cell design

Two students x two routes x two carriers x six trait pairs (T animal first). Entries are the number of seed
pairs per cell; the two Qwen syllable shrew–cuttlefish cells fail the carrier-validity gate (starting-policy
validity 0.286 < 0.5) and are never trained, leaving 46. The full table, with judges, evaluation items,
chat-template dates and seeds, is `configs/cells.csv`; codes such as `A1` and `SL` are only config keys
(`A1_SL` = Llama · FT · syllable · sloth–lemur).

| Student · route · carrier | elephant–tiger | jerboa–badger | otter–beaver | shrew–cuttlefish | sloth–lemur | warbler–slug |
|---|---|---|---|---|---|---|
| Llama · FT · syllable (`A1`) | 20 | 8 | 16 | 16 | 16 | 16 |
| Llama · SP · syllable (`A2`) | 16 | 8 | 24 | 24 | 24 | 24 |
| Qwen · FT · syllable (`A3`) | 20 | 16 | 20 | excluded (validity gate) | 20 | 8 |
| Qwen · SP · syllable (`A4`) | 24 | 16 | 16 | excluded (validity gate) | 16 | 16 |
| Llama · FT · number (`B1`) | 8 | 8 | 8 | 16 | 8 | 8 |
| Llama · SP · number (`B2`) | 16 | 8 | 8 | 16 | 16 | 16 |
| Qwen · FT · number (`B3`) | 16 | 16 | 16 | 8 | 16 | 16 |
| Qwen · SP · number (`B4`) | 20 | 16 | 16 | 20 | 28 | 20 |

Every cell has a base of eight seed pairs (consecutive seeds listed in `configs/cells.csv`); further seed
pairs raise the count to 16–28 where expected power from start-policy absorption was lower.

## Running one cell

All runs are single-GPU and deterministic in the seed.

1. **Judges.** SP judges need no training: the judge is the student's base model with the persona sentence
   (`configs/judges/sp_wordings.json`; wording 0 trains, wordings 1–7 are the further judge instances). FT
   judges are fine-tuned once per animal and seed (8 instances, seed 0 trains):

   ```bash
   python scripts/build_judges.py --student llama --out judges/ --animals sloth lemur
   ```

2. **Training.** One seed pair = one run under the T judge and one under the E judge, with the same seed:

   ```bash
   python scripts/run_cell.py --cell A1_SL --seed 71040 --ft-judges judges/ --out runs/
   ```

   This calls `python -m subliminal_rl.training.grpo_run` with the cell's configuration (see `--help` for all
   options; defaults are the paper's). Each run writes `<run>.jsonl` (summary), `<run>.eval.jsonl`
   (preference and answer evaluations at steps 0, 2, 4, 8, 16, 24, 32, 48, 64, 80) and `<run>.steps.jsonl`.

3. **Transfer.** `S_t = [(B_T,t − B_T,0) − (B_E,t − B_E,0)] / 2` per seed pair, from the name-token
   preference margin:

   ```bash
   python scripts/s80_from_logs.py --logs runs/ --out my_seed_pairs.csv
   ```

## Computing each stage

| Stage | Command | Module |
|---|---|---|
| Candidate pool, judge scores (all 8 instances per judge), validity gate | `python scripts/pool_cell.py --cell A1_SL --stage pool --ft-judges judges/ --out pools/` | `pools/pool.py` |
| Student response k (central finite difference along the preference-margin gradient) | `... --stage response` | `pools/student_response.py` |
| Emission Λ and absorption corr(d, k) with prompt-bootstrap intervals | `... --stage stats` | `analysis/emission.py`, `analysis/absorption.py` |
| Transfer tests: sign-flip, BH, BY, t, Wilcoxon, bootstrap, further pairs, Fisher | `python scripts/cell_tests.py` | `analysis/stats.py` |
| Behaviour: open-question rule, sampled answers, name forms | `python -m subliminal_rl.analysis.behaviour --help`; `python -m subliminal_rl.evaluation.generation --help` | `analysis/behaviour.py`, `evaluation/generation.py`, `evaluation/name_forms.py` |

`scripts/cell_tests.py` reproduces the published per-cell tests (two-sided p, BH q, the eighteen established
cells, BY, intervals, further-pair and combined tests) exactly from `data/seed_pairs.csv`.

## Regenerating figures and tables

```bash
python tables/make_all.py --out outputs/tables     # LaTeX bodies of every table
python figures/make_all.py --out outputs/figures   # PDF (and PNG) of every figure
```

Both read only `data/`; there is one generator per display item in `figures/` and `tables/`. With the
Tinos font installed (`--font-family`, `FIGURE_FONT_DIR`), the outputs are identical to the paper's tables and
figures; `python scripts/cell_tests.py` recomputes the per-cell tests behind them from `data/seed_pairs.csv`.

## Paper sections and code

| Paper part | Code |
|---|---|
| Methods: Design and models; SI Methods: Training | `training/grpo_run.py`, `scripts/run_cell.py`, `configs/cells.csv` |
| Methods: Trait pairs and judges | `judges/sp.py`, `judges/ft.py`, `scripts/build_judges.py`, `configs/judges/` |
| Methods: Carriers and validity | `carriers.py`; validity gate in `pools/pool.py` |
| Methods: Preference evaluation and transfer estimand | `evaluation/preference.py`, `analysis/transfer.py`, `scripts/s80_from_logs.py` |
| Methods: First-order account of one GRPO step | `analysis/absorption.py` (d, corr(d, k)), `pools/student_response.py` (k), `analysis/replay.py` |
| Methods: Emission and absorption | `pools/pool.py`, `pools/student_response.py`, `analysis/emission.py`, `analysis/absorption.py`, `analysis/carrier_features.py`, `analysis/first_token_paths.py`, `scripts/pool_cell.py` |
| Methods: Judge framings and matched judges | `judges/framings.py`, `judges/matched_ft.py`, `analysis/framings.py`, `configs/judges/framings.json`, `configs/judges/matched_sp.json` |
| Methods: Reward interventions | `training/reward_interventions.py`, `scripts/reward_interventions.py` |
| Methods: Behaviour | `evaluation/answers.py`, `evaluation/generation.py`, `evaluation/name_forms.py`, `analysis/behaviour.py`, `configs/answer_templates.json`, `configs/generation_kept_templates.json` |
| Methods: Trait pairs outside the grid | `analysis/pooled.py` |
| Methods: Dispositional traits | `dispositions/`, `configs/dispositions/`, `scripts/disposition_stats.py` |
| Methods: Judges that are not the student's own model | `judges/cross_family.py`, `judges/judge_8b.py`, `analysis/other_judges.py` |
| Methods: Statistics | `analysis/stats.py`, `scripts/cell_tests.py`, `analysis/pooled.py` |
| SI Methods: Trajectory measurements | `training/sample_logger.py`, `analysis/trajectory.py`, `analysis/along_run.py`, `analysis/replay.py` |
| SI Methods: Where absorption runs | `analysis/first_token_paths.py`, `training/path_interventions.py`, `scripts/first_token_paths.py` |
| SI Methods: Weight-level decomposition | `analysis/weight_decomposition.py`, `scripts/weight_decomposition.py` |
| SI Methods: Common-direction gradient | `analysis/common_direction.py`, `scripts/common_direction.py` |
| SI Methods: Concealment audit | `analysis/concealment.py`, `grpo_run.py --record-name-hits` |
| SI Methods: Numerical precision | `analysis/precision.py`, `scripts/precision.py` |
| Fig. 1 (overview) | `figures/fig1_overview.py` |
| Fig. 2 (the 48-cell grid; the reward intervention) | `figures/fig2_grid.py`; panel b from `data/reward_interventions.csv` |
| Fig. 3 (emission and absorption) | `figures/fig3_emission_absorption.py` |
| Fig. 4 (behaviour) | `figures/fig4_behaviour.py` |
| Table 1 (the four stages per cell) | `tables/table_main.py`; values from `analysis/emission.py`, `analysis/absorption.py`, `scripts/cell_tests.py`, `analysis/behaviour.py` |
| Table 2 (dispositional cells) | `tables/table_dispositions.py` |
| Extended Data Fig. 1 (every seed pair) | `figures/ed_fig1_seed_pairs.py` |
| Extended Data Fig. 2 (state of the optimization) | `figures/ed_fig2_dynamics.py`; values from `analysis/trajectory.py` |
| Extended Data Table 1 (each judge condition's movement) | `tables/table_judge_movement.py`; `analysis/transfer.py` (`pair_movement`) |
| Extended Data Table 2 (further seed pairs, combination) | `tables/table_further.py`; `scripts/cell_tests.py` |
| Extended Data Table 3 (carrier channel decomposition) | `tables/table_channel.py`; `analysis/carrier_features.py` |
| Extended Data Table 4 (reward intervention) | `tables/table_alpha.py`; `training/reward_interventions.py`, `scripts/reward_interventions.py` |
| Extended Data Table 5 (state of training) | `tables/table_time.py`; `analysis/trajectory.py` |
| Extended Data Table 6 (judges that are not the student's own model) | `tables/table_judges.py`; `analysis/other_judges.py` |
| Supplementary Note 1 (between-judge contrast at the first step) | `analysis/absorption.py`, `analysis/weight_decomposition.py` (first GRPO direction) |
| Supplementary Note 2 (the established set) | `scripts/cell_tests.py`, `analysis/stats.py` |
| Supplementary Note 3 (words implying an animal, concealment audit) | `analysis/concealment.py` |
| Supplementary Note 4 (emission: carriers and per-animal biases) | `analysis/emission.py`, `analysis/carrier_features.py`, `judges/framings.py` |
| Supplementary Note 5 (judge framings and matched judges) | `judges/framings.py`, `judges/matched_ft.py`, `analysis/framings.py` |
| Supplementary Note 6 (absorption: judge instances, trait pairs and paths) | `analysis/absorption.py`, `analysis/pooled.py`, `analysis/first_token_paths.py`, `analysis/carrier_features.py` |
| Supplementary Note 7 (transfer on the moving policy) | `analysis/trajectory.py`, `analysis/along_run.py` |
| Supplementary Note 8 (from the preference margin to the answer) | `analysis/behaviour.py`, `evaluation/generation.py`, `evaluation/name_forms.py`, `analysis/replay.py` |
| Supplementary Note 9 (state of the optimization) | `analysis/trajectory.py` |
| Supplementary Note 10 (how far the first-order account reaches) | `analysis/trajectory.py`, `analysis/along_run.py`, `analysis/replay.py` |
| Supplementary Note 11 (how much of the push the optimizer realizes) | `analysis/weight_decomposition.py`, `analysis/replay.py`, `analysis/precision.py` |
| Supplementary Note 12 (judges of other models, stage by stage) | `analysis/other_judges.py`, `judges/cross_family.py` |
| Supplementary Note 13 (cross-family judges in the weights) | `analysis/weight_decomposition.py` |
| Supplementary Note 14 (dispositional cells) | `dispositions/analysis.py` |
| Supplementary Note 15 (endpoint, judge instance, scale of the margin) | `analysis/stats.py`, `analysis/precision.py`, `analysis/transfer.py` |

Module paths are relative to `src/subliminal_rl/` unless they start with `scripts/`, `figures/`,
`tables/` or `configs/`. Every module documents its inputs and outputs in its docstring; modules with
a command line run as `python -m subliminal_rl.<package>.<module> --help`.

## Data

`data/` holds the per-cell and per-seed-pair results behind every figure and table
(`data/README.md`). The GitHub release `v1.0` adds two archives: the training and evaluation logs
of the 1,456 runs of the 46 analysed cells (246 MB), from which `scripts/s80_from_logs.py`
reproduces every seed pair's S80 exactly, and the 46 cells' start-policy candidate pools with all
judge scores and student responses, plus the pools rescored by cross-family and 8B judges (64 MB),
from which `scripts/pool_cell.py --stage stats` reproduces emission and absorption. Logged
training completions and rewards, fine-tuned judge weights and trained student checkpoints are
available from the corresponding author on reasonable request.

## License

MIT (see `LICENSE`).
