# Data

## Result files

Small result files from which `figures/make_all.py` and `tables/make_all.py` draw every
figure and table of the paper. All files are plain CSV (UTF-8, comma separated, header
row; an empty field means "not applicable / not reported") except the dispositional
results, which are JSON in the format written by `subliminal_rl.dispositions.analysis`.

Common conventions. A *cell* is one student x route x carrier x trait pair; its key
`<configuration>_<pair>` (for example `A1_SL`) is spelled out in the `student`, `route`,
`carrier` and `trait_pair` columns of `cells.csv`. Configurations: A1 Llama FT syllable,
A2 Llama SP syllable, A3 Qwen FT syllable, A4 Qwen SP syllable, B1-B4 the same with the
number carrier. Pairs: ET elephant-tiger, JB jerboa-badger, OB otter-beaver,
WC shrew-cuttlefish, SL sloth-lemur, WS warbler-slug (the T judge prefers the first
animal). S is half the T-minus-E difference of the change in the preference margin from
step 0 (Methods, equation S); S80 is S at step 80, in nats. Every cell has eight *base*
seed pairs; 35 cells have *further* seed pairs (16 to 28 in all). Values are stored at
full precision; the tables round them.

### cells.csv
One row per design cell (48). Extended Data Table 1, Figs 2-3, Extended Data Figs 1-2.

| column | meaning |
|---|---|
| `cell`, `student`, `route`, `carrier`, `trait_pair` | cell key and its parts (route FT = fine-tuned judges, SP = persona judges) |
| `status` | `analysed`, or `excluded_validity_gate` for the two Qwen syllable shrew-cuttlefish cells (all other columns empty) |
| `n_seed_pairs`, `n_base_pairs`, `n_further_pairs` | seed pairs in the cell (all = base + further) |
| `s80` | mean S80 over all seed pairs |
| `pairs_judge_direction` | seed pairs with S80 > 0 (all pairs) |
| `p_two_sided` | two-sided sign-flip p of the mean S80 over all pairs (enumerated up to 20 pairs, otherwise 200,000 fixed-seed assignments) |
| `q_bh` | Benjamini-Hochberg q of `p_two_sided` over the 46 analysed cells |
| `established` | `yes` where p < 0.05 and q < 0.10 |
| `s80_boot_lo`, `s80_boot_hi` | 95% bootstrap interval of the mean over seed pairs (20,000 resamples) |
| `s80_t_lo`, `s80_t_hi` | 95% t interval |
| `s80_base`, `pairs_judge_direction_base` | mean S80 and pairs with S80 > 0 over the eight base pairs |
| `p_one_sided_T_base`, `p_one_sided_E_base` | exact one-sided sign-flip p of the base-pair mean towards the T judge and towards the E judge |
| `lambda`, `lambda_lo`, `lambda_hi` | emission Lambda at the starting policy with 95% interval (pairings sharing an index excluded, Methods equation Lambda) |
| `lambda_all_pairings`, `..._lo`, `..._hi` | the same over all 64 pairings, including those sharing an index (not reported in the paper) |
| `emission_status` | `present` (lower bound > 0), `negative` (upper bound < 0) or `absent` |
| `absorption`, `absorption_lo`, `absorption_hi` | absorption at the starting policy (training instance) with 95% interval |
| `absorption_status` | `aligned_positive`, `aligned_negative` (interval excludes zero) or `not_significant` |
| `answer_evaluation` | whether the cell's open-question evaluation is `valid` or `degenerate` |
| `validity_step_reaching_0.9` | first evaluation step at which mean carrier validity is at least 0.9 |
| `validity_lowest_run_step80` | carrier validity at step 80 in the cell's lowest run |
| `signal_share_below_half_step` | step (linearly interpolated) at which the reward-signal share falls below 0.5; empty if it never does |
| `step_50pct_s80`, `step_80pct_s80` | first evaluation step at which the base-pair mean S reaches 50% / 80% of its step-80 value |

### seed_pairs.csv
S80 of every seed pair (728 rows): `cell`, `group` (`base` for the first eight, `further`
for the rest), `index` (1-based within the cell, base pairs first), `s80`.
Extended Data Fig. 1; source of the cell statistics above.

### trajectory.csv
Mean S over the eight base seed pairs at the evaluation steps 0, 2, 4, 8, 16, 24, 32, 48,
64 and 80: `cell`, `step`, `s_base`. Extended Data Fig. 2c.

### training_state.csv
State of training by step: `cell`, `step`, `validity` (mean carrier validity of the
completions, logged at the evaluation steps 2-80) and `reward_signal_share` (share of
groups whose eight completions do not all receive the same reward, logged every ten
steps). Extended Data Table 6, Extended Data Fig. 2a-b.

### judge_movement.csv
Change of the preference margin from step 0 to step 80 under each judge, eight base seed
pairs (Extended Data Table 2): `delta_B_T`, `delta_B_E` (mean change of the T-trained and
E-trained students), `common_push` (their mean), `s` (half their difference),
`pairs_same_direction` (base pairs in which both students move the same way, out of
`n_base_pairs`).

### further_pairs.csv
Transfer on a cell's further seed pairs and the combination of the two groups (Extended
Data Table 3). Further pairs alone (35 cells): `s80_further` with bootstrap 95% interval
`s80_further_lo/hi`, `further_judge_direction` (pairs with S80 > 0), two-sided sign-flip
`p_further`, BH `q_further` over the 35 cells, `meets_criterion_further` (p < 0.05 and
q < 0.10). All 46 cells: `p_base` (exact two-sided p of the base eight), `p_combined`
(Fisher combination of the one-sided p of the base and further groups per direction,
two-sided as twice the smaller; a cell without further pairs enters with p = 1 for that
group), `q_combined` (BH over the 46), `meets_criterion_combined`.

### channel_decomposition.csv
Carrier channel decomposition of absorption (Extended Data Table 4). Training-instance
absorption (start-policy standardized score difference) and its split into `feature_part`
(carried by the carrier's surface features: 26 letter frequencies and the 100 commonest
bigrams for syllables, 14 number statistics for numbers) and `residual`, the feature share,
`judge_coefficient` and `student_coefficient`; the same for the instance-mean score
difference (`instance_mean_*`), the correlation of the two score differences and the
number of stored judge instances.

### judge_conditions.csv
Judges that are not the student's own model (Extended Data Table 7, Fig. 3c), one row per
route x student x judge (`own`, `other_family_1b`, `same_family_8b`, `other_family_8b`):
`lambda_mean` (mean emission over the cells), `cells_lambda_lo_positive` of `n_cells`,
`trait_alignment_r` (correlation of the judge's trait difference with that of the own-model
judge), `abs_absorption_mean`, `cells_absorption_sign_as_own`, `judge_term`,
`absorption_s80_r` (across-cell correlation of absorption with S80, per student, pooled
over the two routes), `transfer_established` of `transfer_cells_trained`.

### reward_interventions.csv
Training on c + alpha s (Extended Data Table 5), eight seed pairs per condition: `alpha`,
`s80`, `pairs_judge_direction` (where reported), `difference_from_alpha1` with paired
`p_paired`, final `kl` and carrier `validity` (empty where not recorded).

### behaviour.csv
Answer-level result of each cell as given in Extended Data Table 1 (`answer_level`) and the
table's footnote letter and text (`note`, `note_text`).

### answer_forms.csv, gradient_overlap.csv, behaviour_ladder.csv
Fig. 4. `answer_forms.csv`: margin log p(T form) - log p(E form) of each answer form
(capital, capital plural, lowercase, with article) in three Llama SP number cells, and
whether the sampled direct-answer choice changed (nominal). `gradient_overlap.csv`: cosine
at the starting policy between the gradient of the preference margin and that of each
form. `behaviour_ladder.csv`: number of cells measured at each layer from transfer to a
sampled answer (stages are not nested).

### dispositions/
Dispositional cells (Table 1; Supplementary Note on dispositional traits), as written by
`subliminal_rl.dispositions.analysis` (see its docstring for definitions). Cells are keyed
`<student>|<carrier>|<axis>`, e.g. `Qwen|number|rude-polite`.

- `disp_grid.json`: per cell, absorption `U` with interval, baseline margin `B0`, effect
  size d of each evaluation (`transfer`, `fc` forced choice, `open` open question) at each
  step, and the steps where each is established (`ok`; for `choice`: step, shift in
  percentage points, share of items on which the two students differ, q).
- `disp_endpoint.json`: transfer at step 16, step 80 and averaged over steps 8-80, each
  with d, p, BH q across the twelve cells and the verdict.
- `disp_lambda.json`: emission Lambda (eight wordings per judge) with 95% interval.
- `disp_screen.json`: the 24 screened axes (Qwen syllable candidate pool): emission ratio,
  absorption with interval and the finite-difference step used, baseline margin.

## Large artefacts (to be deposited at Zenodo)

The artefacts below are not in this repository. They are available from the
corresponding author on reasonable request and will be deposited at Zenodo on
publication. Sizes are approximate; file formats are those written by the code in
`src/` (names given as `<...>`).

| Artefact | Written by | Contents | Size |
|---|---|---|---|
| Training logs of the grid | `training/grpo_run.py` | 728 seed pairs of the 46 cells (1,456 runs): `<run>.eval.jsonl` (preference and answer evaluations at 10 steps, ~0.22 MB each), `<run>.steps.jsonl`, `<run>.jsonl` | ~0.35 GB |
| Logged training completions and rewards | `training/sample_logger.py`, `--record-name-hits`, reward-intervention dumps | every prompt, completion and reward of the trajectory runs (~1.9 GB), framing runs (~0.5 GB), cross-family runs (~0.4 GB) and reward-intervention runs (~1.8 GB); rescored samples (~0.5 GB); per-update completion records of the syllable cells (~0.2 GB) | ~5.3 GB |
| Candidate pools of the 46 cells | `pools/pool.py`, `pools/student_response.py` | 3,072 starting-policy completions per cell, scores of every judge instance, student log-likelihood, student response k at six step sizes | ~0.15 GB |
| Rescored pools | `judges/cross_family.py`, `judges/framings.py` | pools scored by cross-family and 8B judges (~0.23 GB), by the three framings for 36 animals and by matched judges (~0.2 GB) | ~0.45 GB |
| FT judge weights | `judges/ft.py` | 12 animals x 8 fine-tuning seeds x 2 base models = 192 checkpoints, fp32 (Llama 4.7 GB, Qwen 6.5 GB each) | ~1.08 TB |
| 8B FT judge weights | `judges/judge_8b.py` | training instance (seed 0) of each of the 12 animals for Llama-3.1-8B-Instruct and Qwen3-8B, bf16 (seeds 1-7 were deleted after scoring the pools) | ~0.36 TB |
| Matched FT judge weights | `judges/matched_ft.py` | 96 checkpoints | ~0.22 TB |
| Logs of the further experiments | `training/grpo_run.py` and the modules named | cross-family and 8B training (~0.5 GB), reward interventions and matched judges (~0.2 GB), framing ladder (~0.13 GB), dispositional cells (~0.46 GB, plus ~45 MB pool statistics), generation-test samples (~0.1 GB), name-form, along-the-run, replay and weight-decomposition results (a few MB) | ~1.4 GB |
| First GRPO directions | `analysis/weight_decomposition.py` | bf16 direction vectors per route and judge family (same-model and cross-family) | ~0.18 TB |
| End-of-training students of the weight decomposition | `training/grpo_run.py --save-final` | partially retained (most checkpoints were deleted after decomposition); the decomposition results themselves are in the logs above | ≤0.64 TB |

The base models are public (`meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-1.7B`,
`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B`); the exact student snapshots are listed in
`src/subliminal_rl/constants.py`.
