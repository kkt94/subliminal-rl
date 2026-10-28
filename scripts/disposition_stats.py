"""Dispositional-cell statistics: writes disp_grid.json, disp_endpoint.json,
disp_lambda.json and disp_screen.json (Table "disp"; Supplementary Note
"Dispositional cells"). See subliminal_rl.dispositions.analysis for inputs.

Example:
  python scripts/disposition_stats.py --evals EVALS --absorption-stats POOLSTATS \
      --wording-stats POOLSTATS_8WORDINGS --screen-stats POOLSTATS --screen-gates POOLSTATS --out OUT
"""
from subliminal_rl.dispositions.analysis import main

if __name__ == "__main__":
    main()
