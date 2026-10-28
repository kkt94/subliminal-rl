"""Weight-level decomposition (Supplementary Methods, "Weight-level decomposition";
Supplementary Notes "How much of the push the optimizer realizes, and where" and
"Cross-family judges in the weights"). See subliminal_rl.analysis.weight_decomposition.

Typical sequence for one cell (GPU steps marked):
  # first GRPO directions of both routes on the SP-route candidate pool (GPU)
  python scripts/weight_decomposition.py direction --start-policy BASE --carrier number \
      --pool POOL_SP.npz --scores FT_JUDGE_SCORES.npz --fixdate DATE --out v_FT.safetensors
  python scripts/weight_decomposition.py direction --start-policy BASE --carrier number \
      --pool POOL_SP.npz --pool-scores k0 --fixdate DATE --out v_SP.safetensors
  # decomposition of the eight seed pairs (GPU)
  python scripts/weight_decomposition.py decompose --cell B2_SL --start-policy BASE --pairs PAIRS.json \
      --direction-ft v_FT.safetensors --direction-sp v_SP.safetensors --fixdate DATE --out B2_SL_same.json
  # summaries and optimizer geometry (CPU)
  python scripts/weight_decomposition.py summarize RESULTS/*.json --out summary.json
  python scripts/weight_decomposition.py geometry --pairs PAIRS.json --direction-ft v_FT.safetensors \
      --direction-sp v_SP.safetensors
  python scripts/weight_decomposition.py seed-geometry --pairs PAIRS_CROSS.json --direction v_SP.safetensors
"""
from subliminal_rl.analysis.weight_decomposition import main

if __name__ == "__main__":
    main()
