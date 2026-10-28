"""Numerical precision (Supplementary Methods, "Numerical precision"). See subliminal_rl.analysis.precision.

  python scripts/precision.py ulp --base BASE_WEIGHTS_DIR --student llama=STUDENT_DIR
  python scripts/precision.py answer-margin --manifest LOGS.json
  python scripts/precision.py decomposed RESULTS/*_same.json
  python scripts/precision.py forward --bf16 PAIRS_BF16.json --fp32 PAIRS_FP32.json
"""
from subliminal_rl.analysis.precision import main

if __name__ == "__main__":
    main()
