"""Common-direction gradient h_C, response k_C and the push u_{C,z} (Supplementary Methods,
"Common-direction gradient"). See subliminal_rl.analysis.common_direction.

  python scripts/common_direction.py response --start-policy BASE --carrier syllable-llama --pair WS \
      --pool POOL.npz --fixdate DATE --out kC.npz                                   # GPU
  python scripts/common_direction.py push --response kC.npz --pool POOL.npz \
      --scores same=pool --scores cross=CROSS_SCORES.npz --out push.json           # CPU
"""
from subliminal_rl.analysis.common_direction import main

if __name__ == "__main__":
    main()
