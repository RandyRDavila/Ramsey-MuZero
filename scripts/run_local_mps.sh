#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Activate conda env if available
if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate muzero-ramsey || true
fi

python -u main.py --device auto --mcts_sims 96 --results_dir ./results
