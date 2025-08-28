# #!/usr/bin/env bash
# set -euo pipefail
# # Ensure we’re in repo root even if called from elsewhere
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
# cd "$REPO_ROOT"

# # Activate env if conda is available
# if command -v conda >/dev/null 2>&1; then
#   # shellcheck disable=SC1091
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda activate muzero-ramsey || true
# fi

# python -u scripts/ramsey_cli.py
#!/usr/bin/env bash
set -euo pipefail

# macOS: avoid OpenMP duplicate runtime crash; enable MPS fallback
if [[ "$(uname -s)" == "Darwin" ]]; then
  export KMP_DUPLICATE_LIB_OK=TRUE
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
fi

# pretty banner is inside the CLI; just run it
python -u scripts/ramsey_cli.py "$@"
