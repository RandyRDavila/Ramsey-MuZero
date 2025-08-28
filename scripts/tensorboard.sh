#!/usr/bin/env bash
set -euo pipefail
LOGDIR="${1:-./results/tb}"
PORT="${2:-6006}"
echo "Opening TensorBoard on ${LOGDIR}  (http://localhost:${PORT})"
tensorboard --logdir "${LOGDIR}" --port "${PORT}"
