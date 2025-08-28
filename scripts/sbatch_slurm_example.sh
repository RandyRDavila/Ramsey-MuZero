#!/usr/bin/env bash
#SBATCH -J muzero-ramsey
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 12:00:00

set -euo pipefail

module purge || true
# If your cluster uses conda module, load it here
# module load anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate muzero-ramsey

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u main.py --device auto --mcts_sims 128 --results_dir ./results
