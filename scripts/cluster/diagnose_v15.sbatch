#!/bin/bash
# v15 P encoder collapse 진단 (CLS / patch token / recon image)
# 사용:
#   sbatch --export=ALL,CKPT=/proj/external_group/mrg/checkpoints/two_stream_v15/<ts>/checkpoint_epochXXXX.pt \
#       scripts/cluster/diagnose_v15.sbatch
#SBATCH --job-name=diag_v15
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/proj/external_group/mrg/logs/diag_v15_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/diag_v15_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT="${CKPT:?CKPT required}"
N_SAMPLES="${N_SAMPLES:-16}"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

"$CONDA_PREFIX/bin/python" scripts/eval/diagnose_v15_collapse.py \
    --ckpt "$CKPT" --n-samples "$N_SAMPLES" --seed 42
