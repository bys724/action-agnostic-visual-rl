#!/bin/bash
# v7-big CLS 진단: cosine similarity + decoder swap ablation
#SBATCH --job-name=diag_v7big_cls
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/proj/external_group/mrg/logs/diag_v7big_cls_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/diag_v7big_cls_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT="${CKPT:-/proj/external_group/mrg/checkpoints/two_stream_v7big_isolated/20260419_220105/checkpoint_epoch0004.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

"$CONDA_PREFIX/bin/python" scripts/eval/diagnose_v7big_cls.py \
    --checkpoint "$CKPT" \
    --egodex-dir /proj/external_group/mrg/datasets/egodex/frames/part4 \
    --droid-dir /proj/external_group/mrg/datasets/droid_frames/ext1 \
    --num-samples "$NUM_SAMPLES" \
    --gap 30 --depth 12 --num-stages 2 --sigma 0.03
