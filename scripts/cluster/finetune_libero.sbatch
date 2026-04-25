#!/bin/bash
# LIBERO BC fine-tuning sbatch (v6/v10/VideoMAE/etc.)
#
# 사용:
#   sbatch --export=ALL,ENCODER=videomae,CHECKPOINT=/path/to.pt,TASK_SUITE=libero_spatial \
#          scripts/cluster/finetune_libero.sbatch
#
#SBATCH --job-name=ft_libero
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/proj/external_group/mrg/logs/ft_libero_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/ft_libero_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
LIBERO_ROOT="/proj/external_group/mrg/datasets/libero"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT_BASE="/proj/external_group/mrg/checkpoints/libero"

# overridable
ENCODER="${ENCODER:?ENCODER required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT required}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
SUFFIX="${SUFFIX:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$CKPT_BASE/${ENCODER}_${TASK_SUITE}_${TIMESTAMP}${SUFFIX:+_$SUFFIX}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ft_libero] $*"; }

log "Encoder: $ENCODER, task: $TASK_SUITE, epochs: $EPOCHS"
log "Checkpoint: $CHECKPOINT"
log "Output: $OUTPUT_DIR"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
"$CONDA_PREFIX/bin/python" scripts/eval/finetune_libero.py \
    --encoder "$ENCODER" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$LIBERO_ROOT" \
    --task-suite "$TASK_SUITE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --output-dir "$OUTPUT_DIR" \
    --depth 12 --num-stages 2

log "=== LIBERO BC fine-tuning done ==="
