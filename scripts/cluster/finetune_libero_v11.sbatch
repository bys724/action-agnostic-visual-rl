#!/bin/bash
# LIBERO BC fine-tuning sbatch — Two-Stream v11 전용
#
# 사용:
#   sbatch --export=ALL,CHECKPOINT=/path/to/v11_ckpt.pt,TASK_SUITE=libero_spatial \
#          scripts/cluster/finetune_libero_v11.sbatch
#   # cls_mode 변경:
#   sbatch --export=ALL,CHECKPOINT=...,CLS_MODE=patch_mean_p_features_tk \
#          scripts/cluster/finetune_libero_v11.sbatch
#
#SBATCH --job-name=ft_libero_v11
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/proj/external_group/mrg/logs/ft_libero_v11_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/ft_libero_v11_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
LIBERO_ROOT="/proj/external_group/mrg/datasets/libero"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT_BASE="/proj/external_group/mrg/checkpoints/libero"

# overridable
CHECKPOINT="${CHECKPOINT:?CHECKPOINT required}"
CLS_MODE="${CLS_MODE:-patch_mean_concat_enc_phase3}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
P_DEPTH="${P_DEPTH:-12}"
M_DEPTH="${M_DEPTH:-6}"
SUFFIX="${SUFFIX:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$CKPT_BASE/two_stream_v11_${TASK_SUITE}_${TIMESTAMP}${SUFFIX:+_$SUFFIX}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ft_libero_v11] $*"; }

log "=== v11 LIBERO BC fine-tuning start ==="
log "Checkpoint: $CHECKPOINT"
log "Task: $TASK_SUITE, cls_mode: $CLS_MODE, epochs: $EPOCHS"
log "p_depth: $P_DEPTH, m_depth: $M_DEPTH"
log "Output: $OUTPUT_DIR"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
"$CONDA_PREFIX/bin/python" scripts/eval/finetune_libero_v11.py \
    --encoder two-stream-v11 \
    --checkpoint "$CHECKPOINT" \
    --data-root "$LIBERO_ROOT" \
    --task-suite "$TASK_SUITE" \
    --cls-mode "$CLS_MODE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --p-depth "$P_DEPTH" \
    --m-depth "$M_DEPTH" \
    --output-dir "$OUTPUT_DIR"

log "=== v11 LIBERO BC fine-tuning done ==="
