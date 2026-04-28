#!/bin/bash
# LIBERO 공식 BC-Transformer + 우리 인코더 어댑터 — 클러스터 학습
#
# 사용:
#   sbatch --export=ALL,ENCODER=two-stream-v11,CHECKPOINT=/proj/.../v11_ep44.pt \
#          scripts/cluster/finetune_libero_bct.sbatch
#   # baseline (no encoder ckpt 필요):
#   sbatch --export=ALL,ENCODER=dinov2 scripts/cluster/finetune_libero_bct.sbatch
#   # V-JEPA 2.1 (seq_len=25):
#   sbatch --export=ALL,ENCODER=vjepa2-1,SEQ_LEN=25 scripts/cluster/finetune_libero_bct.sbatch
#
#SBATCH --job-name=ft_libero_bct
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/proj/external_group/mrg/logs/ft_libero_bct_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/ft_libero_bct_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
LIBERO_ROOT="/proj/external_group/mrg/datasets/libero"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT_BASE="/proj/external_group/mrg/checkpoints/libero_bct"

# overridable
ENCODER="${ENCODER:?ENCODER required (two-stream-v11|videomae-ours|dinov2|siglip|vc1|vjepa2-1)}"
CHECKPOINT="${CHECKPOINT:-}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
SEQ_LEN="${SEQ_LEN:-10}"   # V-JEPA = 25
SEED="${SEED:-0}"
SUFFIX="${SUFFIX:-}"
TASK_IDS="${TASK_IDS:-}"               # sanity용 (e.g., "0")
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"  # sanity용

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$CKPT_BASE/${ENCODER}_${TASK_SUITE}_seed${SEED}_${TIMESTAMP}${SUFFIX:+_$SUFFIX}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ft_libero_bct] $*"; }

log "=== LIBERO BC-T fine-tuning start ==="
log "Encoder: $ENCODER"
log "Checkpoint: ${CHECKPOINT:-<none>}"
log "Task: $TASK_SUITE | epochs: $EPOCHS | seq_len: $SEQ_LEN | seed: $SEED"
log "Output: $OUTPUT_DIR"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# HF cache 사용 (CLIP/DINOv2/SigLIP 미리 캐시됨) — 동시 다운로드 race 회피
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"

# Compose CLI args
ARGS=(
    --encoder "$ENCODER"
    --task-suite "$TASK_SUITE"
    --data-root "$LIBERO_ROOT"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --seq-len "$SEQ_LEN"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
)
if [[ -n "$CHECKPOINT" ]]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$TASK_IDS" ]]; then
    # space로 분리된 ID들. eval로 word splitting
    ARGS+=(--task-ids $TASK_IDS)
fi
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
    ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi

"$CONDA_PREFIX/bin/python" scripts/eval/finetune_libero_bct.py "${ARGS[@]}"

log "=== LIBERO BC-T fine-tuning done ==="
