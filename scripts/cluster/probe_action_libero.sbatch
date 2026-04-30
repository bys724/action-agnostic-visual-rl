#!/bin/bash
# LIBERO Action Probing (Phase 2 보강) — 한 잡 = (encoder × suite) × 4 gaps loop.
#
# 사용:
#   # v11 ep44 × spatial (4 gaps)
#   sbatch --export=ALL,ENCODER=two-stream-v11,\
# CHECKPOINT=/proj/external_group/mrg/checkpoints/two_stream_v11/20260426_014333/checkpoint_epoch0044.pt,\
# TASK_SUITE=libero_spatial \
#          scripts/cluster/probe_action_libero.sbatch
#
#   # baselines (HF cache 사용, ckpt 불필요)
#   sbatch --export=ALL,ENCODER=dinov2,TASK_SUITE=libero_spatial scripts/cluster/probe_action_libero.sbatch
#
#   # sanity (1 task × 5 demos × gap=20만)
#   sbatch --export=ALL,ENCODER=videomae-ours,\
# CHECKPOINT=/proj/external_group/mrg/checkpoints/videomae/20260415_012017/best_model.pt,\
# TASK_SUITE=libero_spatial,TASK_IDS=0,MAX_DEMOS=5,GAPS=20,SUFFIX=sanity \
#          scripts/cluster/probe_action_libero.sbatch
#
#SBATCH --job-name=probe_libero
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/proj/external_group/mrg/logs/probe_libero_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/probe_libero_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
LIBERO_ROOT="/proj/external_group/mrg/datasets/libero"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
OUT_BASE="$PROJECT_ROOT/paper_artifacts/libero_action_probing"

ENCODER="${ENCODER:?ENCODER required}"
CHECKPOINT="${CHECKPOINT:-}"
TASK_SUITE="${TASK_SUITE:?TASK_SUITE required}"
VIEW="${VIEW:-agentview_rgb}"
GAPS="${GAPS:-1 13 20 40}"
TASK_IDS="${TASK_IDS:-}"
MAX_DEMOS="${MAX_DEMOS:-}"
V11_MODE="${V11_MODE:-abd_prime}"
SUFFIX="${SUFFIX:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_BASE/${ENCODER}_${TASK_SUITE}_${TIMESTAMP}${SUFFIX:+_$SUFFIX}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [probe_libero] $*"; }

log "=== LIBERO Action Probing ==="
log "Encoder: $ENCODER"
log "Checkpoint: ${CHECKPOINT:-<none>}"
log "Task suite: $TASK_SUITE | view: $VIEW | gaps: $GAPS"
log "Output: $OUT_DIR"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# HF cache for DINOv2/SigLIP
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"

mkdir -p "$OUT_DIR"

ARGS=(
    --encoder "$ENCODER"
    --task-suite "$TASK_SUITE"
    --data-root "$LIBERO_ROOT"
    --view "$VIEW"
    --output-dir "$OUT_DIR"
    --gaps $GAPS
)
if [[ -n "$CHECKPOINT" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "$TASK_IDS" ]]; then ARGS+=(--task-ids $TASK_IDS); fi
if [[ -n "$MAX_DEMOS" ]]; then ARGS+=(--max-demos-per-task "$MAX_DEMOS"); fi
if [[ "$ENCODER" == "two-stream-v11" ]]; then ARGS+=(--v11-mode "$V11_MODE"); fi

"$CONDA_PREFIX/bin/python" scripts/eval/probe_action_libero.py "${ARGS[@]}"

log "=== done ==="
