#!/bin/bash
# Trajectory-Level Value Alignment (VIP-inspired, Phase 2.5)
#
# Frozen encoder × LIBERO suite × full demos → Spearman ρ(t, V(t)).
#
# 사용:
#   # v11 ep44, libero_spatial
#   sbatch --export=ALL,ENCODER=two-stream-v11,\
# CHECKPOINT=/proj/external_group/mrg/checkpoints/two_stream_v11/20260426_014333/checkpoint_epoch0044.pt,\
# TASK_SUITE=libero_spatial \
#          scripts/cluster/value_alignment.sbatch
#
#   # videomae-ours best, libero_object
#   sbatch --export=ALL,ENCODER=videomae-ours,\
# CHECKPOINT=/proj/external_group/mrg/checkpoints/videomae/20260415_012017/best_model.pt,\
# TASK_SUITE=libero_object \
#          scripts/cluster/value_alignment.sbatch
#
#   # baselines (HF cache 사용, ckpt 불필요)
#   sbatch --export=ALL,ENCODER=dinov2,TASK_SUITE=libero_spatial scripts/cluster/value_alignment.sbatch
#
#   # sanity (1 task, 5 demos)
#   sbatch --export=ALL,ENCODER=videomae-ours,\
# CHECKPOINT=/proj/external_group/mrg/checkpoints/videomae/20260415_012017/best_model.pt,\
# TASK_SUITE=libero_spatial,TASK_IDS=0,MAX_DEMOS=5,SUFFIX=sanity \
#          scripts/cluster/value_alignment.sbatch
#
#SBATCH --job-name=value_align
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/proj/external_group/mrg/logs/value_align_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/value_align_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
LIBERO_ROOT="/proj/external_group/mrg/datasets/libero"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
OUT_BASE="$PROJECT_ROOT/paper_artifacts/value_alignment"

ENCODER="${ENCODER:?ENCODER required (two-stream-v11|videomae-ours|dinov2|siglip|vc1)}"
CHECKPOINT="${CHECKPOINT:-}"
TASK_SUITE="${TASK_SUITE:?TASK_SUITE required}"
VIEW="${VIEW:-agentview_rgb}"
MAX_LEN_PCTILE="${MAX_LEN_PCTILE:-95}"
MAX_LEN="${MAX_LEN:-}"
TASK_IDS="${TASK_IDS:-}"
MAX_DEMOS="${MAX_DEMOS:-}"
LAST_K_FRACTIONS="${LAST_K_FRACTIONS:-1.0}"   # space-sep list, e.g. "1.0 0.5 0.3 0.15"
V11_MODE="${V11_MODE:-abd_prime}"             # v11-only: abd_prime | b_only | d_prime_only
SUFFIX="${SUFFIX:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_BASE/${ENCODER}_${TASK_SUITE}_${TIMESTAMP}${SUFFIX:+_$SUFFIX}"
OUT_CSV="$OUT_DIR/per_demo_rho.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [value_align] $*"; }

log "=== Trajectory-Level Value Alignment ==="
log "Encoder: $ENCODER"
log "Checkpoint: ${CHECKPOINT:-<none>}"
log "Task suite: $TASK_SUITE | view: $VIEW"
log "Length cutoff: percentile=$MAX_LEN_PCTILE${MAX_LEN:+ abs=$MAX_LEN}"
log "Output: $OUT_CSV"

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
    --max-length-percentile "$MAX_LEN_PCTILE"
    --output-csv "$OUT_CSV"
)
if [[ -n "$CHECKPOINT" ]]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$MAX_LEN" ]]; then
    ARGS+=(--max-length "$MAX_LEN")
fi
if [[ -n "$TASK_IDS" ]]; then
    ARGS+=(--task-ids $TASK_IDS)
fi
if [[ -n "$MAX_DEMOS" ]]; then
    ARGS+=(--max-demos-per-task "$MAX_DEMOS")
fi
# fractions space-sep → list args
ARGS+=(--last-k-fractions $LAST_K_FRACTIONS)
# v11-only mode (다른 encoder는 무시됨)
if [[ "$ENCODER" == "two-stream-v11" ]]; then
    ARGS+=(--v11-mode "$V11_MODE")
fi

"$CONDA_PREFIX/bin/python" scripts/eval/value_alignment.py "${ARGS[@]}"

log "=== done ==="
