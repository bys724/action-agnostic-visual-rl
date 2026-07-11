#!/bin/bash
# JHMDB correspondence label-propagation (readout-free PCK) — docs/correspondence_eval_plan.md
#
# 사용:
#   # CoMP P-stream
#   sbatch --export=ALL,ENCODER=parvo,PARVO_STREAM=p,\
# CHECKPOINT=/proj/external_group/mrg/checkpoints/two_stream_v15b_step1_comp_mae_s/20260629_101634/latest.pt \
#          scripts/cluster/correspondence_labelprop.sbatch
#
#   # baseline (HF cache, ckpt 불필요)
#   sbatch --export=ALL,ENCODER=dinov2 scripts/cluster/correspondence_labelprop.sbatch
#
#   # sanity (5클립)
#   sbatch --export=ALL,ENCODER=dinov2,MAX_VIDEOS=5,SUFFIX=sanity \
#          scripts/cluster/correspondence_labelprop.sbatch

#SBATCH --job-name=corr_jhmdb
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/proj/external_group/mrg/logs/corr_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/corr_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
JHMDB_ROOT="/proj/external_group/mrg/datasets/jhmdb"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
OUT_BASE="$PROJECT_ROOT/paper_artifacts/correspondence_labelprop"

ENCODER="${ENCODER:?ENCODER required}"
CHECKPOINT="${CHECKPOINT:-}"
PARVO_STREAM="${PARVO_STREAM:-p}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
SUFFIX="${SUFFIX:-}"
# 🔴 parity: 전파 하이퍼는 기본값 고정 (encoder별 변경 금지 — eval_protocols §0)

TAG="$ENCODER"; [ "$ENCODER" = "parvo" ] && TAG="${ENCODER}_${PARVO_STREAM}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_BASE/${TAG}_${TIMESTAMP}${SUFFIX:+_$SUFFIX}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [corr] $*"; }
log "=== JHMDB correspondence: $TAG → $OUT_DIR ==="

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
ARGS=(--encoder "$ENCODER" --jhmdb-root "$JHMDB_ROOT" --output-dir "$OUT_DIR"
      --parvo-stream "$PARVO_STREAM" --max-videos "$MAX_VIDEOS")
[ -n "$CHECKPOINT" ] && ARGS+=(--checkpoint "$CHECKPOINT")

"$CONDA_PREFIX/bin/python" scripts/eval/correspondence_labelprop.py "${ARGS[@]}"
log "=== Done ==="
