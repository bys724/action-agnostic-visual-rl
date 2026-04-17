#!/bin/bash
# DROID 프레임 + action 추출 (CPU, core_s 파티션)
#
# Usage:
#   sbatch scripts/cluster/extract_droid.sbatch
#   # 특정 구간만:
#   sbatch --export=ALL,START=1000,LIMIT=500 scripts/cluster/extract_droid.sbatch
#
# 출력:
#   /proj/external_group/mrg/datasets/droid_frames/{ext1,ext2,wrist}/ep_XXXXXX/
#   /proj/external_group/mrg/datasets/droid_frames/ext1/ep_XXXXXX/action.npy

#SBATCH --job-name=aavrl_droid_extract
#SBATCH --partition=core_s,normal_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=/proj/external_group/mrg/logs/droid_extract_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/droid_extract_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
DROID_DIR="/proj/external_group/mrg/datasets/droid/1.0.1"
OUTPUT_DIR="/proj/external_group/mrg/datasets/droid_frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"

START="${START:-0}"
LIMIT="${LIMIT:-}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [droid] $*"; }

log "=== DROID extraction start ==="
log "DROID dir: $DROID_DIR"
log "Output dir: $OUTPUT_DIR"
log "Start: $START, Limit: ${LIMIT:-all}"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

# 1. 프레임 추출
FRAME_ARGS=(
    --droid-dir "$DROID_DIR"
    --output-dir "$OUTPUT_DIR"
    --img-size 256
    --start "$START"
    --yes
)
[ -n "$LIMIT" ] && FRAME_ARGS+=(--limit "$LIMIT")

log "Step 1: Frame extraction"
"$CONDA_PREFIX/bin/python" scripts/data/extract_droid_frames.py "${FRAME_ARGS[@]}"

# 2. Action 추출
log "Step 2: Action extraction"
"$CONDA_PREFIX/bin/python" scripts/data/extract_droid_actions.py \
    --droid-dir "$DROID_DIR" \
    --output-dir "$OUTPUT_DIR/ext1"

log "=== DROID extraction done ==="
