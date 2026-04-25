#!/bin/bash
# DROID action probing sbatch (cross-domain)
# 사용:
#   sbatch --export=ALL,ENCODER=videomae,CHECKPOINT=/path/to.pt scripts/cluster/probe_action_droid.sbatch
#
#SBATCH --job-name=probe_droid
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/proj/external_group/mrg/logs/probe_droid_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/probe_droid_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
DROID_ROOT="/proj/external_group/mrg/datasets/droid_frames/ext1"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"

# overridable
ENCODER="${ENCODER:?ENCODER required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT required}"
CLS_MODE="${CLS_MODE:-patch_mean_concat}"
GAP="${GAP:-1}"
MAX_EPISODES="${MAX_EPISODES:-200}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [probe_droid] $*"; }

log "Encoder: $ENCODER, cls_mode: $CLS_MODE, gap: $GAP"
log "Checkpoint: $CHECKPOINT"
log "DROID root: $DROID_ROOT, max_episodes: $MAX_EPISODES"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
"$CONDA_PREFIX/bin/python" scripts/eval/probe_action_droid.py \
    --encoder "$ENCODER" \
    --checkpoint "$CHECKPOINT" \
    --droid-root "$DROID_ROOT" \
    --cls-mode "$CLS_MODE" \
    --gap "$GAP" \
    --max-episodes "$MAX_EPISODES" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --depth 12 --num-stages 2

log "=== DROID probing done ==="
