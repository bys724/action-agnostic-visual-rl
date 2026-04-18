#!/bin/bash
# Action probing sbatch (EgoDex test split, Two-Stream v4)
#
# 목적: Two-Stream best_model(epoch 48)이 action 정보를 인코딩하는지 sanity probing.
# 이 probing은 Two-Stream vs VideoMAE-ours controlled comparison의 축 1 evidence.
#
# Usage:
#   sbatch scripts/cluster/probe_action.sbatch
#   # override 가능:
#   sbatch --export=ALL,CHECKPOINT=/path/to/best.pt,SPLIT=test,CLS_MODE=patch_mean_concat \
#          scripts/cluster/probe_action.sbatch
#
# 모니터링: tail -f /proj/external_group/mrg/logs/probe_<jobid>.out

#SBATCH --job-name=aavrl_probe
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/proj/external_group/mrg/logs/probe_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/probe_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
EGODEX_ROOT="/proj/external_group/mrg/datasets/egodex/raw"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"

# overridable
ENCODER="${ENCODER:-two-stream}"
CHECKPOINT="${CHECKPOINT:-/proj/external_group/mrg/checkpoints/two_stream/20260410_160810/best_model.pt}"
SPLIT="${SPLIT:-test}"
CLS_MODE="${CLS_MODE:-patch_mean_concat}"
GAP="${GAP:-10}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [probe] $*"; }

log "=== Action probing start ==="
log "Encoder: $ENCODER"
log "Checkpoint: $CHECKPOINT"
log "Split: $SPLIT, cls_mode: $CLS_MODE, gap: $GAP"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Compute nodes can't reach HF Hub — force offline (cache must be pre-populated on login node)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
"$CONDA_PREFIX/bin/python" scripts/eval/probe_action.py \
    --encoder "$ENCODER" \
    --checkpoint "$CHECKPOINT" \
    --egodex-root "$EGODEX_ROOT" \
    --frames-root "$FRAMES_ROOT" \
    --egodex-split "$SPLIT" \
    --cls-mode "$CLS_MODE" \
    --gap "$GAP" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --depth 12 --num-stages 2

log "=== Probing done ==="
