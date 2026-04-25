#!/bin/bash
# Action probing sbatch — Two-Stream v11 전용
#
# 목적: v11 (motion-routing + dual-target) 체크포인트의 action information 검증.
#
# Usage:
#   sbatch scripts/cluster/probe_v11.sbatch
#   # override:
#   sbatch --export=ALL,CHECKPOINT=/path/to/ckpt.pt,CLS_MODE=patch_mean_p_features_tk,SPLIT=test \
#          scripts/cluster/probe_v11.sbatch
#
# 모니터링: tail -f /proj/external_group/mrg/logs/probe_<jobid>.out

#SBATCH --job-name=aavrl_probe_v11
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
CHECKPOINT="${CHECKPOINT:?CHECKPOINT env var required}"
SPLIT="${SPLIT:-test}"
CLS_MODE="${CLS_MODE:-patch_mean_p_features_tk}"
GAP="${GAP:-10}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
P_DEPTH="${P_DEPTH:-12}"
M_DEPTH="${M_DEPTH:-6}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [probe_v11] $*"; }

log "=== v11 action probing start ==="
log "Checkpoint: $CHECKPOINT"
log "Split: $SPLIT, cls_mode: $CLS_MODE, gap: $GAP"
log "p_depth: $P_DEPTH, m_depth: $M_DEPTH"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Compute nodes can't reach HF Hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
"$CONDA_PREFIX/bin/python" scripts/eval/probe_action_v11.py \
    --encoder two-stream-v11 \
    --checkpoint "$CHECKPOINT" \
    --egodex-root "$EGODEX_ROOT" \
    --frames-root "$FRAMES_ROOT" \
    --egodex-split "$SPLIT" \
    --cls-mode "$CLS_MODE" \
    --gap "$GAP" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --p-depth "$P_DEPTH" \
    --m-depth "$M_DEPTH"

log "=== Probing done ==="
