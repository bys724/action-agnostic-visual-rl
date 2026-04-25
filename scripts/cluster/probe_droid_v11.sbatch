#!/bin/bash
# DROID action probing sbatch — Two-Stream v11 전용 (cross-domain)
#
# 목적: v11 (motion-routing + dual-target) 체크포인트의 DROID action 정보 검증.
#
# Usage:
#   sbatch --export=ALL,CHECKPOINT=/path/to/ckpt.pt scripts/cluster/probe_droid_v11.sbatch
#   # override:
#   sbatch --export=ALL,CHECKPOINT=/path/to/ckpt.pt,CLS_MODE=patch_mean_concat_enc_phase3,GAP=2 \
#          scripts/cluster/probe_droid_v11.sbatch
#
# 모니터링: tail -f /proj/external_group/mrg/logs/probe_droid_v11_<jobid>.out
#
#SBATCH --job-name=probe_droid_v11
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/proj/external_group/mrg/logs/probe_droid_v11_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/probe_droid_v11_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
DROID_ROOT="/proj/external_group/mrg/datasets/droid_frames/ext1"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"

# overridable
CHECKPOINT="${CHECKPOINT:?CHECKPOINT env var required}"
CLS_MODE="${CLS_MODE:-patch_mean_m_enc}"
GAP="${GAP:-1}"
MAX_EPISODES="${MAX_EPISODES:-200}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
P_DEPTH="${P_DEPTH:-12}"
M_DEPTH="${M_DEPTH:-6}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [probe_droid_v11] $*"; }

log "=== v11 DROID action probing start ==="
log "Checkpoint: $CHECKPOINT"
log "DROID root: $DROID_ROOT, max_episodes: $MAX_EPISODES"
log "cls_mode: $CLS_MODE, gap: $GAP"
log "p_depth: $P_DEPTH, m_depth: $M_DEPTH"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Compute nodes can't reach HF Hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
"$CONDA_PREFIX/bin/python" scripts/eval/probe_action_droid_v11.py \
    --encoder two-stream-v11 \
    --checkpoint "$CHECKPOINT" \
    --droid-root "$DROID_ROOT" \
    --cls-mode "$CLS_MODE" \
    --gap "$GAP" \
    --max-episodes "$MAX_EPISODES" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --p-depth "$P_DEPTH" \
    --m-depth "$M_DEPTH"

log "=== DROID v11 probing done ==="
