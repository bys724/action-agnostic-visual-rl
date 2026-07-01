#!/bin/bash
# CALVIN Action Probing (§C10 DROID 보강) — 한 잡 = (encoder × split) × 4 gaps loop.
#
# 사용:
#   # v15 ep50 × training split (4 gaps loop)
#   sbatch --export=ALL,ENCODER=two-stream-v11,\
# CHECKPOINT=/proj/external_group/mrg/checkpoints/two_stream_v15/20260511_045319/latest.pt,\
# SPLIT=training \
#          scripts/cluster/probe_action_calvin.sbatch
#
#   # baseline (HF cache 사용, ckpt 불필요)
#   sbatch --export=ALL,ENCODER=siglip,SPLIT=training scripts/cluster/probe_action_calvin.sbatch
#
#   # sanity (validation = 소수 episode, --max-episodes 10)
#   sbatch --export=ALL,ENCODER=vc1,SPLIT=validation,MAX_EPISODES=10,SUFFIX=sanity \
#          scripts/cluster/probe_action_calvin.sbatch
#
#SBATCH --job-name=probe_calvin
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/proj/external_group/mrg/logs/probe_calvin_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/probe_calvin_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CALVIN_ROOT="/proj/external_group/mrg/datasets/calvin/task_ABCD_D"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
OUT_BASE="$PROJECT_ROOT/paper_artifacts/calvin_action_probing"

ENCODER="${ENCODER:?ENCODER required}"
CHECKPOINT="${CHECKPOINT:-}"
SPLIT="${SPLIT:-training}"
VIEW="${VIEW:-rgb_static}"
GAPS="${GAPS:-10 20 30 45}"
MAX_EPISODES="${MAX_EPISODES:-200}"
FRAME_STRIDE="${FRAME_STRIDE:-10}"
V11_MODE="${V11_MODE:-p_t_p_tk}"
VIDEOMAE_MODE="${VIDEOMAE_MODE:-paired}"
SUFFIX="${SUFFIX:-}"
CROSS_FOLDER="${CROSS_FOLDER:-}"           # 0이 아닌 임의 값이면 --cross-folder flag 추가
READOUT="${READOUT:-mean}"                 # mean | attentive (STEP 0 게이트)
PARVO_MODE="${PARVO_MODE:-p_t_p_tk}"       # parvo: p_t_p_tk | p_t_m
VIDEOMAE_ENCODER="${VIDEOMAE_ENCODER:-adapter}"  # videomae-ours: adapter(legacy) | vla
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_BASE/${ENCODER}_${SPLIT}_${TIMESTAMP}${SUFFIX:+_$SUFFIX}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [probe_calvin] $*"; }

log "=== CALVIN Action Probing ==="
log "Encoder: $ENCODER | Checkpoint: ${CHECKPOINT:-<none>}"
log "Split: $SPLIT | view: $VIEW | gaps: $GAPS | max_episodes: $MAX_EPISODES"
log "Output: $OUT_DIR"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR"

ARGS=(
    --encoder "$ENCODER"
    --data-root "$CALVIN_ROOT"
    --split "$SPLIT"
    --view "$VIEW"
    --gaps $GAPS
    --max-episodes "$MAX_EPISODES"
    --frame-stride "$FRAME_STRIDE"
    --output-dir "$OUT_DIR"
)
ARGS+=(--readout "$READOUT" --probe-weight-decay "$PROBE_WEIGHT_DECAY")
if [[ -n "$CHECKPOINT" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ "$ENCODER" == "two-stream-v11" ]]; then ARGS+=(--v11-mode "$V11_MODE"); fi
if [[ "$ENCODER" == "parvo" ]]; then ARGS+=(--parvo-mode "$PARVO_MODE"); fi
if [[ "$ENCODER" == "videomae-ours" ]]; then ARGS+=(--videomae-mode "$VIDEOMAE_MODE" --videomae-encoder "$VIDEOMAE_ENCODER"); fi
if [[ -n "$CROSS_FOLDER" ]]; then ARGS+=(--cross-folder); fi

"$CONDA_PREFIX/bin/python" scripts/eval/probe_action_calvin.py "${ARGS[@]}"

log "=== done ==="
