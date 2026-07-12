#!/bin/bash
# SSv2 2-frame linear probe (경로 1) — docs/correspondence_eval_plan.md §7
#
# 사용:
#   sbatch --export=ALL,ENCODER=parvo,PARVO_MODE=p_t_m,\
# CHECKPOINT=/proj/external_group/mrg/checkpoints/two_stream_v15b_step1_comp_mae_s/20260629_101634/latest.pt \
#          scripts/cluster/ssv2_linear_probe.sbatch
#   sbatch --export=ALL,ENCODER=dinov2 scripts/cluster/ssv2_linear_probe.sbatch
#   # sanity: MAX_CLIPS=500,SUFFIX=sanity

#SBATCH --job-name=ssv2_lp
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/proj/external_group/mrg/logs/ssv2_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/ssv2_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
SSV2_ROOT="/proj/external_group/mrg/datasets/ssv2"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
OUT_BASE="$PROJECT_ROOT/paper_artifacts/ssv2_linear_probe"

ENCODER="${ENCODER:?ENCODER required}"
CHECKPOINT="${CHECKPOINT:-}"
PARVO_MODE="${PARVO_MODE:-p_t_m}"
MAX_CLIPS="${MAX_CLIPS:-0}"
WORKERS="${WORKERS:-14}"
READOUT="${READOUT:-mean}"
SUFFIX="${SUFFIX:-}"

TAG="$ENCODER"; [ "$ENCODER" = "parvo" ] && TAG="${ENCODER}_${PARVO_MODE}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_BASE/${TAG}_${TIMESTAMP}${SUFFIX:+_$SUFFIX}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ssv2] $*"; }
log "=== SSv2 linear probe: $TAG → $OUT_DIR ==="

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
ARGS=(--encoder "$ENCODER" --ssv2-root "$SSV2_ROOT" --output-dir "$OUT_DIR"
      --parvo-mode "$PARVO_MODE" --max-clips "$MAX_CLIPS" --workers "$WORKERS"
      --readout "$READOUT")
[ -n "$CHECKPOINT" ] && ARGS+=(--checkpoint "$CHECKPOINT")

"$CONDA_PREFIX/bin/python" scripts/eval/ssv2_linear_probe.py "${ARGS[@]}"
log "=== Done ==="
