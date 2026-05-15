#!/bin/bash
# C6 frame_t vs frame_tk reconstruction quality v11 vs v15
# paper_experiments_plan §6 — design intent realization (asymmetric → symmetric path)
#
# 사용:
#   sbatch scripts/cluster/recon_quality.sbatch
#   sbatch --export=NUM_SAMPLES=400 scripts/cluster/recon_quality.sbatch
#
#SBATCH --job-name=recon_quality
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/proj/external_group/mrg/logs/recon_quality_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/recon_quality_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"

V11_CKPT="${V11_CKPT:-/proj/external_group/mrg/checkpoints/two_stream_v11/20260426_014333/checkpoint_epoch0044.pt}"
V15_CKPT="${V15_CKPT:-/proj/external_group/mrg/checkpoints/two_stream_v15/20260511_045319/latest.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-42}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$PROJECT_ROOT/paper_artifacts/recon_quality/v11_ep44_vs_v15_ep50_${TIMESTAMP}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [recon_quality] $*"; }
log "=== C6 recon quality v11 vs v15 ==="
log "V11: $V11_CKPT"
log "V15: $V15_CKPT"
log "num_samples=$NUM_SAMPLES, batch=$BATCH_SIZE, seed=$SEED"
log "out: $OUT_DIR"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR"

"$CONDA_PREFIX/bin/python" scripts/eval/recon_quality_v11_vs_v15.py \
    --v11-ckpt "$V11_CKPT" \
    --v15-ckpt "$V15_CKPT" \
    --egodex-frames-root "$FRAMES_ROOT" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR"

log "=== done ==="
