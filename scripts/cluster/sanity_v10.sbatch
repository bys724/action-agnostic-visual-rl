#!/bin/bash
# Two-Stream v10 sanity: v6 base + mask_p 0.5 → 0.75
# 확인 포인트:
#   · forward/backward OK
#   · loss_m, loss_p 안정적 수렴 (둘 다 frame_{t+k} reconstruction이므로 v6와 유사 예상)
#   · feat_std, cos_intra (collapse 조짐 없음)
#SBATCH --job-name=aavrl_v10_sanity
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/proj/external_group/mrg/logs/sanity_v10_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/sanity_v10_%j.err

set -euo pipefail

MAX_VIDEOS="${MAX_VIDEOS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-2}"
SUFFIX="${SUFFIX:-}"

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/two_stream_v10_sanity${SUFFIX:+_$SUFFIX}"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v10 sanity test ==="
echo "v6 base + mask_p 0.5→0.75 (MAE-style P mask)"
echo "Suffix=$SUFFIX | videos=$MAX_VIDEOS | batch=$BATCH_SIZE | epochs=$EPOCHS"
echo ""

LOSS_WEIGHT_P="${LOSS_WEIGHT_P:-1.0}"

"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model two-stream \
    --depth 12 --num-stages 2 \
    --use-ape --rotation-aug \
    --mask-ratio 0.5 --mask-ratio-p 0.75 \
    --v9-p-target future --v9-loss-weight-p "$LOSS_WEIGHT_P" \
    --train-data egodex --egodex-root "$FRAMES_ROOT" \
    --egodex-splits part1 --max-videos "$MAX_VIDEOS" \
    --max-gap 60 --sample-dist triangular --sample-center 30 \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr 2e-4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-interval 999 \
    --eval-interval 999 \
    --no-multi-gpu

echo ""
echo "=== sanity done ==="
