#!/bin/bash
# Two-Stream v11 sanity: Motion-Guided Attention Routing + Dual-Target
# 확인 포인트:
#   · forward/backward OK (3-phase decoder)
#   · L_t, L_tk 각각 epoch별 단조 감소
#   · feat_std_p > 0.1 (collapse 방어)
#   · cos_intra_p < 0.95 (uniformity 방어)
#   · ratio L_tk/L_t (Phase 3가 Phase 1보다 높아야 정상 — motion 주입 후 더 어려움)
#SBATCH --job-name=aavrl_v11_sanity
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/proj/external_group/mrg/logs/sanity_v11_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/sanity_v11_%j.err

set -euo pipefail

MAX_VIDEOS="${MAX_VIDEOS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
SUFFIX="${SUFFIX:-}"

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/two_stream_v11_sanity${SUFFIX:+_$SUFFIX}"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v11 sanity test ==="
echo "Motion-Guided Attention Routing + Dual-Target Reconstruction"
echo "Suffix=$SUFFIX | videos=$MAX_VIDEOS | batch=$BATCH_SIZE | epochs=$EPOCHS"
echo ""

"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model two-stream-v11 \
    --depth 12 \
    --v11-m-depth 6 \
    --rotation-aug \
    --mask-ratio 0.3 --mask-ratio-p 0.75 \
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
