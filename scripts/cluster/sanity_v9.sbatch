#!/bin/bash
# Two-Stream v9 sanity: 1 GPU, 소규모 데이터, 2 epoch
# 목적:
# - forward/backward 무결성, residual P target 동작 확인
# - loss_m (full RGB) vs loss_p_raw (residual MSE) 크기 비교로 loss_weight_p 결정
# - Collapse detector(feat_std, cos_intra) 초기값 확인
# 확인 포인트:
#   · loss_m ≈ 0.03~0.08 (v6 초반 수준 예상)
#   · loss_p_raw ≈ ?? — residual MSE라 작을 것으로 예상 (0.001~0.01?)
#   · ratio p/m = 결정 기준: loss_weight_p * ratio ≈ 1.0이 되도록 조정
#   · feat_std_p > 0.1, cos_intra_p < 0.95 (초기부터 collapse 조짐 없어야)
#SBATCH --job-name=aavrl_v9_sanity
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/proj/external_group/mrg/logs/sanity_v9_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/sanity_v9_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/two_stream_v9_sanity"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v9 sanity test ==="
echo "Check: residual P target, loss_m vs loss_p_raw ratio, feat_std, cos_intra"
echo "loss_weight_p=1.0으로 시작해 balance 확인 후 본 학습에서 조정"
echo ""

# Note: loss_weight_p 1.0으로 sanity 돌려 raw ratio 측정 후,
#       본 학습에서는 ratio * loss_weight_p ≈ 1.0이 되도록 조정.
#       예: raw ratio 0.1이면 loss_weight_p=10
"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model two-stream \
    --depth 12 --num-stages 2 \
    --use-ape --rotation-aug \
    --mask-ratio 0.3 --mask-ratio-p 0.75 \
    --v9-residual-p \
    --v9-loss-weight-p 1.0 \
    --train-data egodex --egodex-root "$FRAMES_ROOT" \
    --egodex-splits part1 --max-videos 20 \
    --max-gap 60 --sample-dist triangular --sample-center 30 \
    --epochs 2 --batch-size 8 --lr 2e-4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-interval 999 \
    --eval-interval 999 \
    --no-multi-gpu

echo ""
echo "=== sanity done ==="
