#!/bin/bash
# Two-Stream v8 sanity: 1 GPU, 소규모 데이터, 2 epoch
# 목적:
# - forward/backward 무결성, teacher forward, EMA update 동작 확인
# - **collapse 방어(A+B+D)가 실제로 작동하는지 진단**
#
# 3차 sanity (L_P L2 normalize 적용 후):
# - 1차(λ=0.5) collapse → 2차(λ=0.2) 돌파 → full run ep8 R² = -0.22 (scale mismatch 진단)
# - 이번 변경: prediction head 출력 + teacher 양쪽 L2 normalize → L_P ∈ [0, 4], 학습 후 ~0.2
# - 동반 조정: λ_max 0.05 (normalize 후 L_P가 훨씬 작은 scale이므로 λ·L_P ≈ 2~3·L_M 목표)
# 확인 포인트:
#   · L_P 초기 ≈ 1.0~2.0 (normalized, 랜덤 상태), epoch 진행에 따라 감소 기대
#   · L_M 기존과 동일 (~0.01 시작)
#   · std_s > 0.02, cos_st < 0.95 (이전 기준 유지)
#SBATCH --job-name=aavrl_v8_sanity
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/proj/external_group/mrg/logs/sanity_v8_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/sanity_v8_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/two_stream_v8_sanity"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v8 sanity test ==="
echo "Check: forward/backward, L_M/L_P, EMA update, detach isolation"
echo ""

"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model two-stream \
    --depth 12 --num-stages 2 \
    --use-ape \
    --v8-mode \
    --v8-lambda-max 0.05 \
    --v8-lambda-warmup-epochs 0 \
    --v8-ema-tau-base 0.996 \
    --v8-alpha-var 0.1 \
    --v8-var-target 1.0 \
    --mask-ratio 0.3 --mask-ratio-p 0.5 \
    --rotation-aug \
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
