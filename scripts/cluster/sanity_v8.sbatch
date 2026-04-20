#!/bin/bash
# Two-Stream v8 sanity: 1 GPU, 소규모 데이터, 2 epoch
# 목적:
# - forward/backward 무결성, teacher forward, EMA update 동작 확인
# - **collapse 방어(A+B+D)가 실제로 작동하는지 진단**
#
# 5차 sanity (BYOL form + M trivialization 대책 적용 후):
# - 1차(λ=0.5) collapse → 2차(λ=0.2) 돌파 → full run ep8 R² = -0.22 (scale + detach 복합 원인)
# - 4차 sanity: BYOL form + λ=0.05 scale 검증 통과
# - 5차 변경: cls_m_grad_ratio=0.3 (L_P 일부 역류 허용) + mask_ratio_m 0.3→0.5 (L_M 난이도 상향)
# 확인 포인트:
#   · L_P 초기 ≈ 1.0~2.0 (BYOL form), epoch 진행에 따라 감소
#   · L_M 초기 ≈ 0.03~0.08 (mask 상향으로 전보다 높게 시작), 0.0008 같은 trivial level로 급락 금지
#   · std_s > 0.02, cos_st < 0.95
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
    --v8-cls-m-grad-ratio 0.3 \
    --mask-ratio 0.5 --mask-ratio-p 0.5 \
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
