#!/bin/bash
# Two-Stream v8 sanity: 1 GPU, 소규모 데이터, 2 epoch
# 목적:
# - forward/backward 무결성, teacher forward, EMA update 동작 확인
# - **collapse 방어(A+B+D)가 실제로 작동하는지 진단**
#
# 1차 sanity(warmup=0, λ_max=0.5) 결과: ep2에서 std_s 0.094→0.002 (collapse 전형).
# 본 2차는 A(warmup=0, 단 sanity 2ep라 효과 제한), B(λ_max 0.2), D(α_var 0.1) 적용.
# 판단 기준: std_s > 0.02, cos_st < 0.95, L_var가 활성화되어 collapse 방향으로 증가하지 않음.
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
    --v8-lambda-max 0.2 \
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
