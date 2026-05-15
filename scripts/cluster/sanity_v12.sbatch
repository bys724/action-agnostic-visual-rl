#!/bin/bash
# [DEPRECATED] v12는 폐기 lineup (sanity 단계 cls_p collapse). 학습 안 함, 코드만 reference 보존.
# Two-Stream v12 sanity: v11 + Semantic Residual + EMA Teacher
# 확인 포인트 (vault v12 메모 Phase A diagnostic monitoring 기반):
#   · forward/backward OK (v11 path + residual path + teacher forward)
#   · L_recon (= L_t + L_tk), L_residual 단조 감소
#   · ||sem_m||, ||pred||, ||teacher|| > 0 (collapse 진행 시 0 근처)
#   · std_pred > 0.1 (variance 유지)
#   · cos(p_t, pred) < 0.95 (P time-invariance collapse 방어)
#   · cos(m, Δp) > 0 trend (residual equation 학습 진행 신호)
#   · ema_momentum 0.996 → 0.9999 schedule 확인
#SBATCH --job-name=aavrl_v12_sanity
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/proj/external_group/mrg/logs/sanity_v12_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/sanity_v12_%j.err

set -euo pipefail

MAX_VIDEOS="${MAX_VIDEOS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
SUFFIX="${SUFFIX:-}"

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/two_stream_v12_sanity${SUFFIX:+_$SUFFIX}"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v12 sanity test ==="
echo "v11 + CLS-level Semantic Residual + EMA Teacher"
echo "Suffix=$SUFFIX | videos=$MAX_VIDEOS | batch=$BATCH_SIZE | epochs=$EPOCHS"
echo ""

"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model two-stream-v12 \
    --depth 12 \
    --v11-m-depth 6 \
    --rotation-aug \
    --mask-ratio 0.3 --mask-ratio-p 0.75 \
    --v12-residual-weight 0.05 \
    --v12-vicreg-var-weight 1.0 \
    --v12-vicreg-cov-weight 1.0 \
    --v12-ema-momentum-init 0.996 \
    --v12-ema-momentum-final 0.9999 \
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
