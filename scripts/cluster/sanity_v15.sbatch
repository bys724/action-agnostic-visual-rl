#!/bin/bash
# Two-Stream v15 sanity (v14 진단 후 redesign)
# Layered specialization: predictor-only V-JEPA P + V-JEPA-M (masked) + DINO at M_decoder CLS
#
# 목적 (Sanity): 5 loss path가 정상 학습되는지 확인. multi-crop N=1 (baseline).
# 본 학습 들어가기 전 표준 hyperparameter 검증 + ep당 시간 측정.
#
# 체크 포인트:
#   · L_t, L_tk_recon 단조 감소 (P MAE 정상)
#   · L_pred 단조 감소 (V-JEPA P 학습 정상; predictor-only이라 MAE-driven student와 분리)
#   · L_m_jepa 단조 감소 (V-JEPA M 마스킹 reconstruction 정상)
#   · L_dino < log(K)=log(1024)≈6.93 (prototype 학습)
#   · cos_intra_p < 0.95 / cos_intra_dino < 0.95 (collapse 아님)
#   · cos(pred,target) < 0.99 (V-JEPA predictor identity 아님)
#   · ||dino_center|| 안정 update
#   · ep당 시간 측정 → 본 학습 --time 결정 (v14 대비 +25~30% 추정)
#
# 사용:
#   sbatch scripts/cluster/sanity_v15.sbatch                    # default 200vid × 5ep
#   sbatch --export=MAX_VIDEOS=300,EPOCHS=8 scripts/cluster/sanity_v15.sbatch
#
#SBATCH --job-name=aavrl_v15_sanity
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/proj/external_group/mrg/logs/sanity_v15_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/sanity_v15_%j.err

set -euo pipefail

MAX_VIDEOS="${MAX_VIDEOS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
SUFFIX="${SUFFIX:-}"

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/two_stream_v15_sanity${SUFFIX:+_$SUFFIX}"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v15 sanity test ==="
echo "Layered specialization: predictor-only V-JEPA + V-JEPA-M + DINO at M_decoder CLS"
echo "Suffix=$SUFFIX | videos=$MAX_VIDEOS | batch=$BATCH_SIZE | epochs=$EPOCHS"
echo ""

"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model two-stream-v15 \
    --depth 12 \
    --v11-m-depth 6 \
    --v11-routing-mode v_from_p \
    --rotation-aug \
    --mask-ratio-p 0.75 \
    --v15-lambda-pred 1.0 \
    --v15-lambda-m-jepa 1.0 \
    --v15-lambda-dino 1.0 \
    --v15-mask-ratio-m-jepa 0.5 \
    --v15-dino-n-crop 1 \
    --v15-num-prototypes 1024 \
    --v15-dino-teacher-temp 0.04 \
    --v15-dino-student-temp 0.1 \
    --v15-dino-center-momentum 0.9 \
    --v15-ema-momentum-init 0.996 \
    --v15-ema-momentum-final 0.9999 \
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
