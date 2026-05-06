#!/bin/bash
# Two-Stream v14 sanity (Phase A — design 노트 §15)
# Stream-wise paradigm specialization: P=MAE+V-JEPA, M=DINO
#
# 목적 (Phase A): 4 loss가 정상 학습되는지만 확인. multi-crop N=1 (baseline).
# 본 학습 들어가기 전 sanity로 표준 hyperparameter 검증 + ep당 시간 측정.
#
# 체크 포인트 (노트 §15):
#   · L_t, L_tk_recon 단조 감소 (reconstruction 정상)
#   · L_pred 단조 감소 (V-JEPA 학습 정상)
#   · L_dino < log(K)=log(1024)≈6.93 으로 멀어짐 (prototype 학습 신호 발현)
#   · cos_intra_p < 0.95 (P encoder collapse 아님)
#   · cos_intra_dino < 0.95, std_student_dino > 0.1 (M CLS DINO collapse 아님)
#   · cos(pred,target) < 0.99 (V-JEPA predictor identity 아님)
#   · ||dino_center|| 안정 update
#   · ep당 시간 측정 → 본 학습 --time 결정 (4 loss + 3 teacher EMA, ~+60% v11 대비)
#
#SBATCH --job-name=aavrl_v14_sanity
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/proj/external_group/mrg/logs/sanity_v14_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/sanity_v14_%j.err

set -euo pipefail

MAX_VIDEOS="${MAX_VIDEOS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
SUFFIX="${SUFFIX:-}"

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/two_stream_v14_sanity${SUFFIX:+_$SUFFIX}"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v14 sanity test (Phase A) ==="
echo "Stream-wise paradigm specialization: P=MAE+V-JEPA, M=DINO"
echo "Suffix=$SUFFIX | videos=$MAX_VIDEOS | batch=$BATCH_SIZE | epochs=$EPOCHS"
echo ""

"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model two-stream-v14 \
    --depth 12 \
    --v11-m-depth 6 \
    --v11-routing-mode v_from_p \
    --rotation-aug \
    --mask-ratio-p 0.75 \
    --v14-lambda-pred 1.0 \
    --v14-lambda-dino 1.0 \
    --v14-dino-n-crop 1 \
    --v14-num-prototypes 1024 \
    --v14-dino-teacher-temp 0.04 \
    --v14-dino-student-temp 0.1 \
    --v14-dino-center-momentum 0.9 \
    --v14-ema-momentum-init 0.996 \
    --v14-ema-momentum-final 0.9999 \
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
