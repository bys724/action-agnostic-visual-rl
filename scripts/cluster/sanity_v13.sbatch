#!/bin/bash
# Two-Stream v13 sanity: dual-frame reconstruction + motion-routed latent prediction + DINO global CLS
# 확인 포인트 (post-v12 sanity 진단 결과 → cls_p collapse 본질 해결 검증):
#   · forward/backward OK (frame_t / frame_{t+k} 모두 student P encoder 통과)
#   · L_t, L_tk, L_pred_patch, L_pred_cls 모두 단조 감소
#   · cos_intra_p < 0.95  (P encoder CLS collapse 차단 — v11/v12에서 이 metric=1.000으로 dead 였음)
#   · cos_intra_pred_cls < 0.95  (motion-routed predicted CLS collapse 방어)
#   · std_pred_cls > 0.1  (DINO centering 작동 확인)
#   · ||pred_cls|| ~ ||target_cls||  (학생/teacher magnitude 정합)
#   · ||dino_center|| 안정 (running mean update)
#   · ema_momentum 0.996 → 0.9999 schedule
#SBATCH --job-name=aavrl_v13_sanity
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/proj/external_group/mrg/logs/sanity_v13_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/sanity_v13_%j.err

set -euo pipefail

MAX_VIDEOS="${MAX_VIDEOS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
SUFFIX="${SUFFIX:-}"

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/two_stream_v13_sanity${SUFFIX:+_$SUFFIX}"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v13 sanity test ==="
echo "Dual-frame reconstruction + Motion-Routed Latent + DINO Global CLS"
echo "Suffix=$SUFFIX | videos=$MAX_VIDEOS | batch=$BATCH_SIZE | epochs=$EPOCHS"
echo ""

"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model two-stream-v13 \
    --depth 12 \
    --v11-m-depth 6 \
    --rotation-aug \
    --mask-ratio 0.3 --mask-ratio-p 0.75 \
    --v13-patch-pred-weight 1.5 \
    --v13-cls-pred-weight 0.005 \
    --v13-dino-center-momentum 0.95 \
    --v13-num-prototypes 4096 \
    --v13-dino-teacher-temp 0.04 \
    --v13-dino-student-temp 0.1 \
    --v13-ema-momentum-init 0.996 \
    --v13-ema-momentum-final 0.9999 \
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
