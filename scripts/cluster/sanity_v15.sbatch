#!/bin/bash
# Two-Stream v15 final sanity test
# Layered specialization with predictor-only V-JEPA + V-JEPA-M (Option B) + L_compose + 3-frame triple
# 사용자 spec: docs/v15_compositional_aux_design.md
#
# 목적: 4 loss path가 정상 학습되는지 + L_compose 핵심 sanity (m_long-m_short ≈ m_target).
#
# 체크 포인트:
#   · L_t/L_tn/L_tm 단조 감소 (P MAE 3 frame 모두 정상)
#   · L_pred 단조 감소 (V-JEPA P 3 segment 평균; cos(pred,tgt)<0.99 identity 회피)
#   · L_m_jepa 단조 감소 (V-JEPA-M Option B trivial 회피)
#   · L_compose 단조 감소 (composition_head=linear_residual → M_encoder가 가산성 학습)
#   · cos_intra_p < 0.95 (P encoder collapse 회피)
#   · ep당 시간 측정 → 본 학습 --time 결정 (옵션 B 3-frame 확장)
#
# 사용:
#   sbatch scripts/cluster/sanity_v15.sbatch                           # default 200vid × 5ep
#   sbatch --export=MAX_VIDEOS=300,EPOCHS=10 scripts/cluster/sanity_v15.sbatch
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

MODEL="${MODEL:-two-stream-v15}"            # v15b sanity: MODEL=two-stream-v15b
MAX_VIDEOS="${MAX_VIDEOS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
SUFFIX="${SUFFIX:-}"
# v15b recon-first hard-gate. sanity는 gate=0 권장 (student-anchor V-JEPA 경로를 sanity 내에서 활성화).
V15_GATE_EPOCHS="${V15_GATE_EPOCHS:-0}"
# EMA init: v15b=0.996 (recipe ②), v15=0.999. 명시 override 없으면 모델별 default.
if [ "$MODEL" = "two-stream-v15b" ]; then _EMA_INIT_DEFAULT="0.996"; else _EMA_INIT_DEFAULT="0.999"; fi

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/${MODEL//-/_}_sanity${SUFFIX:+_$SUFFIX}"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v15 final sanity ==="
echo "Layered specialization: predictor-only V-JEPA + V-JEPA-M (Option B) + L_compose"
echo "3-frame triple training (옵션 B): P MAE × 3 + V-JEPA P × 3 segment + V-JEPA M × 1 + L_compose"
echo "Suffix=$SUFFIX | videos=$MAX_VIDEOS | batch=$BATCH_SIZE | epochs=$EPOCHS"
echo ""

"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model "$MODEL" \
    --depth 12 \
    --v11-m-depth 6 \
    --v11-routing-mode "${V11_ROUTING_MODE:-v_from_p}" \
    --rotation-aug \
    --mask-ratio-p 0.75 \
    --v15-lambda-pred "${V15_LAMBDA_PRED:-1.0}" \
    --v15-lambda-pred-warmup-start "${V15_LAMBDA_PRED_WARMUP_START:-0.01}" \
    --v15-lambda-pred-warmup-epochs "${V15_LAMBDA_PRED_WARMUP_EPOCHS:-3}" \
    --v15-lambda-m-jepa "${V15_LAMBDA_M_JEPA:-1.0}" \
    --v15-lambda-m-jepa-warmup-start "${V15_LAMBDA_M_JEPA_WARMUP_START:-0.01}" \
    --v15-lambda-m-jepa-warmup-epochs "${V15_LAMBDA_M_JEPA_WARMUP_EPOCHS:-3}" \
    --v15-lambda-compose "${V15_LAMBDA_COMPOSE:-1.0}" \
    --v15-lambda-compose-warmup-start "${V15_LAMBDA_COMPOSE_WARMUP_START:-0.01}" \
    --v15-lambda-compose-warmup-epochs "${V15_LAMBDA_COMPOSE_WARMUP_EPOCHS:-3}" \
    --v15-mask-ratio-m-jepa 0.5 \
    --v15-composition-mode "${V15_COMP_MODE:-linear_residual}" \
    --v15-ema-momentum-init "${V15_EMA_INIT:-$_EMA_INIT_DEFAULT}" \
    --v15-ema-momentum-final "${V15_EMA_FINAL:-0.9999}" \
    --v15-lambda-gate-epochs "$V15_GATE_EPOCHS" \
    --train-data egodex --egodex-root "$FRAMES_ROOT" \
    --egodex-splits part1 --max-videos "$MAX_VIDEOS" \
    --max-gap "${MAX_GAP:-30}" --sample-dist triangular --sample-center "${SAMPLE_CENTER:-15}" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr 2e-4 \
    --num-workers "${NUM_WORKERS:-4}" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-interval "${SAVE_INTERVAL:-999}" \
    --eval-interval 999 \
    --no-multi-gpu

echo ""
echo "=== sanity done ==="
