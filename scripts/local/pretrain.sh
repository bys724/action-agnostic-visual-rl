#!/bin/bash
# H100 워크스테이션 로컬 학습 스크립트
#
# Usage (repo root에서 실행):
#   bash scripts/local/pretrain.sh                          # two-stream + videomae 병렬
#   bash scripts/local/pretrain.sh --model two-stream-v15b  # 특정 모델만 (v15b = 로컬 재학습)
#   bash scripts/local/pretrain.sh --model two-stream-v15b --sanity  # Sanity test
#
# 환경:
#   H100 x2 (81GB each), EgoDex 프레임 로컬 저장
#   two-stream: GPU 0,1 (DataParallel)
#   videomae: GPU 0 (single GPU, two-stream 끝나면 시작) 또는 병렬

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"      # .../scripts/local
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # repo root (scripts/local → 2단계 위)
EGODEX_ROOT="/mnt/data/egodex_frames"
CHECKPOINT_DIR="/mnt/data/checkpoints"
LOG_DIR="/mnt/data/logs/pretrain"
EPOCHS=30
BATCH_SIZE=32  # H100 81GB → 여유 있게 32
SANITY=false
MODEL=""
# two-stream-v15b 재학습 전용 설정 (env override 가능).
# v15b = v15 아키텍처 동일 + collapse 방지 레시피: ① recon-first hard-gate ② EMA momentum 낮춤 ⑤ lr scaling.
GPU_LIST="${GPU_LIST:-0,1}"          # 기본 2 GPU DataParallel
LR="${LR:-}"                         # ⑤ 비우면 batch에서 자동 linear scaling (cluster 256/1e-4 기준). env로 override 가능
V15_GATE_EPOCHS="${V15_GATE_EPOCHS:-10}"   # ① gate 동안 λ_pred/λ_m_jepa/λ_compose=0 (MAE만), 이후 warmup ramp
V15_EMA_INIT="${V15_EMA_INIT:-0.996}"      # ② teacher momentum 초기값 (cluster 0.999 → 낮춤)
NUM_WORKERS="${NUM_WORKERS:-8}"            # v15 triple은 num_workers=16에서 hang 사고 → 8 고정 (cluster 동일)
SANITY_VIDEOS="${SANITY_VIDEOS:-5}"        # sanity max-videos (v15b 권장 50)

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sanity)
            SANITY=true
            EPOCHS=1
            shift ;;
        --model)
            MODEL="$2"
            shift 2 ;;
        --epochs)
            EPOCHS="$2"
            shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2 ;;
        --splits)
            TRAIN_PARTS="$2"
            shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

# ── EgoDex 데이터 구조 맞추기 ─────────────────────────────────────────────────
# pretrain.py는 --egodex-root 아래에 split 디렉토리(part1, part4, test)를 기대
# 로컬 구조: /mnt/data/egodex_frames_part1/, /mnt/data/egodex_frames_part4/
# → symlink로 연결
mkdir -p "$EGODEX_ROOT"
for PART_DIR in /mnt/data/egodex_frames_part*/; do
    PART_NAME=$(basename "$PART_DIR" | sed 's/egodex_frames_//')
    LINK="$EGODEX_ROOT/$PART_NAME"
    if [ ! -e "$LINK" ]; then
        ln -sf "$PART_DIR" "$LINK"
        echo "Linked: $LINK → $PART_DIR"
    fi
done

# test split: 추출된 프레임 디렉토리 사용
TEST_FRAMES="$EGODEX_ROOT/test_frames"
if [ -d "$TEST_FRAMES" ] && [ ! -e "$EGODEX_ROOT/test" ]; then
    ln -sf "$TEST_FRAMES" "$EGODEX_ROOT/test"
    echo "Linked: $EGODEX_ROOT/test → $TEST_FRAMES"
fi

# 사용 가능한 split 확인 (--splits로 명시한 경우 자동 감지 생략)
# 가용한 part* 전부 사용 (정렬: part1..partN). test 등 비-part 디렉토리는 제외.
if [ -z "$TRAIN_PARTS" ]; then
    for PART_DIR in "$EGODEX_ROOT"/part*/; do
        PART=$(basename "$PART_DIR")
        if [ -d "$PART_DIR" ] && [ "$(ls -A "$PART_DIR" 2>/dev/null)" ]; then
            if [ -n "$TRAIN_PARTS" ]; then
                TRAIN_PARTS="$TRAIN_PARTS,$PART"
            else
                TRAIN_PARTS="$PART"
            fi
        fi
    done
fi

echo "============================================================"
echo "H100 Local Training"
echo "============================================================"
echo "  Train splits: $TRAIN_PARTS"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Logs: $LOG_DIR"
echo "============================================================"

# ── 학습 함수 ─────────────────────────────────────────────────────────────────
run_training() {
    local MODEL_NAME="$1"
    local GPU_IDS="$2"
    local EXTRA_ARGS="${3:-}"
    local MODEL_SLUG="${MODEL_NAME//-/_}"
    local LOG_FILE="$LOG_DIR/train_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S).log"
    local MODEL_CKPT="$CHECKPOINT_DIR/$MODEL_SLUG"

    mkdir -p "$MODEL_CKPT"

    echo ""
    echo "--- Training: $MODEL_NAME (GPU: $GPU_IDS) ---"
    echo "  Log: $LOG_FILE"

    local SANITY_ARGS=""
    if $SANITY; then
        SANITY_ARGS="--max-videos $SANITY_VIDEOS"
    fi

    # multi-gpu 판별
    local MULTI_GPU_ARG=""
    if [[ "$GPU_IDS" != *","* ]]; then
        MULTI_GPU_ARG="--no-multi-gpu"
    fi

    CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u "$PROJECT_ROOT/scripts/pretrain.py" \
        --model "$MODEL_NAME" \
        --train-data egodex \
        --egodex-root "$EGODEX_ROOT" \
        --egodex-splits "$TRAIN_PARTS" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --checkpoint-dir "$MODEL_CKPT" \
        --eval-interval 5 \
        --save-interval 2 \
        $MULTI_GPU_ARG \
        $SANITY_ARGS \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"

    echo "--- $MODEL_NAME complete ---"
}

# ── 실행 ──────────────────────────────────────────────────────────────────────
if [[ -n "$MODEL" ]]; then
    # 단일 모델 — GPU_LIST(기본 0,1) DataParallel 활용
    EXTRA=""
    if [[ "$MODEL" == "two-stream-v15b" ]]; then
        # v15b 재학습 레시피: recon-first hard-gate(①) + EMA momentum 낮춤(②) + lr scaling(⑤).
        # routing_mode/max_gap/sample_center/mask 등은 pretrain.py v15 default가 cluster 본 학습과 동일.
        # warmup-start=0 + gate-epochs N → 첫 N ep λ=0(MAE만), 이후 0→1.0 ramp.
        # ⑤ lr linear scaling: cluster (global batch 256 = 32/GPU × 8, lr 1e-4) 기준.
        # 주의: pretrain.py는 --batch-size를 per-GPU로 취급 (DataParallel: effective = batch × device_count).
        # → global batch = BATCH_SIZE × GPU수.
        N_GPU=$(echo "$GPU_LIST" | tr ',' ' ' | wc -w)
        GLOBAL_BATCH=$(( BATCH_SIZE * N_GPU ))
        if [[ -z "$LR" ]]; then
            LR=$(python3 -c "print(f'{1e-4 * $GLOBAL_BATCH / 256:.3e}')")
        fi
        echo "  [v15b] lr=$LR  (global batch $GLOBAL_BATCH = ${BATCH_SIZE}/GPU × ${N_GPU} GPU, cluster 1e-4/256 linear scaling)"
        EXTRA="--lr $LR \
            --num-workers $NUM_WORKERS \
            --v15-lambda-gate-epochs $V15_GATE_EPOCHS \
            --v15-lambda-pred-warmup-start 0 \
            --v15-lambda-m-jepa-warmup-start 0 \
            --v15-lambda-compose-warmup-start 0 \
            --v15-ema-momentum-init $V15_EMA_INIT"
    fi
    run_training "$MODEL" "$GPU_LIST" "$EXTRA"
else
    # 전략 A: two-stream(GPU 0) + videomae(GPU 1) 병렬 실행
    echo "Running two-stream (GPU 0) and videomae (GPU 1) in parallel"
    run_training "two-stream" "0" "--no-multi-gpu" &
    PID_TWO_STREAM=$!
    run_training "videomae" "1" "--no-multi-gpu" &
    PID_VIDEOMAE=$!

    wait $PID_TWO_STREAM
    wait $PID_VIDEOMAE
fi

echo ""
echo "============================================================"
echo "All training complete!"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "============================================================"
