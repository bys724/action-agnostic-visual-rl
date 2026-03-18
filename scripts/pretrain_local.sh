#!/bin/bash
# H100 워크스테이션 로컬 학습 스크립트
#
# Usage:
#   bash scripts/pretrain_local.sh                    # two-stream + videomae 병렬
#   bash scripts/pretrain_local.sh --model two-stream  # 특정 모델만
#   bash scripts/pretrain_local.sh --sanity            # Sanity test
#
# 환경:
#   H100 x2 (81GB each), EgoDex 프레임 로컬 저장
#   two-stream: GPU 0,1 (DataParallel)
#   videomae: GPU 0 (single GPU, two-stream 끝나면 시작) 또는 병렬

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EGODEX_ROOT="/mnt/data/egodex_frames"
CHECKPOINT_DIR="/mnt/data/checkpoints"
LOG_DIR="/mnt/data/logs/pretrain"
EPOCHS=30
BATCH_SIZE=32  # H100 81GB → 여유 있게 32
SANITY=false
MODEL=""

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

# test split (data/egodex/test → symlink)
LOCAL_TEST="/home/etri/bys/action-agnostic-visual-rl/data/egodex/test"
if [ -d "$LOCAL_TEST" ] && [ ! -e "$EGODEX_ROOT/test" ]; then
    ln -sf "$LOCAL_TEST" "$EGODEX_ROOT/test"
    echo "Linked: $EGODEX_ROOT/test → $LOCAL_TEST"
fi

# 사용 가능한 split 확인
TRAIN_PARTS=""
for PART in part1 part4; do
    if [ -d "$EGODEX_ROOT/$PART" ] && [ "$(ls -A "$EGODEX_ROOT/$PART" 2>/dev/null)" ]; then
        if [ -n "$TRAIN_PARTS" ]; then
            TRAIN_PARTS="$TRAIN_PARTS,$PART"
        else
            TRAIN_PARTS="$PART"
        fi
    fi
done

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
        SANITY_ARGS="--max-videos 5"
    fi

    # multi-gpu 판별
    local MULTI_GPU_ARG=""
    if [[ "$GPU_IDS" != *","* ]]; then
        MULTI_GPU_ARG="--no-multi-gpu"
    fi

    CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 "$PROJECT_ROOT/scripts/pretrain.py" \
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
    # 단일 모델
    if [[ "$MODEL" == "videomae" ]]; then
        run_training "$MODEL" "0" "--no-multi-gpu"
    else
        run_training "$MODEL" "0,1"
    fi
else
    # two-stream + videomae 병렬 (H100 x2)
    # two-stream: GPU 0,1 (DataParallel)
    # videomae: GPU 1 (two-stream 끝난 후 → 순차 실행이 안전)
    # H100 2장이라 병렬은 VRAM 충돌 위험 → 순차 실행
    echo "Running two-stream then videomae (sequential, 2 GPUs)"
    run_training "two-stream" "0,1"
    run_training "videomae" "0" "--no-multi-gpu"
fi

echo ""
echo "============================================================"
echo "All training complete!"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "============================================================"
