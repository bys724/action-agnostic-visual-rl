#!/bin/bash
# AWS EC2 EgoDex Pretraining Launcher
#
# Usage:
#   ./run_aws_training.sh [--sanity] [--model MODEL] [--no-shutdown]
#
# Options:
#   --sanity       Quick sanity test (5 videos, 1 epoch each)
#   --model MODEL  Train only specified model (two-stream|single-stream|videomae)
#   --no-shutdown  Do not shut down instance after completion
#
# Environment variables:
#   S3_BUCKET      S3 bucket for checkpoints (default: bys724-research-2026)
#   EGODEX_ROOT    Local EgoDex root path (default: /workspace/data/egodex)
#   CODE_DIR       Code directory (default: /workspace)

set -e

# ── 설정 ──────────────────────────────────────────────────────────────────────
S3_BUCKET="${S3_BUCKET:-bys724-research-2026}"
EGODEX_ROOT="${EGODEX_ROOT:-/workspace/data/egodex}"
CHECKPOINT_DIR="/workspace/data/checkpoints"
CODE_DIR="${CODE_DIR:-/workspace}"
LOG_DIR="/workspace/data/logs"
EPOCHS=30  # 50 → 30 (faster iteration, can extend later)
BATCH_SIZE=32
SANITY=false
SHUTDOWN=true
MODEL=""  # 빈 값 = 전체 3개 모델 순차 학습

mkdir -p "$LOG_DIR"

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
        --no-shutdown)
            SHUTDOWN=false
            shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--sanity] [--model MODEL] [--no-shutdown]"
            exit 1 ;;
    esac
done

# ── 코드 업데이트 ─────────────────────────────────────────────────────────────
echo "=== [1/4] Pulling latest code ==="
cd "$CODE_DIR"
git pull origin main || echo "Warning: git pull failed (continuing with existing code)"

# ── EgoDex S3 sync ────────────────────────────────────────────────────────────
echo ""
echo "=== [2/4] Syncing EgoDex dataset from S3 ==="
echo "  test split  : s3://${S3_BUCKET}/datasets/egodex-test/ → ${EGODEX_ROOT}/test/"
echo "  full split  : s3://${S3_BUCKET}/datasets/egodex-full/ → ${EGODEX_ROOT}/"

mkdir -p "${EGODEX_ROOT}/test"

if $SANITY; then
    # sanity: train용 part1 5개 task + eval용 test 3개 task
    echo "  [Sanity mode] Syncing 5 tasks from egodex-full/part1/ (train) and 3 tasks from egodex-test/ (eval)..."
    mkdir -p "${EGODEX_ROOT}/part1"

    TRAIN_TASKS=$(aws s3 ls "s3://${S3_BUCKET}/datasets/egodex-full/part1/" | awk '{print $2}' | head -5)
    for TASK in $TRAIN_TASKS; do
        aws s3 sync "s3://${S3_BUCKET}/datasets/egodex-full/part1/${TASK}" \
                    "${EGODEX_ROOT}/part1/${TASK}" \
                    --quiet
    done

    EVAL_TASKS=$(aws s3 ls "s3://${S3_BUCKET}/datasets/egodex-test/" | awk '{print $2}' | head -3)
    for TASK in $EVAL_TASKS; do
        aws s3 sync "s3://${S3_BUCKET}/datasets/egodex-test/${TASK}" \
                    "${EGODEX_ROOT}/test/${TASK}" \
                    --quiet
    done
else
    # test split sync (background)
    echo "  Syncing test split in background..."
    aws s3 sync "s3://${S3_BUCKET}/datasets/egodex-test/" \
                "${EGODEX_ROOT}/test/" \
                --quiet &
    TEST_SYNC_PID=$!

    # full split sync (foreground, 학습 전 완료 필요)
    echo "  Syncing full split (1.84TB, this will take a while)..."
    aws s3 sync "s3://${S3_BUCKET}/datasets/egodex-full/" \
                "${EGODEX_ROOT}/" \
                --quiet

    echo "  Waiting for test split sync to complete..."
    wait $TEST_SYNC_PID
fi

echo "  EgoDex sync complete."

# ── Background S3 Sync (Spot instance 대응) ───────────────────────────────────
echo ""
echo "Starting background S3 sync for checkpoint safety..."
(
  while true; do
    sleep 600  # 10분마다 sync
    if [ -d "$CHECKPOINT_DIR" ]; then
      aws s3 sync "$CHECKPOINT_DIR" "s3://${S3_BUCKET}/checkpoints/" --quiet --exclude "*.log"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint synced to S3" >> "$LOG_DIR/s3_sync.log"
    fi
  done
) &
SYNC_PID=$!
echo "  Background sync started (PID: $SYNC_PID)"
echo "  Syncing every 10 minutes to s3://${S3_BUCKET}/checkpoints/"

# ── 학습 실행 ─────────────────────────────────────────────────────────────────
echo ""
echo "=== [3/4] Starting pretraining ==="

run_training() {
    local MODEL_NAME="$1"
    local EXTRA_ARGS="${2:-}"
    local MODEL_SLUG="${MODEL_NAME//-/_}"
    local LOG_FILE="$LOG_DIR/train_${MODEL_SLUG}.log"
    local MODEL_CHECKPOINT_DIR="${CHECKPOINT_DIR}/${MODEL_SLUG}"

    echo ""
    echo "--- Training: $MODEL_NAME ---"

    # Auto-resume: S3 또는 로컬에서 최신 checkpoint 찾기
    RESUME_ARG=""

    # 1. 로컬에서 latest.pt 찾기
    if [ -d "$MODEL_CHECKPOINT_DIR" ]; then
        LATEST_LOCAL=$(find "$MODEL_CHECKPOINT_DIR" -name "latest.pt" -type f 2>/dev/null | sort -r | head -1)
        if [ -n "$LATEST_LOCAL" ]; then
            echo "  Found local checkpoint: $LATEST_LOCAL"
            RESUME_ARG="--resume $LATEST_LOCAL"
        fi
    fi

    # 2. S3에서 checkpoint 다운로드 시도 (로컬에 없으면)
    if [ -z "$RESUME_ARG" ] && [ -n "$S3_BUCKET" ]; then
        echo "  Checking S3 for existing checkpoints..."
        S3_PATH="s3://${S3_BUCKET}/checkpoints/${MODEL_SLUG}/"

        # S3에 checkpoint가 있는지 확인
        if aws s3 ls "$S3_PATH" >/dev/null 2>&1; then
            echo "  Found checkpoints in S3, downloading..."
            mkdir -p "$MODEL_CHECKPOINT_DIR"
            aws s3 sync "$S3_PATH" "$MODEL_CHECKPOINT_DIR" --quiet

            # 다운로드된 latest.pt 찾기
            LATEST_LOCAL=$(find "$MODEL_CHECKPOINT_DIR" -name "latest.pt" -type f 2>/dev/null | sort -r | head -1)
            if [ -n "$LATEST_LOCAL" ]; then
                echo "  Resuming from S3 checkpoint: $LATEST_LOCAL"
                RESUME_ARG="--resume $LATEST_LOCAL"
            fi
        else
            echo "  No existing checkpoints found in S3, starting from scratch"
        fi
    fi

    if $SANITY; then
        EXTRA_ARGS="$EXTRA_ARGS --max-videos 5"
    fi

    python "$CODE_DIR/scripts/train_long.py" \
        --model "$MODEL_NAME" \
        --train-data egodex \
        --egodex-root "$EGODEX_ROOT" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --checkpoint-dir "$MODEL_CHECKPOINT_DIR" \
        --s3-bucket "$S3_BUCKET" \
        --s3-prefix "checkpoints" \
        $RESUME_ARG \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"

    echo "--- $MODEL_NAME training complete. Log: $LOG_FILE ---"
}

if [[ -z "$MODEL" ]]; then
    # 3개 모델 순차 학습
    run_training "two-stream"
    run_training "single-stream"
    run_training "videomae"
else
    run_training "$MODEL"
fi

# ── 완료 및 종료 ──────────────────────────────────────────────────────────────
echo ""
echo "=== [4/4] All training complete ==="

# Background sync 종료 및 최종 sync
if [ -n "$SYNC_PID" ]; then
    echo "Stopping background sync..."
    kill $SYNC_PID 2>/dev/null

    echo "Performing final checkpoint sync to S3..."
    aws s3 sync "$CHECKPOINT_DIR" "s3://${S3_BUCKET}/checkpoints/" --exclude "*.log"
fi

echo "  Checkpoints: s3://${S3_BUCKET}/checkpoints/"
echo "  View sync log: $LOG_DIR/s3_sync.log"

if $SHUTDOWN; then
    echo "  Shutting down instance in 60 seconds (Ctrl+C to cancel)..."
    sleep 60
    sudo shutdown -h now
fi
