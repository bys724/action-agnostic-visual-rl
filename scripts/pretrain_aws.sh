#!/bin/bash
# AWS EC2 EgoDex Pretraining Launcher
#
# Usage:
#   bash scripts/pretrain_aws.sh [--sanity] [--model MODEL] [--no-shutdown]
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

# ── S3 병렬 설정 (소량 파일 다운로드 속도 개선) ────────────────────────────────
aws configure set default.s3.max_concurrent_requests 100
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB

# ── 설정 ──────────────────────────────────────────────────────────────────────
S3_BUCKET="${S3_BUCKET:-bys724-research-2026}"
EGODEX_ROOT="${EGODEX_ROOT:-/workspace/data/egodex}"
CHECKPOINT_DIR="/workspace/data/checkpoints"
CODE_DIR="${CODE_DIR:-/workspace/action-agnostic-visual-rl}"
LOG_DIR="/workspace/data/logs"
EPOCHS=30  # 50 → 30 (faster iteration, can extend later)
BATCH_SIZE=24  # g5.8xlarge (3 GPUs) → 8 per GPU
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

# ── EgoDex 프레임 S3 sync ─────────────────────────────────────────────────────
# 사전 추출된 프레임(256x256 JPG)을 S3에서 다운로드
# S3 구조: egodex_frames_partN/task_name/video_name/frame_XXXXXX.jpg
#          egodex_frames_test/task_name/video_name/frame_XXXXXX.jpg
echo ""
echo "=== [2/4] Syncing EgoDex frames from S3 ==="

# 학습에 사용할 part 목록 (콤마 구분, 기본: part1만)
# 예: TRAIN_PARTS=part1,part2,part3 ./scripts/pretrain_aws.sh
TRAIN_PARTS="${TRAIN_PARTS:-part1}"

if $SANITY; then
    # sanity: train용 part1에서 5개 task + eval용 test에서 3개 task
    echo "  [Sanity mode] Syncing 5 train tasks + 3 eval tasks..."
    mkdir -p "${EGODEX_ROOT}/test"

    FIRST_PART="${TRAIN_PARTS%%,*}"
    mkdir -p "${EGODEX_ROOT}/${FIRST_PART}"
    TRAIN_TASKS=$(aws s3 ls "s3://${S3_BUCKET}/egodex_frames_${FIRST_PART}/" | awk '{print $2}' | head -5)
    for TASK in $TRAIN_TASKS; do
        aws s3 sync "s3://${S3_BUCKET}/egodex_frames_${FIRST_PART}/${TASK}" \
                    "${EGODEX_ROOT}/${FIRST_PART}/${TASK}" \
                    --quiet
    done

    EVAL_TASKS=$(aws s3 ls "s3://${S3_BUCKET}/egodex_frames_test/" | awk '{print $2}' | head -3)
    for TASK in $EVAL_TASKS; do
        aws s3 sync "s3://${S3_BUCKET}/egodex_frames_test/${TASK}" \
                    "${EGODEX_ROOT}/test/${TASK}" \
                    --quiet
    done
else
    # test split sync (background)
    mkdir -p "${EGODEX_ROOT}/test"
    echo "  Syncing test split in background..."
    aws s3 sync "s3://${S3_BUCKET}/egodex_frames_test/" \
                "${EGODEX_ROOT}/test/" \
                --quiet &
    TEST_SYNC_PID=$!

    # train parts sync (TRAIN_PARTS는 콤마 구분)
    for PART in ${TRAIN_PARTS//,/ }; do
        PART_DIR="${EGODEX_ROOT}/${PART}"
        if [ -d "$PART_DIR" ] && [ "$(ls -A "$PART_DIR" 2>/dev/null)" ]; then
            echo "  ${PART} already exists ($(du -sh "$PART_DIR" | cut -f1)), skipping..."
        else
            echo "  Syncing ${PART} from s3://${S3_BUCKET}/egodex_frames_${PART}/ ..."
            mkdir -p "$PART_DIR"
            aws s3 sync "s3://${S3_BUCKET}/egodex_frames_${PART}/" \
                        "$PART_DIR/" \
                        --quiet
        fi
    done

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

    # Deep Learning AMI: /opt/pytorch 가상환경의 python 사용
    PYTHON="${PYTHON:-/opt/pytorch/bin/python3}"
    export PYTHONPATH="$CODE_DIR:$PYTHONPATH"
    $PYTHON "$CODE_DIR/scripts/pretrain.py" \
        --model "$MODEL_NAME" \
        --train-data egodex \
        --egodex-root "$EGODEX_ROOT" \
        --egodex-splits "$TRAIN_PARTS" \
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
