#!/bin/bash
#
# 전체 학습 파이프라인: 3개 모델 순차 학습
# Two-Stream → Single-Stream → VideoMAE
#

set -e  # 에러 발생 시 중단

# 설정
EPOCHS=30
BATCH_SIZE=24  # per-GPU (effective: 96 with 4 GPUs)
EGODEX_ROOT="/workspace/data/egodex"
S3_BUCKET="bys724-research-2026"
LOG_DIR="/workspace/data/logs"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting Full Training Pipeline"
echo "============================================================"
echo "Epochs: $EPOCHS"
echo "Batch size (per-GPU): $BATCH_SIZE"
echo "Effective batch size: $((BATCH_SIZE * 4))"
echo "Dataset: EgoDex part1"
echo "S3 Bucket: $S3_BUCKET"
echo "============================================================"
echo ""

# Python 경로
PYTHON="/opt/pytorch/bin/python3"

# 1. Two-Stream 모델 학습
echo "[1/3] Training Two-Stream model..."
echo "Started at: $(date)"
$PYTHON /workspace/action-agnostic-visual-rl/scripts/train_long.py \
    --model two-stream \
    --train-data egodex \
    --egodex-root "$EGODEX_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-dir /workspace/data/checkpoints/two_stream \
    --s3-bucket "$S3_BUCKET" \
    --s3-prefix checkpoints 2>&1 | tee "$LOG_DIR/train_two_stream.log"

echo "Two-Stream completed at: $(date)"
echo ""

# 2. Single-Stream 모델 학습
echo "[2/3] Training Single-Stream model..."
echo "Started at: $(date)"
$PYTHON /workspace/action-agnostic-visual-rl/scripts/train_long.py \
    --model single-stream \
    --train-data egodex \
    --egodex-root "$EGODEX_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-dir /workspace/data/checkpoints/single_stream \
    --s3-bucket "$S3_BUCKET" \
    --s3-prefix checkpoints 2>&1 | tee "$LOG_DIR/train_single_stream.log"

echo "Single-Stream completed at: $(date)"
echo ""

# 3. VideoMAE 모델 학습
echo "[3/3] Training VideoMAE model..."
echo "Started at: $(date)"
$PYTHON /workspace/action-agnostic-visual-rl/scripts/train_long.py \
    --model videomae \
    --train-data egodex \
    --egodex-root "$EGODEX_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-dir /workspace/data/checkpoints/videomae \
    --s3-bucket "$S3_BUCKET" \
    --s3-prefix checkpoints 2>&1 | tee "$LOG_DIR/train_videomae.log"

echo "VideoMAE completed at: $(date)"
echo ""

# 완료
echo "============================================================"
echo "All models trained successfully!"
echo "Completed at: $(date)"
echo "============================================================"
echo ""
echo "Checkpoints saved to S3: s3://$S3_BUCKET/checkpoints/"
echo "Logs saved to: $LOG_DIR/"
