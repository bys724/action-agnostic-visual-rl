#!/bin/bash
# EgoDex 파트별 처리: S3 다운로드 → 프레임 추출(센터크롭 256x256) → S3 업로드 → 로컬 삭제
#
# Usage:
#   ./scripts/process_egodex_part.sh part2
#   for p in part2 part3 part4 part5; do ./scripts/process_egodex_part.sh $p; done

set -e

PART=$1
if [[ -z "$PART" ]]; then
    echo "Usage: $0 <part_name>  (e.g. part2)"
    exit 1
fi

AWS=~/.local/bin/aws
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

S3_BUCKET="s3://bys724-research-2026"
S3_SRC="$S3_BUCKET/datasets/egodex-full/$PART/"
S3_DST="$S3_BUCKET/egodex_frames_$PART/"

LOCAL_RAW="$PROJECT_DIR/data/egodex/$PART"
LOCAL_FRAMES="$PROJECT_DIR/data/egodex_frames_$PART"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Step 1: S3에서 원본 다운로드 (이미 있으면 생략)
log "=== Step 1/4: Downloading $PART from S3 ==="
if [ -d "$LOCAL_RAW" ] && [ "$(ls -A "$LOCAL_RAW" 2>/dev/null)" ]; then
    log "Raw data already exists locally, skipping download"
else
    mkdir -p "$LOCAL_RAW"
    $AWS s3 sync "$S3_SRC" "$LOCAL_RAW/" --quiet
fi
log "Download complete. Size: $(du -sh "$LOCAL_RAW" | cut -f1)"

# Step 2: 프레임 추출 (65:35 비대칭 크롭, 256x256)
log "=== Step 2/4: Extracting frames ==="
VIDEO_COUNT=$(find "$LOCAL_RAW" -name "*.mp4" | wc -l)
log "Videos found: $VIDEO_COUNT"
python3 "$SCRIPT_DIR/data/extract_frames.py" \
    --egodex-root "$PROJECT_DIR/data/egodex" \
    --split "$PART" \
    --output-dir "$LOCAL_FRAMES" \
    --yes
log "Extraction complete. Size: $(du -sh "$LOCAL_FRAMES" | cut -f1)"

# Step 3: S3 업로드 (task 단위 순차 — 동시 업로드는 실패 위험)
log "=== Step 3/4: Uploading frames to $S3_DST ==="
for TASK_DIR in "$LOCAL_FRAMES"/*/; do
    TASK_NAME=$(basename "$TASK_DIR")
    log "  Uploading task: $TASK_NAME"
    if ! $AWS s3 cp "$TASK_DIR" "$S3_DST$TASK_NAME/" --recursive --quiet; then
        log "  Retry: $TASK_NAME"
        $AWS s3 cp "$TASK_DIR" "$S3_DST$TASK_NAME/" --recursive --quiet
    fi
done
log "Upload complete."

# Step 4: 로컬 정리
log "=== Step 4/4: Cleaning up local files ==="
rm -rf "$LOCAL_RAW"
rm -rf "$LOCAL_FRAMES"
log "Cleanup done."

log "=== $PART finished ==="
