#!/bin/bash
# EgoDex 파트별 처리: S3 다운로드 → 프레임 추출 → S3 업로드 → 로컬 삭제
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

# Step 1: S3에서 원본 다운로드
log "=== Step 1/4: Downloading $PART from S3 ==="
mkdir -p "$LOCAL_RAW"
$AWS s3 sync "$S3_SRC" "$LOCAL_RAW/" --quiet
log "Download complete. Size: $(du -sh "$LOCAL_RAW" | cut -f1)"

# Step 2: 프레임 추출
log "=== Step 2/4: Extracting frames ==="
VIDEO_COUNT=$(find "$LOCAL_RAW" -name "*.mp4" | wc -l)
log "Videos found: $VIDEO_COUNT"
python3 "$SCRIPT_DIR/data/extract_frames.py" \
    --egodex-root "$PROJECT_DIR/data/egodex" \
    --split "$PART" \
    --output-dir "$LOCAL_FRAMES" \
    --yes
log "Extraction complete. Size: $(du -sh "$LOCAL_FRAMES" | cut -f1)"

# Step 3: 추출 프레임 S3 업로드 (task 단위 병렬 업로드 — s3 sync는 소규모 파일 수천만개에 극도로 느림)
log "=== Step 3/4: Uploading frames to $S3_DST ==="
UPLOAD_PIDS=()
for TASK_DIR in "$LOCAL_FRAMES"/*/; do
    TASK_NAME=$(basename "$TASK_DIR")
    log "  Uploading task: $TASK_NAME"
    $AWS s3 cp "$TASK_DIR" "$S3_DST$TASK_NAME/" --recursive --quiet &
    UPLOAD_PIDS+=($!)
done
# 모든 업로드 완료 대기
UPLOAD_FAIL=0
for PID in "${UPLOAD_PIDS[@]}"; do
    if ! wait "$PID"; then
        UPLOAD_FAIL=1
    fi
done
if [[ $UPLOAD_FAIL -ne 0 ]]; then
    log "ERROR: Some uploads failed. Skipping cleanup."
    exit 1
fi
log "Upload complete."

# Step 4: 로컬 정리
log "=== Step 4/4: Cleaning up local files ==="
rm -rf "$LOCAL_RAW"
rm -rf "$LOCAL_FRAMES"
log "Cleanup done."

log "=== $PART finished ==="
