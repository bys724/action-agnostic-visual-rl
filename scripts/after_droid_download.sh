#!/bin/bash
# DROID 다운로드 완료 대기 → EgoDex part2,3,5 다운로드+추출 (업로드 생략)
# part4는 로컬에 이미 있으므로 제외
#
# Usage:
#   nohup bash scripts/after_droid_download.sh > /mnt/data/after_droid.log 2>&1 &

set -e

AWS=~/.local/bin/aws
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/mnt/data"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# === Step 1: DROID 다운로드 완료 대기 ===
log "Waiting for DROID download (gsutil) to finish..."
while pgrep -f 'gsutil.*rsync.*droid' > /dev/null 2>&1; do
    sleep 60
done
log "DROID download complete."

# === Step 2: EgoDex part2,3,5 순차 다운로드+추출 ===
S3_BUCKET="s3://bys724-research-2026"

for PART in part2 part3 part5; do
    LOCAL_RAW="$DATA_DIR/egodex/$PART"
    LOCAL_FRAMES="$DATA_DIR/egodex_frames_$PART"
    S3_SRC="$S3_BUCKET/datasets/egodex-full/$PART/"

    log "=== $PART: Start ==="

    # 이미 추출된 프레임이 있으면 건너뛰기
    if [ -d "$LOCAL_FRAMES" ] && [ "$(ls -A "$LOCAL_FRAMES" 2>/dev/null)" ]; then
        log "$PART: Frames already exist ($LOCAL_FRAMES), skipping."
        continue
    fi

    # Step 2a: S3에서 원본 다운로드
    if [ -d "$LOCAL_RAW" ] && [ "$(ls -A "$LOCAL_RAW" 2>/dev/null)" ]; then
        log "$PART: Raw data already exists locally, skipping download."
    else
        log "$PART: Downloading raw data from $S3_SRC"
        mkdir -p "$LOCAL_RAW"
        $AWS s3 sync "$S3_SRC" "$LOCAL_RAW/" --quiet
    fi
    log "$PART: Raw data size: $(du -sh "$LOCAL_RAW" | cut -f1)"

    # Step 2b: 프레임 추출 (센터크롭, 256x256)
    VIDEO_COUNT=$(find "$LOCAL_RAW" -name "*.mp4" | wc -l)
    log "$PART: Extracting frames ($VIDEO_COUNT videos)"
    python3 "$SCRIPT_DIR/data/extract_frames.py" \
        --egodex-root "$DATA_DIR/egodex" \
        --split "$PART" \
        --output-dir "$LOCAL_FRAMES" \
        --yes

    log "$PART: Extraction complete. Size: $(du -sh "$LOCAL_FRAMES" | cut -f1)"
    log "=== $PART: Done ==="
done

log "=== All EgoDex parts (2,3,5) complete ==="
