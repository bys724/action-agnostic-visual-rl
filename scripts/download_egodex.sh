#!/bin/bash
# EgoDex 공식 CDN에서 다운로드 → 압축 해제 → 프레임 추출
# 다운로드 완료된 part부터 바로 추출 시작 (파이프라인)
#
# Usage:
#   bash scripts/download_egodex.sh part2 part3 part5

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/mnt/data"
CDN_BASE="https://ml-site.cdn-apple.com/datasets/egodex"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ $# -eq 0 ]; then
    echo "Usage: $0 <part1|part2|part3|part4|part5|test> ..."
    exit 1
fi

for PART in "$@"; do
    ZIP_FILE="$DATA_DIR/$PART.zip"
    RAW_DIR="$DATA_DIR/egodex/$PART"
    FRAMES_DIR="$DATA_DIR/egodex_frames_$PART"

    # 이미 프레임 추출 완료된 경우 스킵
    if [ -d "$FRAMES_DIR" ] && [ "$(ls -A "$FRAMES_DIR" 2>/dev/null)" ]; then
        log "[$PART] Frames already exist, skipping"
        continue
    fi

    # Step 1: 다운로드
    if [ -d "$RAW_DIR" ] && [ "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
        log "[$PART] Raw data already exists, skipping download"
    else
        log "[$PART] Downloading from CDN..."
        curl -L -o "$ZIP_FILE" "$CDN_BASE/$PART.zip"
        log "[$PART] Download complete: $(du -sh "$ZIP_FILE" | cut -f1)"

        # Step 2: 압축 해제
        log "[$PART] Extracting..."
        mkdir -p "$DATA_DIR/egodex"
        unzip -q "$ZIP_FILE" -d "$DATA_DIR/egodex/"
        log "[$PART] Extraction complete: $(du -sh "$RAW_DIR" | cut -f1)"

        # zip 삭제
        rm -f "$ZIP_FILE"
        log "[$PART] Removed zip file"
    fi

    # Step 3: 프레임 추출
    log "[$PART] Extracting frames (center crop → 256x256)..."
    python3 "$SCRIPT_DIR/data/extract_frames.py" \
        --egodex-root "$DATA_DIR/egodex" \
        --split "$PART" \
        --output-dir "$FRAMES_DIR" \
        --yes
    log "[$PART] Frames complete: $(du -sh "$FRAMES_DIR" | cut -f1)"

    log "[$PART] Done!"
done

log "=== All parts complete ==="
