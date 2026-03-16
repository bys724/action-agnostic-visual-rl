#!/bin/bash
# DROID 프레임 추출 → tar → S3 업로드 자동 파이프라인
#
# Usage:
#   bash scripts/process_droid.sh           # 전체 파이프라인
#   bash scripts/process_droid.sh --tar-only  # tar+업로드만 (추출 완료 후)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/mnt/data"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

TAR_ONLY=false
for arg in "$@"; do
    case $arg in
        --tar-only) TAR_ONLY=true ;;
    esac
done

# Step 1: 프레임 추출
if ! $TAR_ONLY; then
    log "=== Step 1: DROID frame extraction ==="
    python3 "$SCRIPT_DIR/data/extract_droid_frames.py" --yes \
        2>&1 | tee "$LOG_DIR/droid_extract.log"
    log "=== Step 1 complete ==="
fi

# Step 2: tar + S3 업로드
log "=== Step 2: tar + S3 upload ==="
bash "$SCRIPT_DIR/tar_and_upload.sh" droid \
    2>&1 | tee "$LOG_DIR/droid_tar_upload.log"
log "=== Step 2 complete ==="

log "=== All done: DROID pipeline ==="
