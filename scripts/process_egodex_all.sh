#!/bin/bash
# EgoDex 전체 part tar+upload 파이프라인
# part1 tar+upload가 이미 진행 중이므로, 그 이후부터 순차 실행
#
# 순서: part4(로컬) → part2(다운로드) → part3 → part5 → test
#
# Usage:
#   bash scripts/process_egodex_all.sh              # part4부터 순차 실행
#   bash scripts/process_egodex_all.sh --wait-part1  # part1 완료 대기 후 시작

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/mnt/data/egodex_all_pipeline.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# part1 대기 옵션
if [[ "$1" == "--wait-part1" ]]; then
    PART1_PID=$(pgrep -f 'tar_and_upload.sh egodex part1' || true)
    if [ -n "$PART1_PID" ]; then
        log "Waiting for part1 tar+upload (PID $PART1_PID) to finish..."
        while kill -0 "$PART1_PID" 2>/dev/null; do sleep 60; done
        log "part1 finished."
    else
        log "part1 not running, proceeding."
    fi
fi

# part4: 로컬에 있으므로 바로 tar+upload
log "=== part4: tar + upload (local) ==="
bash "$SCRIPT_DIR/tar_and_upload.sh" egodex part4 2>&1 | tee -a "$LOG"

# part2: S3에서 다운로드 → tar → upload
log "=== part2: download + tar + upload ==="
bash "$SCRIPT_DIR/tar_and_upload.sh" egodex part2 --download 2>&1 | tee -a "$LOG"

# part3
log "=== part3: download + tar + upload ==="
bash "$SCRIPT_DIR/tar_and_upload.sh" egodex part3 --download 2>&1 | tee -a "$LOG"

# part5
log "=== part5: download + tar + upload ==="
bash "$SCRIPT_DIR/tar_and_upload.sh" egodex part5 --download 2>&1 | tee -a "$LOG"

# test
log "=== test: download + tar + upload ==="
bash "$SCRIPT_DIR/tar_and_upload.sh" egodex test --download 2>&1 | tee -a "$LOG"

log "=== All EgoDex parts complete ==="
