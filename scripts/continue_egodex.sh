#!/bin/bash
# EgoDex 전체 재추출 (256x256 센터크롭) 및 S3 업로드
# part1 우선 처리 (AWS 학습 조기 시작 가능)
set -euo pipefail
LOG="/tmp/egodex_process.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cd /home/etri/bys/action-agnostic-visual-rl

log "=== EgoDex 전체 재추출 시작 (256x256 센터크롭) ==="

for p in part1 part2 part3 part4 part5; do
    log "=== Starting $p ==="
    bash scripts/process_egodex_part.sh "$p" 2>&1 | tee -a "$LOG"
    log "=== Finished $p ==="
done

log "=== ALL PARTS DONE ==="
