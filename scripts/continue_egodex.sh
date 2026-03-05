#!/bin/bash
# EgoDex 전체 재추출 (256x256 센터크롭) 및 S3 업로드
# part2는 로컬에 원본 있음, part1/3/4/5는 S3에서 다운로드 필요
set -e
LOG="/tmp/egodex_process.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cd /home/etri/bys/action-agnostic-visual-rl

log "=== EgoDex 전체 재추출 시작 (256x256 센터크롭) ==="

# part2: 원본이 로컬에 있으므로 추출 → 업로드 → 정리
log "=== Starting part2 (로컬 원본 사용) ==="
bash scripts/process_egodex_part.sh part2 2>&1 | tee -a "$LOG"
log "=== Finished part2 ==="

# part1, 3, 4, 5: S3에서 다운로드 → 추출 → 업로드 → 정리
for p in part1 part3 part4 part5; do
    log "=== Starting $p ==="
    bash scripts/process_egodex_part.sh "$p" 2>&1 | tee -a "$LOG"
    log "=== Finished $p ==="
done

log "=== ALL PARTS DONE ==="
