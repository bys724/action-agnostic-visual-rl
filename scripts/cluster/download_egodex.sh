#!/bin/bash
# IBS 클러스터용 EgoDex CDN 다운로드 스크립트
#
# 로컬 워크스테이션 스크립트 (scripts/local/download_egodex.sh)와의 차이:
#   - 다운로드만 수행 (압축 해제 + 프레임 추출은 scripts/cluster/extract_egodex.sbatch가 담당)
#   - 저장 경로가 /proj/external_group/mrg/datasets/egodex/
#   - 로그인 노드에서 nohup 백그라운드로 실행 (CPU 잡 불필요, 네트워크 I/O만)
#   - curl -C - (resume) 지원: 끊겨도 이어받기
#
# Usage:
#   bash scripts/cluster/download_egodex.sh part1 part2 part3 part4 part5
#   nohup bash scripts/cluster/download_egodex.sh part1 part2 part3 part4 part5 > download.log 2>&1 &

set -e

DATA_ROOT="/proj/external_group/mrg/datasets/egodex"
ZIP_DIR="$DATA_ROOT/zips"
LOG_DIR="/proj/external_group/mrg/logs"
CDN_BASE="https://ml-site.cdn-apple.com/datasets/egodex"

mkdir -p "$ZIP_DIR" "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [ $# -eq 0 ]; then
    echo "Usage: $0 <part1|part2|part3|part4|part5|test> ..."
    echo "Example: $0 part1 part2 part3 part4 part5"
    exit 1
fi

for PART in "$@"; do
    ZIP_FILE="$ZIP_DIR/$PART.zip"

    log "[$PART] Downloading from CDN to $ZIP_FILE"
    # -C - : resume support, 이미 있는 파일이면 이어받기
    # -L   : follow redirects
    # -f   : fail on HTTP errors (e.g., 404)
    curl -L -C - -f -o "$ZIP_FILE" "$CDN_BASE/$PART.zip"
    log "[$PART] Download complete: $(du -sh "$ZIP_FILE" | cut -f1)"
done

log "=== All downloads complete ==="
log "Next: sbatch scripts/cluster/extract_egodex.sbatch"
