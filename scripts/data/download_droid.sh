#!/bin/bash
# DROID 데이터셋 다운로드 스크립트
# GCS에서 RLDS 포맷으로 다운로드 (gs://gresearch/robotics/droid)
# gsutil의 resumable download를 활용하여 중단 시 이어받기 가능
#
# 사용법:
#   bash scripts/data/download_droid.sh [--sample]
#   --sample: droid_100 (2GB 샘플) 다운로드 (테스트용)

set -euo pipefail

DEST="/mnt/data/droid"
LOG="/mnt/data/droid_download.log"
SAMPLE=false

for arg in "$@"; do
    case $arg in
        --sample) SAMPLE=true ;;
    esac
done

# gsutil 설치 확인 및 설치
if ! command -v gsutil &>/dev/null; then
    echo "[$(date '+%H:%M:%S')] gsutil not found. Installing..."
    pip install gsutil 2>&1 | tail -3
    if ! command -v gsutil &>/dev/null; then
        echo "ERROR: gsutil installation failed. Try: pip install gsutil"
        exit 1
    fi
    echo "[$(date '+%H:%M:%S')] gsutil installed."
fi

if $SAMPLE; then
    SRC="gs://gresearch/robotics/droid_100"
    DEST="/mnt/data/droid_100"
    LOG="/mnt/data/droid_100_download.log"
    echo "[$(date '+%H:%M:%S')] Downloading DROID sample (droid_100, ~2GB)..."
else
    SRC="gs://gresearch/robotics/droid"
    echo "[$(date '+%H:%M:%S')] Downloading DROID full dataset (~1.7TB)..."
fi

echo "[$(date '+%H:%M:%S')] Source: $SRC"
echo "[$(date '+%H:%M:%S')] Destination: $DEST"
echo ""

mkdir -p "$DEST"

# gsutil cp -r: resumable download 지원
# -m: 멀티스레드 병렬 다운로드
# 실패 시 자동 재시도 (최대 5회)
MAX_RETRIES=5
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "[$(date '+%H:%M:%S')] Download attempt $((RETRY + 1))/$MAX_RETRIES" | tee -a "$LOG"

    if gsutil -m cp -r "$SRC/*" "$DEST/" 2>&1 | tee -a "$LOG"; then
        echo "[$(date '+%H:%M:%S')] Download complete!" | tee -a "$LOG"
        echo "[$(date '+%H:%M:%S')] Location: $DEST" | tee -a "$LOG"
        du -sh "$DEST" | tee -a "$LOG"
        exit 0
    fi

    RETRY=$((RETRY + 1))
    if [ $RETRY -lt $MAX_RETRIES ]; then
        WAIT=$((30 * RETRY))
        echo "[$(date '+%H:%M:%S')] Download interrupted. Retrying in ${WAIT}s..." | tee -a "$LOG"
        sleep $WAIT
    fi
done

echo "[$(date '+%H:%M:%S')] ERROR: Download failed after $MAX_RETRIES attempts." | tee -a "$LOG"
exit 1
