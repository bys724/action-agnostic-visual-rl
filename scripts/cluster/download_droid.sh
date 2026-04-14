#!/bin/bash
# IBS 클러스터용 DROID 다운로드 스크립트
# GCS의 RLDS 포맷 (gs://gresearch/robotics/droid, ~1.7 TB)
#
# gsutil은 conda env (aavrl-extract)에 설치되어 있음
# 로그인 노드에서 nohup 백그라운드로 실행 (네트워크 I/O만)
#
# Usage:
#   bash scripts/cluster/download_droid.sh
#   bash scripts/cluster/download_droid.sh --sample   # droid_100 (~2GB)
#   nohup bash scripts/cluster/download_droid.sh > droid.log 2>&1 &

set -euo pipefail

# ── 로그인 노드 실행 금지 ─────────────────────────────────────────────────────
# gsutil -m은 기본 수백 스레드. thread-count 제한을 걸어도 로그인 노드의
# 공용 자원을 고갈시켜 접속 장애를 유발할 수 있음. sbatch로 compute 노드 사용.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    HOST=$(hostname)
    if [[ "$HOST" =~ ^olaf[0-9]+$ ]]; then
        echo "ERROR: This script must not run on login node ($HOST)." >&2
        echo "       sbatch로 compute 노드에서 실행하세요." >&2
        echo "       예: sbatch -p normal_cpu -t 24:00:00 --wrap='bash $0 $*'" >&2
        exit 1
    fi
fi

DEST="/proj/external_group/mrg/datasets/droid"
LOG_DIR="/proj/external_group/mrg/logs"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-extract"
SAMPLE=false

for arg in "$@"; do
    case $arg in
        --sample) SAMPLE=true ;;
    esac
done

if $SAMPLE; then
    SRC="gs://gresearch/robotics/droid_100"
    DEST="/proj/external_group/mrg/datasets/droid_100"
    LOG="$LOG_DIR/download_droid_100.log"
    SIZE_DESC="~2 GB sample"
else
    SRC="gs://gresearch/robotics/droid"
    LOG="$LOG_DIR/download_droid.log"
    SIZE_DESC="~1.7 TB full"
fi

mkdir -p "$DEST" "$LOG_DIR"

# conda 환경 로드 (gsutil)
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== DROID download start ==="
log "Source: $SRC"
log "Destination: $DEST"
log "Size: $SIZE_DESC"

# gsutil rsync: 이미 있는 파일은 건너뛰고 나머지만 다운로드 (resumable)
# -m: 멀티스레드 병렬, -r: 재귀
MAX_RETRIES=5
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    log "Download attempt $((RETRY + 1))/$MAX_RETRIES"
    # thread/process count 제한: 기본 수백 → 로그인 노드 thread limit 회피
    if gsutil -o "GSUtil:parallel_thread_count=8" \
              -o "GSUtil:parallel_process_count=4" \
              -m rsync -r "$SRC" "$DEST/"; then
        log "Download complete"
        du -sh "$DEST"
        exit 0
    fi
    RETRY=$((RETRY + 1))
    if [ $RETRY -lt $MAX_RETRIES ]; then
        WAIT=$((30 * RETRY))
        log "Download interrupted. Retrying in ${WAIT}s..."
        sleep $WAIT
    fi
done

log "ERROR: Download failed after $MAX_RETRIES attempts"
exit 1
