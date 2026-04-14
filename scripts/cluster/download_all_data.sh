#!/bin/bash
# IBS 클러스터용 전체 데이터 다운로드 마스터 스크립트
#
# 동작:
#   1. EgoDex part1~5 다섯 개를 모두 병렬로 다운로드 (curl -C - resume)
#   2. 모든 part가 완료되면 (expected size 도달)
#   3. DROID 다운로드 시작 (gsutil rsync)
#
# Usage:
#   nohup bash scripts/cluster/download_all_data.sh \
#       > /proj/external_group/mrg/logs/download_all.log 2>&1 &
#   disown
#
# 모니터링:
#   tail -f /proj/external_group/mrg/logs/download_all.log
#   ls -lh /proj/external_group/mrg/datasets/egodex/zips/
#   ps -fu $USER | grep -E 'curl|gsutil' | grep -v grep
#
# 끊어졌을 때:
#   같은 명령으로 다시 실행. curl/gsutil 모두 resume 지원.

set -uo pipefail

# ── 로그인 노드 실행 금지 ─────────────────────────────────────────────────────
# 다수 병렬 curl + gsutil이 로그인 노드 프로세스/스레드 limit을 초과하여
# 접속 장애를 유발한 전례 있음 (관리자가 강제 kill). 항상 compute 노드에서 실행.
# sbatch 환경 아니면 중단 (SLURM_JOB_ID 없으면 로그인 노드로 간주).
if [ -z "${SLURM_JOB_ID:-}" ]; then
    HOST=$(hostname)
    if [[ "$HOST" =~ ^olaf[0-9]+$ ]]; then
        echo "ERROR: This script must not run on login node ($HOST)." >&2
        echo "       다수 병렬 curl/gsutil이 스레드 limit을 초과해 접속 장애를 유발합니다." >&2
        echo "       sbatch로 compute 노드(normal_cpu 등)에 제출하세요." >&2
        echo "       예: sbatch -p normal_cpu -t 24:00:00 --wrap='bash $0'" >&2
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/proj/external_group/mrg/datasets/egodex"
ZIP_DIR="$DATA_ROOT/zips"
LOG_DIR="/proj/external_group/mrg/logs"
CDN_BASE="https://ml-site.cdn-apple.com/datasets/egodex"

mkdir -p "$ZIP_DIR" "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [master] $*"; }

# ── EgoDex part 메타데이터 ────────────────────────────────────────────────────
# 각 part의 expected size (bytes) — content-length로 사전 확인한 값
declare -A PART_SIZES=(
    [part1]=321588941964
    [part2]=327274733628
    [part3]=326263668370
    [part4]=329365549875
    [part5]=331277374642
    [test]=17304529397
)

PARTS=(part1 part2 part3 part4 part5 test)

# ── Step 1: EgoDex 5 parts 병렬 다운로드 ──────────────────────────────────────
log "=== EgoDex parallel download start ==="
log "Total expected: $(numfmt --to=iec $(( 321588941964 + 327274733628 + 326263668370 + 329365549875 + 331277374642 + 17304529397 )))"

PIDS=()
for PART in "${PARTS[@]}"; do
    ZIP_FILE="$ZIP_DIR/${PART}.zip"
    PART_LOG="$LOG_DIR/download_${PART}.log"
    EXPECTED=${PART_SIZES[$PART]}

    # 이미 완료된 경우 스킵
    if [ -f "$ZIP_FILE" ]; then
        ACTUAL=$(stat -c%s "$ZIP_FILE")
        if [ "$ACTUAL" -ge "$EXPECTED" ]; then
            log "[$PART] Already complete ($ACTUAL bytes), skipping"
            continue
        fi
    fi

    log "[$PART] Starting curl -C - (expected $(numfmt --to=iec $EXPECTED))"
    (
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$PART] curl start (resume)" >> "$PART_LOG"
        curl -L -C - -f -o "$ZIP_FILE" "$CDN_BASE/${PART}.zip" >> "$PART_LOG" 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$PART] curl exit=$?" >> "$PART_LOG"
    ) &
    PIDS+=($!)
done

# ── Step 2: 모든 EgoDex 다운로드 완료 대기 ────────────────────────────────────
log "Waiting for ${#PIDS[@]} parallel downloads to complete..."
FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=$((FAIL+1))
    fi
done

# 완료 후 사이즈 검증
log "=== EgoDex download phase complete ==="
ALL_OK=true
for PART in "${PARTS[@]}"; do
    ZIP_FILE="$ZIP_DIR/${PART}.zip"
    EXPECTED=${PART_SIZES[$PART]}
    if [ ! -f "$ZIP_FILE" ]; then
        log "[$PART] MISSING"
        ALL_OK=false
        continue
    fi
    ACTUAL=$(stat -c%s "$ZIP_FILE")
    if [ "$ACTUAL" -ge "$EXPECTED" ]; then
        log "[$PART] OK ($(numfmt --to=iec $ACTUAL))"
    else
        log "[$PART] INCOMPLETE ($ACTUAL / $EXPECTED bytes)"
        ALL_OK=false
    fi
done

if ! $ALL_OK; then
    log "ERROR: Some EgoDex parts incomplete. Re-run this script to resume."
    exit 1
fi

# ── Step 3: DROID 다운로드 시작 ───────────────────────────────────────────────
log "=== Starting DROID download ==="
bash "$SCRIPT_DIR/download_droid.sh"

log "=== All downloads complete ==="
