#!/bin/bash
# tar EgoDex frames to external drive with resume protection
#
# Resume logic:
#   - <part>.tar.done marker file → skip (already complete)
#   - <part>.tar.tmp → incomplete, will be overwritten on retry
#   - <part>.tar exists but no .done → treat as incomplete (safe default)

set -u  # 'e' 제외: 한 part 실패해도 다음 part 계속

SRC_ROOT="/mnt/data"
DST_ROOT="/mnt/external/egodex_frames"
PARTS=(part1 part2 part3 part4 part5)
LOG="${DST_ROOT}/tar.log"

mkdir -p "$DST_ROOT"
echo "=== $(date '+%F %T') tar_to_external.sh start ===" | tee -a "$LOG"

for p in "${PARTS[@]}"; do
    SRC_DIR="egodex_frames_${p}"
    DST_TMP="${DST_ROOT}/${p}.tar.tmp"
    DST_FINAL="${DST_ROOT}/${p}.tar"
    DONE_MARKER="${DST_ROOT}/${p}.tar.done"

    if [[ -f "$DONE_MARKER" ]]; then
        echo "[$(date '+%T')] SKIP $p (already done)" | tee -a "$LOG"
        continue
    fi

    if [[ ! -d "${SRC_ROOT}/${SRC_DIR}" ]]; then
        echo "[$(date '+%T')] MISSING src $SRC_DIR, skip" | tee -a "$LOG"
        continue
    fi

    # 이전 실패 잔여물 정리
    rm -f "$DST_TMP" "$DST_FINAL"

    echo "[$(date '+%T')] START $p" | tee -a "$LOG"
    START_TS=$(date +%s)

    if tar --sort=inode -cf "$DST_TMP" -C "$SRC_ROOT" "$SRC_DIR" 2>>"$LOG"; then
        mv "$DST_TMP" "$DST_FINAL"
        SIZE=$(du -h "$DST_FINAL" | cut -f1)
        # 원본 파일 수 카운트 (검증용)
        SRC_COUNT=$(find "${SRC_ROOT}/${SRC_DIR}" -type f 2>/dev/null | wc -l)
        TAR_COUNT=$(tar tf "$DST_FINAL" 2>/dev/null | grep -v '/$' | wc -l)
        ELAPSED=$(( $(date +%s) - START_TS ))
        echo "${SRC_COUNT}" > "${DONE_MARKER}"
        echo "[$(date '+%T')] DONE  $p  size=$SIZE  src_files=$SRC_COUNT  tar_entries=$TAR_COUNT  elapsed=${ELAPSED}s" | tee -a "$LOG"
        if [[ "$SRC_COUNT" != "$TAR_COUNT" ]]; then
            echo "[$(date '+%T')] WARN  $p  file count mismatch (src=$SRC_COUNT tar=$TAR_COUNT)" | tee -a "$LOG"
        fi
    else
        echo "[$(date '+%T')] FAIL  $p  (see $LOG)" | tee -a "$LOG"
        # 실패한 tmp는 다음 실행 시 덮어쓰기되므로 남겨둠 (디버깅용)
    fi
done

echo "=== $(date '+%F %T') tar_to_external.sh end ===" | tee -a "$LOG"
echo
echo "Summary:"
ls -lh "${DST_ROOT}/"*.tar 2>/dev/null
ls "${DST_ROOT}/"*.done 2>/dev/null | wc -l | xargs -I{} echo "{}/5 parts completed"
