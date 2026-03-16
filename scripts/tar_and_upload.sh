#!/bin/bash
# 추출된 프레임을 task별 tar로 묶어서 S3에 업로드
# 완료 후 S3의 기존 개별 파일 삭제
#
# Usage:
#   ./scripts/tar_and_upload.sh egodex part1          # 로컬에 있는 경우
#   ./scripts/tar_and_upload.sh egodex part2 --download  # S3에서 다운로드 필요한 경우
#   ./scripts/tar_and_upload.sh bridge_v2              # Bridge V2

set -e

AWS=~/.local/bin/aws
DATA_DIR="/mnt/data"
S3_BUCKET="s3://bys724-research-2026"
TAR_DIR="$DATA_DIR/tarballs"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

upload_tarballs() {
    local LOCAL_FRAMES="$1"    # 프레임 디렉토리 경로
    local S3_TAR_DST="$2"     # S3 tarball 목적지
    local S3_FRAMES_SRC="$3"  # S3 기존 개별 파일 경로 (삭제용)
    local TAR_WORK="$4"       # 로컬 tar 임시 디렉토리

    mkdir -p "$TAR_WORK"

    local TASK_COUNT=$(ls -d "$LOCAL_FRAMES"/*/ 2>/dev/null | wc -l)
    local DONE=0

    for TASK_DIR in "$LOCAL_FRAMES"/*/; do
        TASK_NAME=$(basename "$TASK_DIR")
        TAR_FILE="$TAR_WORK/$TASK_NAME.tar"
        DONE=$((DONE + 1))

        # 이미 S3에 tar가 있으면 건너뛰기
        if $AWS s3 ls "$S3_TAR_DST$TASK_NAME.tar" &>/dev/null; then
            log "  [$DONE/$TASK_COUNT] Skip (already uploaded): $TASK_NAME"
            continue
        fi

        log "  [$DONE/$TASK_COUNT] Tar: $TASK_NAME"
        tar cf "$TAR_FILE" -C "$LOCAL_FRAMES" "$TASK_NAME"

        TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
        log "  [$DONE/$TASK_COUNT] Upload ($TAR_SIZE): $TASK_NAME.tar"
        $AWS s3 cp "$TAR_FILE" "$S3_TAR_DST$TASK_NAME.tar" --quiet

        # 업로드 완료 후 로컬 tar 삭제 (디스크 절약)
        rm -f "$TAR_FILE"
    done

    log "=== All tarballs uploaded to $S3_TAR_DST ==="
}

if [[ "$1" == "egodex" ]]; then
    PART="$2"
    if [[ -z "$PART" ]]; then
        echo "Usage: $0 egodex <partN> [--download]"
        exit 1
    fi

    LOCAL_FRAMES="$DATA_DIR/egodex_frames_$PART"
    S3_TAR_DST="$S3_BUCKET/egodex_tarballs/$PART/"
    S3_FRAMES_SRC="$S3_BUCKET/egodex_frames_$PART/"
    TAR_WORK="$TAR_DIR/egodex_$PART"

    # S3에서 다운로드 필요한 경우
    if [[ "$3" == "--download" ]]; then
        if [ -d "$LOCAL_FRAMES" ] && [ "$(ls -A "$LOCAL_FRAMES" 2>/dev/null)" ]; then
            log "Frames already exist locally: $LOCAL_FRAMES"
        else
            log "=== Downloading frames from S3: $S3_FRAMES_SRC ==="
            mkdir -p "$LOCAL_FRAMES"
            $AWS s3 sync "$S3_FRAMES_SRC" "$LOCAL_FRAMES/" --quiet
            log "Download complete."
        fi
    fi

    if [ ! -d "$LOCAL_FRAMES" ]; then
        echo "Error: $LOCAL_FRAMES not found. Use --download to fetch from S3."
        exit 1
    fi

    log "=== EgoDex $PART: tar + upload ==="
    upload_tarballs "$LOCAL_FRAMES" "$S3_TAR_DST" "$S3_FRAMES_SRC" "$TAR_WORK"

    # 기존 S3 개별 파일 삭제
    log "=== Deleting old S3 frames: $S3_FRAMES_SRC ==="
    $AWS s3 rm "$S3_FRAMES_SRC" --recursive --quiet
    log "=== Done: EgoDex $PART ==="

elif [[ "$1" == "bridge_v2" ]]; then
    LOCAL_FRAMES="$DATA_DIR/bridge_v2_frames"
    S3_TAR_DST="$S3_BUCKET/bridge_v2_tarballs/"
    S3_FRAMES_SRC="$S3_BUCKET/bridge_v2_frames/"
    TAR_WORK="$TAR_DIR/bridge_v2"

    # Bridge V2는 traj가 24,827개 → traj별 tar는 비효율
    # 1000개씩 묶어서 chunk_000.tar, chunk_001.tar, ...
    if [ ! -d "$LOCAL_FRAMES" ]; then
        log "=== Downloading bridge_v2 frames from S3 ==="
        mkdir -p "$LOCAL_FRAMES"
        $AWS s3 sync "$S3_FRAMES_SRC" "$LOCAL_FRAMES/" --quiet
        log "Download complete."
    fi

    mkdir -p "$TAR_WORK"
    log "=== Bridge V2: tar + upload (1000 traj per chunk) ==="

    # trajectory 목록 생성
    TRAJ_LIST=$(mktemp)
    ls -d "$LOCAL_FRAMES"/traj_*/ 2>/dev/null | sort > "$TRAJ_LIST"
    TOTAL=$(wc -l < "$TRAJ_LIST")
    CHUNK_SIZE=5000
    CHUNK_IDX=0

    while IFS= read -r LINE; do
        TRAJS+=("$LINE")
        if [[ ${#TRAJS[@]} -ge $CHUNK_SIZE ]]; then
            CHUNK_NAME=$(printf "chunk_%03d" $CHUNK_IDX)
            TAR_FILE="$TAR_WORK/$CHUNK_NAME.tar"

            if $AWS s3 ls "$S3_TAR_DST$CHUNK_NAME.tar" &>/dev/null; then
                log "  Skip (already uploaded): $CHUNK_NAME"
            else
                START=$((CHUNK_IDX * CHUNK_SIZE + 1))
                END=$((START + ${#TRAJS[@]} - 1))
                log "  Tar: $CHUNK_NAME (traj $START-$END of $TOTAL)"

                # tar에 추가할 경로만 전달
                TRAJ_NAMES=()
                for T in "${TRAJS[@]}"; do
                    TRAJ_NAMES+=("$(basename "$T")")
                done
                tar cf "$TAR_FILE" -C "$LOCAL_FRAMES" "${TRAJ_NAMES[@]}"

                TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
                log "  Upload ($TAR_SIZE): $CHUNK_NAME.tar"
                $AWS s3 cp "$TAR_FILE" "$S3_TAR_DST$CHUNK_NAME.tar" --quiet
                rm -f "$TAR_FILE"
            fi

            CHUNK_IDX=$((CHUNK_IDX + 1))
            TRAJS=()
        fi
    done < "$TRAJ_LIST"

    # 남은 trajectory 처리
    if [[ ${#TRAJS[@]} -gt 0 ]]; then
        CHUNK_NAME=$(printf "chunk_%03d" $CHUNK_IDX)
        TAR_FILE="$TAR_WORK/$CHUNK_NAME.tar"

        if ! $AWS s3 ls "$S3_TAR_DST$CHUNK_NAME.tar" &>/dev/null; then
            TRAJ_NAMES=()
            for T in "${TRAJS[@]}"; do
                TRAJ_NAMES+=("$(basename "$T")")
            done
            log "  Tar: $CHUNK_NAME (last ${#TRAJS[@]} trajectories)"
            tar cf "$TAR_FILE" -C "$LOCAL_FRAMES" "${TRAJ_NAMES[@]}"
            TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
            log "  Upload ($TAR_SIZE): $CHUNK_NAME.tar"
            $AWS s3 cp "$TAR_FILE" "$S3_TAR_DST$CHUNK_NAME.tar" --quiet
            rm -f "$TAR_FILE"
        fi
    fi

    rm -f "$TRAJ_LIST"

    # 기존 S3 개별 파일 삭제
    log "=== Deleting old S3 frames: $S3_FRAMES_SRC ==="
    $AWS s3 rm "$S3_FRAMES_SRC" --recursive --quiet
    log "=== Done: Bridge V2 ==="

elif [[ "$1" == "droid" ]]; then
    # DROID: 카메라별(ext1, ext2, wrist) × 5000 에피소드 청크로 tar
    LOCAL_FRAMES="$DATA_DIR/droid_frames"
    S3_TAR_DST="$S3_BUCKET/droid_tarballs/"
    TAR_WORK="$TAR_DIR/droid"

    if [ ! -d "$LOCAL_FRAMES" ]; then
        echo "Error: $LOCAL_FRAMES not found. Run extract_droid_frames.py first."
        exit 1
    fi

    mkdir -p "$TAR_WORK"

    for CAM in ext1 ext2 wrist; do
        CAM_DIR="$LOCAL_FRAMES/$CAM"
        if [ ! -d "$CAM_DIR" ]; then
            log "Warning: $CAM_DIR not found, skipping."
            continue
        fi

        log "=== DROID $CAM: tar + upload (5000 ep per chunk) ==="

        EP_LIST=$(mktemp)
        ls -d "$CAM_DIR"/ep_*/ 2>/dev/null | sort > "$EP_LIST"
        TOTAL=$(wc -l < "$EP_LIST")
        CHUNK_SIZE=5000
        CHUNK_IDX=0
        EPS=()

        while IFS= read -r LINE; do
            EPS+=("$LINE")
            if [[ ${#EPS[@]} -ge $CHUNK_SIZE ]]; then
                CHUNK_NAME="${CAM}_chunk_$(printf '%03d' $CHUNK_IDX)"
                TAR_FILE="$TAR_WORK/$CHUNK_NAME.tar"

                if $AWS s3 ls "$S3_TAR_DST$CHUNK_NAME.tar" &>/dev/null; then
                    log "  Skip (already uploaded): $CHUNK_NAME"
                else
                    START=$((CHUNK_IDX * CHUNK_SIZE + 1))
                    END=$((START + ${#EPS[@]} - 1))
                    log "  Tar: $CHUNK_NAME (ep $START-$END of $TOTAL)"

                    EP_NAMES=()
                    for E in "${EPS[@]}"; do
                        EP_NAMES+=("$(basename "$E")")
                    done
                    tar cf "$TAR_FILE" -C "$CAM_DIR" "${EP_NAMES[@]}"

                    TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
                    log "  Upload ($TAR_SIZE): $CHUNK_NAME.tar"
                    $AWS s3 cp "$TAR_FILE" "$S3_TAR_DST$CHUNK_NAME.tar" --quiet
                    rm -f "$TAR_FILE"
                fi

                CHUNK_IDX=$((CHUNK_IDX + 1))
                EPS=()
            fi
        done < "$EP_LIST"

        # 남은 에피소드 처리
        if [[ ${#EPS[@]} -gt 0 ]]; then
            CHUNK_NAME="${CAM}_chunk_$(printf '%03d' $CHUNK_IDX)"
            TAR_FILE="$TAR_WORK/$CHUNK_NAME.tar"

            if ! $AWS s3 ls "$S3_TAR_DST$CHUNK_NAME.tar" &>/dev/null; then
                EP_NAMES=()
                for E in "${EPS[@]}"; do
                    EP_NAMES+=("$(basename "$E")")
                done
                log "  Tar: $CHUNK_NAME (last ${#EPS[@]} episodes)"
                tar cf "$TAR_FILE" -C "$CAM_DIR" "${EP_NAMES[@]}"
                TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
                log "  Upload ($TAR_SIZE): $CHUNK_NAME.tar"
                $AWS s3 cp "$TAR_FILE" "$S3_TAR_DST$CHUNK_NAME.tar" --quiet
                rm -f "$TAR_FILE"
            fi
        fi

        rm -f "$EP_LIST"
    done

    log "=== Done: DROID ==="

else
    echo "Usage:"
    echo "  $0 egodex <partN> [--download]"
    echo "  $0 bridge_v2"
    echo "  $0 droid"
    exit 1
fi
