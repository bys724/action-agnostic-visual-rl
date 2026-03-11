#!/bin/bash
# Bridge V2 처리: ZIP 해제 → 프레임 추출(리사이즈 256x256) → S3 업로드(원본+프레임) → 로컬 정리
#
# Usage:
#   bash scripts/process_bridge_v2.sh          # demos + scripted 모두
#   bash scripts/process_bridge_v2.sh demos    # demos만
#   bash scripts/process_bridge_v2.sh scripted # scripted만
#
# 데이터 위치: /mnt/data/bridge_v2/
# S3 업로드: 원본 ZIP + 추출 프레임 모두 업로드

set -e

MODE=${1:-all}  # all, demos, scripted

AWS=~/.local/bin/aws
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

S3_BUCKET="s3://bys724-research-2026"
DATA_DIR="/mnt/data/bridge_v2"
FRAMES_DIR="/mnt/data/bridge_v2_frames"
LOG_FILE="$PROJECT_DIR/data/process_bridge_v2.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ZIP 파일 결정
ZIPS=()
if [[ "$MODE" == "all" || "$MODE" == "demos" ]]; then
    ZIPS+=("demos_8_17.zip")
fi
if [[ "$MODE" == "all" || "$MODE" == "scripted" ]]; then
    ZIPS+=("scripted_6_18.zip")
fi

log "=== Bridge V2 processing started (mode: $MODE) ==="
log "Data dir: $DATA_DIR"
log "Frames dir: $FRAMES_DIR"

# Step 1: ZIP 해제
log "=== Step 1/4: Unzipping ==="
RAW_DIR="$DATA_DIR/raw"
for ZIP in "${ZIPS[@]}"; do
    ZIP_PATH="$DATA_DIR/$ZIP"
    if [ ! -f "$ZIP_PATH" ]; then
        log "ERROR: $ZIP_PATH not found"
        exit 1
    fi
    log "Unzipping $ZIP ($(du -sh "$ZIP_PATH" | cut -f1))..."
    # -n: 이미 존재하는 파일 스킵 (중단 후 재시작 시 이어서)
    unzip -n "$ZIP_PATH" -d "$DATA_DIR/" 2>&1 | tail -1
    log "Unzip $ZIP done."
done
log "Raw data size: $(du -sh "$RAW_DIR" 2>/dev/null | cut -f1 || echo 'N/A')"

# Step 2: 프레임 추출
log "=== Step 2/4: Extracting frames ==="
mkdir -p "$FRAMES_DIR"
python3 "$SCRIPT_DIR/data/extract_bridge_frames.py" \
    --bridge-root "$DATA_DIR" \
    --output-dir "$FRAMES_DIR" \
    --yes
log "Extraction complete. Size: $(du -sh "$FRAMES_DIR" | cut -f1)"

# Step 3: S3 업로드
log "=== Step 3/4: Uploading to S3 ==="

# 3a. 원본 ZIP 업로드
log "Uploading original ZIPs..."
for ZIP in "${ZIPS[@]}"; do
    ZIP_PATH="$DATA_DIR/$ZIP"
    S3_RAW="$S3_BUCKET/datasets/bridge_v2/$ZIP"
    # 이미 업로드되었는지 확인
    if $AWS s3 ls "$S3_RAW" &>/dev/null; then
        log "  $ZIP already on S3, skipping"
    else
        log "  Uploading $ZIP..."
        $AWS s3 cp "$ZIP_PATH" "$S3_RAW"
        log "  $ZIP upload done."
    fi
done

# 3b. 추출 프레임 업로드 (traj 단위 순차 — EgoDex 방식)
S3_FRAMES="$S3_BUCKET/bridge_v2_frames/"
log "Uploading extracted frames to $S3_FRAMES..."
TOTAL_TRAJS=$(find "$FRAMES_DIR" -maxdepth 1 -type d -name "traj_*" | wc -l)
UPLOADED=0
for TRAJ_DIR in "$FRAMES_DIR"/traj_*/; do
    TRAJ_NAME=$(basename "$TRAJ_DIR")
    UPLOADED=$((UPLOADED + 1))
    if (( UPLOADED % 1000 == 0 )); then
        log "  Progress: $UPLOADED/$TOTAL_TRAJS"
    fi
    if ! $AWS s3 cp "$TRAJ_DIR" "$S3_FRAMES$TRAJ_NAME/" --recursive --quiet; then
        log "  Retry: $TRAJ_NAME"
        $AWS s3 cp "$TRAJ_DIR" "$S3_FRAMES$TRAJ_NAME/" --recursive --quiet
    fi
done
log "Frame upload complete. ($UPLOADED trajectories)"

# Step 4: 로컬 정리 (원본 해제분만 삭제, ZIP은 보존)
log "=== Step 4/4: Cleaning up ==="
rm -rf "$RAW_DIR"
rm -rf "$FRAMES_DIR"
log "Cleanup done. (ZIPs preserved in $DATA_DIR)"

log "=== Bridge V2 processing finished ==="
