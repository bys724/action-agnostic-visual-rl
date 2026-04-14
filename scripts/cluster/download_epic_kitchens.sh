#!/bin/bash
# EPIC-KITCHENS-100 다운로드 스크립트 (mp4 videos만, 프레임/flow 제외)
#
# 실행 위치: 로그인 노드 (compute 노드는 외부 네트워크 제한).
# 동시에 다른 대용량 다운로드 돌리지 말 것 (프로세스 누적 방지).
#
# 다운로드 방식: 공식 Python 스크립트 (epic_downloader.py). rsync 기반 내부 구현.
# 데이터: EPIC-KITCHENS-100 videos (~400 GB 추정), 참가자별 .mp4 파일.
# 메타데이터: epic-kitchens-100-annotations (CSV 등) 는 별도로 git clone 권장.
#
# Usage:
#   bash scripts/cluster/download_epic_kitchens.sh
#   nohup bash scripts/cluster/download_epic_kitchens.sh \
#       > /proj/external_group/mrg/logs/download_epic_kitchens.log 2>&1 &
#   disown
#
# 모니터링:
#   tail -f /proj/external_group/mrg/logs/download_epic_kitchens.log
#   ps -fu $USER | grep -E 'python|curl' | grep -v grep
#
# 끊어졌을 때: 같은 명령으로 다시 실행. 공식 스크립트가 resume 지원.

set -euo pipefail

DATA_ROOT="/proj/external_group/mrg/datasets/epic_kitchens_100"
LOG_DIR="/proj/external_group/mrg/logs"
TOOL_DIR="/proj/external_group/mrg/tools/epic-kitchens-download-scripts"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-extract"

mkdir -p "$DATA_ROOT" "$LOG_DIR" "$(dirname "$TOOL_DIR")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [epic-k] $*"; }

# conda (Python 3.5+ 필요, 외부 라이브러리 불필요하지만 일관성 위해 로드)
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# ── 공식 다운로드 스크립트 확보 ───────────────────────────────────────────────
if [ ! -d "$TOOL_DIR" ]; then
    log "Cloning epic-kitchens-download-scripts ..."
    git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts.git "$TOOL_DIR"
else
    log "Tool dir exists: $TOOL_DIR (skip clone)"
fi

# ── 다운로드 실행 ─────────────────────────────────────────────────────────────
# --videos: mp4만 (frames/flow 제외)
# --extension-only: EPIC-KITCHENS-100만 (EPIC-KITCHENS-55 제외)
# --output-path: 저장 경로
log "=== EPIC-KITCHENS-100 videos download start ==="
log "Dest: $DATA_ROOT"

cd "$TOOL_DIR"
python epic_downloader.py \
    --videos \
    --extension-only \
    --output-path "$DATA_ROOT"

log "=== Download phase done ==="
du -sh "$DATA_ROOT" 2>/dev/null | sed 's/^/  total: /' | while read l; do log "$l"; done

# ── Annotations (CSV) 별도 확보 ───────────────────────────────────────────────
ANN_DIR="$DATA_ROOT/annotations"
if [ ! -d "$ANN_DIR" ]; then
    log "Cloning epic-kitchens-100-annotations ..."
    git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-100-annotations.git "$ANN_DIR"
fi

log "=== All done ==="
