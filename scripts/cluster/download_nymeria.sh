#!/bin/bash
# Nymeria 다운로드 스크립트 (video_main_rgb + body_motion + narration subset, ~2.14 TB)
#
# 실행 위치: 로그인 노드 (compute 노드는 외부 네트워크 제한).
# 다른 대용량 다운로드와 동시 실행 금지 (프로세스 누적 방지).
#
# JSON manifest URL의 유효기간 14일. 만료 시:
#   https://www.projectaria.com/datasets/nymeria/ 재방문 → 이메일 재입력 → 새 JSON 받기
#
# Usage:
#   bash scripts/cluster/download_nymeria.sh
#   nohup bash scripts/cluster/download_nymeria.sh \
#       > /proj/external_group/mrg/logs/download_nymeria.log 2>&1 &
#   disown
#
# 모니터링:
#   tail -f /proj/external_group/mrg/logs/download_nymeria.log
#   du -sh /proj/external_group/mrg/datasets/nymeria/data

set -euo pipefail

DATA_ROOT="/proj/external_group/mrg/datasets/nymeria"
MANIFEST="$DATA_ROOT/manifests/nymeria_v0.0_urls_video_motion.json"
DEST="$DATA_ROOT/data"
LOG_DIR="/proj/external_group/mrg/logs"
TOOL_DIR="/proj/external_group/mrg/tools/nymeria_dataset"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-extract"

mkdir -p "$DEST" "$LOG_DIR" "$(dirname "$TOOL_DIR")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [nymeria] $*"; }

# conda 환경 (download.py가 tqdm 등 사용)
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# ── 공식 다운로드 스크립트 확보 ───────────────────────────────────────────────
if [ ! -d "$TOOL_DIR" ]; then
    log "Cloning nymeria_dataset ..."
    git clone --depth 1 https://github.com/facebookresearch/nymeria_dataset.git "$TOOL_DIR"
    # nymeria 패키지 의존성: tqdm, requests 등. 보통 aavrl-extract에 이미 있음.
else
    log "Tool dir exists: $TOOL_DIR (skip clone)"
fi

# ── Manifest 확인 ─────────────────────────────────────────────────────────────
if [ ! -f "$MANIFEST" ]; then
    log "ERROR: Manifest not found: $MANIFEST"
    log "       먼저 filter_nymeria_manifest.py 실행 필요"
    exit 1
fi

log "=== Nymeria subset download start ==="
log "Manifest: $MANIFEST"
log "Dest: $DEST"

# ── 다운로드 실행 ─────────────────────────────────────────────────────────────
# download.py는 manifest의 각 URL을 순차 fetch (기본적으로 requests 기반).
# 끊어지면 같은 명령으로 재실행 시 file_size/sha1 비교로 skip.
#
# download.py는 시작 시 "Proceed: [y/n]" 인터랙티브 확인을 요구함 → nohup 환경에서
# stdin이 닫혀 있으면 멈춤. "y\n"을 파이프로 공급해 비대화식 실행.
#
# get_groups()는 모든 group을 반환하지만, 우리 filtered JSON에 없는 group은
# WARN_NOTFOUND로 skip되므로 subset 다운로드가 안전하게 동작함.
cd "$TOOL_DIR"
echo "y" | python download.py -i "$MANIFEST" -o "$DEST"

log "=== Nymeria download done ==="
du -sh "$DEST" 2>/dev/null | sed 's/^/  total: /' | while read l; do log "$l"; done
