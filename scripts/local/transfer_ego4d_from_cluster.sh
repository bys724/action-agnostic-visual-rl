#!/usr/bin/env bash
#
# Ego4D frames: cluster -> local workstation pull (manifest 기반 재개)
#
# Source : <cluster>:/proj/external_group/mrg/datasets/ego4d/frames/
#          (9821 UUID dirs, ~4 TB / ~9천만 jpg 추정)
# Dest   : /mnt/data/ego4d/frames/  (default; DEST_LOCAL=... override)
#
# 전략:
#   - UUID 폴더 단위 streaming tar over SSH
#     (개별 jpg rsync = 9천만 stat syscall = 비현실적)
#   - manifest 파일에 완료된 UUID 기록 -> 끊기면 빠진 폴더만 재시도
#   - SSH ControlMaster로 N개 병렬 전송이 TCP 1개를 공유 (인증 0회 추가)
#   - mbuffer / parallel은 있으면 사용, 없으면 graceful fallback
#
# 사전 준비 (워크스테이션):
#   1. ~/.ssh/config 에 cluster_login 호스트 등록 (또는 CLUSTER=... 환경변수 override):
#        Host cluster_login
#            HostName <cluster_host>
#            User bys724
#            ServerAliveInterval 30
#            ServerAliveCountMax 120
#            TCPKeepAlive yes
#            ControlMaster auto
#            ControlPath ~/.ssh/cm-%r@%h:%p
#            ControlPersist 1h
#   2. (optional) GNU parallel, mbuffer 설치 — 없으면 xargs/직접파이프로 fallback
#
# 실행:
#   tmux new -s ego4d
#   bash scripts/local/transfer_ego4d_from_cluster.sh --sanity   # 앞 20 폴더만 (throughput 측정)
#   bash scripts/local/transfer_ego4d_from_cluster.sh            # 본 전송
#
#   # 네트워크 단절 자동 복구 wrapper:
#   until bash scripts/local/transfer_ego4d_from_cluster.sh; do
#       echo "[$(date)] retry in 60s"; sleep 60
#   done
#
# 검증 (전송 종료 후):
#   ssh cluster_login \
#     "cd /proj/external_group/mrg/datasets/ego4d/frames \
#      && find . -type f -printf '%P %s\n' | LC_ALL=C sort" > /tmp/ego4d_src.lst
#   ( cd /mnt/data/ego4d/frames \
#     && find . -type f -printf '%P %s\n' | LC_ALL=C sort ) > /tmp/ego4d_dst.lst
#   diff /tmp/ego4d_src.lst /tmp/ego4d_dst.lst    # 출력 0줄이면 완전 일치

set -uo pipefail

CLUSTER="${CLUSTER:-cluster_login}"
SRC_REMOTE="${SRC_REMOTE:-/proj/external_group/mrg/datasets/ego4d/frames}"
DEST_LOCAL="${DEST_LOCAL:-/mnt/data/ego4d/frames}"
MANIFEST="${MANIFEST:-$HOME/ego4d_done.txt}"
UUID_LIST="${UUID_LIST:-$HOME/ego4d_uuids.txt}"
PARALLEL_N="${PARALLEL_N:-6}"

mkdir -p "$DEST_LOCAL"
touch "$MANIFEST"

# UUID 목록 캐시 (1회만; 갱신 원하면 $UUID_LIST 삭제 후 재실행)
if [[ ! -s "$UUID_LIST" ]]; then
    echo "[$(date '+%F %T')] fetching UUID list from $CLUSTER:$SRC_REMOTE ..."
    ssh "$CLUSTER" "ls $SRC_REMOTE" > "$UUID_LIST"
fi

# mbuffer 있으면 throughput 변동 흡수
if command -v mbuffer >/dev/null 2>&1; then
    BUF_CMD="mbuffer -q -m 512M"
else
    BUF_CMD="cat"
fi
export BUF_CMD

# --sanity: 앞 20 폴더만 (throughput 측정용)
LIST_FILE="$UUID_LIST"
if [[ "${1:-}" == "--sanity" ]]; then
    LIST_FILE=$(mktemp)
    head -20 "$UUID_LIST" > "$LIST_FILE"
    echo "[$(date '+%F %T')] sanity mode: $(wc -l < "$LIST_FILE") UUIDs"
fi

transfer_one() {
    local d=$1
    grep -qxF "$d" "$MANIFEST" && return 0
    if ssh "$CLUSTER" "tar -C $SRC_REMOTE -cf - $d" \
        | eval "$BUF_CMD" \
        | tar -C "$DEST_LOCAL" -xf -; then
        flock "$MANIFEST" -c "echo $d >> $MANIFEST"
        return 0
    else
        echo "[$(date '+%F %T')] FAIL $d" >&2
        return 1
    fi
}
export -f transfer_one
export CLUSTER SRC_REMOTE DEST_LOCAL MANIFEST

total=$(wc -l < "$LIST_FILE")
done_n=$(wc -l < "$MANIFEST")
echo "[$(date '+%F %T')] start: $done_n / $total already done, parallel=$PARALLEL_N"

if command -v parallel >/dev/null 2>&1; then
    parallel -j "$PARALLEL_N" transfer_one :::: "$LIST_FILE"
else
    # xargs fallback
    xargs -a "$LIST_FILE" -I{} -P "$PARALLEL_N" \
        bash -c 'transfer_one "$@"' _ {}
fi

done_n=$(wc -l < "$MANIFEST")
echo "[$(date '+%F %T')] end:   $done_n / $total"

# 전체 완료 시 0 / 일부 실패 시 non-zero -> wrapper의 until loop가 재시도
[[ "$done_n" -eq "$total" ]]
