#!/usr/bin/env bash
#
# Ego4D frames: cluster → H100 via ETRI streaming (no local disk usage)
#
# 본 워크스테이션(ETRI)이 cluster와 H100(10.254.39.136) 양쪽 접근 가능하지만
# H100은 cluster 직결 불가, ETRI 디스크는 1TB로 4TB 수용 부족 → ETRI를
# pure pipe로 두고 tar stream을 cluster → ETRI → H100 그대로 흘림.
#
# 의존성: ~/.ssh/config 에 `cluster_login` + `h100` alias 등록.
# manifest는 transfer_ego4d_from_cluster.sh 와 동일 포맷(완료 UUID 한 줄씩),
# 두 스크립트가 같은 manifest를 공유해도 안전.
#
# 실행 (tmux 안에서):
#   bash scripts/local/stream_ego4d_cluster_to_h100.sh --sanity   # 앞 20 UUID
#   bash scripts/local/stream_ego4d_cluster_to_h100.sh            # 본 전송
#   until bash scripts/local/stream_ego4d_cluster_to_h100.sh; do sleep 60; done

set -uo pipefail

CLUSTER="${CLUSTER:-cluster_login}"
H100="${H100:-h100}"
SRC_REMOTE="${SRC_REMOTE:-/proj/external_group/mrg/datasets/ego4d/frames}"
DEST_REMOTE="${DEST_REMOTE:-/mnt/data/ego4d/frames}"
MANIFEST="${MANIFEST:-$HOME/ego4d_done.txt}"
UUID_LIST="${UUID_LIST:-$HOME/ego4d_uuids.txt}"
PARALLEL_N="${PARALLEL_N:-4}"

touch "$MANIFEST"
ssh "$H100" "mkdir -p $DEST_REMOTE"

# UUID 목록 캐시 (1회). 갱신은 $UUID_LIST 삭제 후 재실행.
if [[ ! -s "$UUID_LIST" ]]; then
    echo "[$(date '+%F %T')] fetching UUID list from $CLUSTER:$SRC_REMOTE ..."
    ssh "$CLUSTER" "ls $SRC_REMOTE" > "$UUID_LIST"
fi

# --sanity: 앞 20 UUID만 (throughput 측정용)
LIST_FILE="$UUID_LIST"
if [[ "${1:-}" == "--sanity" ]]; then
    LIST_FILE=$(mktemp)
    head -20 "$UUID_LIST" > "$LIST_FILE"
    echo "[$(date '+%F %T')] sanity mode: $(wc -l < "$LIST_FILE") UUIDs"
fi

transfer_one() {
    local d=$1
    grep -qxF "$d" "$MANIFEST" && return 0
    # cluster tar | ETRI pipe | H100 tar -xf  (ETRI 디스크 안 거침)
    if ssh "$CLUSTER" "tar -C $SRC_REMOTE -cf - $d" \
        | ssh "$H100" "tar -C $DEST_REMOTE -xf -"; then
        flock "$MANIFEST" -c "echo $d >> $MANIFEST"
        return 0
    else
        echo "[$(date '+%F %T')] FAIL $d" >&2
        return 1
    fi
}
export -f transfer_one
export CLUSTER H100 SRC_REMOTE DEST_REMOTE MANIFEST

total=$(wc -l < "$LIST_FILE")
done_n=$(wc -l < "$MANIFEST")
echo "[$(date '+%F %T')] start: $done_n / $total already done, parallel=$PARALLEL_N"

if command -v parallel >/dev/null 2>&1; then
    parallel -j "$PARALLEL_N" transfer_one :::: "$LIST_FILE"
else
    xargs -a "$LIST_FILE" -I{} -P "$PARALLEL_N" \
        bash -c 'transfer_one "$@"' _ {}
fi

done_n=$(wc -l < "$MANIFEST")
echo "[$(date '+%F %T')] end:   $done_n / $total"

[[ "$done_n" -eq "$total" ]]
