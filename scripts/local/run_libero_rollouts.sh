#!/bin/bash
# LIBERO BC-T rollouts: 모든 (suite × seed) ckpt을 2 GPU 병렬 실행.
#
# Usage:
#   bash scripts/local/run_libero_rollouts.sh <encoder_prefix> [num_trials]
#
# Example:
#   bash scripts/local/run_libero_rollouts.sh vc1 50
#
# 검색 패턴: /mnt/data/checkpoints/libero_bct/<encoder_prefix>_libero_*_v3/best.pt
# GPU 분배: round-robin (i%2). 9 ckpt 기준 ~12-17h wall time.

set -u

ENCODER_PREFIX=${1:?"usage: $0 <encoder_prefix> [num_trials]"}
NUM_TRIALS=${2:-50}

cd "$(dirname "$0")/../.."

CKPT_BASE=/mnt/data/checkpoints/libero_bct
TAG="${ENCODER_PREFIX}_v3_t${NUM_TRIALS}"
RESULTS_DIR="data/libero/results/${TAG}"
VIDEO_DIR="data/libero/videos/${TAG}"
LOG_DIR="${RESULTS_DIR}/_logs"

# data/libero/는 docker container 내부에서 root로 생성·소유. 호스트 etri 권한 없음.
# → 모든 file ops를 docker exec 안에서 실행 (컨테이너에서 mkdir, redirect).
docker exec libero-eval mkdir -p "/workspace/$RESULTS_DIR" "/workspace/$VIDEO_DIR" "/workspace/$LOG_DIR"

run_one() {
    local gpu=$1; local ckpt_path=$2; local suite=$3
    local stem; stem=$(basename "$(dirname "$ckpt_path")")
    echo "[GPU$gpu $(date '+%F %T')] START: $stem ($suite, $NUM_TRIALS trials)"
    docker exec -e CUDA_VISIBLE_DEVICES=$gpu libero-eval bash -c "
        python src/eval_libero.py \
            --checkpoint '$ckpt_path' \
            --task-suite '$suite' \
            --num-trials '$NUM_TRIALS' \
            --quiet \
            --output-dir '$RESULTS_DIR' \
            --video-dir '$VIDEO_DIR/$stem' \
            > '$LOG_DIR/${stem}.log' 2>&1
    "
    local rc=$?
    echo "[GPU$gpu $(date '+%F %T')] DONE rc=$rc: $stem"
}

# 모든 ckpt 수집
ALL_CKPTS=()
for d in "$CKPT_BASE/${ENCODER_PREFIX}"_libero_*_v3/; do
    [ -f "${d}best.pt" ] && ALL_CKPTS+=("${d}best.pt")
done
echo "Found ${#ALL_CKPTS[@]} ckpts for encoder=${ENCODER_PREFIX}"
[ ${#ALL_CKPTS[@]} -eq 0 ] && { echo "no ckpts found, abort"; exit 1; }

# GPU 0: 짝수 인덱스, GPU 1: 홀수 인덱스
(
    i=0
    for ckpt in "${ALL_CKPTS[@]}"; do
        if [ $((i % 2)) -eq 0 ]; then
            suite=$(basename "$(dirname "$ckpt")" | grep -oE 'libero_(spatial|object|goal|10|90)' | head -1)
            run_one 0 "$ckpt" "$suite"
        fi
        i=$((i+1))
    done
) &
P0=$!

(
    i=0
    for ckpt in "${ALL_CKPTS[@]}"; do
        if [ $((i % 2)) -eq 1 ]; then
            suite=$(basename "$(dirname "$ckpt")" | grep -oE 'libero_(spatial|object|goal|10|90)' | head -1)
            run_one 1 "$ckpt" "$suite"
        fi
        i=$((i+1))
    done
) &
P1=$!

wait $P0 $P1
echo "[$(date '+%F %T')] All rollouts complete → $RESULTS_DIR"

# 자동 집계 — 부모 dir 전체 스캔. 이전 인코더 결과를 보존하고 합쳐 CSV 갱신.
# encoder_type 메타 없는 옛 JSON은 aggregator 기본 동작으로 제외됨.
echo "Aggregating to paper_artifacts/libero_rollout/ ..."
python3 scripts/eval/aggregate_libero_rollouts.py \
    --input-dir data/libero/results \
    --output-dir paper_artifacts/libero_rollout
