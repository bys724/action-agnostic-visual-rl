#!/bin/bash
# Ad-hoc fill chain for v15-mp missing rollouts (2026-05-19):
# - libero_object seed0 (누락, summary.csv n=2 → n=3로 채움)
# - libero_goal seed0/1/2 (전부 누락)
# Runs 2 GPU round-robin, then aggregates.
set -u

cd /home/etri/bys/action-agnostic-visual-rl
LOG=logs/v15mp_fill_4ckpt_chain.log

ITEMS=(
    "two-stream-v15-mp_libero_object_seed0_20260513_073914_v15ep50|libero_object|two-stream-v15-mp_v15ep50_t50"
    "two-stream-v15-mp_libero_goal_seed0_20260513_130106_v15ep50|libero_goal|two-stream-v15-mp_v15ep50_t50"
    "two-stream-v15-mp_libero_goal_seed1_20260513_154017_v15ep50|libero_goal|two-stream-v15-mp_v15ep50_t50"
    "two-stream-v15-mp_libero_goal_seed2_20260513_154017_v15ep50|libero_goal|two-stream-v15-mp_v15ep50_t50"
)

echo "[$(date '+%F %T')] v15-mp fill chain start, ${#ITEMS[@]} ckpts, 2 GPU round-robin" >> "$LOG"

run_one() {
    local gpu=$1
    local item=$2
    local stem suite tag
    IFS='|' read -r stem suite tag <<< "$item"

    local ckpt="/mnt/data/checkpoints/libero_bct/${stem}/best.pt"
    local results_dir="data/libero/results/${tag}"
    local video_dir="data/libero/videos/${tag}/${stem}"
    local log_dir="${results_dir}/_logs"

    docker exec libero-eval mkdir -p "/workspace/${results_dir}" "/workspace/${video_dir}" "/workspace/${log_dir}"

    echo "[GPU$gpu $(date '+%F %T')] START: $stem ($suite, 50 trials)" >> "$LOG"
    docker exec -e CUDA_VISIBLE_DEVICES=$gpu -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 libero-eval bash -c "
        python src/eval_libero.py \
            --checkpoint '$ckpt' \
            --task-suite '$suite' \
            --num-trials 50 \
            --quiet \
            --output-dir '$results_dir' \
            --video-dir '$video_dir' \
            > '${log_dir}/${stem}.log' 2>&1
    "
    local rc=$?
    echo "[GPU$gpu $(date '+%F %T')] DONE rc=$rc: $stem" >> "$LOG"
}

# GPU0: items 0, 2 | GPU1: items 1, 3
(
    for i in 0 2; do
        run_one 0 "${ITEMS[$i]}"
    done
) &
P0=$!

(
    for i in 1 3; do
        run_one 1 "${ITEMS[$i]}"
    done
) &
P1=$!

wait $P0 $P1

echo "[$(date '+%F %T')] all rollouts done, aggregating..." >> "$LOG"
python3 scripts/eval/aggregate_libero_rollouts.py \
    --input-dir data/libero/results \
    --output-dir paper_artifacts/libero_rollout >> "$LOG" 2>&1
echo "[$(date '+%F %T')] aggregate done" >> "$LOG"
