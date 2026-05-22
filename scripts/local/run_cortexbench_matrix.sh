#!/bin/bash
# CortexBench §C11 full matrix: N enc × 7 task × 3 seed BC trainings
# Multi-GPU round-robin in cortexbench-eval container. Resumable (skip완료 잡).
#
# Usage:
#   bash scripts/local/run_cortexbench_matrix.sh                # 전체 5 enc (105잡)
#   bash scripts/local/run_cortexbench_matrix.sh --aggregate    # 학습 안 돌리고 집계만
#   ENCODERS="siglip_base dinov2_base vc1_vitb" bash scripts/local/run_cortexbench_matrix.sh
#                                                               # 분산용 — 특정 encoder만 (63잡)
#   NUM_GPUS=1 ENCODERS="vc1_vitb" bash scripts/local/run_cortexbench_matrix.sh
#                                                               # 단일 GPU 머신 (직렬 21잡)
#
# 분산 운영 예시 (docs/setup/CORTEXBENCH_GUIDE.md 참고):
#   머신 A (H100×2): ENCODERS="v15_p_only videomae_ours"
#   머신 B (H100×2): ENCODERS="siglip_base dinov2_base vc1_vitb"
#   머신 C (A6000×1, NUM_GPUS=1): ENCODERS="vc1_vitb"
#   → 양쪽 paper_artifacts/cortexbench/ rsync 후 aggregate
#
# 학습 잡 출력: paper_artifacts/cortexbench/<enc>/<task>/seed_<n>/
# 집계 출력:    paper_artifacts/cortexbench/{per_run,per_task,summary}.csv

set -u

cd "$(dirname "$0")/../.."

# Default 5 encoder. ENCODERS env로 override 가능 (분산 운영).
DEFAULT_ENCODERS="v15_p_only videomae_ours siglip_base dinov2_base vc1_vitb"
read -ra ENCODERS <<< "${ENCODERS:-$DEFAULT_ENCODERS}"

# GPU 수 (단일 GPU 머신은 NUM_GPUS=1 로 직렬 실행).
NUM_GPUS=${NUM_GPUS:-2}

# task entries: "suite:env:config-name"
TASKS=(
    "adroit:pen-v0:Adroit_BC_config.yaml"
    "adroit:relocate-v0:Adroit_BC_config.yaml"
    "metaworld:assembly-v2-goal-observable:Metaworld_BC_config.yaml"
    "metaworld:bin-picking-v2-goal-observable:Metaworld_BC_config.yaml"
    "metaworld:button-press-topdown-v2-goal-observable:Metaworld_BC_config.yaml"
    "metaworld:drawer-open-v2-goal-observable:Metaworld_BC_config.yaml"
    "metaworld:hammer-v2-goal-observable:Metaworld_BC_config.yaml"
)
SEEDS=(100 200 300)

OUT_ROOT=paper_artifacts/cortexbench
LOG_DIR=$OUT_ROOT/_logs
AGG_SCRIPT=scripts/eval/aggregate_cortexbench.py

mkdir -p "$LOG_DIR"

if [ "${1:-}" = "--aggregate" ]; then
    echo "[aggregate-only mode]"
    python3 "$AGG_SCRIPT" --root "$OUT_ROOT"
    exit $?
fi

# 컨테이너 살아 있는지 확인
if ! docker ps --format '{{.Names}}' | grep -q '^cortexbench-eval$'; then
    echo "[err] cortexbench-eval 컨테이너 실행 안 됨. 먼저: docker compose up -d cortexbench"
    exit 1
fi

is_done() {
    local rd=$1
    [ -f "$rd/_DONE" ]
}

run_one() {
    local idx=$1 gpu=$2 enc=$3 suite=$4 task=$5 cfg=$6 seed=$7 data_dir=$8
    local run_dir=$OUT_ROOT/$enc/$task/seed_$seed
    local log=$LOG_DIR/${enc}_${task}_seed${seed}.log

    if is_done "$run_dir"; then
        echo "[$idx GPU$gpu] SKIP (done): $enc $task seed=$seed"
        return 0
    fi

    mkdir -p "$run_dir"
    echo "[$idx GPU$gpu $(date '+%F %T')] START: $enc $task seed=$seed → $log"

    docker exec \
        -e V15_CKPT=/mnt/data/cortexbench/checkpoints/v15_ep50_latest.pt \
        -e VIDEOMAE_OURS_CKPT=/mnt/data/cortexbench/checkpoints/videomae_ours_best.pt \
        -e PYTHONPATH=/workspace \
        -e WANDB_MODE=offline \
        -e CUDA_VISIBLE_DEVICES=$gpu \
        cortexbench-eval bash -c "
            cd /workspace/external/eai-vc/cortexbench/mujoco_vc/visual_imitation && \
            python hydra_launcher.py --config-name '$cfg' \
                env='$task' embedding='$enc' seed='$seed' \
                data_dir='$data_dir/' \
                wandb.project='cortexbench_c11' wandb.entity='local' \
                hydra.run.dir='/workspace/$run_dir'
        " > "$log" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        touch "$run_dir/_DONE"
    fi
    echo "[$idx GPU$gpu $(date '+%F %T')] DONE rc=$rc: $enc $task seed=$seed"
}

# 전체 잡 리스트 구축 (idx, enc, suite, task, cfg, seed, data_dir)
JOBS=()
idx=0
for ENC in "${ENCODERS[@]}"; do
    for T_ENTRY in "${TASKS[@]}"; do
        IFS=':' read -r SUITE TASK CFG <<< "$T_ENTRY"
        if [ "$SUITE" = "adroit" ]; then
            DATA_DIR=/mnt/data/cortexbench/datasets/adroit-expert-v1.0
        else
            DATA_DIR=/mnt/data/cortexbench/datasets/metaworld-expert-v1.0
        fi
        for SEED in "${SEEDS[@]}"; do
            JOBS+=("$idx|$ENC|$SUITE|$TASK|$CFG|$SEED|$DATA_DIR")
            idx=$((idx + 1))
        done
    done
done

TOTAL=${#JOBS[@]}
echo "[$(date '+%F %T')] Total jobs: $TOTAL  (NUM_GPUS=$NUM_GPUS)"

# NUM_GPUS 워커가 round-robin으로 IDX 분배 (NUM_GPUS=1이면 직렬).
WORKER_PIDS=()
for ((G=0; G<NUM_GPUS; G++)); do
    (
        for JOB in "${JOBS[@]}"; do
            IFS='|' read -r IDX ENC SUITE TASK CFG SEED DATA_DIR <<< "$JOB"
            if [ $((IDX % NUM_GPUS)) -eq "$G" ]; then
                run_one "$IDX" "$G" "$ENC" "$SUITE" "$TASK" "$CFG" "$SEED" "$DATA_DIR"
            fi
        done
    ) &
    WORKER_PIDS+=($!)
done

wait "${WORKER_PIDS[@]}"
echo "[$(date '+%F %T')] All jobs complete"

echo "Aggregating to $OUT_ROOT/{per_run,per_task,summary}.csv ..."
python3 "$AGG_SCRIPT" --root "$OUT_ROOT"
