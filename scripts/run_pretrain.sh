#!/usr/bin/env bash
# =============================================================================
# BridgeV2 Pre-training + EgoDex Action Probing Pipeline
#
# Phase 0: 이전 테스트 체크포인트 정리 (~28.5 GB)
# Phase 1: BridgeV2 사전학습 (two-stream, single-stream, videomae) ~24h
# Phase 2: EgoDex action probing (5종 인코더) ~4h
# Phase 3: 결과 요약
#
# Usage:
#   bash scripts/run_pretrain.sh           # 전체 파이프라인
#   bash scripts/run_pretrain.sh --skip-cleanup   # 정리 건너뛰기
#   bash scripts/run_pretrain.sh --phase 2        # 특정 Phase부터 시작
# =============================================================================
set -euo pipefail

WORKSPACE="/home/etri/action-agnostic-visual-rl"
CKPT_DIR="$WORKSPACE/data/checkpoints"
PRETRAIN_DIR="$CKPT_DIR/pretrain"
PROBE_DIR="$WORKSPACE/data/probing_results"
LOG_DIR="$WORKSPACE/data/logs/pretrain_$(date +%Y%m%d_%H%M%S)"

DOCKER_IMAGE="libero-env:latest"
DOCKER_COMMON="docker run --rm --gpus all --shm-size=32gb --ipc=host \
  -v $WORKSPACE:/workspace \
  -e PYTHONPATH=/workspace \
  $DOCKER_IMAGE"

SKIP_CLEANUP=false
START_PHASE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-cleanup) SKIP_CLEANUP=true; shift ;;
        --phase) START_PHASE=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"
}

# Helper: find latest timestamped checkpoint directory
find_latest_checkpoint() {
    local base_dir="$1"
    # Find the most recently created timestamped subdirectory
    ls -1d "$base_dir"/2* 2>/dev/null | sort | tail -1
}

log "=========================================="
log "BridgeV2 Pre-training Pipeline"
log "=========================================="
log "Workspace: $WORKSPACE"
log "Log dir:   $LOG_DIR"
log "Start phase: $START_PHASE"
log ""

# =============================================================================
# Phase 0: Checkpoint Cleanup
# =============================================================================
if [[ $START_PHASE -le 0 ]] && [[ $SKIP_CLEANUP == false ]]; then
    log "Phase 0: Checkpoint cleanup"
    log "------------------------------------------"

    CLEANUP_TARGETS=(
        "$CKPT_DIR/two_stream"
        "$CKPT_DIR/single_stream"
        "$CKPT_DIR/videomae"
        "$PRETRAIN_DIR/two_stream"
    )

    # Also clean LIBERO test checkpoints (glob pattern)
    for d in "$CKPT_DIR"/libero_two-stream_*/ "$CKPT_DIR"/libero_single-stream_*/ "$CKPT_DIR"/libero_videomae_*/; do
        [[ -d "$d" ]] && CLEANUP_TARGETS+=("${d%/}")
    done

    # Clean old probing results
    [[ -d "$PROBE_DIR" ]] && CLEANUP_TARGETS+=("$PROBE_DIR")

    TOTAL_SIZE=0
    for target in "${CLEANUP_TARGETS[@]}"; do
        if [[ -e "$target" ]]; then
            size=$(du -sm "$target" 2>/dev/null | cut -f1)
            TOTAL_SIZE=$((TOTAL_SIZE + size))
            log "  DELETE: $target (${size} MB)"
        fi
    done

    if [[ $TOTAL_SIZE -gt 0 ]]; then
        log "  Total: ${TOTAL_SIZE} MB to free"
        for target in "${CLEANUP_TARGETS[@]}"; do
            if [[ -e "$target" ]]; then
                rm -rf "$target"
            fi
        done
        log "  Cleanup done."
    else
        log "  Nothing to clean."
    fi
    log ""
fi

# =============================================================================
# Phase 1: BridgeV2 Pre-training (3 models, sequential)
# =============================================================================
if [[ $START_PHASE -le 1 ]]; then
    log "Phase 1: BridgeV2 Pre-training"
    log "=========================================="

    MODELS=("two-stream" "single-stream" "videomae")
    # Underscore versions for directory names
    MODEL_DIRS=("two_stream" "single_stream" "videomae")

    for i in "${!MODELS[@]}"; do
        model="${MODELS[$i]}"
        model_dir="${MODEL_DIRS[$i]}"
        ckpt_base="$PRETRAIN_DIR/$model_dir"
        log_file="$LOG_DIR/train_${model_dir}.log"

        log ""
        log "Phase 1.$((i+1)): Training $model"
        log "  Checkpoint dir: $ckpt_base"
        log "  Log: $log_file"
        log "  Started: $(date)"

        $DOCKER_COMMON python scripts/train_long.py \
            --model "$model" \
            --epochs 2 \
            --batch-size 32 \
            --train-data bridge \
            --eval-interval 2 \
            --checkpoint-dir "/workspace/data/checkpoints/pretrain/$model_dir" \
            2>&1 | tee "$log_file"

        # Verify checkpoint was created
        latest_dir=$(find_latest_checkpoint "$ckpt_base")
        if [[ -z "$latest_dir" ]] || [[ ! -f "$latest_dir/best_model.pt" ]]; then
            log "ERROR: best_model.pt not found for $model in $ckpt_base"
            exit 1
        fi
        log "  Completed: $(date)"
        log "  Checkpoint: $latest_dir/best_model.pt"
    done

    log ""
    log "Phase 1 complete. All 3 models trained."
    log ""
fi

# =============================================================================
# Phase 2: Action Probing (5 encoders)
# =============================================================================
if [[ $START_PHASE -le 2 ]]; then
    log "Phase 2: EgoDex Action Probing"
    log "=========================================="

    # Check libero-eval container is running
    if ! docker ps --format '{{.Names}}' | grep -q '^libero-eval$'; then
        log "WARNING: libero-eval container not running. Starting with docker run..."
        # Fall back to docker run with single GPU
        PROBE_CMD="docker run --rm --gpus '\"device=0\"' --shm-size=16gb --ipc=host \
            -v $WORKSPACE:/workspace \
            -e PYTHONPATH=/workspace \
            $DOCKER_IMAGE"
    else
        PROBE_CMD="docker exec libero-eval"
    fi

    # --- Custom encoders (need checkpoint) ---
    CUSTOM_ENCODERS=("two-stream" "single-stream" "videomae")
    CUSTOM_DIRS=("two_stream" "single_stream" "videomae")

    for i in "${!CUSTOM_ENCODERS[@]}"; do
        enc="${CUSTOM_ENCODERS[$i]}"
        enc_dir="${CUSTOM_DIRS[$i]}"
        log_file="$LOG_DIR/probe_${enc_dir}.log"

        # Find the latest checkpoint
        latest_dir=$(find_latest_checkpoint "$PRETRAIN_DIR/$enc_dir")
        if [[ -z "$latest_dir" ]] || [[ ! -f "$latest_dir/best_model.pt" ]]; then
            log "ERROR: No checkpoint found for $enc at $PRETRAIN_DIR/$enc_dir"
            exit 1
        fi

        # Convert host path to container path
        ckpt_container="/workspace/data/checkpoints/pretrain/$enc_dir/$(basename "$latest_dir")/best_model.pt"

        log ""
        log "Phase 2: Probing $enc"
        log "  Checkpoint: $ckpt_container"
        log "  Log: $log_file"

        $PROBE_CMD python scripts/probe_action.py \
            --encoder "$enc" \
            --checkpoint "$ckpt_container" \
            --epochs 20 \
            --batch-size 64 \
            --output-dir /workspace/data/probing_results \
            2>&1 | tee "$log_file"

        log "  Done: $enc"
    done

    # --- Pretrained baselines ---
    BASELINES=("clip" "dinov2")

    for enc in "${BASELINES[@]}"; do
        log_file="$LOG_DIR/probe_${enc}.log"

        log ""
        log "Phase 2: Probing $enc (pretrained baseline)"
        log "  Log: $log_file"

        $PROBE_CMD python scripts/probe_action.py \
            --encoder "$enc" \
            --epochs 20 \
            --batch-size 64 \
            --output-dir /workspace/data/probing_results \
            2>&1 | tee "$log_file"

        log "  Done: $enc"
    done

    log ""
    log "Phase 2 complete. All 5 encoders probed."
    log ""
fi

# =============================================================================
# Phase 3: Results Summary
# =============================================================================
if [[ $START_PHASE -le 3 ]]; then
    log "Phase 3: Results Summary"
    log "=========================================="

    if [[ ! -d "$PROBE_DIR" ]] || [[ -z "$(ls "$PROBE_DIR"/probe_*.json 2>/dev/null)" ]]; then
        log "No probing results found in $PROBE_DIR"
        exit 1
    fi

    # Print results table
    echo ""
    echo "| Encoder       | Probe  | R²     | MSE      | Cos Sim | PASS? |"
    echo "|---------------|--------|--------|----------|---------|-------|"

    for f in "$PROBE_DIR"/probe_*.json; do
        enc=$(python3 -c "import json; d=json.load(open('$f')); print(d['encoder'])")
        probe=$(python3 -c "import json; d=json.load(open('$f')); print(d['probe'])")
        r2=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['r2']:.4f}\")")
        mse=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['mse']:.6f}\")")
        cos=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['cosine_sim']:.4f}\")")
        pass_fail=$(python3 -c "import json; d=json.load(open('$f')); print('PASS' if d['r2'] > 0.7 else 'FAIL')")
        printf "| %-13s | %-6s | %s | %s | %s | %-5s |\n" "$enc" "$probe" "$r2" "$mse" "$cos" "$pass_fail"
    done

    echo ""
    log "Pipeline complete!"
    log "Logs: $LOG_DIR"
    log "Results: $PROBE_DIR"
fi
