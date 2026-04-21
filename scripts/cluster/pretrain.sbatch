#!/bin/bash
# IBS 클러스터 EgoDex 사전학습 sbatch 스크립트 (2 노드 × 4 H100 DDP)
#
# pretrain.py 가 SLURM_PROCID / LOCAL_ID / NTASKS 를 보고 DDP 자동 초기화
# (torchrun 불필요). MODEL 에 따라 모델별 인자를 자동 분기.
#
# 본 학습 (50 epoch):
#   sbatch --partition=AIP_long --time=3-12:00:00 \
#          --export=ALL,MODEL=two-stream,EPOCHS=50,USE_SCRATCH=0 \
#          scripts/cluster/pretrain.sbatch
#   sbatch --partition=AIP_long --time=3-12:00:00 \
#          --export=ALL,MODEL=v-jepa,EPOCHS=50,USE_SCRATCH=0 \
#          scripts/cluster/pretrain.sbatch
#
# Sanity (2 노드 8 GPU, 1 epoch):
#   sbatch --nodes=2 --ntasks-per-node=4 --time=00:30:00 \
#          --export=MODEL=two-stream,EPOCHS=1 scripts/cluster/pretrain.sbatch
#
# 모니터링: `squeue --me`, `tail -f /proj/external_group/mrg/logs/pretrain_*.out`
# 청구 기록 (필수): docs/cluster_sessions.md 진행 중 표에 즉시 기록.

#SBATCH --job-name=aavrl_pretrain
#SBATCH --partition=AIP
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=23:30:00
#SBATCH --output=/proj/external_group/mrg/logs/pretrain_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/pretrain_%x_%j.err

# NOTE: --gres=gpu:4 대신 --gpus-per-task=1을 쓰면 NCCL의
# nvmlDeviceGetHandleByPciBusId()가 실패함 (CUDA_VISIBLE_DEVICES 제한 때문에
# NCCL이 GPU topology 탐색 시 non-visible PCI ID를 참조). --gres 사용 필수.

set -euo pipefail

# ── 경로 ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
DATA_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CHECKPOINT_ROOT="/proj/external_group/mrg/checkpoints"
LOG_DIR="/proj/external_group/mrg/logs"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"

mkdir -p "$CHECKPOINT_ROOT" "$LOG_DIR"

# ── 사용자 override 가능 변수 (sbatch --export=...) ──────────────────────────
MODEL="${MODEL:-two-stream}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
LR="${LR:-2e-4}"   # 8 GPU global batch 512 ~ sqrt(4) * 1e-4 (square-root scaling)
SPLITS="${SPLITS:-part1,part2,part3,part4,part5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# CHECKPOINT_SUFFIX: sanity 잡처럼 독립 체크포인트 디렉토리가 필요할 때 설정
# (기본값 비어있음 → 본 학습은 $CHECKPOINT_ROOT/${MODEL}/에 공유)
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-}"
CKPT_DIR="$CHECKPOINT_ROOT/${MODEL//-/_}${CHECKPOINT_SUFFIX:+_$CHECKPOINT_SUFFIX}"

# ── 분산 환경 정보 ───────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [pretrain] $*"; }

log "=== Pretrain job start ==="
log "Job: ${SLURM_JOB_ID:-N/A}"
log "Partition: ${SLURM_JOB_PARTITION:-N/A}"
log "Nodes: ${SLURM_JOB_NUM_NODES:-1} ($(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | tr '\n' ' '))"
log "Tasks per node: ${SLURM_NTASKS_PER_NODE:-1}, total tasks: ${SLURM_NTASKS:-1}"
log "GPUs per task: ${SLURM_GPUS_ON_NODE:-?}, CPUs per task: ${SLURM_CPUS_PER_TASK:-?}"
log "Model: $MODEL, epochs: $EPOCHS, per-GPU batch: $BATCH_SIZE_PER_GPU, lr: $LR"
log "Splits: $SPLITS"

# ── conda 환경 (각 task가 활성화) ────────────────────────────────────────────
module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
log "Conda env: $CONDA_PREFIX"
python --version

# ── Stage 4.5: scratch 가용성 확인 (GPU 노드에서만 의미 있음) ────────────────
# 모든 task가 출력하면 노이즈 → rank 0(SLURM_PROCID=0 또는 미설정)만
if [ "${SLURM_PROCID:-0}" = "0" ]; then
    log "── scratch availability check ──"
    if [ -d /scratch ]; then
        log "  /scratch exists. df:"
        df -h /scratch 2>&1 | sed 's/^/    /' | while read l; do log "$l"; done
        log "  ls /scratch:"
        ls -la /scratch 2>&1 | head -5 | sed 's/^/    /' | while read l; do log "$l"; done
    else
        log "  /scratch does NOT exist on this node"
    fi
    log "──────────────────────────────────"
fi

# ── NCCL + DDP 환경 ──────────────────────────────────────────────────────────
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker}
# MASTER_PORT: 잡마다 고유 포트로 충돌 방지 (동일 노드에 여러 DDP 잡 시)
export MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}

# ── Scratch stage-in (선택적) ────────────────────────────────────────────────
# USE_SCRATCH=1로 설정하면 /proj에서 /scratch로 데이터 복사 후 학습.
# GPU 학습은 random small file read → local NVMe가 GPFS보다 유리.
# 각 노드별로 독립 복사 필요 (node-local storage).
# 복사 완료 후 DATA_ROOT를 scratch 경로로 교체.
#
# scratch 용량: 7 TB (Stage 4.5 확인). EgoDex frames ~1.25 TB → 충분.
# 복사 속도: GPFS read 16 GB/s → scratch write 2 GB/s 병목 → ~10분.
#
# Usage: sbatch --export=ALL,...,USE_SCRATCH=1 scripts/cluster/pretrain.sbatch
USE_SCRATCH="${USE_SCRATCH:-0}"
SCRATCH_DIR="/scratch/tmp/$USER/$SLURM_JOB_ID"

if [ "$USE_SCRATCH" = "1" ]; then
    SCRATCH_DATA="$SCRATCH_DIR/egodex_frames"
    mkdir -p "$SCRATCH_DATA"

    # 모든 task 중 rank 0만 stage-in. 나머지는 대기.
    # 멀티노드에서는 각 노드의 rank 0 (SLURM_LOCALID=0)이 각각 복사.
    if [ "${SLURM_LOCALID:-0}" = "0" ]; then
        log "── Scratch stage-in start ──"
        log "  src: $DATA_ROOT → dst: $SCRATCH_DATA"
        STAGE_IN_START=$(date +%s)

        # rsync로 프레임만 복사 (raw mp4, zips는 불필요)
        for SPLIT in $(echo "$SPLITS" | tr ',' ' ') test; do
            SRC="$DATA_ROOT/$SPLIT"
            if [ -d "$SRC" ]; then
                log "  copying $SPLIT ..."
                cp -a "$SRC" "$SCRATCH_DATA/" 2>&1 || {
                    log "  WARNING: cp failed for $SPLIT, falling back to GPFS"
                    USE_SCRATCH=0
                    break
                }
            fi
        done

        STAGE_IN_END=$(date +%s)
        STAGE_IN_SEC=$((STAGE_IN_END - STAGE_IN_START))
        log "  Stage-in done: ${STAGE_IN_SEC}s ($(awk -v t=$STAGE_IN_SEC 'BEGIN{printf "%.1f", t/60}') min)"
        log "── Scratch stage-in complete ──"
    fi

    # 모든 rank가 stage-in 끝날 때까지 대기
    srun --ntasks-per-node=1 --nodes=$SLURM_JOB_NUM_NODES true 2>/dev/null || true
    sleep 5  # 파일시스템 sync 여유

    if [ "$USE_SCRATCH" = "1" ]; then
        DATA_ROOT="$SCRATCH_DATA"
        log "Using scratch data: $DATA_ROOT"
        log "── Scratch 이점 분석 ──"
        log "  GPFS baseline throughput (BF16 sanity, batch 4): 62.2 samples/sec (1 GPU)"
        log "  본 학습 1 epoch 완료 후 samples_per_sec를 확인하여 GPFS 대비 비교."
        log "  TensorBoard perf/samples_per_sec 또는 로그의 [Time] Throughput 참조."
        log "────────────────────────"
    fi

    # 체크포인트는 /proj(CKPT_DIR)에 직접 저장 — scratch 유실 위험 없음.
    # scratch는 데이터 읽기 전용으로만 사용.
fi

# ── Model-specific arguments ────────────────────────────────────────────────
# Two-Stream: M/P mask ratios + max-gap 60 (our long-range temporal choice)
# V-JEPA:     max-gap 16 (matches original 16-frame clip receptive field);
#             mask/dual-mask config is set inside VJEPAModel defaults.
if [ "$MODEL" = "two-stream" ]; then
    MODEL_ARGS=(
        --depth 12 --num-stages 2
        --mask-ratio 0.3 --mask-ratio-p 0.5
        --max-gap 60 --sample-dist triangular --sample-center 30
    )
elif [ "$MODEL" = "two-stream-v8" ]; then
    # v8: L_M + λ·L_P [+ α·L_var] (EMA teacher on P-stream, collapse 방어 포함)
    # 2차 재설정 (1차 full run ep8 probing R² = -0.22, collapse 확정):
    # [a] L_P BYOL form — L_P = 2·(1 - cos_sim) ∈ [0, 4], scale 문제 해소
    # [b] λ_max 0.05 — λ·L_P ≈ 2~5·L_M (L_P 약간 우위, L_M anchor 유지)
    # [c] cls_m grad ratio 0.3 — 완전 detach 대신 partial flow 허용
    #     (L_M task 자체가 trivial하므로 L_P pull 없으면 M stream이 low-level shortcut에 수렴)
    # [d] mask_ratio_m 0.3 → 0.5 — M task 난이도 상향으로 L_M 지나친 저점 수렴 방지
    # 참고: docs/architecture/v8_siamsimmae.md, 조정 여지 + 실행 기록 섹션
    MODEL_ARGS=(
        --depth 12 --num-stages 2
        --use-ape --rotation-aug
        --v8-mode
        --v8-lambda-max 0.05
        --v8-lambda-warmup-epochs 10
        --v8-ema-tau-base 0.996
        --v8-pred-head-ratio 2.0
        --v8-alpha-var 0.1
        --v8-var-target 1.0
        --v8-cls-m-grad-ratio 0.3
        --mask-ratio 0.5 --mask-ratio-p 0.5
        --max-gap 60 --sample-dist triangular --sample-center 30
    )
    # v8 체크포인트는 별도 디렉토리
    CKPT_DIR="$CHECKPOINT_ROOT/two_stream_v8${CHECKPOINT_SUFFIX:+_$CHECKPOINT_SUFFIX}"
elif [ "$MODEL" = "two-stream-v9" ]; then
    # v9: v4/v6 구조 + P decoder target = frame_t (MAE 방식) + P mask 0.75
    # 설계 근거:
    # - v8 L_P + EMA teacher 접근 폐기 (static salience collapse 확정)
    # - residual target 시도 → P encoder collapse 관찰 (cos_intra_p=1.000, std_p=0.028)
    #   decoder가 "0 출력" trivial minimum으로 빠져 encoder 학습 정지
    # - MAE 방식(frame_t 복원) 회귀: P가 semantic 학습을 강제로 하게 됨 (MAE 검증된 방식)
    # - Motion 활용은 M decoder에서 CLS exchange로 P semantic 활용 (P→M 방향)
    # - CLS exchange 양방향 유지: M stream이 L_M 위해 P semantic 적극 활용, P는 frame_t
    #   복원만 책임지므로 M 정보 선택적 활용
    # - P mask 0.75 (MAE 표준), M mask 0.3 유지
    MODEL_ARGS=(
        --depth 12 --num-stages 2
        --use-ape --rotation-aug
        --mask-ratio 0.3 --mask-ratio-p 0.75
        --v9-p-target current --v9-loss-weight-p 1.0
        --max-gap 60 --sample-dist triangular --sample-center 30
    )
    CKPT_DIR="$CHECKPOINT_ROOT/two_stream_v9${CHECKPOINT_SUFFIX:+_$CHECKPOINT_SUFFIX}"
elif [ "$MODEL" = "v-jepa" ]; then
    MODEL_ARGS=(
        --depth 12
        --max-gap 16 --sample-dist triangular --sample-center 8
    )
elif [ "$MODEL" = "videomae" ]; then
    # mask-ratio 0.5: 2-frame에 공식 0.75 과도 (V-JEPA 실패 사례 참조)
    MODEL_ARGS=(
        --depth 12
        --mask-ratio 0.5
        --max-gap 60 --sample-dist triangular --sample-center 30
    )
else
    echo "Unknown MODEL=$MODEL" >&2
    exit 1
fi

log "Model args: ${MODEL_ARGS[*]}"

# ── MODEL alias 해석 ────────────────────────────────────────────────────────
# scripts/pretrain.py의 --model은 ['two-stream','v-jepa','videomae']만 받음.
# two-stream-v8은 --model two-stream + --v8-mode 조합이므로 여기서 번역.
if [ "$MODEL" = "two-stream-v8" ] || [ "$MODEL" = "two-stream-v9" ]; then
    MODEL_NAME_FOR_PY="two-stream"
else
    MODEL_NAME_FOR_PY="$MODEL"
fi

# ── srun으로 task 분산 실행 ──────────────────────────────────────────────────
# CRITICAL: srun이 새 shell을 띄우면서 PATH 전파 안 됨 → 절대 경로 사용.
srun --kill-on-bad-exit=1 --export=ALL \
    "$CONDA_PREFIX/bin/python" -u "$PROJECT_ROOT/scripts/pretrain.py" \
    --model "$MODEL_NAME_FOR_PY" \
    --train-data egodex \
    --egodex-root "$DATA_ROOT" \
    --egodex-splits "$SPLITS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE_PER_GPU" \
    --lr "$LR" \
    "${MODEL_ARGS[@]}" \
    --checkpoint-dir "$CKPT_DIR" \
    --eval-interval 5 \
    $EXTRA_ARGS

log "=== Pretrain job end ==="
