#!/bin/bash
# IBS 클러스터 EgoDex 사전학습 sbatch 스크립트
#
# 분산 학습 (DDP) 설정:
#   --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 → 총 8 GPU
#   각 task가 1 GPU 점유. srun이 각 노드에서 task 4개씩 spawn → 총 8 task
#   pretrain.py가 SLURM_PROCID/LOCAL_ID/NTASKS를 보고 DDP 자동 초기화 (torchrun 불필요).
#
# 청구 (월 누적 ceil, GPU·일 단위):
#   2 노드 × 4 H100 = 8 GPU
#   1시간당 8 GPU·h = 0.33 일 → 월말 합산 후 ceil
#   --time=23:30:00 = 23.5h × 8 = 188 GPU·h ≈ 7.83 일 → 8일 청구 (488,000원)
#
# Usage:
#   # Sanity (1 노드 1 GPU 짧게 — DDP가 1 GPU에서도 작동)
#   sbatch --nodes=1 --ntasks-per-node=1 --time=00:30:00 \
#          --export=MODEL=two-stream,EPOCHS=1,EXTRA_ARGS=--max-videos=10 \
#          scripts/cluster/pretrain.sbatch
#
#   # Sanity (1 노드 4 GPU)
#   sbatch --nodes=1 --ntasks-per-node=4 --time=00:30:00 \
#          --export=MODEL=two-stream,EPOCHS=1 scripts/cluster/pretrain.sbatch
#
#   # Sanity (2 노드 8 GPU NCCL 검증)
#   sbatch --nodes=2 --ntasks-per-node=4 --time=00:30:00 \
#          --export=MODEL=two-stream,EPOCHS=1 scripts/cluster/pretrain.sbatch
#
#   # 본 학습 (2 노드 8 GPU 50 epoch)
#   sbatch --export=MODEL=two-stream,EPOCHS=50 scripts/cluster/pretrain.sbatch
#   sbatch --export=MODEL=videomae,EPOCHS=50   scripts/cluster/pretrain.sbatch
#
# 모니터링:
#   squeue --me
#   tail -f /proj/external_group/mrg/logs/pretrain_*.out
#
# 청구 기록 (필수): docs/cluster_sessions.md "진행 중" 표에 즉시 기록.

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
    SCRATCH_CKPT="$SCRATCH_DIR/checkpoints"
    mkdir -p "$SCRATCH_DATA" "$SCRATCH_CKPT"

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

    # 체크포인트 stage-out: 잡 종료 시 scratch → /proj 복사
    trap '
        if [ "${SLURM_LOCALID:-0}" = "0" ] && [ -d "$SCRATCH_CKPT" ]; then
            echo "[$(date "+%Y-%m-%d %H:%M:%S")] [pretrain] Stage-out: $SCRATCH_CKPT → $CKPT_DIR"
            rsync -a "$SCRATCH_CKPT/" "$CKPT_DIR/" 2>&1 || true
        fi
    ' EXIT
fi

# ── srun으로 task 분산 실행 ──────────────────────────────────────────────────
# CRITICAL: srun이 새 shell을 띄우면서 PATH 전파 안 됨 → 절대 경로 사용.
srun --kill-on-bad-exit=1 --export=ALL \
    "$CONDA_PREFIX/bin/python" -u "$PROJECT_ROOT/scripts/pretrain.py" \
    --model "$MODEL" \
    --train-data egodex \
    --egodex-root "$DATA_ROOT" \
    --egodex-splits "$SPLITS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE_PER_GPU" \
    --lr "$LR" \
    --depth 12 --num-stages 2 \
    --mask-ratio 0.3 --mask-ratio-p 0.5 \
    --max-gap 60 --sample-dist triangular --sample-center 30 \
    --checkpoint-dir "$([ "$USE_SCRATCH" = "1" ] && echo "$SCRATCH_CKPT" || echo "$CKPT_DIR")" \
    --eval-interval 5 \
    $EXTRA_ARGS

log "=== Pretrain job end ==="
