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

# ── NCCL 환경 ────────────────────────────────────────────────────────────────
# IB가 있으면 사용, 없으면 ethernet
export NCCL_DEBUG=INFO   # 첫 sanity는 verbose로, 안정 후 WARN으로 낮추기
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker}
# Master addr/port는 pretrain.py가 SLURM_JOB_NODELIST에서 추출

# ── srun으로 task 분산 실행 ──────────────────────────────────────────────────
# 각 task가 자기 LOCAL_RANK GPU 점유 → DDP init은 pretrain.py 내부에서 처리
# train_data=egodex, eval은 자동 (test split)
#
# CRITICAL: srun이 새 shell을 띄우면서 login shell PATH가 전파 안 됨.
# "python"으로만 쓰면 execve 실패 → $CONDA_PREFIX/bin/python 절대 경로 사용.
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
    --checkpoint-dir "$CKPT_DIR" \
    --eval-interval 5 \
    $EXTRA_ARGS

log "=== Pretrain job end ==="
