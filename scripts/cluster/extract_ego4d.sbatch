#!/bin/bash
# IBS 클러스터 Ego4D 프레임 추출 sbatch
#
# 입력: /proj/external_group/mrg/datasets/ego4d/v2/full_scale/<uuid>.mp4 (9,823 mp4, 7.2 TB)
# 출력: /proj/external_group/mrg/datasets/ego4d/frames/<uuid>/frame_*.jpg (256², 30 fps 그대로)
#
# 핵심 설계:
#   - 144 worker 병렬 (multiprocessing.Pool, scripts/data/extract_ego4d_frames.py)
#   - cv2.VideoCapture full decode (subsample 없음)
#   - 이미 추출된 디렉토리 skip → resume 가능 (timeout 시 재제출만)
#
# Usage:
#   # Sanity (10 videos, ~5분)
#   sbatch --time=00:30:00 --export=LIMIT=10,SUFFIX=sanity scripts/cluster/extract_ego4d.sbatch
#
#   # Full (9823 videos)
#   sbatch scripts/cluster/extract_ego4d.sbatch
#
# 모니터링:
#   squeue --me
#   tail -f /proj/external_group/mrg/logs/extract_ego4d_<JobID>.out

#SBATCH --job-name=ego4d_extract
#SBATCH --partition=normal_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=12:00:00
#SBATCH --output=/proj/external_group/mrg/logs/extract_ego4d_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/extract_ego4d_%j.err

set -euo pipefail

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATA_ROOT="/proj/external_group/mrg/datasets/ego4d"
VIDEO_DIR="$DATA_ROOT/v2/full_scale"
FRAMES_DIR="$DATA_ROOT/frames${SUFFIX:+_$SUFFIX}"
PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-extract"

mkdir -p "$FRAMES_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

JOB_START=$(date +%s)

NUM_WORKERS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"

log "=== Ego4D extraction start ==="
log "Node: $(hostname)"
log "Allocated CPUs: $NUM_WORKERS"
log "Job: ${SLURM_JOB_ID:-N/A}"
log "Video dir : $VIDEO_DIR"
log "Frames dir: $FRAMES_DIR"
log "Limit     : ${LIMIT:-(none, full)}"

# ── conda 환경 ───────────────────────────────────────────────────────────────
module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
log "Conda env: $CONDA_PREFIX"
python --version

# ── 추출 ─────────────────────────────────────────────────────────────────────
EXTRA_ARGS=()
if [ -n "${LIMIT:-}" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

time python "$PROJECT_ROOT/scripts/data/extract_ego4d_frames.py" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$FRAMES_DIR" \
    --num-workers "$NUM_WORKERS" \
    --yes \
    "${EXTRA_ARGS[@]}"

# ── 검증 ─────────────────────────────────────────────────────────────────────
N_VIDEOS=$(find "$VIDEO_DIR" -maxdepth 1 -name "*.mp4" | wc -l)
N_FRAME_DIRS=$(find "$FRAMES_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
TOTAL_SIZE=$(du -sh "$FRAMES_DIR" | cut -f1)

JOB_END=$(date +%s)
ELAPSED=$((JOB_END - JOB_START))
log "=== Extraction summary ==="
log "Videos found     : $N_VIDEOS"
log "Frame directories: $N_FRAME_DIRS"
log "Output size      : $TOTAL_SIZE"
log "Elapsed          : $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
