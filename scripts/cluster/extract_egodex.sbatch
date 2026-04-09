#!/bin/bash
# IBS 클러스터 EgoDex 프레임 추출 sbatch 스크립트
#
# 동작:
#   1. zip 파일이 raw/$PART/로 압축 해제됨 (이미 풀려 있으면 스킵)
#   2. scripts/data/extract_frames.py로 프레임 추출 (144 worker 병렬)
#   3. 추출 결과 검증
#
# 파티션 선택 전략:
#   모든 CPU 노드가 동일 하드웨어 (Intel Xeon Platinum 8360Y, 36C×2 socket = 144 logical CPU).
#   --partition=core_s,normal_cpu 멀티 지정 → Slurm이 먼저 가용한 쪽으로 자동 배치.
#     - core_s: PriorityJobFactor 200 (압도적), MaxTime 2h
#     - normal_cpu: PriorityJobFactor 20, MaxTime 3 days, exclusive
#   --time=02:00:00은 두 파티션 모두 허용. 추출이 2h 안에 끝나야 함.
#   2h를 넘길 가능성이 보이면 normal_cpu 단독으로 시간 늘려서 재제출.
#
# 저장소 주의:
#   - olaf-c 노드에는 /scratch가 없음 (관리자 확인). GPFS(/proj) 직접 사용.
#   - GPFS write 16 GB/s로 충분히 빠름. scratch 우회 불필요.
#
# Usage:
#   # 단일 part (테스트용 — 첫 측정 권장)
#   sbatch --export=PART=part1 scripts/cluster/extract_egodex.sbatch
#
#   # 여러 part 동시 추출 (sbatch array, 노드 1개씩 병렬)
#   sbatch --array=1-5 scripts/cluster/extract_egodex.sbatch
#
#   # 일부만 (예: 다운로드 끝난 1, 2번만)
#   sbatch --array=1,2 scripts/cluster/extract_egodex.sbatch
#
#   # 2h를 초과해야 하는 경우 normal_cpu 단독 + 시간 연장
#   sbatch --partition=normal_cpu --time=04:00:00 --array=1-5 scripts/cluster/extract_egodex.sbatch
#
# 제출 후 모니터링:
#   squeue --me
#   tail -f /proj/external_group/mrg/logs/extract_egodex_*.out

#SBATCH --job-name=egodex_extract
#SBATCH --partition=core_s,normal_cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/proj/external_group/mrg/logs/extract_egodex_%A_%a.out
#SBATCH --error=/proj/external_group/mrg/logs/extract_egodex_%A_%a.err

set -euo pipefail

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATA_ROOT="/proj/external_group/mrg/datasets/egodex"
ZIP_DIR="$DATA_ROOT/zips"
RAW_DIR="$DATA_ROOT/raw"
FRAMES_DIR="$DATA_ROOT/frames"
PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-extract"

# ── PART 결정 (array index 또는 --export=PART) ────────────────────────────────
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    PART="part${SLURM_ARRAY_TASK_ID}"
fi
PART="${PART:?Error: PART not set. Use --export=PART=partN or --array=1-5}"

ZIP_FILE="$ZIP_DIR/$PART.zip"
RAW_PART_DIR="$RAW_DIR/$PART"
FRAMES_PART_DIR="$FRAMES_DIR/$PART"

mkdir -p "$RAW_DIR" "$FRAMES_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$PART] $*"; }

JOB_START_EPOCH=$(date +%s)

log "=== Extraction job start ==="
log "Node: $(hostname), CPUs: ${SLURM_CPUS_ON_NODE:-unknown}"
log "Partition: ${SLURM_JOB_PARTITION:-N/A}"
log "Job: ${SLURM_JOB_ID:-N/A} (array task ${SLURM_ARRAY_TASK_ID:-N/A})"
log "Time limit: ${SLURM_JOB_END_TIME:-N/A}"

# ── conda 환경 활성화 ─────────────────────────────────────────────────────────
module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
log "Activated conda env: $CONDA_PREFIX"
python --version

# ── Step 1: unzip ────────────────────────────────────────────────────────────
if [ -d "$RAW_PART_DIR" ] && [ -n "$(ls -A "$RAW_PART_DIR" 2>/dev/null)" ]; then
    log "Raw data already exists at $RAW_PART_DIR, skipping unzip"
else
    if [ ! -f "$ZIP_FILE" ]; then
        log "ERROR: zip file not found: $ZIP_FILE"
        log "Run scripts/cluster/download_egodex.sh $PART first"
        exit 1
    fi
    log "Unzipping $ZIP_FILE → $RAW_DIR/"
    time unzip -q "$ZIP_FILE" -d "$RAW_DIR/"
    log "Unzip complete: $(du -sh "$RAW_PART_DIR" | cut -f1)"
fi

# ── Step 2: 프레임 추출 ───────────────────────────────────────────────────────
log "Extracting frames → $FRAMES_PART_DIR (workers=${SLURM_CPUS_ON_NODE})"
mkdir -p "$FRAMES_PART_DIR"

time python "$PROJECT_ROOT/scripts/data/extract_frames.py" \
    --egodex-root "$RAW_DIR" \
    --split "$PART" \
    --output-dir "$FRAMES_PART_DIR" \
    --num-workers "$SLURM_CPUS_ON_NODE" \
    --yes

# ── Step 3: 검증 ──────────────────────────────────────────────────────────────
N_VIDEOS=$(find "$RAW_PART_DIR" -name "*.mp4" | wc -l)
N_FRAME_DIRS=$(find "$FRAMES_PART_DIR" -mindepth 2 -maxdepth 2 -type d | wc -l)
TOTAL_SIZE=$(du -sh "$FRAMES_PART_DIR" | cut -f1)

JOB_END_EPOCH=$(date +%s)
ELAPSED=$((JOB_END_EPOCH - JOB_START_EPOCH))
ELAPSED_HMS=$(printf '%dh %dm %ds' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))

log "=== Extraction complete ==="
log "Videos found:        $N_VIDEOS"
log "Frame directories:   $N_FRAME_DIRS"
log "Frames total size:   $TOTAL_SIZE"
log "Total elapsed:       $ELAPSED_HMS ($ELAPSED sec)"
log "Throughput:          $(awk -v v=$N_VIDEOS -v t=$ELAPSED 'BEGIN{if(t>0)printf "%.2f videos/sec", v/t}')"

if [ "$N_VIDEOS" -ne "$N_FRAME_DIRS" ]; then
    log "WARNING: video count != frame dir count (some extractions may have failed)"
    exit 2
fi

log "All videos extracted successfully"
