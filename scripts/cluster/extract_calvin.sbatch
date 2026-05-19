#!/bin/bash
# CALVIN task_ABCD_D.zip 압축 해제 (§C10 dataset prep).
#
# task_ABCD_D.zip은 ~700 GB 압축, 풀면 ~1-2 TB 수준 추정. 수많은 작은 .npz
# 파일이라 메타데이터 부담 큼 → CPU 큐에서 처리, login node 안 씀.
#
# Usage:
#   sbatch scripts/cluster/extract_calvin.sbatch
#
# 종료 후 검증:
#   ls /proj/external_group/mrg/datasets/calvin/task_ABCD_D/ → training/, validation/ 등 존재 확인
#   ls /proj/external_group/mrg/datasets/calvin/task_ABCD_D/training/ | head → episode_XXXXXXX.npz 패턴 확인

#SBATCH --job-name=extract_calvin
#SBATCH --partition=long_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/proj/external_group/mrg/logs/extract_calvin_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/extract_calvin_%j.err

set -euo pipefail

CALVIN_DIR=/proj/external_group/mrg/datasets/calvin
ZIPFILE=task_ABCD_D.zip

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [extract_calvin] $*"; }

cd "$CALVIN_DIR"

log "=== CALVIN unzip 시작 ==="
log "ZIP file: $(ls -lh $ZIPFILE | awk '{print $5}')"

# unzip은 single-thread지만 sbatch 자원이 보장됨
log "Running unzip (single-thread, 수 시간 소요 예상)..."
time unzip -q $ZIPFILE

log "=== 압축 해제 완료. directory 구조 확인 ==="
ls -la $CALVIN_DIR/ | head -20
echo "---"
log "task_ABCD_D 내부:"
ls -la $CALVIN_DIR/task_ABCD_D/ 2>/dev/null | head -20
echo "---"
log "training/ 처음 5개 파일:"
ls $CALVIN_DIR/task_ABCD_D/training/ 2>/dev/null | head -5
echo "---"
log "validation/ 처음 5개 파일:"
ls $CALVIN_DIR/task_ABCD_D/validation/ 2>/dev/null | head -5
echo "---"
log "ep_start_end_ids.npy 존재?"
find $CALVIN_DIR/task_ABCD_D -name "ep_start_end_ids.npy" 2>/dev/null | head -5

log "=== done. 압축 해제 검증 후 zip 삭제 권장 (700GB 회수) ==="
