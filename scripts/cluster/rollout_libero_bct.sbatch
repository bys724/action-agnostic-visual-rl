#!/bin/bash
# LIBERO BC-T closed-loop rollout — 클러스터판 (로컬 = scripts/local/run_libero_rollouts.sh).
# 렌더 기본 = osmesa(CPU) + mig-3g.40gb (정책 forward만 GPU). 10-03 스모크 40843414: ~30 s/에피소드.
# MIG에서 EGL은 벤더 로드 실패(MESA-LOADER nouveau, SFA 39514248 선례) → EGL은 normal(V100)에서만.
# ⚠️ 비교할 팔끼리는 같은 RENDER로 돌릴 것 (렌더러가 다르면 픽셀이 달라짐 — SFA 측정 함정).
#
# 사용:
#   sbatch --export=ALL,CKPT=/proj/.../libero_bct/<run>/best.pt,TASK_SUITE=libero_object \
#          scripts/cluster/rollout_libero_bct.sbatch
#   # 스모크: TASK_IDS="0" NUM_TRIALS=2
#
#SBATCH --job-name=rollout_bct
#SBATCH --partition=mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/external_group/mrg/logs/rollout_bct_%x_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/rollout_bct_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"

# CKPT 또는 CKPT_SUFFIX(학습 잡 SUFFIX — afterok 체인에서 타임스탬프 디렉터리를 실행 시점에 해석)
if [[ -z "${CKPT:-}" && -n "${CKPT_SUFFIX:-}" ]]; then
    CKPT=$(ls -d /proj/external_group/mrg/checkpoints/libero_bct/*_"$CKPT_SUFFIX"/best.pt | tail -1)
fi
CKPT="${CKPT:?CKPT 또는 CKPT_SUFFIX 필요 (BC-T best.pt)}"
TASK_SUITE="${TASK_SUITE:-libero_object}"
NUM_TRIALS="${NUM_TRIALS:-50}"        # eval_protocols §6: 50 trial × 10 task = 500 ep
TASK_IDS="${TASK_IDS:-}"              # 스모크용 (e.g., "0")
ROLL_SEED="${ROLL_SEED:-7}"           # eval_protocols §6: rollout RNG 고정 seed=7
RENDER="${RENDER:-osmesa}"
OUT_BASE="${OUT_BASE:-/proj/external_group/mrg/results/libero_rollouts}"

STEM=$(basename "$(dirname "$CKPT")")
ABLATE_EXTRA="${ABLATE_EXTRA:-}"      # E3 진단: 비현재 스트림(과거 P / M) 0
TAG="${STEM}_t${NUM_TRIALS}${TASK_IDS:+_tasks${TASK_IDS// /-}}${ABLATE_EXTRA:+_ablate}"

log() { echo "[$(date '+%F %T')] [rollout_bct] $*"; }
log "ckpt=$CKPT suite=$TASK_SUITE trials=$NUM_TRIALS tasks=${TASK_IDS:-all} render=$RENDER"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=$RENDER PYOPENGL_PLATFORM=$RENDER
[ "$RENDER" = "osmesa" ] && export LD_LIBRARY_PATH=/proj/external_group/mrg/lib/osmesa:${LD_LIBRARY_PATH:-}

cd "$PROJECT_ROOT"
ARGS=(--checkpoint "$CKPT" --task-suite "$TASK_SUITE" --num-trials "$NUM_TRIALS"
      --seed "$ROLL_SEED" --quiet
      --output-dir "$OUT_BASE/$TAG" --video-dir "$OUT_BASE/$TAG/videos")
[ -n "$TASK_IDS" ] && ARGS+=(--task-ids $TASK_IDS)
[ -n "$ABLATE_EXTRA" ] && ARGS+=(--ablate-extra)

"$CONDA_PREFIX/bin/python" -u src/eval_libero.py "${ARGS[@]}"
log "done → $OUT_BASE/$TAG"
