#!/bin/bash
# Region-wise M-recon patch error (scripts/eval/mrecon_region_error.py) — LIBERO re-render + seg.
# Render = osmesa (EGL fails on MIG). Sample/mask seed are fixed by script defaults → same for every ckpt.
#
# 사용:
#   sbatch --partition=<p> --export=ALL,CKPT=/proj/.../latest.pt scripts/cluster/mrecon_region_error.sbatch
#   # 추가 인자: EXTRA="--task-ids 0 1 --demos 2"
#
#SBATCH --job-name=mrecon_region
#SBATCH --partition=mig-1g.10gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/external_group/mrg/logs/mrecon_region_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/mrecon_region_%j.err

set -euo pipefail
PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CKPT="${CKPT:?CKPT 필요}"
EXTRA="${EXTRA:-}"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /proj/external_group/mrg/conda_envs/aavrl-train

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/proj/external_group/mrg/lib/osmesa:${LD_LIBRARY_PATH:-}
cd "$PROJECT_ROOT"
"$CONDA_PREFIX/bin/python" -u scripts/eval/mrecon_region_error.py --checkpoint "$CKPT" $EXTRA
