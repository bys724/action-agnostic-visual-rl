#!/bin/bash
# Layer-1 observer Fuse (scripts/eval/observer_fuse.py) — frozen CoMP encoder, LIBERO-object.
# Frozen token cache (~11GB fp16) lives on GPU → mig-3g.40gb is enough.
#
# 사용:
#   sbatch --export=ALL,CKPT=/proj/.../checkpoint_epoch0010.pt,TAG=c1dn scripts/cluster/observer_fuse.sbatch
#   # smoke: EXTRA="--max-demos 8 --steps 200"
#
#SBATCH --job-name=observer_fuse
#SBATCH --partition=mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/proj/external_group/mrg/logs/observer_fuse_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/observer_fuse_%j.err

set -euo pipefail
PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CKPT="${CKPT:-}"   # DINO 칸(EXTRA="--p-source dino --m-source raw")은 생략
TAG="${TAG:?TAG 필요}"
EXTRA="${EXTRA:-}"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /proj/external_group/mrg/conda_envs/aavrl-train

cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$CONDA_PREFIX/bin/python" -u scripts/eval/observer_fuse.py ${CKPT:+--checkpoint "$CKPT"} --tag "$TAG" $EXTRA
