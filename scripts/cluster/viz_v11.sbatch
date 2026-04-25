#!/bin/bash
# Two-Stream v11 attention + reconstruction visualization
# 사용:
#   sbatch --export=ALL,CKPT=...,TAG=v11_ep8 scripts/cluster/viz_v11.sbatch
# 환경변수:
#   CKPT     (필수): v11 checkpoint path
#   TAG      (기본 v11_ep8): output 파일 식별자
#   P_DEPTH  (기본 12): P encoder depth (ckpt config와 매칭)
#   M_DEPTH  (기본 6):  M encoder depth
#SBATCH --job-name=viz_v11
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/proj/external_group/mrg/logs/viz_v11_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/viz_v11_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT="${CKPT:?CKPT required}"
TAG="${TAG:-v11_ep8}"
P_DEPTH="${P_DEPTH:-12}"
M_DEPTH="${M_DEPTH:-6}"
EGODEX_DIR="/proj/external_group/mrg/datasets/egodex/frames/part4"
DROID_DIR="/proj/external_group/mrg/datasets/droid_frames/ext1"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

OUTPUT="${PROJECT_ROOT}/docs/architecture/attn_${TAG}.png"
"$CONDA_PREFIX/bin/python" scripts/eval/visualize_attn_v11.py \
    --checkpoint "$CKPT" \
    --egodex-dir "$EGODEX_DIR" \
    --droid-dir "$DROID_DIR" \
    --output "$OUTPUT" \
    --gap 30 --p-depth "$P_DEPTH" --m-depth "$M_DEPTH"
echo "Saved: $OUTPUT"
