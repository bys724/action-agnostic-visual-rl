#!/bin/bash
# [DEPRECATED] v14 폐기 lineup viz. 코드만 reference 보존.
# Two-Stream v14 visualization: nomask reconstruction + attention map (motion-routing focus)
# 사용:
#   sbatch --export=ALL,CKPT=...,TAG=v14_ep4 scripts/cluster/viz_v14.sbatch
# 환경변수:
#   CKPT (필수): v14 checkpoint path
#   TAG  (기본 v14_main): attention map 파일 식별자
#SBATCH --job-name=viz_v14
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/proj/external_group/mrg/logs/viz_v14_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/viz_v14_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT="${CKPT:?CKPT required}"
TAG="${TAG:-v14_main}"
EGODEX_DIR="/proj/external_group/mrg/datasets/egodex/frames/part4"
DROID_DIR="/proj/external_group/mrg/datasets/droid_frames/ext1"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

# (1) No-mask reconstruction (test set 4 random sample)
"$CONDA_PREFIX/bin/python" scripts/eval/visualize_v14_no_mask.py \
    --ckpt "$CKPT" --num-samples 4 --seed 42

# (2) Attention map (motion-routing 2 iter focus, deterministic high-change pair)
ATTN_OUT="${PROJECT_ROOT}/paper_artifacts/v14_main_train_samples/attn_${TAG}.png"
"$CONDA_PREFIX/bin/python" scripts/eval/visualize_attn_v14.py \
    --checkpoint "$CKPT" \
    --egodex-dir "$EGODEX_DIR" \
    --droid-dir "$DROID_DIR" \
    --output "$ATTN_OUT" \
    --gap 30
echo "Saved attn: $ATTN_OUT"
