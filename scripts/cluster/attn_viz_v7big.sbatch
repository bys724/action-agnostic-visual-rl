#!/bin/bash
# v7-big 3-CLS attention viz — specialization 진단
#SBATCH --job-name=attn_v7big
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/proj/external_group/mrg/logs/attn_v7big_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/attn_v7big_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT="${CKPT:-/proj/external_group/mrg/checkpoints/two_stream_v7big_sigma003/20260419_023107/checkpoint_epoch0004.pt}"
EPOCH_TAG="${EPOCH_TAG:-ep4}"

OUTPUT="${PROJECT_ROOT}/docs/architecture/attn_v7big_${EPOCH_TAG}.png"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

"$CONDA_PREFIX/bin/python" scripts/eval/visualize_attn_compare_v7big.py \
    --checkpoint "$CKPT" \
    --egodex-dir /proj/external_group/mrg/datasets/egodex/frames/part4 \
    --droid-dir /proj/external_group/mrg/datasets/droid_frames/ext1 \
    --output "$OUTPUT" \
    --gap 30 --depth 12 --num-stages 2 --sigma 0.03

echo "Saved: $OUTPUT"
