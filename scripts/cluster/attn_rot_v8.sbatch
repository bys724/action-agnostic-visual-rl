#!/bin/bash
# v8 rotation diagnostic — attn_compare_v8와 동일 샘플, 0/90/180/270 회전 반응
#SBATCH --job-name=attn_rot_v8
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/proj/external_group/mrg/logs/attn_rot_v8_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/attn_rot_v8_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT="${CKPT:-/proj/external_group/mrg/checkpoints/two_stream_v8/20260421_054901/checkpoint_epoch0012.pt}"
EPOCH_TAG="${EPOCH_TAG:-ep12_run2}"

OUTPUT_DIR="${PROJECT_ROOT}/docs/architecture/rotation_v8"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

"$CONDA_PREFIX/bin/python" scripts/eval/visualize_attn_rotation_v8.py \
    --checkpoint "$CKPT" \
    --egodex-dir /proj/external_group/mrg/datasets/egodex/frames/part4 \
    --droid-dir /proj/external_group/mrg/datasets/droid_frames/ext1 \
    --output-dir "$OUTPUT_DIR" \
    --gap 30 --depth 12 --num-stages 2 \
    --epoch-tag "$EPOCH_TAG"

echo "Saved to: $OUTPUT_DIR"
