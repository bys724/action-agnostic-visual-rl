#!/bin/bash
# v6 ep20 rotation diagnostic — EgoDex + DROID
#SBATCH --job-name=rotdiag_v6_ep20
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/proj/external_group/mrg/logs/rotdiag_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/rotdiag_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT="/proj/external_group/mrg/checkpoints/two_stream_v6_ape_mask50_rotaug/20260416_203440/checkpoint_epoch0020.pt"

EGODEX_T="/proj/external_group/mrg/datasets/egodex/frames/part4/pick_place_food/338/frame_000030.jpg"
EGODEX_TK="/proj/external_group/mrg/datasets/egodex/frames/part4/pick_place_food/338/frame_000060.jpg"
DROID_T="/proj/external_group/mrg/datasets/droid_frames/ext1/ep_000002/frame_000030.jpg"
DROID_TK="/proj/external_group/mrg/datasets/droid_frames/ext1/ep_000002/frame_000060.jpg"

OUTDIR="${PROJECT_ROOT}/docs/architecture"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== EgoDex (sample 338) ==="
"$CONDA_PREFIX/bin/python" scripts/eval/visualize_attn_rotation.py \
    --checkpoints "$CKPT" \
    --labels "v6_rotaug_ep20_EgoDex" \
    --frame-t  "$EGODEX_T" \
    --frame-tk "$EGODEX_TK" \
    --output-dir "$OUTDIR"

echo "=== DROID (ep_000002) ==="
"$CONDA_PREFIX/bin/python" scripts/eval/visualize_attn_rotation.py \
    --checkpoints "$CKPT" \
    --labels "v6_rotaug_ep20_DROID" \
    --frame-t  "$DROID_T" \
    --frame-tk "$DROID_TK" \
    --output-dir "$OUTDIR"

echo "=== done ==="
