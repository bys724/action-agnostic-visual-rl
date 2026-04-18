#!/bin/bash
# v7-big sanity: 1 GPU, 50 steps, 작은 batch로 shape/NaN/perf 체크
#SBATCH --job-name=aavrl_v7big_sanity
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/proj/external_group/mrg/logs/sanity_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/sanity_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
EGODEX_ROOT="/proj/external_group/mrg/datasets/egodex/raw"
FRAMES_ROOT="/proj/external_group/mrg/datasets/egodex/frames"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CHECKPOINT_DIR="/proj/external_group/mrg/checkpoints/two_stream_v7big_sanity"

# Clean previous sanity checkpoints (avoid resume contamination)
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "=== v7-big sanity test ==="
echo "Model params, loss, GPU util check"
echo ""

"$CONDA_PREFIX/bin/python" scripts/pretrain.py \
    --model two-stream \
    --depth 12 --num-stages 2 \
    --use-ape \
    --v7-big --sigma 0.03 \
    --mask-ratio 0.1 --mask-ratio-p 0.5 \
    --rotation-aug \
    --train-data egodex --egodex-root "$EGODEX_ROOT" \
    --egodex-splits part1 --max-videos 20 \
    --max-gap 60 --sample-dist triangular --sample-center 30 \
    --epochs 1 --batch-size 8 --lr 1e-4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-interval 999 \
    --eval-interval 999 \
    --no-multi-gpu

echo ""
echo "=== sanity done ==="
