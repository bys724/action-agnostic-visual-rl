#!/bin/bash
# Two-Stream v15 visualization: nomask reconstruction × 3 frame + motion routing × 3 segment + L_compose
# 사용:
#   sbatch --export=ALL,CKPT=/proj/external_group/mrg/checkpoints/two_stream_v15/<ts>/checkpoint_epochXXXX.pt scripts/cluster/viz_v15.sbatch
#SBATCH --job-name=viz_v15
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/proj/external_group/mrg/logs/viz_v15_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/viz_v15_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT="${CKPT:?CKPT required}"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

"$CONDA_PREFIX/bin/python" scripts/eval/visualize_v15_no_mask.py \
    --ckpt "$CKPT" --num-egodex 2 --num-droid 2 --seed 42
