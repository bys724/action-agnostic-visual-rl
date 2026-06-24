#!/bin/bash
# MCP-MAE(§9 pixel-pred) reconstruction/prediction 가시화 (MIG 최소사양).
# Usage:
#   sbatch --export=ALL,CKPT=<pixel_pred.pt>,OUT=paper_artifacts/.../x.png,NO_SOBEL=1,\
#          EMBED_DIM=384,NUM_HEADS=6,M_DEPTH=6,NUM_EGODEX=6 scripts/cluster/viz_mcp_mae.sbatch
#SBATCH --job-name=viz_mcp_mae
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/proj/external_group/mrg/logs/viz_mcp_mae_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/viz_mcp_mae_%j.err
set -euo pipefail
module load miniforge3/24.11.3-2
conda activate /proj/external_group/mrg/conda_envs/aavrl-train
cd /proj/home/mrg/bys724/action-agnostic-visual-rl

PYTHONPATH=. python scripts/eval/visualize_mcp_mae.py \
    --ckpt "${CKPT:?CKPT required}" \
    --out "${OUT:-paper_artifacts/mcp_mae_sanity_recon/recon.png}" \
    --embed-dim "${EMBED_DIM:-384}" --num-heads "${NUM_HEADS:-6}" --m-depth "${M_DEPTH:-6}" \
    --routing-source "${ROUTING_SOURCE:-m}" \
    --num-per-source "${NUM_PER_SOURCE:-2}" --max-videos "${MAX_VIDEOS:-60}" \
    --seen-split "${SEEN_SPLIT:-part1}" --unseen-split "${UNSEEN_SPLIT:-part4}" \
    ${NO_SOBEL:+--no-sobel} ${UNSEEN_DROID:+--unseen-droid}
echo "done."
