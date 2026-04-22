#!/bin/bash
# Two-Stream attention compare + rotation diagnostic (variant-agnostic)
# 사용:
#   sbatch --export=ALL,CKPT=...,TAG=v10_ep4,P_TARGET=future scripts/cluster/viz_two_stream.sbatch
# 환경변수:
#   CKPT       (필수): checkpoint path
#   TAG        (기본 v9_resnorm_ep8): output 파일/디렉토리 식별자
#   P_TARGET   (기본 residual): future | current | residual
#SBATCH --job-name=viz_ts
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/proj/external_group/mrg/logs/viz_ts_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/viz_ts_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"
CKPT="${CKPT:?CKPT required}"
TAG="${TAG:-v9_resnorm_ep8}"
P_TARGET="${P_TARGET:-residual}"
EGODEX_DIR="/proj/external_group/mrg/datasets/egodex/frames/part4"
DROID_DIR="/proj/external_group/mrg/datasets/droid_frames/ext1"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

# ── 1. Attention compare (4×6 grid, 2 EgoDex + 2 DROID) ──
OUTPUT="${PROJECT_ROOT}/docs/architecture/attn_${TAG}.png"
"$CONDA_PREFIX/bin/python" scripts/eval/visualize_attn_compare.py \
    --checkpoint "$CKPT" \
    --egodex-dir "$EGODEX_DIR" \
    --droid-dir "$DROID_DIR" \
    --output "$OUTPUT" \
    --p-target "$P_TARGET" \
    --gap 30 --depth 12 --num-stages 2
echo "Saved: $OUTPUT"

# ── 2. Extract deterministic pair paths (same seed as compare) ──
PAIRS=$("$CONDA_PREFIX/bin/python" - <<PY
import sys
sys.path.insert(0, '.')
from scripts.eval.visualize_attn_compare import find_high_change_pairs
ego = find_high_change_pairs("$EGODEX_DIR", 30, 2, 30, 42)
drd = find_high_change_pairs("$DROID_DIR", 30, 2, 30, 43)
for p in ego + drd:
    print(p[0], p[1])
PY
)

# ── 3. Rotation diagnostic per pair ──
ROT_DIR="${PROJECT_ROOT}/docs/architecture/rotation_${TAG}"
mkdir -p "$ROT_DIR"
echo "$PAIRS" | while read FT FTK; do
    [ -z "$FT" ] && continue
    "$CONDA_PREFIX/bin/python" scripts/eval/visualize_attn_rotation.py \
        --checkpoints "$CKPT" \
        --labels "$TAG" \
        --frame-t "$FT" --frame-tk "$FTK" \
        --output-dir "$ROT_DIR" \
        --p-target "$P_TARGET"
done
echo "Rotation figures saved to: $ROT_DIR"
