#!/bin/bash
# PCA overlay 가시화 (representation_visualization_plan #3, paper_artifacts/visualizations/pca_overlay/).
#
# 변경 이력:
# - v1 (2026-05-27): LIBERO HDF5 + 3 encoder
# - v2 (2026-05-27): 5 encoder (ours/dinov2/siglip/vc1/videomae-ours) + 다중 source (libero/egodex/droid/calvin/npy)
#
# 사용 예:
#   # LIBERO demo
#   sbatch --export=ALL,SOURCE=libero,DEMO_KEY=demo_0,OUT_TAG=libero_spatial_task0_demo0 \
#          scripts/cluster/viz_pca_overlay.sbatch
#
#   # EgoDex demo (task dir 한 개)
#   sbatch --export=ALL,SOURCE=egodex,\
# EGODEX_TASK_DIR=/proj/external_group/mrg/datasets/egodex/frames/part1/add_remove_lid,\
# OUT_TAG=egodex_add_remove_lid scripts/cluster/viz_pca_overlay.sbatch
#
#   # DROID episode
#   sbatch --export=ALL,SOURCE=droid,\
# DROID_EPISODE_DIR=/proj/external_group/mrg/datasets/droid_frames/ext1/ep_000000,\
# OUT_TAG=droid_ep000000 scripts/cluster/viz_pca_overlay.sbatch
#
#   # CALVIN segment
#   sbatch --export=ALL,SOURCE=calvin,\
# CALVIN_SPLIT=/proj/external_group/mrg/datasets/calvin/task_ABCD_D/validation,\
# CALVIN_START=10000,CALVIN_END=10120,OUT_TAG=calvin_val_seg10000 \
#          scripts/cluster/viz_pca_overlay.sbatch
#
#SBATCH --job-name=viz_pca
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/proj/external_group/mrg/logs/viz_pca_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/viz_pca_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"

SOURCE="${SOURCE:-libero}"
OUT_TAG="${OUT_TAG:?OUT_TAG required}"

# Common
ENCODERS="${ENCODERS:-ours dinov2 siglip vc1 videomae-ours}"
MAX_FRAMES="${MAX_FRAMES:-}"
FPS="${FPS:-10}"
ALPHA="${ALPHA:-0.40}"
FRAME_STRIDE="${FRAME_STRIDE:-5}"

# LIBERO
HDF5_DIR="${HDF5_DIR:-/proj/external_group/mrg/datasets/libero/libero_spatial}"
HDF5_NAME="${HDF5_NAME:-pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5}"
DEMO_KEY="${DEMO_KEY:-demo_0}"
VIEW="${VIEW:-agentview_rgb}"

# EgoDex / DROID / CALVIN
EGODEX_TASK_DIR="${EGODEX_TASK_DIR:-}"
DROID_EPISODE_DIR="${DROID_EPISODE_DIR:-}"
CALVIN_SPLIT="${CALVIN_SPLIT:-}"
CALVIN_START="${CALVIN_START:-0}"
CALVIN_END="${CALVIN_END:-120}"

# NPY (rollout video frames)
NPY_PATH="${NPY_PATH:-}"

OUT_DIR="$PROJECT_ROOT/paper_artifacts/visualizations/pca_overlay/$OUT_TAG"

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$PROJECT_ROOT"

ARGS=(
    --source "$SOURCE"
    --out-dir "$OUT_DIR"
    --encoders $ENCODERS
    --fps "$FPS"
    --alpha "$ALPHA"
    --frame-stride "$FRAME_STRIDE"
)
case "$SOURCE" in
    libero) ARGS+=(--hdf5 "$HDF5_DIR/$HDF5_NAME" --demo-key "$DEMO_KEY" --view "$VIEW") ;;
    egodex) ARGS+=(--task-dir "$EGODEX_TASK_DIR") ;;
    droid)  ARGS+=(--episode-dir "$DROID_EPISODE_DIR") ;;
    calvin) ARGS+=(--calvin-split "$CALVIN_SPLIT" --calvin-start "$CALVIN_START" --calvin-end "$CALVIN_END") ;;
    npy)    ARGS+=(--npy-path "$NPY_PATH") ;;
esac
if [[ -n "$MAX_FRAMES" ]]; then ARGS+=(--max-frames "$MAX_FRAMES"); fi

echo "[viz_pca] source=$SOURCE out=$OUT_DIR encoders=$ENCODERS"
"$CONDA_PREFIX/bin/python" -u scripts/viz/pca_overlay.py "${ARGS[@]}"
echo "[viz_pca] done"
