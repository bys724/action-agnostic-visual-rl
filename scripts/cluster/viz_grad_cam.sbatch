#!/bin/bash
# Grad-CAM arrow 가시화 (paper representation_visualization_plan #1, prototype).
#
# 사용:
#   sbatch --export=ALL,DEMO_KEY=demo_0,OUT_TAG=libero_spatial_task0_demo0 \
#          scripts/cluster/viz_grad_cam.sbatch
#
#SBATCH --job-name=viz_grad_cam
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/proj/external_group/mrg/logs/viz_gc_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/viz_gc_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CONDA_ENV="/proj/external_group/mrg/conda_envs/aavrl-train"

HDF5_DIR="${HDF5_DIR:-/proj/external_group/mrg/datasets/libero/libero_spatial}"
HDF5_NAME="${HDF5_NAME:-pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5}"
DEMO_KEY="${DEMO_KEY:-demo_0}"
OUT_TAG="${OUT_TAG:-libero_spatial_task0_demo0}"
GAP="${GAP:-20}"
PROBE_EPOCHS="${PROBE_EPOCHS:-80}"
MULTI_DEMO="${MULTI_DEMO:-}"
ENCODER="${ENCODER:-v15}"

OUT_DIR="$PROJECT_ROOT/paper_artifacts/visualizations/grad_cam_arrow/$OUT_TAG"

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
    --hdf5 "$HDF5_DIR/$HDF5_NAME"
    --demo-key "$DEMO_KEY"
    --out-dir "$OUT_DIR"
    --encoder "$ENCODER"
    --gap "$GAP"
    --probe-epochs "$PROBE_EPOCHS"
)
if [[ -n "$MULTI_DEMO" ]]; then ARGS+=(--multi-demo); fi

echo "[viz_gc] encoder=$ENCODER hdf5=$HDF5_NAME demo=$DEMO_KEY gap=$GAP multi=$MULTI_DEMO out=$OUT_DIR"
"$CONDA_PREFIX/bin/python" -u scripts/viz/grad_cam_arrow.py "${ARGS[@]}"
echo "[viz_gc] done"
