#!/bin/bash
# CortexBench (eai-vc) third_party + local package install.
# Runs inside container after build, against host-mounted /workspace/external/eai-vc.
# Re-runnable: `pip install -e` is idempotent.
set -e

EAI_VC=/workspace/external/eai-vc
cd "$EAI_VC"

# 0) H100 sm_90 지원: conda yaml의 pytorch=1.13 → torch 2.1.2+cu118 (py3.8 호환 마지막
#    버전대 중 sm_90 PTX 포함). dm-control 1.0.5 + Adroit/Meta-World 환경에는 영향 없음.
echo "[install_eai_vc] torch 2.1.2+cu118 upgrade (H100 sm_90)..."
pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2

echo "[install_eai_vc] mujoco-py..."
pip install -e ./third_party/mujoco-py

echo "[install_eai_vc] mj_envs (Adroit + others)..."
pip install -e ./third_party/mj_envs

echo "[install_eai_vc] mjrl (BC learner)..."
pip install -e ./third_party/mjrl

echo "[install_eai_vc] metaworld..."
pip install -e ./third_party/metaworld

echo "[install_eai_vc] dmc2gym..."
pip install -e ./third_party/dmc2gym

echo "[install_eai_vc] vc_models (encoder loading API)..."
pip install -e ./vc_models

echo "[install_eai_vc] cortexbench/mujoco_vc (visual IL tasks)..."
pip install -e ./cortexbench/mujoco_vc

echo "[install_eai_vc] Compiling mujoco-py (first import builds cython)..."
python -c "import mujoco_py; print('mujoco_py OK, path:', mujoco_py.__file__)"

echo "[install_eai_vc] Done."
