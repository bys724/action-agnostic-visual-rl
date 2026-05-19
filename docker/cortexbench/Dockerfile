# CortexBench (eai-vc) Evaluation Environment
# Manipulation subset only (Adroit + Meta-World) — habitat/trifinger excluded.
# Original eai-vc spec (paper §C11 protocol 일치):
#   - Python 3.8 + PyTorch 1.13 + cudatoolkit 11.6
#   - MuJoCo 200 (legacy) + mujoco-py (eai-vc fork)
# Risk: H100 sm_90 PTX JIT fallback (PyTorch 1.13에 sm_90 코드 없음).
#   Frozen encoder forward + linear policy training만 하므로 작동 추정.
#   Sanity 단계에서 실패 시 cu117 또는 pt2.1 patch 검토.

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps for mujoco200 rendering + python build + compilers
RUN apt-get update && apt-get install -y \
    git wget curl vim unzip build-essential \
    libgl1-mesa-glx libosmesa6-dev libglew-dev libglfw3-dev libegl1 \
    patchelf libglib2.0-0 libsm6 libxrender1 libxext6 \
    libxi-dev libxinerama-dev libxcursor-dev libxrandr-dev libxxf86vm-dev \
    libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (mamba) for fast conda env resolution
ENV CONDA_DIR=/opt/conda
RUN wget -qO miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash miniforge.sh -b -p $CONDA_DIR && rm miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# MuJoCo 200 binary + free license key (roboti.us, 2026-05-19 살아있음 확인)
RUN mkdir -p /root/.mujoco && \
    wget -qO /tmp/mujoco200.zip https://www.roboti.us/download/mujoco200_linux.zip && \
    cd /root/.mujoco && unzip -q /tmp/mujoco200.zip && \
    mv mujoco200_linux mujoco200 && rm /tmp/mujoco200.zip && \
    wget -qO /root/.mujoco/mjkey.txt https://www.roboti.us/file/mjkey.txt
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco200

# eai-vc conda env (pruned manipulation subset)
COPY docker/cortexbench/environment-manipulation.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean -ay

# Make eai-vc env default in shell + ENV PATH
SHELL ["/bin/bash", "-c"]
ENV PATH=$CONDA_DIR/envs/eai-vc/bin:$PATH

# Install helper script. third_party/vc_models/mujoco_vc install은 build time이 아닌
# 컨테이너 first-up 시점에 mounted /workspace/external/eai-vc에서 실행.
# 이유: host code 변경시 image rebuild 회피 + submodule 변경 추적 단순화.
COPY docker/cortexbench/install_eai_vc.sh /usr/local/bin/install_eai_vc.sh
RUN chmod +x /usr/local/bin/install_eai_vc.sh

WORKDIR /workspace

# Headless rendering for mujoco
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=all

CMD ["/bin/bash"]
