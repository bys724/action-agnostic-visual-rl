# LIBERO Evaluation Environment
# Based on CUDA 12.1 for H100 (sm_90) compatibility
# Python 3.10 is default on Ubuntu 22.04

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-venv python3-pip \
    git wget curl vim \
    make g++ clang \
    libosmesa6-dev libgl1-mesa-glx libegl1 \
    libglew-dev libglfw3-dev libgles2-mesa-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip

WORKDIR /workspace

# PyTorch 2.1.0 with CUDA 12.1 (H100 compatible)
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# LIBERO core dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    scipy==1.10.1 \
    mujoco==3.2.3 \
    robosuite==1.4.1 \
    numba==0.58.1 \
    opencv-python==4.6.0.66 \
    pillow==10.4.0 \
    matplotlib==3.5.3

# Additional dependencies
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg \
    tqdm tyro pyyaml \
    requests websocket-client \
    pandas h5py

# Custom encoder dependencies (for two-stream, videomae evaluation)
RUN pip install --no-cache-dir \
    'transformers>=4.30.0,<4.43.0' \
    timm>=0.9.0 \
    einops

# Clone and install LIBERO with all dependencies
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/libero && \
    cd /opt/libero && \
    pip install bddl easydict cloudpickle 'gym>=0.21.0,<0.26.0' && \
    pip install .

# Add LIBERO to PYTHONPATH
ENV PYTHONPATH=/opt/libero:$PYTHONPATH

# Setup EGL for headless rendering
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# LIBERO config directory
ENV LIBERO_CONFIG_PATH=/tmp/libero
RUN mkdir -p /tmp/libero

# Copy LIBERO config (paths will be set up when volume is mounted)
COPY docker/libero/libero_config.yaml /tmp/libero/config.yaml

# Environment variables for rendering
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=all

EXPOSE 8888

CMD ["/bin/bash"]
