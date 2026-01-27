# LIBERO Evaluation Environment
# Python 3.8 required for LIBERO compatibility

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA for Python 3.8
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3.8-venv python3.8-distutils \
    git wget curl vim \
    make g++ clang \
    libosmesa6-dev libgl1-mesa-glx libegl1 \
    libglew-dev libglfw3-dev libgles2-mesa-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.8 1

RUN python -m pip install --upgrade pip

WORKDIR /workspace

# PyTorch with CUDA (older version for LIBERO compatibility)
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# LIBERO core dependencies
RUN pip install --no-cache-dir \
    numpy==1.22.4 \
    scipy==1.10.1 \
    mujoco==3.2.3 \
    robosuite==1.4.1 \
    numba==0.53.1 \
    opencv-python==4.6.0.66 \
    pillow==10.4.0 \
    matplotlib==3.5.3

# Additional dependencies
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg \
    tqdm tyro pyyaml \
    requests websocket-client \
    pandas

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
