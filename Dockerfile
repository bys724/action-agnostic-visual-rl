# NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Install system dependencies and Vulkan
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglu1-mesa \
    libglu1-mesa-dev \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    xvfb \
    ffmpeg \
    libvulkan1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy project files
COPY requirements.txt /workspace/requirements.txt
COPY third_party /workspace/third_party

# Install numpy first (version constraint)
RUN pip install numpy==1.24.4

# Install ManiSkill2_real2sim
RUN cd /workspace/third_party/SimplerEnv/ManiSkill2_real2sim && \
    pip install -e .

# Install SimplerEnv
RUN cd /workspace/third_party/SimplerEnv && \
    pip install -e .

# Install project requirements (without numpy since it's already installed)
RUN grep -v numpy requirements.txt > requirements_no_numpy.txt && \
    pip install -r requirements_no_numpy.txt || true

# Copy the rest of the project
COPY . /workspace

# Create directories for experiments and logs
RUN mkdir -p /workspace/experiments /workspace/logs /workspace/checkpoints

# Set up display for rendering (when needed)
ENV DISPLAY=:99

# Entry point for running with virtual display
ENTRYPOINT ["/bin/bash"]