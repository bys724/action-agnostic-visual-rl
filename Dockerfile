FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    xvfb \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies with NumPy 1.x for compatibility
RUN pip install --no-cache-dir "numpy<2.0" scipy

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch with CUDA support first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install ManiSkill3 (required by SimplerEnv)
RUN pip install --upgrade git+https://github.com/haosulab/ManiSkill.git

# Install SimplerEnv from submodule
COPY third_party/SimplerEnv /tmp/SimplerEnv
RUN cd /tmp/SimplerEnv && pip install -e .

# Install OpenCV with GUI support
RUN apt-get update && apt-get install -y python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# Ensure NumPy 1.x is maintained
RUN pip install --force-reinstall "numpy<2.0"

# Environment variables for rendering
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV DISPLAY=:99

# Create entrypoint script
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]