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

# Install NumPy 1.x first (MUST be before other packages)
RUN pip install --no-cache-dir "numpy<2.0" scipy

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Force NumPy 1.x again (in case requirements.txt upgraded it)
RUN pip install --no-cache-dir --force-reinstall "numpy<2.0"

# Install PyTorch with CUDA support first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install ManiSkill3 (required by SimplerEnv)
RUN pip install --upgrade git+https://github.com/haosulab/ManiSkill.git

# Install SimplerEnv from submodule
COPY third_party/SimplerEnv /tmp/SimplerEnv
RUN cd /tmp/SimplerEnv && pip install -e .

# Install OpenCV with GUI support (system package)
RUN apt-get update && apt-get install -y python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# Remove pip opencv-python if accidentally installed and ensure NumPy 1.x
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
    pip install --force-reinstall "numpy<2.0" --no-deps && \
    python -c "import numpy; assert numpy.__version__ < '2.0', f'NumPy {numpy.__version__} >= 2.0!'" && \
    python -c "import cv2; assert hasattr(cv2, 'imshow'), 'OpenCV GUI support missing!'"

# Environment variables for rendering
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV DISPLAY=:99

# Setup bashrc with commonly needed environment variables
RUN echo '# GPU Memory Management' >> /root/.bashrc && \
    echo 'export XLA_PYTHON_CLIENT_PREALLOCATE=false' >> /root/.bashrc && \
    echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> /root/.bashrc && \
    echo '' >> /root/.bashrc && \
    echo '# CUDA Settings' >> /root/.bashrc && \
    echo 'export CUDA_VISIBLE_DEVICES=0' >> /root/.bashrc && \
    echo '' >> /root/.bashrc && \
    echo '# Convenience aliases' >> /root/.bashrc && \
    echo 'alias ll="ls -la"' >> /root/.bashrc && \
    echo 'alias test-simple="python src/eval_simpler.py --model simple --n-episodes 2"' >> /root/.bashrc && \
    echo 'alias test-all="./scripts/test_baseline.sh"' >> /root/.bashrc && \
    echo '' >> /root/.bashrc && \
    echo '# Welcome message' >> /root/.bashrc && \
    echo 'echo "========================================="' >> /root/.bashrc && \
    echo 'echo "SimplerEnv Evaluation Environment"' >> /root/.bashrc && \
    echo 'echo "GPU memory settings configured."' >> /root/.bashrc && \
    echo 'echo "Use test-simple or test-all to run tests"' >> /root/.bashrc && \
    echo 'echo "========================================="' >> /root/.bashrc

# Create entrypoint script
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]