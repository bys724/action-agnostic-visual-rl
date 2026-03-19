# Action-Agnostic Visual RL 개발 환경
# H100 x 2 GPU, PyTorch 2.2 + HuggingFace 생태계
#
# 주요 기능:
# - Vision Encoders: DINOv2, SigLIP, CLIP (timm, transformers)
# - 효율적 학습: DataParallel, multi-GPU
# - 실험 추적: TensorBoard

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl vim htop tmux \
    libgl1-mesa-glx libglib2.0-0 \
    ninja-build cmake \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip

WORKDIR /workspace

# ============================================================
# PyTorch 2.2 with CUDA 12.1 (H100 optimized)
# 후속 패키지가 torch를 덮어쓰지 못하도록 --no-deps로 보호
# ============================================================
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================================
# HuggingFace + Vision Models + Training Utils
# --no-deps 없이 설치하되, torch 버전 고정을 위해 마지막에 재설치
# ============================================================
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    huggingface_hub \
    safetensors \
    tokenizers \
    timm>=0.9.10 \
    einops

# Flash Attention 2 (H100 optimized)
RUN pip install packaging ninja && \
    MAX_JOBS=4 pip install "flash-attn>=2.5.0" --no-build-isolation

# ============================================================
# Experiment Tracking & Data Processing
# ============================================================
RUN pip install --no-cache-dir \
    tensorboard \
    tqdm \
    matplotlib \
    "numpy<2" \
    scipy \
    pillow \
    opencv-python-headless

# ============================================================
# PyTorch 버전 고정: 후속 패키지가 torch를 업그레이드했을 수 있으므로 재설치
# ============================================================
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================================
# Environment Configuration
# ============================================================
ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_VISIBLE_DEVICES=0,1
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1
ENV HF_HOME=/workspace/.cache/huggingface

EXPOSE 6006

WORKDIR /workspace

CMD ["/bin/bash"]
