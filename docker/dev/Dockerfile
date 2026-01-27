# Action-Agnostic Visual RL 개발 환경
# H100 x 2 GPU, PyTorch 2.2 + HuggingFace 생태계
#
# 주요 기능:
# - Vision Encoders: DINOv2, SigLIP, CLIP (timm, transformers)
# - Language Models: Llama, T5 등 (transformers)
# - 효율적 학습: DeepSpeed, FSDP, LoRA (PEFT)
# - 실험 추적: WandB, TensorBoard

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
# ============================================================
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================================
# HuggingFace Ecosystem (Pre-trained Models)
# ============================================================
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    datasets \
    huggingface_hub \
    safetensors \
    tokenizers

# ============================================================
# Vision Models (DINOv2, CLIP, SigLIP)
# ============================================================
RUN pip install --no-cache-dir \
    timm>=0.9.10 \
    open_clip_torch \
    einops

# Flash Attention 2 (H100 optimized)
RUN pip install packaging ninja && \
    pip install "flash-attn>=2.5.0" --no-build-isolation

# ============================================================
# Efficient Training (Multi-GPU, LoRA)
# ============================================================
RUN pip install --no-cache-dir \
    deepspeed \
    peft>=0.11.0 \
    bitsandbytes

# ============================================================
# Experiment Tracking & Utilities
# ============================================================
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    tqdm \
    rich \
    omegaconf \
    hydra-core

# ============================================================
# Data Processing
# ============================================================
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    pillow \
    opencv-python-headless \
    albumentations \
    webdataset

# ============================================================
# Development Tools
# ============================================================
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    matplotlib \
    seaborn

RUN pip install --no-cache-dir \
    pytest \
    black \
    isort \
    mypy

# ============================================================
# Environment Configuration
# ============================================================
ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_VISIBLE_DEVICES=0,1
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1
ENV HF_HOME=/workspace/.cache/huggingface

# Ports
EXPOSE 6006
EXPOSE 8888

WORKDIR /workspace

CMD ["/bin/bash"]
