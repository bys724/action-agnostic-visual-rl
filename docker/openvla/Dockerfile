# OpenVLA Policy Server
# 최적화된 transformers 버전으로 OpenVLA만 실행

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl vim \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python -m pip install --upgrade pip

WORKDIR /app

# PyTorch with CUDA
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# OpenVLA dependencies - 공식 버전 사용
RUN pip install --no-cache-dir \
    transformers==4.40.1 \
    accelerate>=0.30.0 \
    timm==0.9.10 \
    tokenizers==0.19.1 \
    sentencepiece \
    pillow \
    einops \
    safetensors \
    scipy \
    numpy

# API server dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    requests

# transforms3d for action conversion
RUN pip install transforms3d

# Copy server code
COPY docker/openvla/server.py /app/server.py

# Environment
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8001

CMD ["python", "server.py"]
