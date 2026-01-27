# OpenVLA-LIBERO Policy Server
# OpenVLA fine-tuned on LIBERO dataset

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

# OpenVLA dependencies
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

# Copy server code
COPY docker/openvla-libero/server.py /app/server.py

ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8010

CMD ["python", "server.py"]
