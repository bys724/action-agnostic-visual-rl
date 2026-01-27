# Pi0 Policy Server
# JAX + Flax environment for Pi0 inference

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git git-lfs wget curl vim \
    build-essential clang \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip

WORKDIR /app

# JAX with CUDA
RUN pip install --no-cache-dir \
    jax[cuda12]==0.4.30 \
    flax==0.8.5 \
    orbax-checkpoint==0.5.23

# Core dependencies
RUN pip install --no-cache-dir \
    numpy \
    einops \
    pillow \
    transformers \
    safetensors \
    scipy

# API server dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    pydantic \
    requests

# Copy server code (will be created)
COPY docker/pi0/server.py /app/server.py

ENV TOKENIZERS_PARALLELISM=false
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPOSE 8003

CMD ["python", "server.py"]
