# LAPA Policy Server
# JAX/Flax 기반 LAPA 모델 전용 환경

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

# LAPA dependencies - 공식 버전 사용
RUN pip install --no-cache-dir \
    jax[cuda12]==0.4.23 \
    flax==0.7.0 \
    optax==0.1.7 \
    chex==0.1.82 \
    ml_collections \
    scipy==1.12.0

# Transformers - LAPA 공식 버전
RUN pip install --no-cache-dir \
    transformers==4.29.2 \
    sentencepiece \
    pillow \
    einops \
    albumentations \
    opencv-python-headless

# tux (LAPA custom dependency)
RUN pip install --no-cache-dir git+https://github.com/lhao499/tux.git

# API server dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    requests

# transforms3d for action conversion
RUN pip install transforms3d tensorflow

# Copy LAPA source (will be mounted or copied)
COPY third_party/LAPA /app/LAPA

# Copy server code
COPY docker/lapa/server.py /app/server.py

# Add LAPA to path
ENV PYTHONPATH=/app/LAPA:$PYTHONPATH

EXPOSE 8002

CMD ["python", "server.py"]
