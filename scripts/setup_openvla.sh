#!/bin/bash
# OpenVLA 실제 사용을 위한 설정 스크립트

echo "============================================"
echo "OpenVLA Setup for Production"
echo "============================================"

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. GPU 확인
echo -e "\n${YELLOW}1. Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$VRAM" -lt 20000 ]; then
        echo -e "${RED}Warning: GPU has less than 20GB VRAM. OpenVLA-7B requires 20-24GB.${NC}"
    else
        echo -e "${GREEN}✓ GPU memory sufficient${NC}"
    fi
else
    echo -e "${RED}✗ No GPU detected${NC}"
fi

# 2. Python 패키지 업그레이드
echo -e "\n${YELLOW}2. Upgrading packages for OpenVLA...${NC}"
pip install --upgrade transformers accelerate timm tokenizers einops safetensors

# 3. Flash Attention 설치 (선택)
echo -e "\n${YELLOW}3. Flash Attention 2 (optional, for faster inference)${NC}"
echo "Install with: pip install flash-attn --no-build-isolation"
echo "Requires: CUDA 11.6+ and compatible GPU"

# 4. 모델 사전 다운로드 (선택)
echo -e "\n${YELLOW}4. Pre-download model (optional)${NC}"
echo "This will download ~15GB of model files:"
echo "python -c \"from transformers import AutoProcessor; AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)\""

# 5. 환경 변수 설정
echo -e "\n${YELLOW}5. Environment variables${NC}"
echo "export CUDA_VISIBLE_DEVICES=0"
echo "export TRANSFORMERS_CACHE=/workspace/.cache/huggingface"
echo "export HF_HOME=/workspace/.cache/huggingface"

echo -e "\n${GREEN}============================================"
echo "Setup complete! Test with:"
echo "python src/eval_simpler.py --model openvla/openvla-7b --n-episodes 1"
echo "============================================${NC}"