#!/bin/bash
# OpenVLA 모델 테스트 스크립트

echo "============================================"
echo "OpenVLA Model Integration Test"
echo "============================================"

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. OpenVLA 모델로 평가 테스트 (소규모)
echo -e "\n${YELLOW}1. Testing OpenVLA with eval_simpler.py...${NC}"
echo "----------------------------------------"
python src/eval_simpler.py \
    --model "openvla/openvla-7b" \
    --n-episodes 2 \
    --max-steps 100 \
    --output data/results/openvla_test

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ OpenVLA evaluation test passed${NC}"
else
    echo -e "${RED}✗ OpenVLA evaluation test failed${NC}"
    exit 1
fi

# 2. OpenVLA 모델로 trajectory 수집 테스트
echo -e "\n${YELLOW}2. Testing trajectory collection with OpenVLA...${NC}"
echo "----------------------------------------"
python src/collect_trajectories.py \
    --model "openvla/openvla-7b" \
    --n-per-task 1 \
    --max-steps 100 \
    --output data/trajectories/openvla_test

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ OpenVLA trajectory collection test passed${NC}"
else
    echo -e "${RED}✗ OpenVLA trajectory collection test failed${NC}"
    exit 1
fi

# 3. 결과 확인
echo -e "\n${YELLOW}3. Checking output files...${NC}"
echo "----------------------------------------"
if [ -d "data/results/openvla_test" ]; then
    echo -e "${GREEN}✓ OpenVLA results directory exists${NC}"
    ls -la data/results/openvla_test/
else
    echo -e "${YELLOW}⚠ OpenVLA results directory not found (may be normal)${NC}"
fi

if [ -d "data/trajectories/openvla_test" ]; then
    echo -e "${GREEN}✓ OpenVLA trajectories directory exists${NC}"
    ls -la data/trajectories/openvla_test/
else
    echo -e "${YELLOW}⚠ OpenVLA trajectories directory not found (may be normal)${NC}"
fi

echo -e "\n${GREEN}============================================"
echo "OpenVLA Integration Test Complete!"
echo "============================================${NC}"
echo ""
echo "Note: To use OpenVLA in production:"
echo "1. Download model checkpoint: huggingface-cli download openvla/openvla-7b"
echo "2. Or use HuggingFace model ID directly: --model openvla/openvla-7b"
echo "3. For faster inference, consider using openvla/openvla-7b-finetuned"