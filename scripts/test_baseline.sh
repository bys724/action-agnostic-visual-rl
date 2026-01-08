#!/bin/bash
# SimplerEnv 베이스라인 모델 테스트 스크립트

echo "============================================"
echo "SimplerEnv Baseline Model Test"
echo "============================================"

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. SimplePolicy 테스트 (기본 동작 확인)
echo -e "\n${YELLOW}1. Testing with SimplePolicy...${NC}"
echo "----------------------------------------"
python src/eval_simpler.py --model simple --n-episodes 2 --max-steps 100
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ SimplePolicy test passed${NC}"
else
    echo -e "${RED}✗ SimplePolicy test failed${NC}"
    exit 1
fi

# 2. Trajectory 수집 테스트
echo -e "\n${YELLOW}2. Testing trajectory collection...${NC}"
echo "----------------------------------------"
python src/collect_trajectories.py --model simple --n-per-task 2 --max-steps 100
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Trajectory collection test passed${NC}"
else
    echo -e "${RED}✗ Trajectory collection test failed${NC}"
    exit 1
fi

# 3. 결과 확인
echo -e "\n${YELLOW}3. Checking output files...${NC}"
echo "----------------------------------------"
if [ -d "data/results" ]; then
    echo -e "${GREEN}✓ Results directory exists${NC}"
    ls -la data/results/
else
    echo -e "${RED}✗ Results directory not found${NC}"
fi

if [ -d "data/trajectories" ]; then
    echo -e "${GREEN}✓ Trajectories directory exists${NC}"
    ls -la data/trajectories/
else
    echo -e "${RED}✗ Trajectories directory not found${NC}"
fi

echo -e "\n${GREEN}============================================"
echo "Test Complete!"
echo "============================================${NC}"