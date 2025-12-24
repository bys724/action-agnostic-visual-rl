#!/bin/bash
# SIMPLER 환경 설치 스크립트 (conda 없이 pip/venv 사용)

echo "========================================="
echo "Action-Agnostic Visual RL 환경 설치"
echo "========================================="

# 1. Python 버전 확인
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Python 버전: $python_version"

if [[ ! "$python_version" =~ ^3\.(10|11) ]]; then
    echo "경고: Python 3.10 또는 3.11이 권장됩니다."
    read -p "계속하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. 가상환경 생성 (이미 있으면 스킵)
if [ ! -d "venv" ]; then
    echo "가상환경 생성중..."
    python3 -m venv venv
else
    echo "가상환경이 이미 존재합니다."
fi

# 3. 가상환경 활성화
echo "가상환경 활성화..."
source venv/bin/activate

# 4. pip 업그레이드
echo "pip 업그레이드..."
pip install --upgrade pip

# 5. numpy 버전 고정 (중요!)
echo "numpy 1.24.4 설치 (IK 오류 방지)..."
pip install numpy==1.24.4

# 6. 서브모듈 초기화
echo "서브모듈 초기화..."
git submodule update --init --recursive

# 7. ManiSkill2 설치
echo "ManiSkill2 real2sim 설치..."
cd third_party/SimplerEnv/ManiSkill2_real2sim
pip install -e .

# 8. SIMPLER 설치
echo "SIMPLER 설치..."
cd ../
pip install -e .

# 9. 프로젝트 루트로 돌아오기
cd ../../

# 10. 프로젝트 의존성 설치
echo "프로젝트 의존성 설치..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "설치 완료!"
echo "========================================="
echo ""
echo "다음 명령으로 환경을 테스트하세요:"
echo "  source venv/bin/activate"
echo "  python scripts/test_simpler_env.py --list-envs"
echo ""