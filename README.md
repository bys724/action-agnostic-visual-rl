# Action-Agnostic Visual Representation for Robotic Manipulation

## 핵심 아이디어
행동 정보 없이 학습한 시각적 표현이 다양한 로봇과 작업에서 더 잘 일반화된다는 가설 검증

## 실험 환경
- **벤치마크**: SIMPLER (시뮬레이션-실제 전환 평가)
- **시뮬레이터**: SAPIEN (레이 트레이싱 렌더링)
- **GPU 요구사항**: RTX 시리즈 (RT 코어 필수)

## 프로젝트 구조
```
action-agnostic-visual-rl/
├── configs/           # 실험 설정
├── src/              
│   ├── models/       # 시각 인코더, 정책 네트워크
│   ├── algorithms/   # RL 알고리즘
│   ├── envs/         # SIMPLER 환경 래퍼
│   └── utils/        # 유틸리티
├── scripts/          # 실행 스크립트
└── experiments/      # 실험 결과
```

## 설치

### 자동 설치 (권장)
```bash
# 설치 스크립트 실행
bash setup_env.sh
```

### 수동 설치
```bash
# Python venv 사용 (conda 없이)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate

# pip 업그레이드
pip install --upgrade pip

# numpy 버전 고정 (중요!)
pip install numpy==1.24.4

# SIMPLER 서브모듈 설치
cd third_party/SimplerEnv/ManiSkill2_real2sim
pip install -e .
cd ../
pip install -e .
cd ../../

# 나머지 패키지 설치
pip install -r requirements.txt
```

## 실행
```bash
# SIMPLER 환경 테스트
python scripts/test_simpler_env.py --list-envs
python scripts/test_simpler_env.py --env google_robot_pick_coke_can

# 학습 (준비중)
python scripts/train.py --config configs/experiment.yaml

# 평가 (준비중)
python scripts/evaluate.py --checkpoint path/to/model.pt
```

## 주요 실험
1. 베이스라인: 랜덤 초기화, 고정/미세조정 사전학습 모델
2. 제안 방법: 행동 비의존적 시각 표현 학습 (EMA 교사-학생)