# SimplerEnv 베이스라인 테스트 가이드

## 사전 준비

### 1. Docker 환경 시작
```bash
# Docker 이미지 빌드 (최초 1회 또는 Dockerfile 수정 후)
docker build -t simpler-env:latest .

# 평가 컨테이너 시작
docker compose up -d eval

# 컨테이너 접속
docker exec -it simpler-dev bash
```

### 2. 환경 변수 (자동 설정됨)
컨테이너 접속 시 자동으로 설정되는 환경 변수:
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` - JAX 메모리 점진적 할당
- `TF_FORCE_GPU_ALLOW_GROWTH=true` - TensorFlow 메모리 점진적 할당  
- `CUDA_VISIBLE_DEVICES=0` - 첫 번째 GPU 사용

**참고**: 이제 `.bashrc`에 자동 설정되어 수동 입력 불필요

## 테스트 실행

### 0. 기본 SimplerEnv 동작 확인
```bash
# SimplerEnv 환경 테스트 (GUI 포함)
python src/test_simpler_demo.py --gui

# 헤드리스 모드 (서버 환경)
python src/test_simpler_demo.py
```

### 옵션 1: 자동 테스트 스크립트
```bash
# 모든 테스트 자동 실행
./scripts/test_baseline.sh

# OpenVLA 테스트 (GPU 필요)
./scripts/test_openvla.sh
```

### 옵션 2: 개별 테스트

#### 1. SimplePolicy 테스트 (빠른 동작 확인)
```bash
# 기본 정책으로 환경 테스트
python src/eval_simpler.py --model simple --n-episodes 2 --max-steps 100

# Trajectory 수집 테스트
python src/collect_trajectories.py --model simple --n-per-task 2 --max-steps 100
```

예상 출력:
```
Loading policy: simple
============================================================
Evaluating PutSpoonOnTableClothInScene-v1...
  Episode 1/2: Fail
  Episode 2/2: Fail
  Success rate: 0.00%
...
```

#### 2. OpenVLA 모델 테스트 (GPU 필요)
```bash
# OpenVLA import 확인
python -c "from src.policies.openvla import OpenVLAPolicy; print('OpenVLA import OK')"

# OpenVLA 평가 (최소 테스트)
python src/eval_simpler.py --model "openvla/openvla-7b" --n-episodes 1 --max-steps 100

# OpenVLA trajectory 수집
python src/collect_trajectories.py --model "openvla/openvla-7b" --n-per-task 1 --max-steps 100
```

#### 3. Octo 모델 테스트 (의존성 충돌 가능)
```bash
# Octo-small 모델 평가 (자동 다운로드)
python src/eval_simpler.py --model octo-small --n-episodes 4

# Octo-base 모델 평가 (더 큰 모델)
python src/eval_simpler.py --model octo-base --n-episodes 4
```

#### 4. RT-1 모델 테스트 (체크포인트 필요)
```bash
# RT-1 체크포인트 다운로드 (별도 수행 필요)
# gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .

# RT-1 평가
python src/eval_simpler.py --model /path/to/rt1_checkpoint --n-episodes 4
```

#### 5. Trajectory 수집
```bash
# SimplePolicy로 trajectory 수집 (테스트)
python src/collect_trajectories.py --model simple --n-per-task 2

# Octo로 성공 trajectory 수집
python src/collect_trajectories.py --model octo-small --n-per-task 25
```

## 명령어 옵션

### eval_simpler.py
- `--model`: 모델 타입 또는 경로 (simple, octo-small, octo-base, rt1_path)
- `--n-episodes`: 평가할 에피소드 수 (기본: 24)
- `--max-steps`: 에피소드당 최대 스텝 (기본: 300)
- `--output`: 결과 저장 디렉토리 (기본: ./data/results)

### collect_trajectories.py
- `--model`: 모델 타입 또는 경로
- `--n-per-task`: Task당 수집할 trajectory 수 (기본: 25)
- `--max-steps`: 에피소드당 최대 스텝 (기본: 300)
- `--output`: Trajectory 저장 디렉토리 (기본: ./data/trajectories)

## 출력 파일

### 평가 결과 (JSON)
```
data/results/
└── octo-small_20241209_143022.json
    {
      "PutSpoonOnTableClothInScene-v1": 0.25,
      "PutCarrotOnPlateInScene-v1": 0.30,
      "StackGreenCubeOnYellowCubeBakedTexInScene-v1": 0.15,
      "PutEggplantInBasketScene-v1": 0.20,
      "average": 0.225
    }
```

### Trajectory 데이터 (Pickle)
```
data/trajectories/
└── octo-small_100trajs_20241209_143022.pkl
    [
      {
        "task": "...",
        "observations": [...],
        "actions": [...],
        "rewards": [...],
        "language_instruction": "...",
        "success": true,
        "total_reward": 1.0
      },
      ...
    ]
```

## 트러블슈팅

### 1. CUDA/GPU 메모리 부족
```bash
# 메모리 제한 설정
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### 2. 모델 다운로드 실패
- 인터넷 연결 확인
- Hugging Face 토큰 설정 필요할 수 있음

### 3. ImportError
```bash
# 필요한 패키지 설치
pip install simpler-env
pip install mani_skill3
```

### 4. 느린 실행 속도
- `--n-episodes` 줄이기
- `--max-steps` 줄이기
- GPU 사용 확인: `nvidia-smi`

## 예상 성능 (참고)

| Model | Average Success Rate | Notes |
|-------|---------------------|-------|
| Random | ~0% | 랜덤 액션 |
| SimplePolicy | 0-5% | 스크립트 정책 |
| OpenVLA | 20-40% | 7B VLA 모델 (Zero-shot) |
| Octo-small | 15-30% | 경량 모델 |
| Octo-base | 20-40% | 표준 모델 |
| RT-1 | 30-50% | Google 모델 |

## 다음 단계

1. **베이스라인 성능 기록**: 각 모델의 zero-shot 성능 측정
2. **Trajectory 수집**: 성공률 높은 모델로 100개 trajectory 수집
3. **Fine-tuning 준비**: 수집된 데이터로 모델 학습
4. **비교 평가**: Fine-tuning 전후 성능 비교