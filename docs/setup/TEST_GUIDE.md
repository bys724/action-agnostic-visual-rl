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

### 옵션 1: OpenVLA 설치 및 테스트
```bash
# OpenVLA 전체 설치 (모델 + 의존성 + assets)
python scripts/setup_openvla.py

# 설치 확인
python scripts/setup_openvla.py --verify-only

# OpenVLA 평가 테스트
python src/eval_simpler.py --model "openvla/openvla-7b" --n-episodes 1 --max-steps 10
```

### 옵션 2: 개별 테스트

#### 1. 단일 모델 평가 (레거시 모드)
```bash
# OpenVLA 평가
python src/eval_simpler.py --model "openvla/openvla-7b" --n-episodes 2 --max-steps 100

# 또는 타입 명시
python src/eval_simpler.py --model "openvla-7b" --checkpoint "openvla/openvla-7b" --n-episodes 2
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

#### 2. 다중 모델 비교 평가
```bash
# 커맨드라인에서 직접 설정
python src/eval_simpler.py \
    --models "openvla-base:openvla:openvla/openvla-7b" \
             "openvla-ft:openvla:/path/to/finetuned" \
    --n-episodes 24

# JSON 설정 파일 사용
python src/eval_simpler.py --config configs/eval_example.json
```

#### 3. Trajectory 수집
```bash
# OpenVLA trajectory 수집
python src/collect_trajectories.py \
    --type openvla \
    --checkpoint "openvla/openvla-7b" \
    --n-per-task 25

# 커스텀 모델 trajectory 수집
python src/collect_trajectories.py \
    --type custom \
    --checkpoint "/path/to/model" \
    --name "my-model-v1" \
    --n-per-task 10
```

## 명령어 옵션

### eval_simpler.py (다중 모델 지원)
- `--model`: 단일 모델 경로 (레거시 모드)
- `--models`: 모델 목록 "name:type:checkpoint" 형식
- `--config`: JSON 설정 파일 경로
- `--n-episodes`: 평가할 에피소드 수 (기본: 24)
- `--max-steps`: 에피소드당 최대 스텝 (기본: 300)
- `--output`: 결과 저장 디렉토리 (기본: ./data/results)

### collect_trajectories.py
- `--type`: 모델 타입 (openvla, lapa, custom)
- `--checkpoint`: 체크포인트 경로 또는 ID
- `--name`: 출력 파일용 모델 이름 (선택)
- `--n-per-task`: Task당 수집할 trajectory 수 (기본: 25)
- `--max-steps`: 에피소드당 최대 스텝 (기본: 300)
- `--output`: Trajectory 저장 디렉토리 (기본: ./data/trajectories)

## 출력 파일

### 평가 결과 (JSON)
```
data/results/
└── openvla_20241209_143022.json
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
└── openvla_100trajs_20241209_143022.pkl
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
| OpenVLA-7B | 20-40% | Zero-shot 성능 |
| OpenVLA-7B-FT | 40-60% | Fine-tuned (예상) |
| LAPA-7B | 30-50% | Action-free 사전학습 (예상) |
| Custom Model | TBD | 개발 중 |

## 다음 단계

1. **베이스라인 성능 기록**: 각 모델의 zero-shot 성능 측정
2. **Trajectory 수집**: 성공률 높은 모델로 100개 trajectory 수집
3. **Fine-tuning 준비**: 수집된 데이터로 모델 학습
4. **비교 평가**: Fine-tuning 전후 성능 비교