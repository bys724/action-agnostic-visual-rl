# 다중 모델 평가 가이드

## 명령어 형식 설명

### --models 옵션 형식: `"name:type:checkpoint"`

- **name**: 결과에 표시될 모델 식별자
- **type**: 모델 아키텍처 타입 (openvla, lapa, custom)
- **checkpoint**: 모델 경로 또는 HuggingFace ID

## 사용 예시

### 1. 단일 OpenVLA 모델 평가
```bash
# 기본 HuggingFace 모델
python src/eval_simpler.py \
    --model "openvla/openvla-7b" \
    --n-episodes 24 \
    --max-steps 300
```

### 2. 여러 체크포인트 비교
```bash
# 베이스 모델과 fine-tuned 모델 비교
python src/eval_simpler.py \
    --models "base:openvla:openvla/openvla-7b" \
             "ft-v1:openvla:/checkpoints/openvla-ft-epoch10" \
             "ft-v2:openvla:/checkpoints/openvla-ft-epoch20" \
    --n-episodes 24
```

결과 예시:
```
COMPARISON TABLE
================================================================================
      model  avg_success  Task1_success  Task2_success  Task3_success  Task4_success
       base        25.0%          20.0%          30.0%          15.0%          35.0%
      ft-v1        45.0%          40.0%          50.0%          35.0%          55.0%
      ft-v2        52.0%          48.0%          56.0%          45.0%          59.0%
```

### 3. JSON 설정 파일 사용
```bash
# configs/experiment.json 작성
{
  "models": [
    {
      "name": "openvla-base",
      "type": "openvla",
      "checkpoint": "openvla/openvla-7b"
    },
    {
      "name": "openvla-droid-ft",
      "type": "openvla", 
      "checkpoint": "/workspace/checkpoints/openvla-droid"
    },
    {
      "name": "lapa-7b",
      "type": "lapa",
      "checkpoint": "lapa/lapa-7b"
    }
  ]
}

# 실행
python src/eval_simpler.py --config configs/experiment.json
```

### 4. 특정 태스크만 평가
```bash
python src/eval_simpler.py \
    --models "model1:openvla:path1" "model2:openvla:path2" \
    --tasks "PutSpoonOnTableClothInScene-v1" "PutCarrotOnPlateInScene-v1" \
    --n-episodes 10
```

## 출력 파일

### 개별 모델 결과 (JSON)
```
data/results/
├── openvla-base_20241209_143022.json
├── openvla-ft-v1_20241209_143122.json
└── openvla-ft-v2_20241209_143222.json
```

### 비교 결과 (CSV)
```
data/results/
└── comparison_20241209_143022.csv
```

CSV 형식:
```csv
model,type,checkpoint,avg_success,Task1_success,Task1_reward,Task1_steps,...
base,openvla,openvla/openvla-7b,0.25,0.20,0.8,150,...
ft-v1,openvla,/checkpoints/v1,0.45,0.40,1.2,120,...
```

## Trajectory 수집

### 단일 모델로 수집
```bash
python src/collect_trajectories.py \
    --type openvla \
    --checkpoint "openvla/openvla-7b" \
    --name "openvla-base" \
    --n-per-task 25
```

### Fine-tuned 모델로 수집
```bash
python src/collect_trajectories.py \
    --type openvla \
    --checkpoint "/workspace/checkpoints/best_model" \
    --name "openvla-best" \
    --n-per-task 50 \
    --tasks "PutSpoonOnTableClothInScene-v1"
```

## 실제 OpenVLA 사용 준비

### 하드웨어 요구사항
- GPU: 24GB+ VRAM (RTX 3090, A5000, A100)
- RAM: 32GB+
- Storage: 20GB+ (모델 파일)

### 소프트웨어 설정
```bash
# Docker 컨테이너 내에서
./scripts/setup_openvla.sh

# 또는 수동 설정
pip install --upgrade transformers>=4.45.0 accelerate>=0.30.0
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
```

### 모델 사전 다운로드 (선택)
```bash
# 첫 실행 전 모델 다운로드 (약 15GB)
python -c "
from transformers import AutoProcessor, AutoModelForCausalLM
AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
print('Model files downloaded')
"
```

## 성능 최적화

### 1. Flash Attention 2 사용
```bash
pip install flash-attn --no-build-isolation
# 모델 로드 시 자동으로 사용됨
```

### 2. 배치 처리
```python
# 여러 에피소드를 동시에 처리하려면
--n-episodes 1  # GPU 메모리 절약
--n-episodes 4  # 충분한 메모리가 있을 때
```

### 3. Mixed Precision
```python
# bfloat16 사용 (기본값)
# float16보다 안정적
```

## 문제 해결

### CUDA Out of Memory
```bash
# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 또는 더 작은 배치
--n-episodes 1
```

### 모델 로드 실패
```python
# trust_remote_code 확인
# transformers 버전 확인 (>= 4.45.0)
pip install --upgrade transformers
```

### 느린 추론 속도
```bash
# Flash Attention 설치
# GPU 사용 확인
nvidia-smi
```