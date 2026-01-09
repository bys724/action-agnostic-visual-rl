# 체크포인트 관리 가이드

## 개요
모델 체크포인트를 효율적으로 관리하기 위한 시스템입니다. HuggingFace 모델을 로컬에 저장하고, 오프라인에서도 사용할 수 있습니다.

## 디렉토리 구조
```
data/checkpoints/
├── registry.json          # 모든 체크포인트 레지스트리
├── openvla/               # OpenVLA 모델들
│   ├── openvla-7b/       # 기본 7B 모델
│   └── openvla-custom/    # 커스텀 fine-tuned 모델
├── lapa/                  # LAPA 모델들
└── custom/                # 사용자 정의 모델들
```

## 사용 방법

### 1. 모델 다운로드 및 저장

```bash
# OpenVLA 7B 모델 다운로드
python scripts/download_model.py download openvla/openvla-7b --type openvla

# 커스텀 모델 다운로드
python scripts/download_model.py download your-hf-id/your-model --type custom

# 로컬 모델 목록 확인
python scripts/download_model.py list
```

### 2. 로컬 체크포인트 사용

```bash
# 자동으로 로컬 체크포인트 우선 사용 (기본값)
python src/eval_simpler.py --model "openvla/openvla-7b"

# 강제로 온라인에서 다운로드
python src/eval_simpler.py --model "openvla/openvla-7b" --no-use-local

# 로컬 경로 직접 지정
python src/eval_simpler.py --model "./data/checkpoints/openvla/openvla-7b"
```

### 3. 다중 체크포인트 비교

```json
// configs/checkpoint_comparison.json
{
  "models": [
    {
      "name": "openvla-7b-base",
      "type": "openvla",
      "checkpoint": "./data/checkpoints/openvla/openvla-7b"
    },
    {
      "name": "openvla-7b-finetuned",
      "type": "openvla",
      "checkpoint": "./data/checkpoints/openvla/openvla-7b-finetuned"
    },
    {
      "name": "custom-model-v1",
      "type": "custom",
      "checkpoint": "./data/checkpoints/custom/model-v1"
    }
  ]
}
```

```bash
# 비교 실행
python src/eval_simpler.py --config configs/checkpoint_comparison.json
```

## 장점

1. **오프라인 사용**: 인터넷 없이도 모델 사용 가능
2. **빠른 로딩**: 로컬 파일에서 직접 로드하여 다운로드 시간 절약
3. **버전 관리**: 여러 버전의 체크포인트를 동시에 관리
4. **체계적 구조**: 모델 타입별로 정리된 디렉토리 구조
5. **레지스트리**: 모든 체크포인트를 한 곳에서 관리

## Docker에서 사용

```bash
# Docker 컨테이너 실행 시 체크포인트 마운트
docker compose up -d eval

# 컨테이너 내에서 모델 다운로드
docker exec simpler-dev python scripts/download_model.py download openvla/openvla-7b

# 컨테이너 내에서 평가 실행
docker exec simpler-dev python src/eval_simpler.py --model "openvla/openvla-7b"
```

## 레지스트리 파일 구조

`data/checkpoints/registry.json`:
```json
{
  "models": {
    "openvla": {
      "openvla-7b": {
        "path": "./data/checkpoints/openvla/openvla-7b",
        "added": "2024-01-09T10:00:00"
      }
    },
    "custom": {
      "my-model-v1": {
        "path": "./data/checkpoints/custom/my-model-v1", 
        "added": "2024-01-09T11:00:00"
      }
    }
  }
}
```

## 커스텀 체크포인트 추가

Fine-tuning 후 저장한 체크포인트를 추가하려면:

```bash
# 1. 체크포인트를 적절한 위치에 복사
cp -r /path/to/your/checkpoint ./data/checkpoints/custom/my-model

# 2. 레지스트리 업데이트 (자동)
python scripts/download_model.py download my-model --type custom --save-dir ./data/checkpoints

# 3. 사용
python src/eval_simpler.py --model "./data/checkpoints/custom/my-model"
```

## 주의사항

- 체크포인트는 용량이 크므로 (OpenVLA-7B는 약 14GB) 충분한 디스크 공간 확보
- `registry.json`은 버전 관리에 포함시켜 팀원들과 공유
- 실제 체크포인트 파일들은 `.gitignore`에 추가하여 git에서 제외
- Docker 볼륨으로 마운트되므로 호스트와 컨테이너 간 공유됨