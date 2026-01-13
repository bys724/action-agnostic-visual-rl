# 체크포인트 관리

## 디렉토리 구조

```
data/checkpoints/
├── openvla/           # OpenVLA 모델
│   └── openvla-7b/
└── lapa/              # LAPA 모델
    └── params/
```

## 사용법

### 모델 다운로드
```bash
# OpenVLA
python scripts/setup_openvla.py

# LAPA (HuggingFace에서 수동 다운로드)
# https://huggingface.co/latent-action-pretraining/LAPA-7B-openx
```

### 평가 시 사용
```bash
# 로컬 체크포인트
docker exec simpler-dev python src/eval_simpler.py \
    --model "./data/checkpoints/openvla/openvla-7b"

# HuggingFace ID
docker exec simpler-dev python src/eval_simpler.py \
    --model "openvla/openvla-7b"
```

## 주의사항
- 체크포인트는 용량이 큼 (OpenVLA-7B: ~14GB)
- `data/` 폴더는 .gitignore에 포함
