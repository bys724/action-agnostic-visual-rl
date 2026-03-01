# Action-Agnostic Visual Representation Learning

**연구 질문**: 행동 정보 없이 학습한 시각적 표현이 다양한 로봇과 작업에 더 잘 일반화되는가?

## 핵심 아이디어

1. **EgoDex Pretraining**: 대규모 자기중심 비디오로 행동-독립적 시각 표현 학습
2. **Action Probing**: 학습된 표현이 행동 정보를 인코딩하는지 검증
3. **LIBERO Evaluation**: 로봇 조작 태스크에서 성능 평가

## 빠른 시작

### EgoDex Pretraining (AWS)

```bash
# AWS EC2 인스턴스에서
./scripts/run_aws_training.sh  # 3개 모델 순차 학습
```

### LIBERO Evaluation (Docker)

```bash
docker compose up -d libero openvla-libero
docker exec libero-eval python src/eval_libero.py --model openvla
```

## 모델

| 모델 | 설명 | 목적 |
|------|------|------|
| Two-Stream | M/P 채널 분리 + CLS 교환 | 제안 모델 |
| Single-Stream | 단일 RGB 스트림 | Baseline |
| VideoMAE | Masked reconstruction | Comparison |

## 프로젝트 구조

```
├── src/
│   ├── models/          # 모델 구현
│   └── eval_libero.py   # LIBERO 평가
├── scripts/
│   ├── train_long.py    # EgoDex 학습
│   └── run_aws_training.sh
├── docs/
│   ├── RESEARCH_PLAN.md       # 연구 계획
│   └── AWS_INSTANCE_GUIDE.md  # AWS 설정
└── data/                # 데이터셋, 체크포인트 (gitignore)
```

## 문서

- **개발 가이드**: [`CLAUDE.md`](CLAUDE.md) - AI 어시스턴트용 개발 가이드
- **연구 계획**: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)
- **AWS 가이드**: [`docs/AWS_INSTANCE_GUIDE.md`](docs/AWS_INSTANCE_GUIDE.md)
- **LIBERO 평가**: [`docs/setup/LIBERO_TEST_GUIDE.md`](docs/setup/LIBERO_TEST_GUIDE.md)

## 현재 진행 상황 (2026-03-01)

- [x] Two-Stream/Single-Stream/VideoMAE 구현
- [x] AWS 학습 파이프라인 구축
- [x] LIBERO 평가 환경 구축
- [ ] EgoDex part1 사전학습 (진행 중)
- [ ] Action probing 실험
- [ ] LIBERO fine-tuning & 평가

---

> 이 프로젝트는 로봇 조작에서 범용적인 시각 표현을 학습하는 새로운 접근 방식을 탐구합니다.
