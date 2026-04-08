# Action-Agnostic Visual Representation Learning

**연구 질문**: 행동 정보 없이 학습한 시각적 표현이 다양한 로봇과 작업에 더 잘 일반화되는가?

## 핵심 아이디어

1. **EgoDex Pretraining**: 대규모 자기중심 비디오로 행동-독립적 시각 표현 학습
2. **Action Probing**: 학습된 표현이 행동 정보를 인코딩하는지 검증 (EgoDex + DROID)
3. **LIBERO Evaluation**: 로봇 조작 태스크에서 fine-tuning + 시뮬레이터 rollout

## 빠른 시작

### Pre-training

```bash
# Two-Stream v4 (제안 모델)
python scripts/pretrain.py --model two-stream \
    --depth 12 --num-stages 2 \
    --mask-ratio 0.3 --mask-ratio-p 0.5 \
    --max-gap 60 --sample-dist triangular --sample-center 30 \
    --epochs 30

# VideoMAE (baseline)
python scripts/pretrain.py --model videomae --epochs 30
```

### Action Probing

```bash
# EgoDex (within-domain)
python scripts/eval/probe_action.py \
    --encoder two-stream --checkpoint <ckpt> \
    --gap 10 --cls-mode patch_mean_concat

# DROID (cross-domain)
python scripts/eval/probe_action_droid.py \
    --encoder two-stream --checkpoint <ckpt> --gap 10
```

### LIBERO Evaluation

```bash
# Fine-tuning
python scripts/eval/finetune_libero.py --encoder two-stream --checkpoint <ckpt>

# 시뮬레이터 rollout
docker compose up -d libero
docker exec libero-eval python src/eval_libero.py --encoder two-stream
```

## 모델

| 모델 | 설명 | 역할 |
|------|------|------|
| **Two-Stream** | M/P 채널 분리 + CLS exchange + 2D RoPE + MAE masking | 제안 모델 |
| VideoMAE | Masked autoencoder | Comparison baseline |

## 프로젝트 구조

```
├── src/
│   ├── models/          # Two-Stream, VideoMAE
│   ├── datasets/        # EgoDex, Bridge V2, DROID
│   ├── training/        # Pre-training 루프
│   └── eval_libero.py   # LIBERO 시뮬레이터 평가
├── scripts/
│   ├── pretrain.py      # Pre-training 메인
│   ├── eval/            # Probing, fine-tuning, 시각화
│   └── data/            # 데이터 전처리
├── docs/
│   ├── RESEARCH_PLAN.md  # 연구 계획 (마스터 문서)
│   ├── PROBING_GUIDE.md  # Probing 가이드 + 결과
│   └── setup/            # LIBERO 평가 가이드
└── data/                 # 데이터셋, 체크포인트 (gitignore)
```

## 문서

- **연구 계획**: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)
- **개발 가이드**: [`CLAUDE.md`](CLAUDE.md)
- **Probing**: [`docs/PROBING_GUIDE.md`](docs/PROBING_GUIDE.md)
- **LIBERO 평가**: [`docs/setup/LIBERO_TEST_GUIDE.md`](docs/setup/LIBERO_TEST_GUIDE.md)

## 진행 상황 (2026-04-08)

**Phase 1 완료** — 모든 ablation 종료, v4 설정 확정
- [x] Two-Stream / VideoMAE 구현
- [x] Architecture ablation (depth/stages)
- [x] MAE masking + P self-sufficiency 해결
- [x] Gap 분포 실험 (triangular)
- [x] Composition consistency 실험 (효과 없음, 제외)
- [ ] Full training (8x H100): EgoDex part1~5
- [ ] DROID action probing (cross-domain)
- [ ] LIBERO fine-tuning + rollout
