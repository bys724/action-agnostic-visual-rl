# Action-Agnostic Visual RL Research Plan

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?

---

## Phase 1: EgoDex Part1 Pretraining (현재)

### 목표
- EgoDex part1 (~336GB)에서 3가지 모델 사전학습
- 학습 안정성 및 수렴 여부 확인
- Action probing 준비

### 모델

| 모델 | 학습 방식 | 목적 |
|-----|----------|------|
| Two-Stream | Future prediction | 제안 모델 (M/P 채널) |
| Single-Stream | Future prediction | Baseline |
| VideoMAE | Masked reconstruction (75%) | Comparison |

### 학습 설정

```python
# scripts/train_long.py
epochs = 30
batch_size = 24  # g5.12xlarge (4x A10G) → 6/GPU
max_gap = 10
sample_decay = 0.8
loss_decay = 0.5
dataset = "EgoDex part1"
```

### 기대 결과
- 3개 모델 모두 수렴
- Training curve 비교
- Part1만으로 충분한지 검증

---

## Phase 2: Action Probing

### 목표
학습된 표현이 action 정보를 포함하는지 검증

### 방법
1. 인코더 freeze
2. Linear probe로 Bridge V2 action 예측
3. Baseline 모델과 비교

### Baseline 모델
- DINOv2 (ImageNet)
- CLIP (WebData)
- R3M (Ego4D)
- (Optional) RT-2, OpenVLA 인코더

### 평가 지표
- Action prediction accuracy
- Task success rate
- Generalization to unseen tasks

---

## Phase 3: Full EgoDex Training (미래)

### 데이터셋
- EgoDex part1-5 (1.84TB)
- 더 많은 task diversity

### 개선 사항
- Spatial augmentation 추가
  - Random crop
  - Color jitter
  - Horizontal flip
- Extended training (100 epochs)
- Larger batch size

---

## Phase 4: LIBERO Integration (미래)

### 목표
OpenVLA/Pi0 인코더를 학습된 표현으로 대체

### 실험
1. OpenVLA vision encoder → Two-Stream
2. Pi0 vision encoder → Two-Stream
3. LIBERO 4개 suite에서 fine-tuning
4. 원본 모델과 성능 비교

---

## 핵심 구현 결정

### 모델별 학습 방식

**Two-Stream/Single-Stream**:
```python
# Future prediction with gap weighting
img_pred, change_emb = model(img_t, img_tk)
loss = F.mse_loss(img_pred, img_tk)
weighted_loss = (loss * gap_weights).mean()
```

**VideoMAE**:
```python
# Official masked reconstruction
loss, img_pred = model.compute_loss(img_t, img_tk)  # 75% masking
```

**이유**: 각 SSL 방법론의 best practice 유지 → 공정한 비교

### AWS Infrastructure

**Instance**: g5.12xlarge
- 4x NVIDIA A10G
- 48 vCPU, 192GB RAM
- us-west-2 (Oregon)

**Storage**:
- EBS 1TB (code, checkpoints, logs)
- S3 (datasets, backup)

**Cost Optimization**:
- Auto-shutdown after training
- S3 sync every 10min
- On-demand (Spot quota = 0)

---

## 현재 상태 (2026-02-28)

### 완료
- [x] Two-Stream 구현
- [x] Single-Stream 구현
- [x] VideoMAE wrapper 구현
- [x] EgoDex part1 S3 업로드 완료
- [x] Training script 작성 (`scripts/train_long.py`)
- [x] AWS 학습 스크립트 (`scripts/run_aws_training.sh`)
- [x] Part1 skip logic 추가 (불필요한 1.84TB 다운로드 방지)

### 진행 중
- [ ] 새 인스턴스 시작 (EBS 1TB)
- [ ] Single-stream 학습 시작
- [ ] Two-stream 학습
- [ ] VideoMAE 학습

### 다음 단계
- [ ] Action probing 코드 작성
- [ ] Baseline 모델 다운로드
- [ ] Bridge V2 action probing
- [ ] 결과 분석 및 논문 작성

---

## 실험 검증 계획

### 1. 움직임 정보 인코딩 검증

| 실험 | 방법 | 성공 기준 |
|-----|------|----------|
| Optical Flow Probe | Linear probe로 flow 예측 | DINOv2 대비 20%+ 낮은 EPE |
| Temporal Order | 프레임 순서 분류 | 90%+ 정확도 |
| Action Clustering | CLS 임베딩 t-SNE | Task별 클러스터 형성 |

### 2. Ablation Study

| 변형 | 설명 |
|-----|------|
| M-only | P 채널 제거 |
| P-only | M 채널 제거 |
| No Exchange | CLS 교환 없이 concat만 |
| Early Fusion | 6ch 단일 ViT |

---

## Timeline

### Week 1-2: Part1 Pretraining
- 3개 모델 학습 완료
- Training curve 분석
- Checkpoint S3 저장

### Week 3-4: Action Probing
- Linear probe 구현
- Bridge V2 실험
- Baseline 비교

### Week 5-6: Analysis
- 결과 분석
- 논문 초안 작성
- 추가 실험 계획

---

## Notes

- Phase 1은 feasibility check - 빠른 iteration 우선
- Spatial augmentation은 Phase 3에서 추가
- 문서는 실험 진행에 따라 업데이트

**Last Updated**: 2026-02-28
