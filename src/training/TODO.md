# Training Module TODO

## 구현 완료

### Pre-training (Self-supervised)
- ✅ `pretrain.py`: Pre-training 학습 루프
  - Future frame prediction (Two-Stream, Single-Stream)
  - Masked autoencoding (VideoMAE)
  - Dataset: EgoDex
  - Multi-gap sampling, Gap-weighted loss
  - Checkpoint management, Multi-GPU support

## 구현 필요

### 1. Fine-tuning (Supervised Action Prediction)

#### 1.1 Fine-tuning 학습 루프
- ⬜ `finetune.py`: 재사용 가능한 fine-tuning 구현
  - Pre-trained encoder 로딩 (frozen/unfrozen)
  - Action head 학습
  - LIBERO / Bridge V2 데이터셋 지원
  - 체크포인트 관리

**참고 구현**: `scripts/eval/finetune_libero.py` (현재 LIBERO 전용)

#### 1.2 데이터 형식
```python
# Pre-training
batch = (image_current, image_future, gap)  # 비디오만

# Fine-tuning
batch = (image_current, image_future, action, ...)  # 비디오 + 행동
```

#### 1.3 모델 구조
```python
# Fine-tuning: Encoder + Action head
encoder = TwoStreamEncoder(checkpoint_path="...")
action_head = ActionHead(input_dim=encoder.embed_dim, action_dim=7)
action_pred = action_head(encoder(images))
loss = F.mse_loss(action_pred, action_gt)
```

### 2. LIBERO Pre-training (Self-supervised on LIBERO)

현재는 EgoDex만 pre-training하지만, LIBERO 데이터로도 pre-training 가능:

#### 2.1 시나리오
- ⬜ LIBERO 비디오로 pre-training (행동 라벨 미사용)
- ⬜ Bridge V2 비디오로 pre-training (행동 라벨 미사용)
- 목적: 로봇 도메인 특화 표현 학습 vs 범용 표현 비교

#### 2.2 데이터셋 추가
- ⬜ LIBERODataset (비디오만): `src/datasets/libero.py`
  - 현재 구조: 비디오 + 행동 라벨
  - Pre-training용: 비디오만 추출 (행동 무시)
  - Multi-gap sampling 지원

#### 2.3 실험 설계
비교 실험:
1. EgoDex pre-trained → LIBERO fine-tuned (현재 계획)
2. LIBERO pre-trained → LIBERO fine-tuned (도메인 특화)
3. Bridge pre-trained → LIBERO fine-tuned (로봇 데이터 전이)

### 3. Baseline 비교 실험

#### 3.1 비교 대상
현재 연구 질문: **행동 정보 없이 학습한 표현이 더 범용적인가?**

**우리 모델**:
- EgoDex pre-trained encoder + LIBERO fine-tuned
- Action-agnostic 표현 학습

**Baseline**:
- ⬜ **OpenVLA (LIBERO pre-trained)**: 이미 LIBERO 학습됨
- ⬜ **Pi0 (LIBERO pre-trained)**: 이미 LIBERO 학습됨
- ⬜ **From-scratch**: LIBERO만 학습 (pre-training 없음)
- ⬜ **LIBERO pre-trained**: 우리 모델을 LIBERO로 pre-train (시나리오 2.1)

#### 3.2 평가 스크립트
- 🟡 `src/eval_libero.py`: 현재 OpenVLA만 지원
- ⬜ Pi0 통합 필요
- ⬜ 우리 모델 (Two-Stream, Single-Stream, VideoMAE) 통합
- ⬜ From-scratch baseline 추가

#### 3.3 평가 지표
- Success rate (task 성공률)
- Generalization (seen vs unseen tasks)
- Data efficiency (학습 데이터 양 대비 성능)

### 4. Scripts 추가 필요

#### 4.1 Fine-tuning 스크립트
```bash
# ⬜ scripts/finetune.py
python scripts/finetune.py \
    --encoder two-stream \
    --checkpoint data/checkpoints/two_stream/latest.pt \
    --dataset libero \
    --task libero_spatial \
    --epochs 50

# ⬜ scripts/finetune_aws.sh (AWS 자동화)
```

#### 4.2 Baseline 학습 스크립트
```bash
# ⬜ scripts/train_baseline.py
# From-scratch baseline 학습
python scripts/train_baseline.py --dataset libero --model two-stream
```

#### 4.3 비교 평가 스크립트
```bash
# ⬜ scripts/eval/compare_baselines.py
# 여러 모델 한번에 평가 + 결과 비교
python scripts/eval/compare_baselines.py \
    --models two-stream,single-stream,openvla,pi0 \
    --task libero_spatial
```

## 구현 우선순위

### Phase 1 (현재): EgoDex Pre-training
- ✅ EgoDex pre-training 완료 중

### Phase 2: Action Probing
- ✅ 계획 수립됨 (`docs/PROBING_GUIDE.md`)
- ⬜ 실행 대기

### Phase 3: LIBERO Fine-tuning & Evaluation
**우선순위 1**: Fine-tuning 구현
1. ⬜ `src/training/finetune.py` 작성
2. ⬜ `scripts/finetune.py` 작성
3. ⬜ 우리 모델로 LIBERO fine-tuning 실행

**우선순위 2**: Baseline 통합
4. ⬜ `src/eval_libero.py`에 Pi0, 우리 모델 추가
5. ⬜ From-scratch baseline 학습
6. ⬜ 비교 평가 실행

**우선순위 3** (선택): LIBERO Pre-training
7. ⬜ LIBERO/Bridge로 pre-training (도메인 특화 실험)

## 참고 문서

- `scripts/eval/finetune_libero.py`: 현재 LIBERO fine-tuning 구현
- `src/eval_libero.py`: LIBERO 평가 (OpenVLA만)
- `docs/PROBING_GUIDE.md`: Action probing 실험
- `docs/RESEARCH_PLAN.md`: 전체 연구 계획
