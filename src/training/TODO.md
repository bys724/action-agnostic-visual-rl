# Training Module TODO

## 구현 완료

### Pre-training (Self-supervised)
- ✅ `pretrain.py`: Pre-training 학습 루프
  - Future frame prediction (Two-Stream, Single-Stream)
  - Masked autoencoding (VideoMAE)
  - Multi-gap sampling
  - Gap-weighted loss
  - Checkpoint management
  - Multi-GPU support (DataParallel)

## 구현 필요 (Fine-tuning)

### Fine-tuning (Supervised)
- ⬜ `finetune.py`: Fine-tuning 학습 루프
  - Action prediction 지원
  - Pre-trained encoder 로딩 (frozen/unfrozen)
  - Action head 학습
  - Bridge V2 / LIBERO 데이터셋 호환

**참고 구현**: `scripts/eval/finetune_libero.py`
- 현재 LIBERO fine-tuning 코드가 독립적으로 구현되어 있음
- 이를 재사용 가능한 형태로 `src/training/finetune.py`에 통합 필요

### 구현 시 고려사항

#### 1. 데이터 형식
```python
# Pre-training
batch = (image_current, image_future, gap)  # 비디오만

# Fine-tuning
batch = (image_current, image_future, action, ...)  # 비디오 + 행동
```

#### 2. 모델 구조
```python
# Pre-training: 전체 모델
model = TwoStreamModel()
loss, pred_image = model.compute_loss(img_current, img_future)

# Fine-tuning: Encoder + Action head
encoder = TwoStreamEncoder(checkpoint_path="...")
action_head = ActionHead(input_dim=encoder.embed_dim, action_dim=7)
action_pred = action_head(encoder(images))
loss = F.mse_loss(action_pred, action_gt)
```

#### 3. 학습 루프 차이
**Pre-training:**
- Loss: Image reconstruction (MSE)
- Metric: Pixel-level accuracy
- Evaluation: Visual quality

**Fine-tuning:**
- Loss: Action prediction (MSE or Cross-entropy)
- Metric: Action accuracy, Success rate
- Evaluation: Task performance (LIBERO success rate)

### 인터페이스 설계 (제안)

```python
# src/training/finetune.py
def train(
    encoder,                  # Pre-trained encoder
    action_head,              # Action prediction head
    train_dataset,            # Dataset with actions
    num_epochs=100,
    batch_size=32,
    lr=1e-4,
    freeze_encoder=False,     # Freeze encoder weights
    device="cuda",
    checkpoint_dir=None,
    eval_dataset=None,
    **kwargs
):
    """Fine-tuning loop for action prediction."""
    pass

def train_epoch(...):
    """One epoch of fine-tuning."""
    pass

def evaluate(...):
    """Evaluate action prediction accuracy."""
    pass
```

### 사용 예시 (미래)

```bash
# Pre-training (현재 구현됨)
python scripts/pretrain.py --model two-stream --epochs 100

# Fine-tuning (TODO)
python scripts/finetune.py \
    --encoder two-stream \
    --checkpoint data/checkpoints/two_stream/latest.pt \
    --dataset libero \
    --epochs 50
```

## 참고 문서

- `scripts/eval/finetune_libero.py`: 현재 LIBERO fine-tuning 구현
- `docs/PROBING_GUIDE.md`: Action probing 실험 (관련 작업)
