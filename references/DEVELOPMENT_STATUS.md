# Development Status & Next Steps

**Date**: 2026-02-09
**Last Updated**: After Decoder Update implementation

---

## Current Implementation Status

### Completed

**Model Architecture** (`src/models/two_stream.py`):
- Two-Stream Preprocessing (M: 4ch, P: 5ch)
- InterleavedTwoStreamViT (CLS Exchange, 3 stages)
- PixelwiseFusion
- VideoDecoder (Skip + Multi-scale CLS injection)

**Decoder Features**:
- Skip connection from img_t (downsampled to 56x56)
- Multi-scale CLS injection via FiLM-like modulation
  - Level 1 (14->28): final CLS (abstract)
  - Level 2 (28->56): stage2 CLS + skip merge
  - Level 3 (56->112): stage1 CLS (concrete)
  - Level 4 (112->224): No CLS (detail)
- Total params: 192.3M (Encoder: 188.7M, Decoder: 3.6M)

**Training**:
- BridgeDataset implementation
- EgoDexDataset implementation
- Multi-gap sampling (1-10 frames)
- Training pipeline with DataParallel

---

## Current Training Status

**New Training (post-decoder update)**:
- Checkpoint: `data/checkpoints/two_stream_test/`
- Dataset: BridgeV2 (24,828 trajectories)
- Eval: EgoDex test (3,243 videos)
- Loss: ~0.0017-0.0024 (stable)

**Note**: Old checkpoints incompatible due to architecture change.

---

## Next Steps

### Phase 1: Vision Encoder Pretraining (in progress)

1. **Short test** (2 epochs) - Verify training stability
2. **Full training** (350 epochs) with new decoder
3. **Ablation study**:
   - A: Patches only (no skip, no CLS) - baseline
   - B: + img_t skip only
   - C: + img_t + final CLS
   - D: + img_t + intermediate CLS (current)

### Phase 2: Encoder Replacement Experiments

| Experiment | Baseline | Target |
|------------|----------|--------|
| OpenVLA + Our Encoder | 65% | 75% (+10%) |
| Pi0 + Our Encoder | 68% | 77% (+9%) |

### Phase 3: Validation Experiments

1. **Optical Flow Probe**: Linear probe for flow prediction
2. **Temporal Order Classification**: Frame pair ordering
3. **Action Clustering**: t-SNE visualization of CLS embeddings

---

## Quick Commands

### Test Model
```bash
docker exec simpler-eval python3 -c "
from src.models.two_stream import TwoStreamVideoPredictor
import torch

model = TwoStreamVideoPredictor(encoder_type='interleaved').cuda()
img_t = torch.rand(2, 3, 224, 224).cuda()
img_tk = torch.rand(2, 3, 224, 224).cuda()

img_pred, cls_emb = model(img_t, img_tk)
print(f'Forward: {img_pred.shape}, {cls_emb.shape}')

loss = torch.nn.functional.mse_loss(img_pred, img_tk)
loss.backward()
print('Backward OK')
"
```

### Train Model
```bash
docker exec -e CUDA_VISIBLE_DEVICES=1 simpler-eval python /workspace/scripts/train_long.py \
    --epochs 350 --batch-size 32 \
    --checkpoint-dir /workspace/data/checkpoints/two_stream \
    --train-data bridge --bridge-root /workspace/data/datasets/bridge_v2
```

### Resume Training
```bash
docker exec -e CUDA_VISIBLE_DEVICES=1 simpler-eval python /workspace/scripts/train_long.py \
    --resume /workspace/data/checkpoints/two_stream/<timestamp>/latest.pt
```

---

## Key References

- **Main memo**: `references/논문 - Action-Agnostic Visual Behavior Representation.md`
- **Experiment plan**: `docs/research/EXPERIMENT_PLAN.md`
