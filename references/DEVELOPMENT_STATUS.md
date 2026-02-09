# Development Status & Next Steps

**Date**: 2026-02-09
**Last Updated**: After Decoder Design Q&A session

---

## 📊 Current Implementation Status

### ✅ Completed

**Model Architecture** (`src/models/two_stream.py`):
- ✅ Two-Stream Preprocessing (M: 4ch, P: 5ch)
- ✅ InterleavedTwoStreamViT (CLS Exchange, 3 stages)
- ✅ PixelwiseFusion
- ⚠️ VideoDecoder (현재: patches만 사용, **업데이트 필요**)

**Training**:
- ✅ BridgeDataset implementation
- ✅ EgoDexDataset implementation
- ✅ Multi-gap sampling (1-10 frames)
- ✅ Training pipeline with DataParallel

**Checkpoints**:
- Latest: `data/checkpoints/two_stream/20260202_085022/`
- Progress: 46/350 epochs (13%)
- Dataset: Bridge V2 (1.24M train, 324K eval)
- Best eval loss: 0.0000379

---

## 🎯 Current Task: Decoder Update

### Problem

**Current decoder** (line 578-632):
```python
# Patches만 사용, 이전 이미지 없음
def forward(self, patches):
    x = self.input_proj(patches)
    # ... upsampling ...
    return img_pred
```

**Issues**:
1. 모든 decoder level에 P_CLS_final만 반복 주입
2. Encoder는 각 stage마다 진화하는 CLS 사용 (불일치)
3. Skip connection 없음

### Solution: Intermediate CLS + Skip

```python
# Encoder: Save intermediate CLS
P_CLS_stage1 = layer4_output[:, 0]   # Low-level
P_CLS_stage2 = layer8_output[:, 0]   # Mid-level
P_CLS_final  = layer12_output[:, 0]  # High-level

# Decoder: Multi-scale injection
Level 1 (14→28):   P_CLS_final    # Abstract
Level 2 (28→56):   P_CLS_stage2   # Mid
Level 3 (56→112):  P_CLS_stage1   # Concrete
Level 4 (112→224): No CLS         # Detail

# Skip from img_t (56x56)
```

---

## 🛠️ Required Changes

### 1. InterleavedTwoStreamViT.forward()

**Location**: Line ~450

**Change**:
```python
# Add at the end of forward()
p_cls_intermediates = {}

for stage in range(self.num_stages):
    # ... process blocks ...

    # Save intermediate CLS
    if stage == 0:
        p_cls_intermediates['stage1'] = p_tokens[:, 0].clone()
    elif stage == 1:
        p_cls_intermediates['stage2'] = p_tokens[:, 0].clone()

p_cls_intermediates['final'] = p_tokens[:, 0].clone()

return m_tokens, p_tokens, p_cls_intermediates  # NEW!
```

### 2. VideoDecoder (Complete Redesign)

**Location**: Line ~578

**New structure**:
- Input: `patches`, `img_t`, `p_cls_intermediates`
- Skip connection from `img_t` (downsampled to 56x56)
- CLS injection via FiLM-like modulation
- 4 upsampling levels with appropriate CLS

**Key methods**:
```python
def inject_cls(self, feature_map, cls_token, proj_layer):
    """FiLM-like CLS injection."""
    cls_proj = proj_layer(cls_token).view(B, C, 1, 1)
    return feature_map + cls_proj
```

### 3. TwoStreamEncoder.forward()

**Location**: Line ~533

**Change**:
```python
# Return intermediate CLS
return cls_fused, patches_fused, p_cls_intermediates
```

### 4. TwoStreamVideoPredictor.forward()

**Location**: Line ~655

**Change**:
```python
cls_emb, patches, p_cls_intermediates = self.encoder(img_t, img_tk)
img_pred = self.decoder(patches, img_t, p_cls_intermediates)
```

---

## 📚 Key Design Rationale

### Q: "정보를 너무 많이 주면 당연히 잘 되는 것 아닌가?"

**A**:
1. **Pretraining 목적**: Decoder 성능 (X) → Encoder representation 품질 (O)
2. **Task difficulty**: 너무 어려움 (불안정) vs 적절함 (안정) vs 너무 쉬움 (trivial)
3. **Skip ≠ 정답**: U-Net/ResNet처럼 gradient flow 개선용
4. **최종 검증**: LIBERO downstream task에서 encoder만 사용했을 때

**Reference**: 메인 메모 Section 10 "Decoder Design: Intermediate CLS Injection"

---

## 🧪 Testing Plan

### Quick Test
```bash
cd /workspace
docker exec simpler-eval python3 -c "
from src.models.two_stream import TwoStreamVideoPredictor
import torch

model = TwoStreamVideoPredictor(encoder_type='interleaved').cuda()
img_t = torch.rand(2, 3, 224, 224).cuda()
img_tk = torch.rand(2, 3, 224, 224).cuda()

img_pred, cls_emb = model(img_t, img_tk)
print(f'✓ Forward: {img_pred.shape}')

loss = torch.nn.functional.mse_loss(img_pred, img_tk)
loss.backward()
print('✓ Backward OK')
"
```

### Training Test (BridgeV2)
```bash
docker exec simpler-eval python3 src/models/two_stream.py --test train --epochs 1 --batch-size 8
```

---

## 📋 Next Steps After Decoder Update

1. **Sanity check**: Loss가 줄어드는지 확인
2. **Component ablation**:
   - A: Patches만 (no skip, no CLS)
   - B: + img_t only
   - C: + img_t + final CLS
   - D: + img_t + intermediate CLS + skip (현재 구현)
3. **EgoDex pretraining**: Bridge V2 → EgoDex로 전환
4. **LIBERO transfer**: Stage 3 실험

---

## 🔗 Key References

- **Main memo**: `references/논문 - Action-Agnostic Visual Behavior Representation.md`
  - Section 10: Decoder Design Q&A
  - Section "실험 계획": Stage 0-3
  - Section "실험으로 검증 가능한 주장": 무엇을 검증할 수 있는지

- **Concept notes**:
  - `Pixel-wise Channel Fusion for Behavior Representation.md`
  - `Two-Stream Image Preprocessing.md`

---

## ⚠️ Important Notes

1. **Breaking change**: Old checkpoints incompatible
2. **Dataset**: Currently Bridge V2, should switch to EgoDex
3. **Strategy**: Start simple (baseline) → add components → ablate
4. **Final goal**: LIBERO transfer, not pretraining loss
