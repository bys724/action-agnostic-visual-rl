# Development Status & Next Steps

**Date**: 2026-02-09
**Last Updated**: After Experimental Strategy Discussion

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

## 🎯 Current Task: Baseline Implementation & Code Completion

### Goal

**H100 Phase (~3 days)**: 모든 실험 코드 완성 + 빠른 검증

**필요한 구현**:
1. VideoMAE baseline (masked reconstruction)
2. Single-stream baseline (future prediction, P-stream only)
3. Two-stream (future prediction, ours) - 기존 코드 업데이트
4. OpenVLA encoder 교체 코드
5. Bridge V2 짧은 학습 + LIBERO quick test

---

## 🛠️ H100 Phase: Required Implementation

### Task 1: Baseline Implementation

**1.1 VideoMAE Baseline**
```python
# GitHub: MCG-NJU/VideoMAE
# Architecture: Video ViT with masked reconstruction
# Task: Masked patch prediction (90% masking)
# Output: Reconstruction loss
```

**1.2 Single-Stream Baseline**
```python
# P-stream only (spatial structure)
# Reuse Two-Stream encoder code with flag
# Task: Future frame prediction
# Output: img_t+k from img_tk only
```

**1.3 Two-Stream (Update)**
```python
# M-stream + P-stream
# Update decoder: intermediate CLS + skip connection
# Task: Future frame prediction
# Output: img_t+k from img_t + img_tk
```

### Task 2: Short Pretraining (Bridge V2, 5-10 epochs)

```bash
python train.py --method videomae --epochs 10
python train.py --method single_stream --epochs 10
python train.py --method two_stream --epochs 10
```

### Task 3: Quick LIBERO Test

**3.1 OpenVLA Encoder Replacement**
```python
# openvla/openvla-7b 코드 수정
# Vision encoder만 교체 (SigLIP → ours)
# Language model, action head 고정
```

**3.2 Quick Fine-tuning**
```bash
# Bridge V2 subset (10-20% demos)
python finetune_openvla.py --encoder videomae
python finetune_openvla.py --encoder single_stream
python finetune_openvla.py --encoder two_stream
```

**3.3 Early Signal Check**
- Loss convergence speed
- Eval loss comparison
- Sanity check: Two-stream > Single-stream?

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

## 🧪 Quick Test Commands

```bash
# Sanity check
python train.py --method videomae --test --batch-size 8
python train.py --method single_stream --test --batch-size 8
python train.py --method two_stream --test --batch-size 8

# Short training (5-10 epochs)
python train.py --method videomae --epochs 10 --data bridge_v2
python train.py --method single_stream --epochs 10 --data bridge_v2
python train.py --method two_stream --epochs 10 --data bridge_v2

# Quick LIBERO test
python finetune_openvla.py --encoder videomae --demos 0.2
```

---

## 📋 AWS Phase (After H100)

**완성된 코드 실행만**:

1. **EgoDex Full Pretraining**
   - 3개 encoder (VideoMAE, Single-stream, Two-stream)
   - Full dataset (829h, 194 tasks)

2. **LIBERO Full Evaluation**
   - OpenVLA + 각 encoder
   - Success rate 측정 (Spatial, Object, Long)

3. **Component Ablation (문제 발생 시만)**
   - Intermediate CLS vs Final CLS
   - Skip connection 유무
   - Distillation 효과

---

## 🔗 Key References

- **Main memo**: `references/논문 - Action-Agnostic Visual Behavior Representation.md`
  - Section 10: Decoder Design Q&A
  - Section "실험 계획": Two-Phase Strategy

- **GitHub**:
  - VideoMAE: `MCG-NJU/VideoMAE`
  - OpenVLA: `openvla/openvla`

---

## ⚠️ Important Notes

1. **H100 = Code Completion**: 모든 코드 완성 후 AWS로
2. **3-way Comparison**: VideoMAE vs Single-stream vs Two-stream
3. **Early Signal**: H100에서 Quick LIBERO Test로 sanity check
4. **Final Goal**: LIBERO downstream task, not pretraining loss
