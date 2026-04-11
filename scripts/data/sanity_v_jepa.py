#!/usr/bin/env python3
"""V-JEPA sanity check: model build, forward, backward, EMA update.

CPU 또는 저렴한 MIG GPU에서 수 초 안에 실행.
목적: 구조 검증, 메모리/gradient 흐름, EMA 동작 확인.
"""

import sys
sys.path.insert(0, "/proj/home/mrg/bys724/action-agnostic-visual-rl")

import torch
from src.models import VJEPAModel

print("=== V-JEPA sanity check ===")

# 1. Build
m = VJEPAModel(depth=12)
n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in m.parameters())
print(f"Params: trainable={n_trainable:,}, total={n_total:,}")
print(f"  x_encoder: {sum(p.numel() for p in m.x_encoder.parameters()):,}")
print(f"  y_encoder (frozen): {sum(p.numel() for p in m.y_encoder.parameters()):,}")
print(f"  predictor: {sum(p.numel() for p in m.predictor.parameters()):,}")

# 2. Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = m.to(device)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# 3. Forward (eval mode)
m.eval()
B = 2
img_t = torch.randn(B, 3, 224, 224, device=device)
img_tk = torch.randn(B, 3, 224, 224, device=device)
loss, img_pred = m.compute_loss(img_t, img_tk)
print(f"\n[eval] loss={loss.item():.6f}, img_pred shape={img_pred.shape}")

# 4. Backward (train mode) — gradient flow 검증
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
loss, _ = m.compute_loss(img_t, img_tk)
opt.zero_grad()
loss.backward()

x_grad = sum(p.grad.abs().sum().item() for p in m.x_encoder.parameters() if p.grad is not None)
pred_grad = sum(p.grad.abs().sum().item() for p in m.predictor.parameters() if p.grad is not None)
y_grad_count = sum(1 for p in m.y_encoder.parameters() if p.grad is not None)
print(f"\n[train] loss={loss.item():.6f}")
print(f"  x_encoder grad sum: {x_grad:.4f} (should be > 0)")
print(f"  predictor grad sum: {pred_grad:.4f} (should be > 0)")
print(f"  y_encoder grad count: {y_grad_count} (should be 0)")

# 5. Optimizer step + EMA update
opt.step()
y_before = next(iter(m.y_encoder.parameters())).data.clone()
m.update_ema()
y_after = next(iter(m.y_encoder.parameters())).data
ema_diff = (y_after - y_before).abs().sum().item()
print(f"  EMA update changed y param sum: {ema_diff:.6f} (should be > 0)")

# 6. EMA momentum scheduling
m.set_ema_momentum(0, 50)
mom_start = m._ema_momentum
m.set_ema_momentum(49, 50)
mom_end = m._ema_momentum
print(f"  EMA momentum: epoch0={mom_start:.4f}, epoch49={mom_end:.4f}")

# 7. Feature extraction (inference-time, for downstream use)
m.eval()
feat = m.extract_features(img_t, img_tk)
print(f"\n[inference] feature shape: {feat.shape}")
print(f"  → (B={B}, num_patches=196, embed_dim=768)")

# 8. Memory check
if torch.cuda.is_available():
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\nPeak GPU memory: {peak_mb:.1f} MB")

print("\n=== All checks passed ===")
