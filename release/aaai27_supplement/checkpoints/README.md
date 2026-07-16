# Pretrained Checkpoints

Two headline encoders (ViT-S two-stream, 32.3M params each) pretrained on EgoDex
part1 (4.62M frame-pair samples, 50 epochs, 8 GPUs, global batch 1024).
Weights-only: optimizer/scheduler states removed.

| File | Model | Role in paper |
|---|---|---|
| `comp_mae_s.pt` | CoMP-MAE-S | Ours (symmetric cross-reconstruction) |
| `plain_xmae_s.pt` | Plain cross-modal MAE-S | Matched control (M-recon off, standard cross-attn, same params/data) |

## Integrity

Verify after download (`SHA256SUMS` shipped alongside):

```bash
sha256sum -c SHA256SUMS
```

| sha256 | file |
|---|---|
| `c0338ca282c848b72fa3cb5b385683a80d0b9f2778ad3a2ae481c9efe76c032f` | comp_mae_s.pt |
| `4c9bb95dd2b950b06e46e8542823b65a90da5989f1a56afecca22044c2e5c190` | plain_xmae_s.pt |

## Contents & loading

```python
import torch
ckpt = torch.load("comp_mae_s.pt", map_location="cpu")
# keys: epoch (50), model_state_dict, train_loss, eval_loss
model.load_state_dict(ckpt["model_state_dict"])
```

Final losses for reference: CoMP-MAE-S train 0.0143 / eval 0.0180;
plain control train 0.0609 / eval 0.0619 (loss scales differ by objective —
not comparable across the two models).

`*.train_meta.json` files record run-level metadata (epochs, batch size, LR,
dataset sizes) as written at training time.

## Preprocessing parity (required for reproduction)

Inputs are raw RGB in `[0, 1]` (`/255.0` only). Do **not** apply ImageNet
mean/std normalization — the encoders were trained without it and evaluation
with normalized inputs is out-of-distribution.
