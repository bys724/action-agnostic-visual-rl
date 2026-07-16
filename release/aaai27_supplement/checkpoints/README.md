# Pretrained Checkpoints

**Weights are released upon publication.** This anonymized code supplement ships
the full training + evaluation source, the exact hyperparameter configs
(`train/config_*.yaml`), and the aggregated evaluation artifacts, but not the
trained weight files. The reproducibility checklist commits the *source code*;
the trained weights will be made publicly available (full fp32) with the
de-anonymized release.

Two headline encoders (ViT-S two-stream, 32.3M params each) were pretrained on
EgoDex part1 (4.62M frame-pair samples, 50 epochs, 8x H100, global batch 1024).

| File (upon publication) | Model | Role in paper |
|---|---|---|
| `comp_mae_s.pt` | CoMP-MAE-S | Ours (symmetric cross-reconstruction) |
| `plain_xmae_s.pt` | Plain cross-modal MAE-S | Matched control (M-recon off, standard cross-attn, same params/data) |

## Reproducing the weights

Pretraining costs roughly **~13.7 h wall-clock on 8x H100 (~110 GPU-hours)** for
CoMP-MAE-S and **~19.3 h (~154 GPU-hours)** for the plain control at the config
above. See the top-level `README.md` (Pretraining) and
`train/config_{comp_mae_s,plain_control}.yaml` for the exact commands and
hyperparameters. The reported final losses were: CoMP-MAE-S train 0.0143 / eval
0.0180; plain control train 0.0609 / eval 0.0619 (loss scales differ by
objective and are not comparable across the two models).

Even without the weights, the headline **efficiency table** reproduces from the
shipped aggregated probing artifacts (`eval/build_step0_efficiency_table.py`),
and the **rollout significance test** reproduces from the shipped per-task CSV
(`stats/stats_libero_rollout.py`).

## Integrity (for the upon-publication release)

The released weight files will match these SHA-256 sums (also in `SHA256SUMS`):

| sha256 | file |
|---|---|
| `c0338ca282c848b72fa3cb5b385683a80d0b9f2778ad3a2ae481c9efe76c032f` | comp_mae_s.pt |
| `4c9bb95dd2b950b06e46e8542823b65a90da5989f1a56afecca22044c2e5c190` | plain_xmae_s.pt |

## Loading (once weights are available)

```python
import torch
from model.comp_mae import CoMPMAE

model = CoMPMAE(embed_dim=384, num_heads=6, m_depth=6, comp_mae=True,
                routing_mode="v_from_p", pair_mode=True, use_sobel=False,
                masked_anchor=True)
ck = torch.load("comp_mae_s.pt", map_location="cpu")   # keys: epoch, model_state_dict, train_loss, eval_loss
model.load_state_dict(ck["model_state_dict"])
```

Frozen probing / BC evaluation only uses the P (and optionally M) encoder; the
decoder / reconstruction heads are not needed at inference. `*.train_meta.json`
records run-level metadata (epochs, batch size, LR, dataset sizes).

## Preprocessing parity (required for reproduction)

Inputs are raw RGB in `[0, 1]` (`/255.0` only). Do **not** apply ImageNet
mean/std normalization — the encoders were trained without it, and evaluation
with normalized inputs is out-of-distribution.
