#!/usr/bin/env python
"""Token provider for layer 2 (Vault 4th decision 10-10: observer Fuse/z removed; layer 1 = input format).

Layer 2 always receives (anchor P tokens + the M chunk tokens after it). This class gives exactly those, straight
from the frozen CoMP encoder: no Fuse, no dataset-statistic normalization, no z.

  tp = TokenProvider(encoder_ckpt)                      # e.g. C1-DN two_stream_v15b_refine_comp_s_denoise_augtgt/…/checkpoint_epoch0010.pt
  P  = tp.p_tokens(frames)                              # (n, 196, D)  frames = uint8 (n, H, W, 3) agentview
  M  = tp.m_tokens(frames_a, frames_b)                  # (n, 196, D)  M of ΔL(frame_a → frame_b), pairwise
  anchor, chunks = tp.anchor_and_chunks(seq)            # seq = [f_anchor, …, f_t] (k = len−1 ∈ 1..20)
        chunks = [(M tokens (1,196,D), start_unit, length_frames), …]: complete 5-frame chunks + one partial
        last chunk (1–4 frames) — the observer-v4 convention (start unit 0..3, length 1..5)

Preprocessing = the encoder's own: uint8 → [0,1] → bilinear 224 (no ImageNet normalization), P = RGB, M = ΔL
(BT.709 luminance difference). Tokens exclude CLS. dtype = fp32 by default (cast to fp16 for caching:
196×384×2 B ≈ 150 KB per token map).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_action_libero import build_parvo_encoder  # noqa: E402

UNIT, K_MAX = 5, 20


class TokenProvider:
    def __init__(self, encoder_ckpt: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.enc = build_parvo_encoder(encoder_ckpt, self.device)  # frozen, eval
        self.dim = self.enc.pos_embed_p.shape[-1]

    def _x(self, frames):
        f = torch.as_tensor(np.asarray(frames), device=self.device)
        f = f[None] if f.ndim == 3 else f
        return F.interpolate(f.permute(0, 3, 1, 2).float().div_(255.0), size=(224, 224),
                             mode="bilinear", align_corners=False)

    @torch.no_grad()
    def p_tokens(self, frames, batch: int = 128) -> torch.Tensor:
        x = self._x(frames)
        return torch.cat([self.enc._encode_p_unmasked(self.enc.preprocessing.compute_p_channel(x[i:i + batch]))[:, 1:]
                          for i in range(0, len(x), batch)])

    @torch.no_grad()
    def m_tokens(self, frames_a, frames_b, batch: int = 128) -> torch.Tensor:
        a, b = self._x(frames_a), self._x(frames_b)
        assert a.shape == b.shape
        return torch.cat([self.enc._encode_m_unmasked(self.enc.preprocessing.compute_m_channel(a[i:i + batch], b[i:i + batch]))[:, 1:]
                          for i in range(0, len(a), batch)])

    @torch.no_grad()
    def anchor_and_chunks(self, seq):
        """seq = [f_anchor, f_anchor+1, …, f_t]; k = len(seq) − 1 ∈ 1..20."""
        k = len(seq) - 1
        assert 1 <= k <= K_MAX
        q, r = divmod(k, UNIT)
        bounds = [(UNIT * i, UNIT * (i + 1), i, UNIT) for i in range(q)] + ([(UNIT * q, k, q, r)] if r else [])
        m = self.m_tokens([seq[s] for s, _, _, _ in bounds], [seq[e] for _, e, _, _ in bounds])
        return self.p_tokens(seq[0]), [(m[j:j + 1], st, ln) for j, (_, _, st, ln) in enumerate(bounds)]
