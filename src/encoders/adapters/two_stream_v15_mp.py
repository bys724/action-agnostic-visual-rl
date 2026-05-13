"""Two-Stream v15 어댑터 — M(prev,curr) + P(curr) concat mode (C-variant).

A+D' (v11 default)에서 motion-routing 단계를 제거하고 P encoder의 curr-frame만 사용.
- 입력: (prev_obs, curr_obs) 2-frame pair
- M encoder × 1 (m_channel = compute_m_channel(prev, curr)) → patch mean (768)
- P encoder × 1 (p_channel = compute_p_channel(curr)) → patch mean (768)
- 출력: (B, T, 1536) — M_patch_mean ⊕ P_curr_patch_mean
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import sys
import torch

from .base import EncoderAdapter

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TwoStreamV15MPAdapter(EncoderAdapter):
    embed_dim = 768 * 2  # M ⊕ P_curr
    img_size = 224

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        p_depth: int = 12,
        m_depth: int = 6,
        freeze: bool = True,
        device: str = "cpu",
    ):
        super().__init__(freeze=freeze)
        from scripts.eval.probe_action_v11 import (
            _m_encoder_forward,
            _p_encoder_forward,
        )

        if checkpoint_path is None:
            from src.models import TwoStreamV11Model
            self.model = TwoStreamV11Model(
                embed_dim=768, p_depth=p_depth, m_depth=m_depth,
                num_heads=12, mlp_ratio=4.0,
                image_size=224, patch_size=16,
                mask_ratio_m=0.0, mask_ratio_p=0.0,
                decoder_depth_m=3, interpreter_depth=3,
                num_motion_iters=2, rotation_aug=False,
            ).to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            from scripts.eval.probe_action_v11 import load_v11_model
            self.model = load_v11_model(
                checkpoint_path, p_depth=p_depth, m_depth=m_depth, device=device,
            )
        self._m_forward = _m_encoder_forward
        self._p_forward = _p_encoder_forward

        if freeze:
            self.freeze_encoder()

        self.prev_obs: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.prev_obs = None

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape

        if T > 1:
            prev = torch.cat([obs_seq[:, :1], obs_seq[:, :-1]], dim=1)
        else:
            if self.prev_obs is None:
                prev = obs_seq.clone()
            else:
                prev = self.prev_obs
            self.prev_obs = obs_seq.detach()

        img_prev = prev.reshape(B * T, C, H, W)
        img_curr = obs_seq.reshape(B * T, C, H, W)

        m_chan = self.model.preprocessing.compute_m_channel(img_prev, img_curr)
        p_chan_curr = self.model.preprocessing.compute_p_channel(img_curr)

        m_enc = self._m_forward(self.model, m_chan)         # [B*T, 1+N, D]
        p_enc_curr = self._p_forward(self.model, p_chan_curr)

        M_feat = m_enc[:, 1:].mean(dim=1)         # (B*T, 768)
        P_curr = p_enc_curr[:, 1:].mean(dim=1)    # (B*T, 768)

        token = torch.cat([M_feat, P_curr], dim=-1)
        return token.reshape(B, T, -1)
