"""Two-Stream v15 어댑터 — P_t + P_tk concat mode (옵션 B).

v15 LIBERO probing champion mode (`patch_mean_concat_p_t_p_tk`) BC-T 어댑터화.
- 입력: (prev_obs, curr_obs) 2-frame pair
- P encoder × 2 forward (각자 p_channel) → patches mean → concat
- 출력: (B, T, 1536) — P_t patch mean (768) ⊕ P_tk patch mean (768)
- Motion-routing/M encoder 미사용
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


class TwoStreamV15PtPtkAdapter(EncoderAdapter):
    embed_dim = 768 * 2  # P_t ⊕ P_tk
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
        from scripts.eval.probe_action_v11 import _p_encoder_forward

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

        p_chan_prev = self.model.preprocessing.compute_p_channel(img_prev)
        p_chan_curr = self.model.preprocessing.compute_p_channel(img_curr)

        p_enc_prev = self._p_forward(self.model, p_chan_prev)  # [B*T, 1+N, D]
        p_enc_curr = self._p_forward(self.model, p_chan_curr)

        P_t = p_enc_prev[:, 1:].mean(dim=1)   # (B*T, 768) — frame_t patches mean
        P_tk = p_enc_curr[:, 1:].mean(dim=1)  # (B*T, 768) — frame_t+k patches mean

        token = torch.cat([P_t, P_tk], dim=-1)
        return token.reshape(B, T, -1)
