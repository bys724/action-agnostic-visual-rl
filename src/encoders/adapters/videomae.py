"""VideoMAE-ours 어댑터 (2-frame native pair).

src/models/videomae.py의 VideoMAEEncoderForVLA 사용. 입력 (B, T, 3, H, W) →
2-frame pair (img_prev, img_curr) 형성 후 6-channel stacked 입력으로 forward.
출력: (B, T, 768) — patch mean
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


class VideoMAEOursAdapter(EncoderAdapter):
    """VideoMAE-ours 어댑터. Two-Stream v11 어댑터와 같은 pair 형성 정책.

    출력: patch mean (CLS 없음 — VideoMAE-ours는 CLS token 미사용)

    mode:
        - "paired" (BC-T 표준): (prev, curr) 한 번 forward → (B, T, 768)
        - "p_t_p_tk" (probing only): prev/curr 각각 same-frame 복제 forward 후
          patch mean concat → (B, T, 1536). tubelet_size=2가 두 frame을 한
          patch로 묶으므로 단일 frame representation은 same-frame replica로 추출
          (paper_experiments_plan §C7 catalyst evidence).
    """

    embed_dim = 768
    img_size = 224

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        depth: int = 12,
        num_heads: int = 12,
        freeze: bool = True,
        mode: str = "paired",
    ):
        super().__init__(freeze=freeze)
        from src.models.videomae import VideoMAEEncoderForVLA

        assert mode in ("paired", "p_t_p_tk"), f"Unknown mode: {mode}"
        self.mode = mode

        self.model = VideoMAEEncoderForVLA(
            checkpoint_path=checkpoint_path,
            depth=depth, num_heads=num_heads,
            image_size=self.img_size, patch_size=16,
        )
        if freeze:
            self.freeze_encoder()

        self.prev_obs: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.prev_obs = None

    def _encode_pair(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B', 6, H, W) → (B', 768) patch mean."""
        was_training = self.model.training
        self.model.eval()
        with torch.set_grad_enabled(not self._freeze):
            patches = self.model(pixel_values)
        if was_training and not self._freeze:
            self.model.train()
        return patches.mean(dim=1)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape

        if T > 1:
            prev = torch.cat([obs_seq[:, :1], obs_seq[:, :-1]], dim=1)
        else:
            prev = obs_seq.clone() if self.prev_obs is None else self.prev_obs
            self.prev_obs = obs_seq.detach()

        if self.mode == "p_t_p_tk":
            prev_flat = prev.reshape(B * T, C, H, W)
            curr_flat = obs_seq.reshape(B * T, C, H, W)
            pv_prev = torch.cat([prev_flat, prev_flat], dim=1)  # (B*T, 6, H, W) same-frame replica
            pv_curr = torch.cat([curr_flat, curr_flat], dim=1)
            tok_prev = self._encode_pair(pv_prev)               # (B*T, 768)
            tok_curr = self._encode_pair(pv_curr)
            token = torch.cat([tok_prev, tok_curr], dim=-1)     # (B*T, 1536)
            return token.reshape(B, T, -1)

        # default: paired forward (BC-T 표준)
        pixel_values = torch.cat([prev, obs_seq], dim=2)  # (B, T, 6, H, W)
        pixel_values = pixel_values.reshape(B * T, 2 * C, H, W)
        token = self._encode_pair(pixel_values)
        return token.reshape(B, T, -1)
