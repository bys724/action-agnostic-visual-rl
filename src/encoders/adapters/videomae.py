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
    """

    embed_dim = 768
    img_size = 224

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        depth: int = 12,
        num_heads: int = 12,
        freeze: bool = True,
    ):
        super().__init__(freeze=freeze)
        from src.models.videomae import VideoMAEEncoderForVLA

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

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape

        if T > 1:
            prev = torch.cat([obs_seq[:, :1], obs_seq[:, :-1]], dim=1)
        else:
            prev = obs_seq.clone() if self.prev_obs is None else self.prev_obs
            self.prev_obs = obs_seq.detach()

        # 6-channel stacked: VideoMAEEncoderForVLA가 (B, 6, H, W) 입력 받음
        pixel_values = torch.cat([prev, obs_seq], dim=2)  # (B, T, 6, H, W)
        pixel_values = pixel_values.reshape(B * T, 2 * C, H, W)

        # frozen이면 grad 불필요
        was_training = self.model.training
        self.model.eval()
        with torch.set_grad_enabled(not self._freeze):
            patches = self.model(pixel_values)  # (B*T, num_patches, 768)
        if was_training and not self._freeze:
            self.model.train()

        token = patches.mean(dim=1)  # (B*T, 768)
        return token.reshape(B, T, -1)
