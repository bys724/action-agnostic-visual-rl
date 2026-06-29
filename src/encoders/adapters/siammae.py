"""SiamMAE 어댑터 (external reference baseline).

src/models/siammae.py의 SiamMAEEncoderForVLA 사용. SiamMAE는 **프레임별 단독 인코딩**
(siamese 2D ViT)이라 BC-T 타임스텝 특징이 single-frame patch mean으로 깔끔히 추출됨
(prev frame 불필요 — cross-frame 학습은 pretraining에만). docs/siammae_baseline_plan.md.

출력: (B, T, embed_dim) — current frame patch mean. embed_dim = 384(small)/768(base),
체크포인트에서 자동 추론 (size mismatch 시 strict=False silent-skip로 random weight 로드되는 사고 방지).
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


def _infer_siammae_size(checkpoint_path: Optional[str]) -> str:
    """체크포인트 encoder embed_dim으로 size(small=384 / base=768) 추론. 없으면 base."""
    if not checkpoint_path:
        return "base"
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    for k, v in sd.items():
        if k.endswith("patch_embed.projection.weight"):
            return "small" if v.shape[0] == 384 else "base"
    return "base"


class SiamMAEAdapter(EncoderAdapter):
    """SiamMAE 어댑터. 단일 프레임 인코딩 → patch mean → (B, T, D).

    SiamMAE 인코더는 프레임 독립 인코딩이므로 prev frame 불필요 (single_frame 패턴).
    """

    img_size = 224

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        size: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__(freeze=freeze)
        from src.models.siammae import SiamMAEEncoderForVLA

        size = size or _infer_siammae_size(checkpoint_path)
        self.model = SiamMAEEncoderForVLA(
            checkpoint_path=checkpoint_path, size=size,
            image_size=self.img_size, patch_size=16,
        )
        self.embed_dim = self.model.embed_dim  # 384(small) / 768(base)
        if freeze:
            self.freeze_encoder()

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape
        flat = obs_seq.reshape(B * T, C, H, W)
        # SiamMAEEncoderForVLA는 [B,6](prev⊕curr) 중 current만 인코딩 → curr=flat replica.
        pixel_values = torch.cat([flat, flat], dim=1)  # [B*T, 6, H, W]
        was_training = self.model.training
        self.model.eval()
        with torch.set_grad_enabled(not self._freeze):
            tokens = self.model(pixel_values)          # [B*T, N, D]
        if was_training and not self._freeze:
            self.model.train()
        return tokens.mean(dim=1).reshape(B, T, -1)    # [B, T, D]
