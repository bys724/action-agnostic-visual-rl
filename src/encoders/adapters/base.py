"""EncoderAdapter ABC + factory.

모든 어댑터의 공통 인터페이스. BC-T policy의 spatial_encode가 호출.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class EncoderAdapter(nn.Module, ABC):
    """Pre-trained vision encoder를 BC-T spatial token으로 변환.

    Subclass는 다음을 정의:
      · embed_dim   : 출력 토큰 차원 (BC-T input projection이 통일)
      · img_size    : encoder native resolution (LIBERO obs는 이 사이즈로 resize)
      · forward()   : (B, T, 3, H, W) → (B, T, embed_dim)
      · reset()     : rollout 에피소드 시작 시 내부 state 초기화 (default no-op)
    """

    embed_dim: int
    img_size: int = 224

    def __init__(self, freeze: bool = True):
        super().__init__()
        self._freeze = freeze

    def freeze_encoder(self):
        """Encoder 파라미터 전체 freeze."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @abstractmethod
    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq: (B, T, 3, H, W) [0,1] float → (B, T, embed_dim)"""
        raise NotImplementedError

    def reset(self) -> None:
        """Rollout 에피소드 시작 시 호출. Stateful 어댑터만 override."""
        return None


def build_adapter(
    encoder_type: str,
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> EncoderAdapter:
    """Encoder type → 해당 어댑터 인스턴스.

    Args:
        encoder_type: 'parvo-ptptk' | 'videomae-ours' | 'dinov2' | 'siglip'
                      | 'vc1' | 'vjepa2-1'
        checkpoint_path: Encoder ckpt 경로 (필요 시)
        **kwargs: 어댑터별 추가 인자
    """
    encoder_type = encoder_type.lower().replace("_", "-")

    if encoder_type == "parvo-ptptk":
        from .parvo_pt_ptk import ParvoPtPtkAdapter
        return ParvoPtPtkAdapter(checkpoint_path=checkpoint_path, **kwargs)

    if encoder_type == "videomae-ours":
        from .videomae import VideoMAEOursAdapter
        return VideoMAEOursAdapter(checkpoint_path=checkpoint_path, **kwargs)

    if encoder_type == "siammae":
        from .siammae import SiamMAEAdapter
        return SiamMAEAdapter(checkpoint_path=checkpoint_path, **kwargs)

    if encoder_type in ("dinov2", "siglip", "vc1"):
        from .single_frame import SingleFrameAdapter
        return SingleFrameAdapter(
            encoder_type=encoder_type,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

    if encoder_type in ("vjepa2-1", "v-jepa-2-1"):
        from .vjepa2 import VJEPA2Adapter
        return VJEPA2Adapter(checkpoint_path=checkpoint_path, **kwargs)

    raise ValueError(f"Unknown encoder_type: {encoder_type}")
