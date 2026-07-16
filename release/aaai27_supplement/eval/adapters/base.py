"""EncoderAdapter ABC + factory.

Common interface for all encoder adapters. The BC-Transformer policy's
spatial encoder calls into this to turn a pre-trained vision encoder into a
sequence of spatial tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class EncoderAdapter(nn.Module, ABC):
    """Wrap a pre-trained vision encoder as a BC-T spatial token producer.

    A subclass defines:
      - embed_dim : output token dim (the BC-T input projection unifies it)
      - img_size  : encoder native resolution (LIBERO obs are resized to it)
      - forward() : (B, T, 3, H, W) -> (B, T, embed_dim)
      - reset()   : reset internal state at the start of a rollout episode (no-op by default)
    """

    embed_dim: int
    img_size: int = 224

    def __init__(self, freeze: bool = True):
        super().__init__()
        self._freeze = freeze

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @abstractmethod
    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq: (B, T, 3, H, W) float in [0, 1] -> (B, T, embed_dim)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the start of a rollout episode. Override in stateful adapters."""
        return None


def build_adapter(
    encoder_type: str,
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> EncoderAdapter:
    """Encoder type -> adapter instance.

    This release ships the CoMP-MAE adapter ('comp-mae'), which is used for
    the CoMP-MAE / plain-control BC-T finetuning and rollout (the headline
    comparison). Baseline adapters (VideoMAE, DINOv2, SigLIP, VC-1, V-JEPA-2)
    were used to produce the baseline rows and can be added analogously.

    Args:
        encoder_type: 'comp-mae' (CoMP-MAE / plain control).
        checkpoint_path: pretrained encoder checkpoint path.
        **kwargs: adapter-specific arguments.
    """
    encoder_type = encoder_type.lower().replace("_", "-")

    if encoder_type == "comp-mae":
        from .comp_mae_adapter import CoMPMAEAdapter
        return CoMPMAEAdapter(checkpoint_path=checkpoint_path, **kwargs)

    raise ValueError(
        f"Unknown/omitted encoder_type: {encoder_type}. "
        "This release ships 'comp-mae' (CoMP-MAE); baseline adapters are omitted."
    )
