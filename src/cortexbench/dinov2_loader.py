"""CortexBench wrapper for DINOv2 base (§C11).

DINOv2 (facebook/dinov2-base) — self-distillation vision SSL baseline.
Output: last_hidden_state shape=(B, 257, 768) [CLS at idx 0, 256 patches] →
patches mean (skip CLS) = 768-d.

다른 wrapper들과 동일한 patches mean pooling protocol.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class DinoV2Wrapper(nn.Module):
    """DINOv2 vision encoder as single-image embedder."""

    def __init__(self, hf_id: str = "facebook/dinov2-base"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(hf_id)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size  # 768 (ViT-B)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) ImageNet-normalized
        out = self.model(pixel_values=x)
        # DINOv2: CLS at index 0, skip
        return out.last_hidden_state[:, 1:].mean(dim=1)  # (B, 768)


def dinov2_transforms():
    """ImageNet preprocessing (DINOv2 native)."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dinov2(
    hf_id: str = "facebook/dinov2-base",
) -> Tuple[nn.Module, int, callable, dict]:
    """Hydra factory for DINOv2 (CortexBench `_target_`)."""
    model = DinoV2Wrapper(hf_id=hf_id)
    transform = dinov2_transforms()
    metadata = {
        "algo": "dinov2",
        "model": "dinov2-base",
        "data": ["lvd-142m"],
        "comment": "single-image patches mean (skip CLS), self-distillation SSL baseline (§C11)",
    }
    with torch.no_grad():
        zero = transform(Image.new("RGB", (224, 224))).unsqueeze(0)
        d = model(zero).shape[1]
    assert d == model.embedding_dim, f"embedding_dim mismatch: {d} != {model.embedding_dim}"
    return model, model.embedding_dim, transform, metadata
