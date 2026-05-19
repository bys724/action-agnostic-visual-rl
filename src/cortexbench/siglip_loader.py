"""CortexBench wrapper for SigLIP base (§C11).

SigLIPVisionModel (google/siglip-base-patch16-224) — vision-language SSL baseline.
Output: last_hidden_state shape=(B, 196, 768), CLS 토큰 없음 → patches mean = 768-d.

다른 wrapper들과 동일한 patches mean pooling protocol.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class SiglipWrapper(nn.Module):
    """SigLIP vision encoder as single-image embedder."""

    def __init__(self, hf_id: str = "google/siglip-base-patch16-224"):
        super().__init__()
        from transformers import SiglipVisionModel
        self.model = SiglipVisionModel.from_pretrained(hf_id)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size  # 768 (ViT-B)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) — SigLIP-native normalization (mean/std 0.5)
        out = self.model(pixel_values=x)
        # SigLIP 출력엔 CLS 없음 → 모든 token 평균
        return out.last_hidden_state.mean(dim=1)  # (B, 768)


def siglip_transforms():
    """SigLIP native preprocessing — mean/std 0.5 (NOT ImageNet)."""
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_siglip(
    hf_id: str = "google/siglip-base-patch16-224",
) -> Tuple[nn.Module, int, callable, dict]:
    """Hydra factory for SigLIP (CortexBench `_target_`)."""
    model = SiglipWrapper(hf_id=hf_id)
    transform = siglip_transforms()
    metadata = {
        "algo": "siglip",
        "model": "siglip-base-patch16-224",
        "data": ["webli"],
        "comment": "single-image patches mean, vision-language SSL baseline (§C11)",
    }
    with torch.no_grad():
        zero = transform(Image.new("RGB", (224, 224))).unsqueeze(0)
        d = model(zero).shape[1]
    assert d == model.embedding_dim, f"embedding_dim mismatch: {d} != {model.embedding_dim}"
    return model, model.embedding_dim, transform, metadata
