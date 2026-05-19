"""CortexBench wrapper for VideoMAE-ours (§C11).

VideoMAE는 학습 시 tubelet_size=2로 (frame_t, frame_tk) 두 프레임을 spatio-temporal
patch로 묶어 처리. CortexBench는 single image 호출이므로 self-pair 전략 사용
(같은 frame을 두 번 stack → tubelet 통과 → patches mean = 768-d).

v15와 동일한 patches mean pooling protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VideoMAEOursWrapper(nn.Module):
    """VideoMAE-ours as single-image embedder.

    Strategy: self-pair (x, x) → VideoMAE encoder forward → patches mean = 768-d.
    """

    def __init__(self, checkpoint_path: str, embed_dim: int = 768):
        super().__init__()
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.models.videomae import VideoMAEEncoderForVLA

        self.model = VideoMAEEncoderForVLA(checkpoint_path=checkpoint_path, embed_dim=embed_dim)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.embedding_dim = embed_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) already-normalized tensor
        # VideoMAE 입력: (B, 6, H, W) = stack of (current, future). self-pair로 동일 frame 2번.
        pv = torch.cat([x, x], dim=1)            # (B, 6, H, W)
        patches = self.model(pv)                 # (B, num_patches, embed_dim)
        return patches.mean(dim=1)               # (B, 768)


def videomae_transforms():
    """ImageNet preprocessing, 224×224 (CortexBench/PVR convention)."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_videomae_ours(
    checkpoint_path: str,
    embed_dim: int = 768,
) -> Tuple[nn.Module, int, callable, dict]:
    """Hydra factory for VideoMAE-ours (CortexBench `_target_`)."""
    model = VideoMAEOursWrapper(checkpoint_path, embed_dim=embed_dim)
    transform = videomae_transforms()
    metadata = {
        "algo": "videomae-ours",
        "model": "vit_base_patch16",
        "data": ["egodex"],
        "comment": "self-pair single-image embedding, action-agnostic MAE pretrain (§C11)",
    }
    with torch.no_grad():
        zero = transform(Image.new("RGB", (224, 224))).unsqueeze(0)
        d = model(zero).shape[1]
    assert d == model.embedding_dim, f"embedding_dim mismatch: {d} != {model.embedding_dim}"
    return model, model.embedding_dim, transform, metadata
