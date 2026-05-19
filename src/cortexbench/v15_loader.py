"""CortexBench encoder wrapper for Two-Stream v15 (§C11).

CortexBench encoder contract:
- Input: single image tensor (B, 3, H, W), already preprocessed via transform
- Output: embedding tensor (B, embedding_dim)
- Factory returns (model, embedding_dim, transform, metadata) — hydra-instantiated

v15는 학습 시 (prev, curr) paired였지만 **실제 inference에선 P encoder만 단독 사용**:
- M encoder는 학습 시 motion catalyst 역할만 (pretraining trick), inference에서는 무관
- Single-image embedding = P encoder patches mean = 768-d
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class V15POnlyWrapper(nn.Module):
    """v15 P encoder as single-image CortexBench-style embedder.

    Strategy: P encoder만 단일 forward → patches mean (skip CLS) = 768-d.
    M encoder는 학습 시 motion catalyst 역할만 했으므로 inference에서 제외.
    """

    def __init__(self, checkpoint_path: str, p_depth: int = 12, m_depth: int = 6):
        super().__init__()
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.eval.probe_action_v11 import (
            _p_encoder_forward,
            load_v11_model,
        )
        self._p_encoder_forward = _p_encoder_forward
        self.model = load_v11_model(
            checkpoint_path, p_depth=p_depth, m_depth=m_depth, device="cpu",
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.embedding_dim = 768  # P encoder patches mean

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) already-normalized tensor
        # preprocessing(im_t, im_tk): two-stream architecture가 (M, P) channel 분리.
        # P channel만 사용 (im_tk 인수는 P 경로에서 무시되지만 dummy 필요).
        _, p_channel = self.model.preprocessing(x, x)
        p_enc = self._p_encoder_forward(self.model, p_channel)
        # patches mean (skip CLS at index 0)
        return p_enc[:, 1:].mean(dim=1)  # (B, 768)


def v15_transforms():
    """CortexBench/PVR convention: 224×224, ImageNet norm."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_v15_p_only(
    checkpoint_path: str,
    p_depth: int = 12,
    m_depth: int = 6,
) -> Tuple[nn.Module, int, callable, dict]:
    """Hydra factory for v15 P-encoder-only (CortexBench `_target_`).

    Returns:
        (model, embedding_dim, transform, metadata) — vc_models.models.load_model 호환
    """
    model = V15POnlyWrapper(checkpoint_path, p_depth=p_depth, m_depth=m_depth)
    transform = v15_transforms()
    metadata = {
        "algo": "two-stream-v15",
        "model": "vit_base_patch16",
        "data": ["egodex"],
        "comment": "P encoder only (768-d), action-agnostic pretrain (§C11)",
    }
    with torch.no_grad():
        zero = transform(Image.new("RGB", (224, 224))).unsqueeze(0)
        d = model(zero).shape[1]
    assert d == model.embedding_dim, f"embedding_dim mismatch: {d} != {model.embedding_dim}"
    return model, model.embedding_dim, transform, metadata
