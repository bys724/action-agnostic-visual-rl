"""CortexBench encoder wrapper for Two-Stream v15 (§C11).

CortexBench encoder contract:
- Input: single image tensor (B, 3, H, W), already preprocessed via transform
- Output: embedding tensor (B, embedding_dim)
- Factory returns (model, embedding_dim, transform, metadata) — hydra-instantiated

v15 is natively paired (prev, curr) → 본 wrapper는 self-pair 전략 사용
(같은 frame을 prev=curr로 두 번 forward → P_t patches mean ⊕ P_tk patches mean = 1536-d).
Temporal context는 CortexBench 측 fuse_embeddings_flare가 4-timestep delta로 별도 추가.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class V15PtPtkWrapper(nn.Module):
    """v15 encoder as single-image CortexBench-style embedder.

    Strategy: self-pair (prev=curr) → P_t patches mean ⊕ P_tk patches mean = 2 * 768.
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
        self.embedding_dim = 768 * 2  # P_t ⊕ P_tk

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) already-normalized tensor
        _, p_channel = self.model.preprocessing(x, x)  # self-pair
        p_enc = self._p_encoder_forward(self.model, p_channel)
        # P_t = P_tk = same (single image), so duplicate the mean
        feat = p_enc[:, 1:].mean(dim=1)  # (B, 768) — patches mean (skip CLS)
        return torch.cat([feat, feat], dim=-1)  # (B, 1536)


def v15_transforms():
    """Standard ImageNet preprocessing — v15는 EgoDex/LIBERO에서 [0,1] float input.
    여기서는 CortexBench/PVR convention: 224×224, ImageNet norm.
    """
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_v15_ptptk(
    checkpoint_path: str,
    p_depth: int = 12,
    m_depth: int = 6,
) -> Tuple[nn.Module, int, callable, dict]:
    """Hydra factory for v15 P_t+P_tk encoder (CortexBench `_target_`).

    Returns:
        (model, embedding_dim, transform, metadata) — vc_models.models.load_model 호환
    """
    model = V15PtPtkWrapper(checkpoint_path, p_depth=p_depth, m_depth=m_depth)
    transform = v15_transforms()
    metadata = {
        "algo": "two-stream-v15",
        "model": "vit_base_patch16",
        "data": ["egodex"],
        "comment": "self-pair P_t+P_tk, action-agnostic pretrain (§C11)",
    }
    # sanity check embedding_dim
    with torch.no_grad():
        zero = transform(Image.new("RGB", (224, 224))).unsqueeze(0)
        d = model(zero).shape[1]
    assert d == model.embedding_dim, f"embedding_dim mismatch: {d} != {model.embedding_dim}"
    return model, model.embedding_dim, transform, metadata
