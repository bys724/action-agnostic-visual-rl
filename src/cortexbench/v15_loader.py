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

# A6000(sm_86) 호환성: patch_embed_p 출력 std≈24의 큰 attention score에서
# flash/mem_efficient SDPA backend가 NaN 발생 (Block 0부터 폭주).
# math backend의 standard softmax는 numerically stable. H100(sm_90)에서는
# 같은 입력도 안정 backend 자동 선택 — 즉 본 강제는 A6000 호환성 workaround.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class V15POnlyWrapper(nn.Module):
    """v15 P encoder as single-image CortexBench-style embedder.

    Strategy: P encoder만 단일 forward → patches mean (skip CLS) = 768-d.
    M encoder는 학습 시 motion catalyst 역할만 했으므로 inference에서 제외.

    Note: ckpt가 v15 lineage (motion routing + composition head 포함)이므로
    `TwoStreamV15Model` 직접 instantiate. v11 class로 load 시 motion routing
    weight 289개가 unexpected keys로 누락되어 일부 GPU에서 forward 불안정.
    """

    def __init__(self, checkpoint_path: str, p_depth: int = 12, m_depth: int = 6):
        super().__init__()
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.models import TwoStreamV15Model
        from scripts.eval.probe_action_v11 import _p_encoder_forward
        self._p_encoder_forward = _p_encoder_forward
        self.model = TwoStreamV15Model(
            embed_dim=768, p_depth=p_depth, m_depth=m_depth,
            num_heads=12, mlp_ratio=4.0, image_size=224, patch_size=16,
            mask_ratio_m=0.0, mask_ratio_p=0.0,
            decoder_depth_m=3, interpreter_depth=3, num_motion_iters=2,
            rotation_aug=False,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (first 3: {missing[:3]})")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")
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
    """Match EgoDex pretraining input: 224×224, raw [0, 1] tensor.

    학습 파이프라인(src/datasets/base.py)이 frame을 `float / 255.0`만 적용해
    [0, 1] raw로 forward. TwoStreamPreprocessing.compute_p_channel도 입력을
    [0, 1]로 명시 가정 (luminance/Sobel/RGB 정규화 기준). 따라서 inference도
    ImageNet Normalize 없이 [0, 1] 그대로 통과해야 학습 분포와 일치.
    """
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
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
