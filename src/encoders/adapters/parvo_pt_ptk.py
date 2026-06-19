"""Parvo (code v15b, no-Sobel) 어댑터 — P_t ⊕ P_tk concat (BC-T용).

Parvo = TwoStreamV15Model(pair_mode=True, use_sobel=False, masked_anchor=True).
- P channel = RGB 3ch (no-Sobel). 학습 입력과 동일 ([0,1] raw RGB) → preprocessing parity OK.
- 입력: (prev, curr) 2-frame pair → P encoder × 2 (각자 RGB) → patches mean → concat.
- 출력: (B, T, 1536) = P_t patch mean (768) ⊕ P_tk patch mean (768).
- M encoder / motion-routing 미사용 (probe `patch_mean_concat_p_t_p_tk` 동치).

⚠️ 기존 two_stream_v15_pt_ptk.py 어댑터는 v11(Sobel 5ch)을 instantiate해 Parvo에 부적합.
   본 어댑터는 probe_action.py의 parvo 경로와 동일하게 TwoStreamV15Model(use_sobel=False).
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


class ParvoPtPtkAdapter(EncoderAdapter):
    embed_dim = 768 * 2  # P_t ⊕ P_tk
    img_size = 224

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
        device: str = "cpu",
        **kwargs,  # build_adapter가 넘기는 잉여 인자 무시
    ):
        super().__init__(freeze=freeze)
        from src.models.two_stream_v15 import TwoStreamV15Model

        # probe_action.py:load_encoder('parvo')와 동일 구성
        self.model = TwoStreamV15Model(
            pair_mode=True, use_sobel=False, masked_anchor=True,
        ).to(device)
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, _ = self.model.load_state_dict(sd, strict=False)
            enc_missing = [k for k in missing
                           if k.startswith(("blocks_p", "patch_embed_p", "cls_token_p", "pos_embed_p"))]
            assert not enc_missing, f"Parvo: P encoder 가중치 미로드 {enc_missing[:5]}"
        self.model.eval()

        if freeze:
            self.freeze_encoder()

        self.prev_obs: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.prev_obs = None

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape

        # P_t = 이전 프레임, P_tk = 현재 프레임 (학습 pair 순서 일치)
        if T > 1:
            prev = torch.cat([obs_seq[:, :1], obs_seq[:, :-1]], dim=1)
        else:
            if self.prev_obs is None:
                prev = obs_seq.clone()
            else:
                prev = self.prev_obs
            self.prev_obs = obs_seq.detach()

        img_prev = prev.reshape(B * T, C, H, W)
        img_curr = obs_seq.reshape(B * T, C, H, W)

        # no-Sobel: compute_p_channel가 [0,1] RGB 3ch 그대로 반환 (학습 입력과 동일)
        p_prev = self.model.preprocessing.compute_p_channel(img_prev)
        p_curr = self.model.preprocessing.compute_p_channel(img_curr)

        P_t = self.model._encode_p_unmasked(p_prev)[:, 1:].mean(dim=1)   # (B*T, 768)
        P_tk = self.model._encode_p_unmasked(p_curr)[:, 1:].mean(dim=1)

        token = torch.cat([P_t, P_tk], dim=-1)
        return token.reshape(B, T, -1)
