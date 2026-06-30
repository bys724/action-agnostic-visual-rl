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
import torch.nn as nn

from .base import EncoderAdapter

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class ParvoPtPtkAdapter(EncoderAdapter):
    """P_t ⊕ P_tk readout. arch(dim/head/depth)는 ckpt에서 추론 → ViT-S(384,CoMP-MAE-S)/
    ViT-B(768) 자동 대응. pooling='mean'(기존)|'attentive'(stream별 learnable query).
    attentive 시 query만 trainable(encoder frozen) → BC loss로 policy와 동시학습.
    """
    img_size = 224

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
        device: str = "cpu",
        pooling: str = "mean",
        use_m: bool = False,
        **kwargs,  # build_adapter가 넘기는 잉여 인자 무시
    ):
        super().__init__(freeze=freeze)
        from src.models.two_stream_v15 import TwoStreamV15Model

        assert checkpoint_path is not None, "ParvoPtPtkAdapter requires checkpoint"
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # arch 추론 (probe_action.py parvo 경로와 동일). head_dim=64 표준.
        _ed = next(v.shape[-1] for k, v in sd.items() if k == "pos_embed_p")
        _md = len({k.split(".")[1] for k in sd if k.startswith("blocks_m.")})
        _comp = any("m_recon" in k for k in sd)
        self.base_dim = _ed
        self.use_m = use_m
        # P_t ⊕ P_tk (⊕ M) — use_m 시 M(ΔL 현재−직전) stream 추가. instance(ckpt별 384/768)
        self.n_streams = 3 if use_m else 2
        self.embed_dim = _ed * self.n_streams
        self.pooling = pooling

        self.model = TwoStreamV15Model(
            embed_dim=_ed, num_heads=_ed // 64, m_depth=_md, comp_mae=_comp,
            pair_mode=True, use_sobel=False, masked_anchor=True,
        ).to(device)
        missing, _ = self.model.load_state_dict(sd, strict=False)
        enc_missing = [k for k in missing
                       if k.startswith(("blocks_p", "patch_embed_p", "cls_token_p", "pos_embed_p"))]
        assert not enc_missing, f"Parvo: P encoder 가중치 미로드 {enc_missing[:5]}"
        self.model.eval()

        # attentive pooling: stream별(P_t, P_tk[, M]) learnable query 1개씩 (single-head, 최소 capacity)
        if pooling == "attentive":
            self.pool_q = nn.Parameter(torch.randn(self.n_streams, _ed) * 0.02)
            self.pool_scale = _ed ** -0.5
        elif pooling != "mean":
            raise ValueError(f"pooling must be 'mean'|'attentive', got {pooling}")

        if freeze:
            # encoder backbone만 freeze. attentive pool_q는 trainable 유지 → optimizer 자동 수거.
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        self.prev_obs: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        # frozen encoder는 항상 eval (BN/dropout train↔inference parity). pool_q는 Parameter라 무관.
        super().train(mode)
        self.model.eval()
        return self

    def reset(self) -> None:
        self.prev_obs = None

    def _attn_pool(self, tokens: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # tokens (BT, N, D), q (D,) → softmax-weighted patch sum (BT, D)
        attn = (tokens @ q) * self.pool_scale          # (BT, N)
        attn = attn.softmax(dim=-1)
        return (attn.unsqueeze(-1) * tokens).sum(dim=1)

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

        # frozen encoder forward는 grad 불필요 (#3: autograd bookkeeping 제거 → 메모리·속도).
        # pool_q(trainable)는 no_grad 밖에서 encoder *출력*에 작용 → grad는 pool_q로만 흐름.
        with torch.set_grad_enabled(not self._freeze):
            # no-Sobel: compute_p_channel가 [0,1] RGB 3ch 그대로 반환 (학습 입력과 동일)
            p_prev = self.model.preprocessing.compute_p_channel(img_prev)
            p_curr = self.model.preprocessing.compute_p_channel(img_curr)

            tok_t = self.model._encode_p_unmasked(p_prev)[:, 1:]    # (B*T, N, D)
            tok_tk = self.model._encode_p_unmasked(p_curr)[:, 1:]

            feats = [tok_t, tok_tk]
            if self.use_m:
                # M = ΔL(현재, 직전) motion. ⚠️ rollout gap=1(연속프레임) vs 학습 gap~15 분포차 주의
                m_chan = self.model.preprocessing.compute_m_channel(img_prev, img_curr)
                feats.append(self.model._encode_m_unmasked(m_chan)[:, 1:])

        if self.pooling == "attentive":
            pooled = [self._attn_pool(t, self.pool_q[i]) for i, t in enumerate(feats)]
        else:
            pooled = [t.mean(dim=1) for t in feats]

        token = torch.cat(pooled, dim=-1)
        return token.reshape(B, T, -1)
