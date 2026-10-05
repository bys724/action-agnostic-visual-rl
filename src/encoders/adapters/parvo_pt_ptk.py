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
        embed_dim: Optional[int] = None,
        m_depth: Optional[int] = None,
        comp_mae: Optional[bool] = None,
        qk_norm: Optional[bool] = None,
        motion_gap: Optional[int] = None,
        motion_dropout_p: float = 0.0,
        motion_source: str = "rgb_prev",
        **kwargs,  # build_adapter가 넘기는 잉여 인자 무시
    ):
        super().__init__(freeze=freeze)
        from src.models.two_stream_v15 import TwoStreamV15Model

        if checkpoint_path is not None:
            # finetune 경로: pretrain ckpt에서 arch 추론 + encoder 가중치 로드.
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            # arch 추론 (probe_action.py parvo 경로와 동일). head_dim=64 표준.
            _ed = next(v.shape[-1] for k, v in sd.items() if k == "pos_embed_p")
            _md = len({k.split(".")[1] for k in sd if k.startswith("blocks_m.")})
            _comp = any("m_recon" in k for k in sd)
            _qk = any(".q_norm." in k for k in sd)
        else:
            # rollout 등 self-contained 경로: arch를 명시 kwargs로 받고 가중치는
            # 외부(policy_state_dict)에서 덮어씀. 추론 규약은 finetune과 동일.
            assert None not in (embed_dim, m_depth, comp_mae), (
                "checkpoint_path=None이면 embed_dim/m_depth/comp_mae 필수 "
                "(policy_state_dict에서 추론해 전달)"
            )
            sd = None
            _ed, _md, _comp = embed_dim, m_depth, comp_mae
            _qk = bool(qk_norm)
        # qk-norm 전역 스위치 — 모델 생성 전 설정 (pretrain --qk-norm parity).
        # 미설정 시 strict=False가 q_norm/k_norm 가중치를 조용히 드랍 → 아키 불일치 inference.
        from src.models.common import blocks as _blocks
        _blocks.QK_NORM_DEFAULT = _qk
        self.base_dim = _ed
        self.use_m = use_m
        # P_t ⊕ P_tk (⊕ M) — use_m 시 M(ΔL 현재−직전) stream 추가. instance(ckpt별 384/768)
        # E3 팔 (claim_spine_v2 §4.2, 같은 두 프레임 예산): rgb_prev = 외형 2장 [P_prev, P_curr](②, 기존) ·
        # comp_m = 외형 1장 + CoMP M [P_curr, M](④) · none = 외형 1장 [P_curr](①)
        assert motion_source in ("rgb_prev", "comp_m", "none"), motion_source
        assert not (use_m and motion_source != "rgb_prev"), "use_m(외형 2장+M)은 rgb_prev 전용"
        self.motion_source = motion_source
        self.n_streams = {"rgb_prev": 3 if use_m else 2, "comp_m": 2, "none": 1}[motion_source]
        self.embed_dim = _ed * self.n_streams
        self.pooling = pooling
        # E3 (claim_spine_v2 §4.3): motion_gap=g면 입력 T_in = T_out + g 시퀀스를 받아
        # 프레임 j를 j−g와 짝지음 (학습=robomimic frame_stack g+1 앞 패딩, 롤아웃=히스토리 앞 패딩).
        # None이면 기존 동작(시퀀스 내 1스텝 shift, T_out=T_in) 그대로.
        self.motion_gap = motion_gap
        # copycat 대책: 학습 중 현재 프레임 P 외 스트림(과거 프레임 P, M)을 (B·T) 단위 확률 p로 0.
        # rescale 없음(채널 마스킹 의미 유지). eval/rollout에선 비활성.
        self.motion_dropout_p = motion_dropout_p
        self.force_drop_extra = False   # 롤아웃 진단(--ablate-extra): 비현재 스트림 항상 0 = dropout 상태

        self.model = TwoStreamV15Model(
            embed_dim=_ed, num_heads=_ed // 64, m_depth=_md, comp_mae=_comp,
            pair_mode=True, use_sobel=False, masked_anchor=True,
        ).to(device)
        if sd is not None:
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            enc_missing = [k for k in missing
                           if k.startswith(("blocks_p", "patch_embed_p", "cls_token_p", "pos_embed_p"))]
            assert not enc_missing, f"Parvo: P encoder 가중치 미로드 {enc_missing[:5]}"
            _qk_dropped = [k for k in unexpected if ".q_norm." in k or ".k_norm." in k]
            assert not _qk_dropped, f"Parvo: qk-norm 가중치 드랍 {_qk_dropped[:5]}"
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
        if self.motion_gap is not None:
            g = self.motion_gap
            assert T > g, f"motion_gap={g}면 T_in > g 필요 (got T={T})"
            prev, obs_seq = obs_seq[:, :-g], obs_seq[:, g:]
            T = T - g
        elif T > 1:
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

            tok_tk = self.model._encode_p_unmasked(p_curr)[:, 1:]   # (B*T, N, D)
            if self.motion_source == "rgb_prev":
                tok_t = self.model._encode_p_unmasked(p_prev)[:, 1:]
                feats = [tok_t, tok_tk]
            elif self.motion_source == "comp_m":
                m_chan = self.model.preprocessing.compute_m_channel(img_prev, img_curr)
                feats = [tok_tk, self.model._encode_m_unmasked(m_chan)[:, 1:]]
            else:
                feats = [tok_tk]
            if self.use_m:
                # M = ΔL(현재, 직전) motion. ⚠️ rollout gap=1(연속프레임) vs 학습 gap~15 분포차 주의
                m_chan = self.model.preprocessing.compute_m_channel(img_prev, img_curr)
                feats.append(self.model._encode_m_unmasked(m_chan)[:, 1:])

        if self.pooling == "attentive":
            pooled = [self._attn_pool(t, self.pool_q[i]) for i, t in enumerate(feats)]
        else:
            pooled = [t.mean(dim=1) for t in feats]

        if (self.training and self.motion_dropout_p > 0) or self.force_drop_extra:
            # 현재 프레임 P만 유지(rgb_prev면 pooled[1], 그 외 pooled[0]). 나머지(과거 프레임 P, M)만 마스킹
            cur = 1 if self.motion_source == "rgb_prev" else 0
            for i in (i for i in range(len(pooled)) if i != cur):
                p = 1.0 if self.force_drop_extra else self.motion_dropout_p
                keep = torch.rand(pooled[i].shape[0], 1, device=pooled[i].device) >= p
                pooled[i] = pooled[i] * keep

        token = torch.cat(pooled, dim=-1)
        return token.reshape(B, T, -1)
