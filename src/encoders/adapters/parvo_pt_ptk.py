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
import torch.nn.functional as F

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
        anchor_cams: Optional[dict] = None,
        dinov2_cams: tuple = (),
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

        # α 앵커 모드 (docs/alpha_plan.md §3): 카메라별 입력 형식 = 앵커 P (+ 앵커→현재 누적 변화 한 장).
        #   now = 매 스텝 현재 P(★ 와 같은 1 stream) · p = 앵커 P 만 · pm = 앵커 P + CoMP M(ΔL 앵커→t)
        #   · praw = 앵커 P + raw ΔL 16×16 패치(alpha_readout raw 토큰화, 0 패딩만 생략 = 256-d 자기 query)
        #   · mo = 창 슬롯엔 M 만, 현재 앵커 P 는 정책이 창 앞 토큰으로 한 번만 넣음(anchor_p_once · 두 카메라 함께)
        # 앵커 나이는 입력이 아님(표집 분포만) — 어댑터는 앵커 프레임 인덱스만 받는다.
        # ★(anchor_cams=None)는 아래를 전혀 타지 않음 → 파라미터 생성 순서·RNG 소비 불변.
        self.anchor_cams = dict(anchor_cams) if anchor_cams else None
        self.dinov2_cams = tuple(dinov2_cams or ())
        if self.anchor_cams:
            assert motion_source == "none" and not use_m and motion_gap is None and motion_dropout_p == 0, \
                "앵커 모드는 E3 스트림 옵션과 배타 (motion_source=none 기반)"
            assert all(m in ("now", "p", "pm", "praw", "mo") for m in self.anchor_cams.values()), self.anchor_cams
            assert len({m == "mo" for m in self.anchor_cams.values()}) == 1, "mo 는 두 카메라 함께"
            assert set(self.dinov2_cams) <= set(self.anchor_cams), self.dinov2_cams
            self.dino = None
            if self.dinov2_cams:
                # 외부 P = DINOv2-base. alpha_readout/observer_fuse.DinoP 와 같은 규약:
                # [0,1] 224 → bilinear 196 → ImageNet 정규화(DINO 자체 계약) → 14px 패치 196 토큰(768-d)
                from .single_frame import SingleFrameAdapter
                self.dino = SingleFrameAdapter("dinov2", freeze=True)
            dims = {"p": _ed, "dp": 768, "m": _ed, "raw": 256}
            kinds = sorted({k for c in self.anchor_cams for k in self._anchor_streams(c) + [self._p_kind(c)]} - {"p"})
            if pooling == "attentive":
                # C1-DN P query 는 ★ 와 같은 pool_q[0] 재사용, 나머지 stream 종류별 query 1개(카메라 공유 = ★ 규약)
                self.anchor_q = nn.ParameterDict({k: nn.Parameter(torch.randn(dims[k]) * 0.02) for k in kinds})
            self._anchor_dims = dims

    def train(self, mode: bool = True):
        # frozen encoder는 항상 eval (BN/dropout train↔inference parity). pool_q는 Parameter라 무관.
        super().train(mode)
        self.model.eval()
        if getattr(self, "dino", None) is not None:
            self.dino.eval()
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

    # ── α 앵커 모드 ────────────────────────────────────────────────────────
    def _p_kind(self, cam: str) -> str:
        return "dp" if cam in self.dinov2_cams else "p"

    def _anchor_streams(self, cam: str) -> list:
        p = self._p_kind(cam)
        return {"now": [p], "p": [p], "pm": [p, "m"], "praw": [p, "raw"], "mo": ["m"]}[self.anchor_cams[cam]]

    def cam_embed_dim(self, cam: str) -> int:
        return sum(self._anchor_dims[k] for k in self._anchor_streams(cam))

    def cam_p_dim(self, cam: str) -> int:
        return self._anchor_dims[self._p_kind(cam)]

    def _pool_kind(self, kind: str, tok: torch.Tensor) -> torch.Tensor:
        if self.pooling != "attentive":
            return tok.mean(dim=1)
        if kind == "p":
            return self._attn_pool(tok, self.pool_q[0])
        q = self.anchor_q[kind]
        attn = ((tok @ q) * q.shape[-1] ** -0.5).softmax(dim=-1)
        return (attn.unsqueeze(-1) * tok).sum(dim=1)

    def encode_p_pooled(self, cam: str, x: torch.Tensor) -> torch.Tensor:
        """x (N, 3, 224, 224) [0,1] → pooled P (N, D). 롤아웃 캐시 단위(앵커 프레임에서만 호출)."""
        kind = self._p_kind(cam)
        with torch.set_grad_enabled(not self._freeze):
            if kind == "dp":
                tok = self.dino.encode_patch_tokens(
                    F.interpolate(x, size=(196, 196), mode="bilinear", align_corners=False))
            else:
                tok = self.model._encode_p_unmasked(self.model.preprocessing.compute_p_channel(x))[:, 1:]
        return self._pool_kind(kind, tok)

    def encode_change_pooled(self, cam: str, x_anchor: torch.Tensor, x_cur: torch.Tensor) -> Optional[torch.Tensor]:
        """ΔL(앵커 → 현재) 한 장 → pooled 변화 stream (N, D) · 변화 채널 없는 모드면 None."""
        kind = next((k for k in self._anchor_streams(cam) if k in ("m", "raw")), None)
        if kind is None:
            return None
        with torch.set_grad_enabled(not self._freeze):
            dl = self.model.preprocessing.compute_m_channel(x_anchor, x_cur)   # (N, 1, 224, 224)
            if kind == "m":
                tok = self.model._encode_m_unmasked(dl)[:, 1:]
            else:
                tok = F.unfold(dl, kernel_size=16, stride=16).transpose(1, 2)   # (N, 196, 256)
        return self._pool_kind(kind, tok)

    def forward_anchor(self, obs_seq: torch.Tensor, anchor_idx: torch.Tensor, cam: str) -> torch.Tensor:
        """학습 경로. obs_seq (B, T_in, C, H, W) · anchor_idx (B, T) = 각 정책 위치의 앵커 프레임 인덱스(obs_seq 기준).
        정책 위치 = obs_seq 의 마지막 T 프레임. 'now' 카메라는 anchor_idx 무시(현재 P).
        → (B, T, cam_embed_dim). 같은 앵커 프레임은 한 번만 인코딩."""
        B, T_in, C, H, W = obs_seq.shape
        T = anchor_idx.shape[1]
        flat = obs_seq.reshape(B * T_in, C, H, W)
        base = (torch.arange(B, device=obs_seq.device) * T_in)[:, None]
        cur = (base + (T_in - T) + torch.arange(T, device=obs_seq.device)[None]).reshape(-1)
        anc = (base + anchor_idx).reshape(-1)
        feats = []
        if self.anchor_cams[cam] != "mo":   # mo: 슬롯에 P 없음 (앵커 P 는 anchor_p_once)
            src = cur if self.anchor_cams[cam] == "now" else anc
            uniq, inv = torch.unique(src, return_inverse=True)
            feats.append(self.encode_p_pooled(cam, flat[uniq])[inv])
        ch = self.encode_change_pooled(cam, flat[anc], flat[cur])
        if ch is not None:
            feats.append(ch)
        return torch.cat(feats, dim=-1).reshape(B, T, -1)

    def anchor_p_once(self, obs_seq: torch.Tensor, anchor_idx: torch.Tensor, cam: str) -> torch.Tensor:
        """mo 학습 경로: 창 마지막(현재) 위치의 앵커 프레임 P 한 장 → pooled (B, D_P)."""
        B = obs_seq.shape[0]
        return self.encode_p_pooled(cam, obs_seq[torch.arange(B, device=obs_seq.device), anchor_idx[:, -1]])
