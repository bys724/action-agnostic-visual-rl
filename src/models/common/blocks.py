"""
Shared transformer building blocks across two-stream model versions.

Single source of truth for:
- build_2d_rope_freqs / apply_rope : 2D RoPE positional encoding
- TransformerBlock                 : RoPE-capable pre-norm ViT block
- MotionRoutingBlock               : motion-guided attention routing (Q/K=M, V=P)

이전엔 TransformerBlock이 two_stream.py(v4-10), MotionRoutingBlock이 two_stream_v11.py에
정의돼 활성 v15(Parvo/MS-JEPA/MCP-MAE)가 legacy 파일에 import 의존했음. legacy 정리 시
활성 모델이 깨지지 않도록 공용 부품만 여기로 추출 (REFACTOR_PLAN.md S1/S2).
정의는 원본과 byte-동일 — state_dict key/forward 출력 불변.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_rope_freqs(num_patches_per_side: int, dim: int, theta: float = 10000.0):
    """2D RoPE 주파수 테이블 생성.

    패치의 (row, col) 좌표를 dim의 절반씩 나눠 인코딩.
    Returns: [N, dim] complex frequencies (N = num_patches_per_side^2)
    """
    half_dim = dim // 2
    freqs_row = 1.0 / (theta ** (torch.arange(0, half_dim // 2, dtype=torch.float32) / (half_dim // 2)))
    freqs_col = 1.0 / (theta ** (torch.arange(0, half_dim // 2, dtype=torch.float32) / (half_dim // 2)))

    rows = torch.arange(num_patches_per_side, dtype=torch.float32)
    cols = torch.arange(num_patches_per_side, dtype=torch.float32)

    # [H, W, half_dim//2] for row and col separately
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
    grid_r = grid_r.reshape(-1)  # [N]
    grid_c = grid_c.reshape(-1)  # [N]

    # outer product: [N, half_dim//2]
    angles_r = torch.outer(grid_r, freqs_row)
    angles_c = torch.outer(grid_c, freqs_col)

    # concat row and col angles: [N, half_dim]
    angles = torch.cat([angles_r, angles_c], dim=-1)

    # complex form: cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # [N, half_dim]
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, has_cls: bool = True) -> torch.Tensor:
    """RoPE를 attention의 Q, K에 적용.

    Args:
        x: [B, N(+1), H, D_head] — CLS가 있으면 idx 0이 CLS
        freqs_cis: [N, D_head//2] — 패치 위치별 complex frequencies
        has_cls: True면 idx 0(CLS)은 RoPE 적용 안 함

    Returns: [B, N(+1), H, D_head]
    """
    if has_cls:
        cls_tok = x[:, :1]
        patches = x[:, 1:]
    else:
        patches = x

    B, N, H, D = patches.shape

    # [B, N, H, D] → [B, N, H, D//2, 2] → complex
    patches_c = patches.float().reshape(B, N, H, D // 2, 2)
    patches_c = torch.view_as_complex(patches_c)  # [B, N, H, D//2]

    # freqs_cis: [N, D//2] → broadcast: [1, N, 1, D//2]
    freqs = freqs_cis[:N].unsqueeze(0).unsqueeze(2).to(patches_c.device)
    patches_c = patches_c * freqs

    # complex → real
    patches = torch.view_as_real(patches_c).reshape(B, N, H, D).type_as(x)

    if has_cls:
        return torch.cat([cls_tok, patches], dim=1)
    return patches


class TransformerBlock(nn.Module):
    """RoPE 지원 Transformer block (Pre-norm, ViT style)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(
        self, x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            freqs_cis: [N_patches, D_head//2] — None이면 RoPE 없이 동작
            attn_mask: [N, N] or [B, N, N] additive mask (-inf로 차단)
                        — v7-big P stream에서 CLS_P_bg ↔ CLS_P_motion 상호 attention 차단용
        """
        # Self-attention with RoPE
        h = self.norm1(x)
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [B, N, H, D_head]

        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis, has_cls=True)
            k = apply_rope(k, freqs_cis, has_cls=True)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [B, H, N, D_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, N, D)  # [B, N, D]

        x = x + self.proj(attn)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MotionRoutingBlock(nn.Module):
    """Motion-guided attention routing.

    forward(v_owner_state, qk_helper_state): V·residual은 owner, Q/K는 helper(=routing
    pattern 공급). owner/helper에 P/M 중 무엇을 넣을지는 호출부가 결정 (CoMP-MAE 미러링).

    routing_mode == "v_from_p" (v11 default, paper novelty):
        Q, K from helper, V from owner (residual=owner).
        "helper의 attention 그래프를 owner의 value에 적용" — helper가 정의하는 spatial
        routing pattern으로 owner 내용물을 재조합. P-recon: owner=P/helper=M (M→M attn,
        gather P). M-recon(CoMP-MAE): owner=M/helper=P (P→P attn, gather M).

    routing_mode == "v_from_m" (ablation, 표준 cross-attention):
        Q from P (queries), K, V from M (memory).
        P가 M에서 motion 정보를 조회하는 conventional design.
        Paper Table에서 v_from_p와 정량 비교 (NeurIPS A1 ablation).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        routing_mode: str = "v_from_p",
        routing_source: str = "m",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.routing_mode = routing_mode
        # routing_source (v_from_p 전용): Q/K를 어디서 뽑나.
        #   "m" (default) = M(motion/ΔL)이 routing pattern 결정 — 논문 핵심(MCP-MAE/MS-JEPA).
        #   "p" = SiamMAE-analog 대조군. Q/K도 P(RGB)에서 → "ΔL-where vs RGB-where" 단일변수 격리.
        #         V는 두 경우 모두 P 유지, qk_m·norm_m·v_p 모듈 동일 → param-symmetric (입력만 교체).
        self.routing_source = routing_source
        if routing_source not in ("m", "p"):
            raise ValueError(f"Unknown routing_source: {routing_source}. Expected 'm' or 'p'.")
        if routing_source == "p" and routing_mode != "v_from_p":
            raise ValueError("routing_source='p' (SiamMAE-analog)는 routing_mode='v_from_p' 전용.")

        if routing_mode == "v_from_p":
            # Q, K from M; V from P
            self.norm_m = nn.LayerNorm(embed_dim)
            self.qk_m = nn.Linear(embed_dim, embed_dim * 2)
            self.norm_p = nn.LayerNorm(embed_dim)
            self.v_p = nn.Linear(embed_dim, embed_dim)
        elif routing_mode == "v_from_m":
            # Q from P; K, V from M (표준 cross-attention)
            self.norm_p = nn.LayerNorm(embed_dim)
            self.q_p = nn.Linear(embed_dim, embed_dim)
            self.norm_m = nn.LayerNorm(embed_dim)
            self.kv_m = nn.Linear(embed_dim, embed_dim * 2)
        else:
            raise ValueError(
                f"Unknown routing_mode: {routing_mode}. "
                "Expected 'v_from_p' (default) or 'v_from_m' (ablation)."
            )

        # 두 모드 모두 동일한 proj_out + FFN — parameter count 일치
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        # Post-attention FFN
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, v_owner_state: torch.Tensor, qk_helper_state: torch.Tensor) -> torch.Tensor:
        """Routing/cross-attn: V는 owner에서, residual도 owner에 붙음 (V는 복구 대상).

        인자명 generic화 (CoMP-MAE guard 1): 메커니즘은 owner/helper 역할만 결정하고
        P/M 의미는 호출부가 부여한다. v_from_p + source='m' 인스턴스를:
          - P-recon: forward(v_owner=P_state, qk_helper=M_completed) → M→M attn, gather P
          - M-recon: forward(v_owner=M_state, qk_helper=P_full)      → P→P attn, gather M
        같은 모듈을 **인자만 swap**해 미러링 (별도 인스턴스 = param-symmetry). `v_from_m`
        ablation은 V를 helper에서 뽑는 표준 cross-attn (owner=residual target은 동일).

        Args:
            v_owner_state:  [B, N+1, D] — V 소유 stream (residual 대상)
            qk_helper_state:[B, N+1, D] — Q/K helper stream
        Returns:
            owner_state_updated: [B, N+1, D]
        """
        B, N, D = v_owner_state.shape

        if self.routing_mode == "v_from_p":
            # routing_source="p"(SiamMAE-analog): Q/K도 owner에서 (V는 항상 owner). 모듈 동일, 입력만 교체.
            qk_src = v_owner_state if self.routing_source == "p" else qk_helper_state
            qk = self.qk_m(self.norm_m(qk_src)).reshape(
                B, N, 2, self.num_heads, self.head_dim,
            )
            q, k = qk.unbind(dim=2)
            v = self.v_p(self.norm_p(v_owner_state)).reshape(
                B, N, self.num_heads, self.head_dim,
            )
        else:  # v_from_m — 표준 cross-attn: Q=owner, K/V=helper (ablation, M-recon 미사용)
            q = self.q_p(self.norm_p(v_owner_state)).reshape(
                B, N, self.num_heads, self.head_dim,
            )
            kv = self.kv_m(self.norm_m(qk_helper_state)).reshape(
                B, N, 2, self.num_heads, self.head_dim,
            )
            k, v = kv.unbind(dim=2)

        # SDPA
        q = q.transpose(1, 2)  # [B, H, N, D_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, D)

        # Residual + projection (owner에 붙음)
        out = v_owner_state + self.proj_out(attn)

        # FFN + residual
        out = out + self.ffn(self.norm_ffn(out))

        return out
