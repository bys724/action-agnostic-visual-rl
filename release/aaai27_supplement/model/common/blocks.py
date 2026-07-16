"""
Shared transformer building blocks.

- build_2d_rope_freqs / apply_rope : 2D RoPE positional encoding (optional;
      the released models use absolute position embeddings and call the blocks
      with freqs_cis=None, but the RoPE path is kept for completeness).
- TransformerBlock                 : pre-norm ViT block (RoPE-capable).
- MotionRoutingBlock               : value-ownership cross-stream routing, the
      core of CoMP-MAE (see the paper, Sec. Method).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_rope_freqs(num_patches_per_side: int, dim: int, theta: float = 10000.0):
    """Build a 2D RoPE frequency table.

    Each patch's (row, col) coordinate is encoded into one half of `dim`.
    Returns: [N, dim//2] complex frequencies (N = num_patches_per_side^2).
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
    """Apply RoPE to attention Q or K.

    Args:
        x: [B, N(+1), H, D_head] -- index 0 is CLS when has_cls=True.
        freqs_cis: [N, D_head//2] -- per-patch complex frequencies.
        has_cls: if True, index 0 (CLS) is left un-rotated.

    Returns: [B, N(+1), H, D_head]
    """
    if has_cls:
        cls_tok = x[:, :1]
        patches = x[:, 1:]
    else:
        patches = x

    B, N, H, D = patches.shape

    # [B, N, H, D] -> [B, N, H, D//2, 2] -> complex
    patches_c = patches.float().reshape(B, N, H, D // 2, 2)
    patches_c = torch.view_as_complex(patches_c)  # [B, N, H, D//2]

    # freqs_cis: [N, D//2] -> broadcast: [1, N, 1, D//2]
    freqs = freqs_cis[:N].unsqueeze(0).unsqueeze(2).to(patches_c.device)
    patches_c = patches_c * freqs

    # complex -> real
    patches = torch.view_as_real(patches_c).reshape(B, N, H, D).type_as(x)

    if has_cls:
        return torch.cat([cls_tok, patches], dim=1)
    return patches


class TransformerBlock(nn.Module):
    """Pre-norm ViT transformer block (RoPE-capable)."""

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
            freqs_cis: [N_patches, D_head//2] -- None disables RoPE (APE path).
            attn_mask: [N, N] or [B, N, N] additive attention mask (optional).
        """
        # Self-attention (optionally with RoPE)
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
    """Value-ownership cross-stream routing (core CoMP-MAE mechanism).

    forward(v_owner_state, qk_helper_state): the value/residual comes from the
    `owner` stream; Q/K (the routing pattern) come from the `helper` stream. The
    caller decides which of P/M is owner vs. helper, so the same block mirrors
    both reconstruction directions.

    routing_mode == "v_from_p" (value-ownership routing, the paper's design):
        Q, K from the helper; V from the owner (residual = owner). I.e. "apply the
        helper's attention graph to the owner's values" -- reassemble the owner's
        content along the spatial routing pattern the helper defines.
          P-recon: owner=P, helper=M  (M->M attention, gather P)
          M-recon: owner=M, helper=P  (P->P attention, gather M)

    routing_mode == "v_from_m" (ablation, standard cross-attention):
        Q from P (queries); K, V from M (memory). P looks up motion information
        from M in the conventional way. Compared against v_from_p in the paper.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        routing_mode: str = "v_from_p",
        routing_source: str = "m",
        v_source: str = "owner",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.routing_mode = routing_mode
        # v_source (v_from_p only; "value-source scalpel" ablation): where V (and
        #   the v_p/norm_p input) is drawn from.
        #   "owner"  (default) = V from the residual/owner stream (V_M in M-recon).
        #   "helper"           = V from the Q/K helper stream (V_P in M-recon) --
        #     flips only the M-recon value ownership to surgically turn off M
        #     grounding, while the residual/subsequent state stays on the owner
        #     (P-recon difficulty unchanged = difficulty-matched control).
        self.v_source = v_source
        if v_source not in ("owner", "helper"):
            raise ValueError(f"Unknown v_source: {v_source}. Expected 'owner' or 'helper'.")
        if v_source == "helper" and routing_mode != "v_from_p":
            raise ValueError("v_source='helper' (value-source scalpel) requires routing_mode='v_from_p'.")
        # routing_source (v_from_p only): where Q/K are drawn from.
        #   "m" (default) = M (motion/|ΔL|) defines the routing pattern (main design).
        #   "p"           = RGB-routed control: Q/K also from P (RGB) -> isolates the
        #     single variable "|ΔL|-where vs RGB-where". V stays on P in both cases and
        #     the qk/norm/v modules are identical, so it is parameter-symmetric.
        self.routing_source = routing_source
        if routing_source not in ("m", "p"):
            raise ValueError(f"Unknown routing_source: {routing_source}. Expected 'm' or 'p'.")
        if routing_source == "p" and routing_mode != "v_from_p":
            raise ValueError("routing_source='p' (RGB-routed control) requires routing_mode='v_from_p'.")

        if routing_mode == "v_from_p":
            # Q, K from M; V from P
            self.norm_m = nn.LayerNorm(embed_dim)
            self.qk_m = nn.Linear(embed_dim, embed_dim * 2)
            self.norm_p = nn.LayerNorm(embed_dim)
            self.v_p = nn.Linear(embed_dim, embed_dim)
        elif routing_mode == "v_from_m":
            # Q from P; K, V from M (standard cross-attention)
            self.norm_p = nn.LayerNorm(embed_dim)
            self.q_p = nn.Linear(embed_dim, embed_dim)
            self.norm_m = nn.LayerNorm(embed_dim)
            self.kv_m = nn.Linear(embed_dim, embed_dim * 2)
        else:
            raise ValueError(
                f"Unknown routing_mode: {routing_mode}. "
                "Expected 'v_from_p' (default) or 'v_from_m' (ablation)."
            )

        # Both modes share the same proj_out + FFN -> matched parameter count.
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        # Post-attention FFN
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, v_owner_state: torch.Tensor, qk_helper_state: torch.Tensor) -> torch.Tensor:
        """Routing / cross-attention. V is drawn from the owner and the residual
        attaches to the owner (V is the reconstruction target).

        The argument names are generic: the mechanism only fixes the owner/helper
        roles; the P/M meaning is assigned by the caller. A single v_from_p +
        routing_source='m' instance is used as:
          - P-recon: forward(v_owner=P_state, qk_helper=M_completed) -> M->M attn, gather P
          - M-recon: forward(v_owner=M_state, qk_helper=P_full)      -> P->P attn, gather M
        (a separate instance per direction = parameter symmetry). The v_from_m
        ablation instead draws K/V from the helper (standard cross-attention).

        Args:
            v_owner_state:   [B, N+1, D] -- value-owning stream (residual target)
            qk_helper_state: [B, N+1, D] -- Q/K helper stream
        Returns:
            [B, N+1, D] updated owner state.
        """
        B, N, D = v_owner_state.shape

        if self.routing_mode == "v_from_p":
            # routing_source="p" (RGB-routed control): Q/K also from the owner.
            # The module is identical; only the input is swapped.
            qk_src = v_owner_state if self.routing_source == "p" else qk_helper_state
            qk = self.qk_m(self.norm_m(qk_src)).reshape(
                B, N, 2, self.num_heads, self.head_dim,
            )
            q, k = qk.unbind(dim=2)
            # v_source="helper" (scalpel): gather V from the helper; residual stays on owner.
            v_src = v_owner_state if self.v_source == "owner" else qk_helper_state
            v = self.v_p(self.norm_p(v_src)).reshape(
                B, N, self.num_heads, self.head_dim,
            )
        else:  # v_from_m -- standard cross-attn: Q=owner, K/V=helper (ablation)
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

        # Residual + projection (attaches to the owner)
        out = v_owner_state + self.proj_out(attn)

        # FFN + residual
        out = out + self.ffn(self.norm_ffn(out))

        return out
