"""
Base two-stream module (shared encoders, masking, decoder).

This class holds the shared building blocks that CoMP-MAE (CoMPMAE,
see comp_mae.py) builds on: the two independent stream encoders (M/P),
mask tokens, decoder position embeddings, the motion-routing / interpreter
decoder modules, and the pixel reconstruction head, plus a handful of masking
and (un)patchify helpers.

It is not used on its own in this release -- CoMPMAE subclasses it and
provides the CoMP-MAE / plain-control forward passes. Some modules created here
(e.g. `interpreter_1`, the M decoder) belong to an earlier objective and are
deleted or frozen by the subclass; this is documented at each construction site.

Stream conventions (no-Sobel setting used by CoMP-MAE):
    M encoder input = |ΔL| (motion), 1 channel, depth m_depth (small motion sensor)
    P encoder input = RGB  (appearance), 3 channels, depth p_depth (semantic backbone)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import TwoStreamPreprocessing
from .common.blocks import TransformerBlock, MotionRoutingBlock


class TwoStreamBase(nn.Module):
    """Shared two-stream encoder/decoder components (base class).

    Args:
        embed_dim: Transformer embedding dim (M/P share it so routing Q/K/V
            projections match on head_dim).
        p_depth: P encoder depth (default 12, semantic backbone).
        m_depth: M encoder depth (default 6, motion sensor -- kept small).
        num_heads: attention heads.
        mlp_ratio: MLP expansion ratio.
        image_size / patch_size: ViT input geometry.
        mask_ratio_m / mask_ratio_p: per-stream mask ratios.
        decoder_depth_m: M decoder depth.
        interpreter_depth: interpreter_1 / interpreter_2 depth.
        num_motion_iters: motion-routing iterations.
        rotation_aug / independent_rotation_prob: training-time rotation augmentation.
        routing_mode: value-ownership routing mode (see MotionRoutingBlock).
        use_sobel: True  -> M=3ch (|ΔL|+Sobel), P=5ch (Sobel+RGB);
                   False -> M=1ch (|ΔL|),       P=3ch (RGB)   [CoMP-MAE setting].
    """

    def __init__(
        self,
        embed_dim: int = 768,
        p_depth: int = 12,
        m_depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio_m: float = 0.3,
        mask_ratio_p: float = 0.75,
        decoder_depth_m: int = 3,
        interpreter_depth: int = 3,
        num_motion_iters: int = 2,
        rotation_aug: bool = False,
        independent_rotation_prob: float = 0.1,
        routing_mode: str = "v_from_p",
        use_sobel: bool = True,
    ):
        super().__init__()
        self.routing_mode = routing_mode
        self.use_sobel = use_sobel

        self.embed_dim = embed_dim
        self.p_depth = p_depth
        self.m_depth = m_depth
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patches_per_side = image_size // patch_size
        self.mask_ratio_m = mask_ratio_m
        self.mask_ratio_p = mask_ratio_p
        self.num_motion_iters = num_motion_iters
        self.rotation_aug = rotation_aug
        self.independent_rotation_prob = independent_rotation_prob

        # ── Channel preprocessing (RGB pair -> M / P channels) ─────────────
        self.preprocessing = TwoStreamPreprocessing(use_sobel=use_sobel)
        m_in_ch = 3 if use_sobel else 1
        p_in_ch = 5 if use_sobel else 3

        # ── Stream-independent patch embeddings ────────────────────────────
        self.patch_embed_m = nn.Conv2d(
            in_channels=m_in_ch, out_channels=embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.patch_embed_p = nn.Conv2d(
            in_channels=p_in_ch, out_channels=embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # CLS tokens (M=1, P=1)
        self.cls_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_p = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Absolute position embeddings for the encoders (CLS + N patches)
        self.pos_embed_m = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_embed_p = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Encoder transformer blocks (M=m_depth, P=p_depth)
        self.blocks_m = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(m_depth)
        ])
        self.blocks_p = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(p_depth)
        ])
        self.norm_m = nn.LayerNorm(embed_dim)
        self.norm_p = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token_m, std=0.02)
        nn.init.trunc_normal_(self.cls_token_p, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_m, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_p, std=0.02)

        # ── Mask tokens ────────────────────────────────────────────────────
        self.mask_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token_p = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token_m, std=0.02)
        nn.init.trunc_normal_(self.mask_token_p, std=0.02)

        # ── Decoder position embeddings (added after mask-token injection) ──
        self.dec_pos_embed_m = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dec_pos_embed_p = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.dec_pos_embed_m, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed_p, std=0.02)

        # ── Decoder modules ────────────────────────────────────────────────
        # NOTE: `interpreter_1` and the M decoder below belong to an earlier
        #   objective. CoMPMAE DELETES them for both released
        #   objectives (they are absent from the released checkpoints); they are
        #   constructed here only so the base __init__ stays self-contained.
        #   `motion_routing` and `interpreter_2` are kept but FROZEN by the
        #   subclass (present in the checkpoints, no gradient). `recon_head` is
        #   used actively (P-recon pixel head).
        self.m_decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(decoder_depth_m)
        ])
        self.m_decoder_norm = nn.LayerNorm(embed_dim)

        self.interpreter_1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(interpreter_depth)
        ])
        self.interpreter_1_norm = nn.LayerNorm(embed_dim)

        self.motion_routing = nn.ModuleList([
            MotionRoutingBlock(embed_dim, num_heads, mlp_ratio, routing_mode=routing_mode)
            for _ in range(num_motion_iters)
        ])

        self.interpreter_2 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(interpreter_depth)
        ])
        self.interpreter_2_norm = nn.LayerNorm(embed_dim)

        # Pixel reconstruction head: D -> patch_size^2 * 3 (standard MAE head).
        self.recon_head = nn.Linear(embed_dim, patch_size * patch_size * 3)

    # ----------------------------------------------------------------------
    # Masking / sequence helpers (used by the CoMP-MAE forward passes)
    # ----------------------------------------------------------------------

    def _random_mask(self, B: int, device: torch.device, ratio: float) -> torch.Tensor:
        """Random mask: returns [B, N] bool, True=masked."""
        N = self.num_patches
        num_masked = int(ratio * N)
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :N - num_masked], False)
        return mask

    def _encode_stream_visible(
        self,
        tokens: torch.Tensor,        # [B, 1+N, D] CLS + patches with APE already added
        mask: torch.Tensor,          # [B, N] bool, True=masked
        blocks: nn.ModuleList,       # encoder blocks for this stream
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        """Visible-only encoding: drop masked patches, encode CLS + visible.

        Returns [B, 1 + N_vis, D] (CLS + visible patches).
        """
        cls = tokens[:, :1]
        patches = tokens[:, 1:]
        B, N, D = patches.shape

        visible = patches[~mask].reshape(B, -1, D)
        x = torch.cat([cls, visible], dim=1)

        for block in blocks:
            x = block(x, freqs_cis=None)  # APE path (freqs_cis=None)
        x = norm(x)
        return x

    def _inject_mask_tokens(
        self,
        encoded: torch.Tensor,       # [B, 1 + N_vis, D]
        mask: torch.Tensor,          # [B, N] bool
        mask_token: torch.Tensor,    # [1, 1, D]
    ) -> torch.Tensor:
        """Restore the full sequence by inserting mask tokens at masked positions.

        Returns [B, 1 + N, D] (CLS + all patches in original order).
        """
        cls = encoded[:, :1]
        visible = encoded[:, 1:]        # [B, N_vis, D]
        B, _, D = visible.shape
        N = mask.shape[1]

        full_patches = mask_token.expand(B, N, -1).clone()
        full_patches[~mask] = visible.reshape(-1, D)
        return torch.cat([cls, full_patches], dim=1)

    def _unpatchify(self, patch_preds: torch.Tensor) -> torch.Tensor:
        """[B, N, ps^2 * 3] -> [B, 3, H, W]."""
        B, N, _ = patch_preds.shape
        P = self.num_patches_per_side
        ps = self.patch_size
        img = patch_preds.reshape(B, P, P, ps, ps, 3)
        img = img.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, 3, P, ps, P, ps]
        return img.reshape(B, 3, P * ps, P * ps)

    def _patchify(self, img: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, N, ps^2 * C]  (C=3 for RGB, C=1 for |ΔL|)."""
        B, C, H, W = img.shape
        ps = self.patch_size
        P = H // ps
        x = img.reshape(B, C, P, ps, P, ps)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, P, P, ps, ps, C]
        return x.reshape(B, P * P, ps * ps * C)
