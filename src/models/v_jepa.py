"""
V-JEPA (Video Joint-Embedding Predictive Architecture) — 2-frame variant for EgoDex.

Reference:
    Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video."
    arXiv:2404.08471, 2024.

This is our re-implementation of V-JEPA 1 adapted to 2-frame input on EgoDex,
for direct comparison against Two-Stream (same data, different method).

Key differences vs Two-Stream:
- Target is FEATURE prediction (L1 in embedding space), not pixel reconstruction
- Uses EMA y-encoder as target (stop-gradient)
- Separate narrow predictor network

2-frame adaptation:
- Two frames → 1 tubelet group (tubelet_size=2) → 196 spatial tokens
- Masking is spatial-only (single temporal group collapses temporal masking)
- Multi-block masking: sample random spatial blocks, complement of mask = x
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_stream import TransformerBlock, build_2d_rope_freqs


# ============================================================================
# Tubelet (2-frame) patch embedding
# ============================================================================

class TubeletEmbed(nn.Module):
    """3D conv that projects (B, 2, 3, H, W) → (B, 196, embed_dim).

    tubelet_size=2 fuses 2 frames into a single temporal token group.
    Equivalent to a 2D ViT processing a 6-channel stacked image.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        tubelet_size: int = 2,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.tubelet_size = tubelet_size

        # 3D conv: kernel=(tubelet_size, patch_size, patch_size)
        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] where T=2

        Returns:
            tokens: [B, N, D] where N = (T/tubelet_size) * num_patches
                    with T=2, tubelet_size=2 → N = num_patches = 196
        """
        # [B, T, C, H, W] → [B, C, T, H, W] for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.projection(x)  # [B, D, T', H', W']
        B, D, Tp, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, Tp * Hp * Wp, D)  # [B, N, D]
        return x


# ============================================================================
# V-JEPA encoder (ViT backbone)
# ============================================================================

class VJEPAEncoder(nn.Module):
    """Vision Transformer backbone for V-JEPA.

    Standard ViT with 2D RoPE. No CLS token (V-JEPA uses patch tokens only).
    Used as both x-encoder (trained) and y-encoder (EMA target).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        tubelet_size: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = TubeletEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
        )
        self.num_patches = self.patch_embed.num_patches  # 196

        # 2D RoPE (spatial only, since tubelet collapses temporal)
        self.head_dim = embed_dim // num_heads
        freqs = build_2d_rope_freqs(
            num_patches_per_side=self.patch_embed.num_patches_per_side,
            dim=self.head_dim,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        visible_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T=2, C, H, W]
            visible_indices: [B, N_visible] token indices to process.
                             If None, processes all tokens.

        Returns:
            features: [B, N_out, D] where N_out = len(visible_indices) or num_patches
        """
        tokens = self.patch_embed(x)  # [B, num_patches, D]

        if visible_indices is not None:
            # Gather visible tokens per batch
            B, N, D = tokens.shape
            idx = visible_indices.unsqueeze(-1).expand(-1, -1, D)
            tokens = torch.gather(tokens, dim=1, index=idx)  # [B, N_visible, D]
            freqs_cis = self._gather_freqs(visible_indices)
        else:
            freqs_cis = self.freqs_cis

        # Run through blocks (TransformerBlock uses has_cls=True default; we set False)
        for block in self.blocks:
            tokens = self._block_forward(block, tokens, freqs_cis)

        tokens = self.norm(tokens)
        return tokens

    def _gather_freqs(self, visible_indices: torch.Tensor) -> torch.Tensor:
        """Gather RoPE frequencies for visible tokens.

        Note: freqs_cis is [N, head_dim/2] (complex). Different batches may have
        different visible indices, so we handle per-batch. Since RoPE application
        is done inside block.forward, we need a per-batch freqs.

        For simplicity, return full freqs_cis and trust that block uses [:N_visible]
        positions. This WORKS if batch sizes are the same, but may not correctly
        apply position-specific RoPE for different indices. For V-JEPA with global
        masking (same mask per batch), this is fine.

        For per-batch masks, would need to restructure block forward to accept
        position-specific freqs. Deferred to later optimization.
        """
        # Use first batch's indices (assumes shared mask across batch)
        idx = visible_indices[0]  # [N_visible]
        return self.freqs_cis[idx]

    def _block_forward(self, block, tokens, freqs_cis):
        """Run a TransformerBlock without CLS token handling."""
        # TransformerBlock.forward has has_cls=True default in apply_rope.
        # For V-JEPA we have no CLS, so apply RoPE to all tokens.
        h = block.norm1(tokens)
        B, N, D = h.shape
        qkv = block.qkv(h).reshape(B, N, 3, block.num_heads, block.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, N, H, D_head]

        if freqs_cis is not None:
            from .two_stream import apply_rope
            q = apply_rope(q, freqs_cis, has_cls=False)
            k = apply_rope(k, freqs_cis, has_cls=False)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, D)

        tokens = tokens + block.proj(attn)
        tokens = tokens + block.mlp(block.norm2(tokens))
        return tokens


# ============================================================================
# V-JEPA predictor (narrow transformer)
# ============================================================================

class VJEPAPredictor(nn.Module):
    """Predict masked token features from visible tokens.

    Follows V-JEPA paper: narrow transformer (12 blocks, embed_dim=384).
    Takes visible x-encoder features + learnable mask tokens, outputs predictions
    at masked positions.
    """

    def __init__(
        self,
        encoder_embed_dim: int = 768,
        pred_embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_patches: int = 196,
    ):
        super().__init__()
        self.pred_embed_dim = pred_embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.num_patches = num_patches

        # Project encoder features to predictor space
        self.input_proj = nn.Linear(encoder_embed_dim, pred_embed_dim)

        # Learnable mask token (shared)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding (learnable, full length)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Narrow transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(pred_embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(pred_embed_dim)

        # Project back to encoder feature dim (for L1 loss against y-encoder)
        self.output_proj = nn.Linear(pred_embed_dim, encoder_embed_dim)

    def forward(
        self,
        visible_features: torch.Tensor,
        visible_indices: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visible_features: [B, N_visible, D_enc] from x-encoder
            visible_indices: [B, N_visible] positions of visible tokens
            masked_indices: [B, N_masked] positions of masked tokens

        Returns:
            predicted_features: [B, N_masked, D_enc] predictions for masked positions
        """
        B, N_vis, _ = visible_features.shape
        N_masked = masked_indices.shape[1]
        N_total = N_vis + N_masked  # should equal num_patches

        # Project visible features to predictor dim
        vis = self.input_proj(visible_features)  # [B, N_visible, D_pred]

        # Add positional embedding to visible tokens
        vis_pos = self._gather_pos(visible_indices)  # [B, N_visible, D_pred]
        vis = vis + vis_pos

        # Create mask tokens with positional embedding
        mask_pos = self._gather_pos(masked_indices)  # [B, N_masked, D_pred]
        masks = self.mask_token.expand(B, N_masked, -1) + mask_pos  # [B, N_masked, D_pred]

        # Concatenate visible + mask tokens (order: visible first, then masked)
        x = torch.cat([vis, masks], dim=1)  # [B, N_total, D_pred]

        # Run through predictor blocks (no RoPE — pos embedding is additive)
        for block in self.blocks:
            x = self._block_forward(block, x)

        x = self.norm(x)

        # Return only the masked position outputs
        pred_masked = x[:, N_vis:]  # [B, N_masked, D_pred]
        return self.output_proj(pred_masked)  # [B, N_masked, D_enc]

    def _gather_pos(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather positional embedding for given indices.

        Args:
            indices: [B, N] token positions

        Returns:
            [B, N, D_pred]
        """
        B, N = indices.shape
        pos = self.pos_embed.expand(B, -1, -1)  # [B, num_patches, D_pred]
        idx = indices.unsqueeze(-1).expand(-1, -1, self.pred_embed_dim)
        return torch.gather(pos, dim=1, index=idx)

    def _block_forward(self, block, tokens):
        """Predictor uses no RoPE (additive pos embedding instead)."""
        h = block.norm1(tokens)
        B, N, D = h.shape
        qkv = block.qkv(h).reshape(B, N, 3, block.num_heads, block.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, D)

        tokens = tokens + block.proj(attn)
        tokens = tokens + block.mlp(block.norm2(tokens))
        return tokens


# ============================================================================
# Multi-block masking
# ============================================================================

def sample_multi_block_mask(
    num_patches_per_side: int = 14,
    num_blocks: int = 4,
    block_scale_range: tuple = (0.15, 0.2),
    aspect_range: tuple = (0.75, 1.5),
    max_mask_ratio: float = 0.9,
) -> torch.Tensor:
    """Sample a multi-block mask on a num_patches_per_side × num_patches_per_side grid.

    Simplified version of V-JEPA's multi-block masking:
    - Sample `num_blocks` rectangular blocks of random aspect & scale
    - Mark their union as masked (True)
    - Rejection-sample if total mask exceeds max_mask_ratio

    Returns:
        mask: [num_patches_per_side^2] bool tensor. True = masked.
    """
    H = W = num_patches_per_side
    total = H * W
    mask = torch.zeros(H, W, dtype=torch.bool)

    for _ in range(num_blocks):
        # random scale (fraction of area)
        scale = torch.empty(1).uniform_(*block_scale_range).item()
        aspect = torch.empty(1).uniform_(*aspect_range).item()

        block_area = total * scale
        block_h = max(1, int(round((block_area * aspect) ** 0.5)))
        block_w = max(1, int(round((block_area / aspect) ** 0.5)))
        block_h = min(block_h, H)
        block_w = min(block_w, W)

        # random top-left
        top = torch.randint(0, H - block_h + 1, (1,)).item()
        left = torch.randint(0, W - block_w + 1, (1,)).item()

        mask[top:top + block_h, left:left + block_w] = True

    mask = mask.reshape(-1)

    # Cap mask ratio
    ratio = mask.float().mean().item()
    if ratio > max_mask_ratio:
        # Randomly unmask a fraction
        masked_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        num_to_keep = int(total * max_mask_ratio)
        if masked_idx.numel() > num_to_keep:
            perm = torch.randperm(masked_idx.numel())
            drop = masked_idx[perm[num_to_keep:]]
            mask[drop] = False

    return mask


# ============================================================================
# V-JEPA Model (top-level)
# ============================================================================

class VJEPAModel(nn.Module):
    """Full V-JEPA model: x-encoder + predictor + EMA y-encoder.

    Training objective:
        L = || predictor(x_encoder(masked)) − sg(y_encoder(full)) ||_1

    Inference: use x_encoder as the frozen feature extractor.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        pred_embed_dim: int = 384,
        pred_depth: int = 12,
        pred_num_heads: int = 12,
        mask_num_blocks: int = 4,
        mask_scale_range: tuple = (0.15, 0.2),
        max_mask_ratio: float = 0.9,
        ema_momentum_start: float = 0.998,
        ema_momentum_end: float = 1.0,
    ):
        super().__init__()
        self.mask_num_blocks = mask_num_blocks
        self.mask_scale_range = mask_scale_range
        self.max_mask_ratio = max_mask_ratio
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end
        self._ema_momentum = ema_momentum_start

        # x-encoder (online, trained)
        self.x_encoder = VJEPAEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # y-encoder (target, EMA of x-encoder, stop-gradient)
        self.y_encoder = copy.deepcopy(self.x_encoder)
        for p in self.y_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = VJEPAPredictor(
            encoder_embed_dim=embed_dim,
            pred_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=pred_num_heads,
            mlp_ratio=mlp_ratio,
            num_patches=self.x_encoder.num_patches,
        )

        self.num_patches = self.x_encoder.num_patches
        self.num_patches_per_side = self.x_encoder.patch_embed.num_patches_per_side

    @torch.no_grad()
    def update_ema(self, momentum: float = None):
        """Update y-encoder as EMA of x-encoder. Call after each optimizer step."""
        m = momentum if momentum is not None else self._ema_momentum
        for p_y, p_x in zip(self.y_encoder.parameters(), self.x_encoder.parameters()):
            p_y.data.mul_(m).add_(p_x.data, alpha=1 - m)

    def set_ema_momentum(self, epoch: int, total_epochs: int):
        """Linearly anneal EMA momentum from start to end over training."""
        t = min(1.0, max(0.0, epoch / max(1, total_epochs)))
        self._ema_momentum = (
            self.ema_momentum_start * (1 - t) + self.ema_momentum_end * t
        )

    def _sample_masks(self, batch_size: int, device: torch.device):
        """Sample a shared mask for the batch (simpler than per-sample masks)."""
        mask = sample_multi_block_mask(
            num_patches_per_side=self.num_patches_per_side,
            num_blocks=self.mask_num_blocks,
            block_scale_range=self.mask_scale_range,
            max_mask_ratio=self.max_mask_ratio,
        ).to(device)

        masked_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)  # [N_masked]
        visible_idx = torch.nonzero(~mask, as_tuple=False).reshape(-1)  # [N_visible]

        # Expand to batch
        masked_idx = masked_idx.unsqueeze(0).expand(batch_size, -1)
        visible_idx = visible_idx.unsqueeze(0).expand(batch_size, -1)
        return visible_idx, masked_idx

    def compute_loss(self, img_t: torch.Tensor, img_tk: torch.Tensor):
        """Training loss.

        Args:
            img_t:  [B, C, H, W] first frame
            img_tk: [B, C, H, W] second frame

        Returns:
            loss: scalar L1 loss
            img_pred: dummy placeholder (for compatibility with train loop that expects it)
        """
        B = img_t.shape[0]
        device = img_t.device

        # Stack frames into [B, 2, C, H, W]
        frames = torch.stack([img_t, img_tk], dim=1)

        # Sample mask
        visible_idx, masked_idx = self._sample_masks(B, device)

        # x-encoder: visible tokens only
        x_feat = self.x_encoder(frames, visible_indices=visible_idx)
        # [B, N_visible, D_enc]

        # Predictor: predict masked feature
        pred = self.predictor(x_feat, visible_idx, masked_idx)
        # [B, N_masked, D_enc]

        # y-encoder: full sequence, no gradient
        with torch.no_grad():
            y_feat_full = self.y_encoder(frames, visible_indices=None)
            # Gather masked positions
            idx = masked_idx.unsqueeze(-1).expand(-1, -1, y_feat_full.shape[-1])
            y_target = torch.gather(y_feat_full, dim=1, index=idx)
            # [B, N_masked, D_enc]

        # L1 loss in feature space
        loss = F.l1_loss(pred, y_target.detach())

        # dummy image output (for compatibility with save_epoch_samples)
        img_pred = img_tk

        return loss, img_pred

    def forward(self, img_t: torch.Tensor, img_tk: torch.Tensor):
        """Convenience forward for training loop. Delegates to compute_loss."""
        return self.compute_loss(img_t, img_tk)

    @torch.no_grad()
    def extract_features(self, img_t: torch.Tensor, img_tk: torch.Tensor):
        """Inference-time feature extraction (for frozen downstream use).

        Uses x-encoder on full sequence (no masking).

        Returns:
            features: [B, num_patches, embed_dim]
        """
        frames = torch.stack([img_t, img_tk], dim=1)
        return self.x_encoder(frames, visible_indices=None)
