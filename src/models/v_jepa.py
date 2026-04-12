"""
V-JEPA (Bardes et al. 2024, arXiv:2404.08471) — 2-frame variant for EgoDex.

Re-implementation adapted to 2-frame input for direct comparison against
Two-Stream on the same EgoDex data. Tubelet_size=2 collapses the two frames
into one temporal group → 196 spatial tokens; masking is therefore spatial-only.
Temporal long-range context is provided by variable gap sampling (--max-gap 16,
matching V-JEPA's 16-frame clip receptive field).

Design choices vs. the official facebookresearch/jepa repo:
- L1 loss in feature space, target features LayerNorm'd (official forward_target).
- Dual mask per batch: short-range (8 small blocks, 50% masked) + long-range
  (2 large blocks, 60% masked). 원본 V-JEPA(80/85%)보다 완화 — 2-frame
  spatial-only 세팅에서는 temporal redundancy 가 없어 높은 mask ratio 가 학습
  발산을 유발 (1차, 2차 실험에서 확인). Predictor 2회 호출, losses averaged.
- Predictor: narrow transformer (12 × 384), fixed 2D sin-cos PE added to tokens.
- Target encoder: EMA of x-encoder, momentum schedule 0.998→1.0 with
  ipe_scale=1.25 (matches vitl16.yaml).
- Encoder PE: 2D RoPE instead of sin-cos. This is a deliberate deviation —
  RoPE is the current standard for modern ViT backbones (DINOv2, EVA, ViT-22B)
  and does not affect V-JEPA's prediction objective. Paper should note this
  as a "standard ViT modernization".
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_stream import TransformerBlock, build_2d_rope_freqs


# ============================================================================
# Helpers: 2D sin-cos PE (predictor), batched RoPE (encoder per-sample mask)
# ============================================================================

def build_2d_sincos_pe(grid_size: int, embed_dim: int) -> torch.Tensor:
    """Fixed 2D sinusoidal positional embedding (non-learnable).

    Encodes (row, col) grid positions into `embed_dim` with half-row / half-col.
    Standard recipe from MAE / I-JEPA / V-JEPA predictors.

    Returns:
        [N, embed_dim] float tensor where N = grid_size^2.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"
    half = embed_dim // 2  # each axis gets half

    def _1d_sincos(positions: torch.Tensor, dim: int) -> torch.Tensor:
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
        omega = 1.0 / (10000 ** omega)  # [dim/2]
        out = torch.einsum('n,d->nd', positions.float(), omega)  # [N, dim/2]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [N, dim]

    rows = torch.arange(grid_size)
    cols = torch.arange(grid_size)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
    grid_r = grid_r.reshape(-1)
    grid_c = grid_c.reshape(-1)
    pe_r = _1d_sincos(grid_r, half)
    pe_c = _1d_sincos(grid_c, half)
    return torch.cat([pe_r, pe_c], dim=1)  # [N, embed_dim]


def apply_rope_batched(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply 2D RoPE with per-sample frequencies (supports per-sample masking).

    Args:
        x: [B, N, H, D_head]
        freqs_cis: [B, N, D_head//2] complex — different token positions per sample.

    Returns:
        [B, N, H, D_head] with RoPE applied.
    """
    B, N, H, D = x.shape
    x_c = x.float().reshape(B, N, H, D // 2, 2)
    x_c = torch.view_as_complex(x_c)  # [B, N, H, D/2]
    freqs = freqs_cis.unsqueeze(2).to(x_c.device)  # [B, N, 1, D/2]
    x_c = x_c * freqs
    return torch.view_as_real(x_c).reshape(B, N, H, D).type_as(x)


# ============================================================================
# Tubelet (2-frame) patch embedding
# ============================================================================

class TubeletEmbed(nn.Module):
    """3D conv projecting [B, T, C, H, W] → [B, N, D]. tubelet_size=2 fuses 2
    frames into a single temporal token group (equivalent to a 2D ViT over
    6-channel stacked frames)."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        tubelet_size: int = 2,
    ):
        super().__init__()
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C, H, W] → [B, C, T, H, W] → Conv3d → [B, D, T', H', W']
        x = x.permute(0, 2, 1, 3, 4)
        x = self.projection(x)
        B, D, Tp, Hp, Wp = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, Tp * Hp * Wp, D)


# ============================================================================
# V-JEPA encoder (ViT backbone)
# ============================================================================

class VJEPAEncoder(nn.Module):
    """ViT backbone with 2D RoPE, no CLS token. Used as both online (x) and
    EMA target (y) encoder."""

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

        # 2D RoPE (spatial only; tubelet collapses temporal into one group)
        self.head_dim = embed_dim // num_heads
        freqs = build_2d_rope_freqs(
            num_patches_per_side=self.patch_embed.num_patches_per_side,
            dim=self.head_dim,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)  # [N, D_head/2]

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,                    # [B, T=2, C, H, W]
        visible_indices: torch.Tensor = None,  # [B, N_visible] or None for all tokens
    ) -> torch.Tensor:
        """Returns [B, N_visible_or_full, embed_dim]."""
        tokens = self.patch_embed(x)  # [B, N, D]
        B = tokens.shape[0]

        if visible_indices is not None:
            # Per-sample gather of visible tokens and their RoPE frequencies
            D = tokens.shape[-1]
            idx = visible_indices.unsqueeze(-1).expand(-1, -1, D)
            tokens = torch.gather(tokens, dim=1, index=idx)
            freqs_cis = self.freqs_cis[visible_indices]  # [B, N_vis, D_head/2]
        else:
            freqs_cis = self.freqs_cis.unsqueeze(0).expand(B, -1, -1)

        for block in self.blocks:
            tokens = self._block_forward(block, tokens, freqs_cis)

        tokens = self.norm(tokens)
        return tokens

    def _block_forward(self, block, tokens, freqs_cis):
        # TransformerBlock forward without CLS, with per-sample RoPE frequencies
        # (shape [B, N, D_head/2]). Reuses the block's sub-modules directly.
        h = block.norm1(tokens)
        B, N, D = h.shape
        qkv = block.qkv(h).reshape(B, N, 3, block.num_heads, block.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, N, H, D_head]

        q = apply_rope_batched(q, freqs_cis)
        k = apply_rope_batched(k, freqs_cis)

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
    """Narrow transformer (default 12 × 384) that maps visible x-encoder features
    to predictions at masked positions. Uses fixed 2D sin-cos PE (no RoPE)."""

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

        self.input_proj = nn.Linear(encoder_embed_dim, pred_embed_dim)
        self.output_proj = nn.Linear(pred_embed_dim, encoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Fixed 2D sin-cos PE (non-learnable buffer). Added to both visible
        # and mask tokens so the predictor knows which positions to predict.
        grid_size = int(round(num_patches ** 0.5))
        assert grid_size * grid_size == num_patches, "num_patches must be a square"
        pos_embed = build_2d_sincos_pe(grid_size, pred_embed_dim).unsqueeze(0)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(pred_embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(pred_embed_dim)

    def forward(
        self,
        visible_features: torch.Tensor,  # [B, N_vis, D_enc]
        visible_indices: torch.Tensor,   # [B, N_vis]
        masked_indices: torch.Tensor,    # [B, N_masked]
    ) -> torch.Tensor:
        """Returns [B, N_masked, D_enc] predictions at masked positions."""
        B, N_vis, _ = visible_features.shape
        N_masked = masked_indices.shape[1]

        vis = self.input_proj(visible_features) + self._gather_pos(visible_indices)
        masks = self.mask_token.expand(B, N_masked, -1) + self._gather_pos(masked_indices)

        x = torch.cat([vis, masks], dim=1)  # visible first, masked second
        for block in self.blocks:
            x = self._block_forward(block, x)
        x = self.norm(x)

        return self.output_proj(x[:, N_vis:])

    def _gather_pos(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather 2D sin-cos PE at given indices → [B, N, D_pred]."""
        B = indices.shape[0]
        pos = self.pos_embed.expand(B, -1, -1)
        idx = indices.unsqueeze(-1).expand(-1, -1, self.pred_embed_dim)
        return torch.gather(pos, dim=1, index=idx)

    def _block_forward(self, block, tokens):
        # No RoPE — PE is already additive on the input tokens.
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
# Multi-block masking (per-sample, fixed count)
# ============================================================================

def sample_block_mask_fixed(
    num_patches_per_side: int,
    num_blocks: int,
    block_scale_range: tuple,
    aspect_range: tuple,
    target_mask_count: int,
) -> torch.Tensor:
    """Sample a union of random blocks, then adjust to exactly `target_mask_count`.

    Fixed count is required so that batch tensors have uniform shape. The
    adjustment randomly masks/unmasks tokens after the block union if the
    natural mask count doesn't match the target.

    Returns `[num_patches_per_side^2]` bool tensor (True = masked).
    """
    H = W = num_patches_per_side
    total = H * W
    assert 0 < target_mask_count < total, "target_mask_count out of range"
    mask = torch.zeros(H, W, dtype=torch.bool)

    for _ in range(num_blocks):
        scale = torch.empty(1).uniform_(*block_scale_range).item()
        aspect = torch.empty(1).uniform_(*aspect_range).item()
        block_area = total * scale
        block_h = min(H, max(1, int(round((block_area * aspect) ** 0.5))))
        block_w = min(W, max(1, int(round((block_area / aspect) ** 0.5))))
        top = torch.randint(0, H - block_h + 1, (1,)).item()
        left = torch.randint(0, W - block_w + 1, (1,)).item()
        mask[top:top + block_h, left:left + block_w] = True

    mask = mask.reshape(-1)
    n_masked = int(mask.sum().item())

    if n_masked > target_mask_count:
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        drop = idx[torch.randperm(idx.numel())[:n_masked - target_mask_count]]
        mask[drop] = False
    elif n_masked < target_mask_count:
        idx = torch.nonzero(~mask, as_tuple=False).reshape(-1)
        add = idx[torch.randperm(idx.numel())[:target_mask_count - n_masked]]
        mask[add] = True

    return mask


# ============================================================================
# V-JEPA Model (top-level)
# ============================================================================

class VJEPAModel(nn.Module):
    """x-encoder + narrow predictor + EMA y-encoder.

    Objective: L1( predictor(x_encoder(masked)) , LayerNorm(sg(y_encoder(full))) ),
    averaged over two mask types (short-range + long-range).

    Downstream usage: call `extract_features()` with the x-encoder frozen.
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
        # Dual masking config — matches facebookresearch/jepa vitl16.yaml
        # (spatial-only adaptation for 2-frame tubelet; per-block scales are
        #  the spatial_scale of the official 3D multi-block mask)
        short_num_blocks: int = 8,
        short_scale_range: tuple = (0.10, 0.15),
        short_mask_ratio: float = 0.50,            # 2-frame spatial-only 적응: temporal
        long_num_blocks: int = 2,                  # redundancy 없으므로 원본(80/85%)
        long_scale_range: tuple = (0.30, 0.40),    # 대비 대폭 완화. visible 토큰 충분히 확보.
        long_mask_ratio: float = 0.60,
        aspect_range: tuple = (0.75, 1.5),
        ema_momentum_start: float = 0.998,
        ema_momentum_end: float = 1.0,
        ema_ipe_scale: float = 1.25,               # official: stretch schedule 25% past end
    ):
        super().__init__()
        self.short_num_blocks = short_num_blocks
        self.short_scale_range = short_scale_range
        self.short_mask_ratio = short_mask_ratio
        self.long_num_blocks = long_num_blocks
        self.long_scale_range = long_scale_range
        self.long_mask_ratio = long_mask_ratio
        self.aspect_range = aspect_range
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end
        self.ema_ipe_scale = ema_ipe_scale
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
        """Update y-encoder as EMA of x-encoder. Call after every optimizer step."""
        m = momentum if momentum is not None else self._ema_momentum
        for p_y, p_x in zip(self.y_encoder.parameters(), self.x_encoder.parameters()):
            p_y.data.mul_(m).add_(p_x.data, alpha=1 - m)

    def set_ema_momentum(self, epoch: int, total_epochs: int):
        # `ipe_scale=1.25` stretches the schedule 25% past the nominal end,
        # so momentum reaches ~0.9996 (not exactly 1.0) at the final epoch.
        effective_total = max(1.0, total_epochs * self.ema_ipe_scale)
        t = min(1.0, max(0.0, epoch / effective_total))
        self._ema_momentum = (
            self.ema_momentum_start * (1 - t) + self.ema_momentum_end * t
        )

    def _sample_masks_per_sample(
        self,
        batch_size: int,
        num_blocks: int,
        scale_range: tuple,
        target_mask_count: int,
        device: torch.device,
    ):
        """Sample `batch_size` independent fixed-count masks.

        Returns (visible_idx, masked_idx) each of shape [B, N_*].
        """
        vis_list, mask_list = [], []
        for _ in range(batch_size):
            mask_b = sample_block_mask_fixed(
                num_patches_per_side=self.num_patches_per_side,
                num_blocks=num_blocks,
                block_scale_range=scale_range,
                aspect_range=self.aspect_range,
                target_mask_count=target_mask_count,
            )
            mask_list.append(torch.nonzero(mask_b, as_tuple=False).reshape(-1))
            vis_list.append(torch.nonzero(~mask_b, as_tuple=False).reshape(-1))

        return (
            torch.stack(vis_list, dim=0).to(device),
            torch.stack(mask_list, dim=0).to(device),
        )

    def _forward_one_mask(
        self,
        frames: torch.Tensor,
        y_target_normed: torch.Tensor,
        visible_idx: torch.Tensor,
        masked_idx: torch.Tensor,
    ) -> torch.Tensor:
        """x-encoder(visible) → predictor → L1 vs LayerNormed y-target."""
        x_feat = self.x_encoder(frames, visible_indices=visible_idx)
        pred = self.predictor(x_feat, visible_idx, masked_idx)
        idx = masked_idx.unsqueeze(-1).expand(-1, -1, y_target_normed.shape[-1])
        y_target = torch.gather(y_target_normed, dim=1, index=idx)
        return F.l1_loss(pred, y_target.detach())

    def compute_loss(self, img_t: torch.Tensor, img_tk: torch.Tensor):
        """Dual-mask V-JEPA loss.

        Matches `app/vjepa/train.py::forward_target` in the official repo:
        y-encoder runs once on the full sequence, its output is LayerNormed,
        then reused as the target for both the short-range and long-range mask
        passes. The two L1 losses are averaged (not summed).

        Returns (scalar_loss, dummy_image_pred_for_train_loop_compat).
        """
        B = img_t.shape[0]
        device = img_t.device
        frames = torch.stack([img_t, img_tk], dim=1)  # [B, 2, C, H, W]

        with torch.no_grad():
            y_feat_full = self.y_encoder(frames, visible_indices=None)
            y_target_normed = F.layer_norm(y_feat_full, (y_feat_full.shape[-1],))

        n_short = int(round(self.short_mask_ratio * self.num_patches))
        vis_s, mask_s = self._sample_masks_per_sample(
            B, self.short_num_blocks, self.short_scale_range, n_short, device,
        )
        loss_short = self._forward_one_mask(frames, y_target_normed, vis_s, mask_s)

        n_long = int(round(self.long_mask_ratio * self.num_patches))
        vis_l, mask_l = self._sample_masks_per_sample(
            B, self.long_num_blocks, self.long_scale_range, n_long, device,
        )
        loss_long = self._forward_one_mask(frames, y_target_normed, vis_l, mask_l)

        loss = 0.5 * (loss_short + loss_long)
        return loss, img_tk  # second return is a dummy for save_epoch_samples

    def forward(self, img_t: torch.Tensor, img_tk: torch.Tensor):
        return self.compute_loss(img_t, img_tk)

    @torch.no_grad()
    def extract_features(self, img_t: torch.Tensor, img_tk: torch.Tensor):
        """Frozen inference: x-encoder on full (unmasked) sequence. → [B, 196, D]."""
        frames = torch.stack([img_t, img_tk], dim=1)
        return self.x_encoder(frames, visible_indices=None)
