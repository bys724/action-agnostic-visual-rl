#!/usr/bin/env python
"""
VideoMAE Wrapper for BridgeV2 Training

Based on official VideoMAE implementation from:
https://github.com/MCG-NJU/VideoMAE (included as git submodule)

Adapted for 2-frame video prediction task with BridgeV2 data.
Key modification: num_frames=2 instead of default 16.
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from pathlib import Path

# Add VideoMAE to path for reference (we use adapted components)
VIDEOMAE_PATH = Path(__file__).parent.parent.parent / "external" / "VideoMAE"
sys.path.insert(0, str(VIDEOMAE_PATH))

# Import components from official VideoMAE
from modeling_finetune import Block, get_sinusoid_encoding_table
from timm.models.layers import trunc_normal_


# =============================================================================
# Adapted PatchEmbed for 2 frames
# =============================================================================

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for video.
    Modified from official VideoMAE to support num_frames=2.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=2,
        tubelet_size=2,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames

        # Number of patches
        self.num_patches_spatial = (img_size // patch_size) ** 2  # 14x14 = 196
        self.num_patches_temporal = num_frames // tubelet_size  # 2 // 2 = 1
        self.num_patches = self.num_patches_spatial * self.num_patches_temporal

        # 3D convolution for spatio-temporal patch embedding
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            patches: [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, D, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, T'*H'*W', D]
        return x


# =============================================================================
# VideoMAE Encoder (adapted for 2 frames)
# =============================================================================

class VideoMAEEncoder(nn.Module):
    """
    VideoMAE Encoder adapted for 2-frame input.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_frames=2,
        tubelet_size=2,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches

        # Position embedding (sinusoidal)
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.0,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, T, H, W] - video input
            mask: [B, num_patches] - True for masked patches

        Returns:
            x_vis: [B, num_visible, embed_dim] - visible patch embeddings
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, D]

        # Add position embedding
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        # Select visible patches (not masked)
        B, N, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # [B, num_visible, D]

        # Transformer blocks
        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis


# =============================================================================
# VideoMAE Decoder (adapted for 2 frames)
# =============================================================================

class VideoMAEDecoder(nn.Module):
    """
    VideoMAE Decoder for masked patch reconstruction.
    """
    def __init__(
        self,
        patch_size=16,
        num_patches=196,
        embed_dim=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        tubelet_size=2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Output: predict pixels for masked patches
        # 3 channels * tubelet_size * patch_size^2
        self.num_classes = 3 * tubelet_size * patch_size * patch_size

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.0,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes)

    def forward(self, x, num_masked):
        """
        Args:
            x: [B, num_patches, embed_dim] - full sequence with mask tokens
            num_masked: number of masked patches

        Returns:
            pred: [B, num_masked, num_classes] - predicted pixels
        """
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # Only predict for masked tokens (last num_masked positions)
        pred = self.head(x[:, -num_masked:])

        return pred


# =============================================================================
# Complete VideoMAE Model (adapted for 2 frames)
# =============================================================================

class VideoMAEForBridge(nn.Module):
    """
    VideoMAE adapted for BridgeV2 2-frame video prediction.

    Based on official VideoMAE architecture but modified for:
    - num_frames=2 (instead of 16)
    - tubelet_size=2 (merges 2 frames into 1 temporal position)
    - Same interface as TwoStreamVideoPredictor

    Input: img_t, img_tk [B, 3, 224, 224]
    Output: img_pred [B, 3, 224, 224], cls_emb [B, D]
    """

    def __init__(
        self,
        model_size: str = "base",
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 2,
        tubelet_size: int = 2,
        mask_ratio: float = 0.75,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.mask_ratio = mask_ratio

        # Model configs
        configs = {
            "small": {
                "encoder_embed_dim": 384,
                "encoder_depth": 12,
                "encoder_num_heads": 6,
                "decoder_embed_dim": 192,
                "decoder_depth": 4,
                "decoder_num_heads": 3,
            },
            "base": {
                "encoder_embed_dim": 768,
                "encoder_depth": 12,
                "encoder_num_heads": 12,
                "decoder_embed_dim": 384,
                "decoder_depth": 4,
                "decoder_num_heads": 6,
            },
            "large": {
                "encoder_embed_dim": 1024,
                "encoder_depth": 24,
                "encoder_num_heads": 16,
                "decoder_embed_dim": 512,
                "decoder_depth": 4,
                "decoder_num_heads": 8,
            },
        }

        cfg = configs[model_size]
        self.embed_dim = cfg["encoder_embed_dim"]

        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.num_masks = int(self.num_patches * mask_ratio)

        # Encoder
        self.encoder = VideoMAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=cfg["encoder_embed_dim"],
            depth=cfg["encoder_depth"],
            num_heads=cfg["encoder_num_heads"],
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )

        # Decoder
        self.decoder = VideoMAEDecoder(
            patch_size=patch_size,
            num_patches=self.num_patches,
            embed_dim=cfg["decoder_embed_dim"],
            depth=cfg["decoder_depth"],
            num_heads=cfg["decoder_num_heads"],
            tubelet_size=tubelet_size,
        )

        # Encoder to decoder projection
        self.encoder_to_decoder = nn.Linear(
            cfg["encoder_embed_dim"],
            cfg["decoder_embed_dim"],
            bias=False
        )

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg["decoder_embed_dim"]))
        trunc_normal_(self.mask_token, std=0.02)

        # Position embedding for decoder
        self.decoder_pos_embed = get_sinusoid_encoding_table(
            self.num_patches, cfg["decoder_embed_dim"]
        )

        # CLS projection (for downstream compatibility)
        self.cls_proj = nn.Sequential(
            nn.Linear(cfg["encoder_embed_dim"], cfg["encoder_embed_dim"]),
            nn.GELU(),
        )

    def random_masking(self, batch_size, device):
        """
        Generate random mask for tube masking.

        Returns:
            mask: [B, num_patches] - True for masked
            ids_restore: [B, num_patches] - indices to restore order
        """
        N = self.num_patches
        len_keep = N - self.num_masks

        # Random noise for each sample
        noise = torch.rand(batch_size, N, device=device)

        # Sort to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Binary mask: False for kept, True for masked
        mask = torch.ones(batch_size, N, device=device, dtype=torch.bool)
        mask[:, :len_keep] = False

        # Unshuffle mask to original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video without masking (for downstream tasks like OpenVLA).

        Args:
            video: [B, T, C, H, W] or [B, C, T, H, W] - video input

        Returns:
            embeddings: [B, num_patches, embed_dim] - patch embeddings
        """
        # Handle input format
        if video.shape[2] == 3:  # [B, T, C, H, W]
            video = video.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]

        B = video.shape[0]

        # Patch embedding (without masking)
        x = self.encoder.patch_embed(video)  # [B, num_patches, D]

        # Add position embedding
        pos_embed = self.encoder.pos_embed.type_as(x).to(x.device).clone().detach()
        x = x + pos_embed

        # Transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.norm(x)
        return x  # [B, num_patches, embed_dim]

    def forward(self, img_t, img_tk):
        """
        Forward pass.

        Args:
            img_t: [B, 3, H, W] - current frame
            img_tk: [B, 3, H, W] - future frame

        Returns:
            img_pred: [B, 3, H, W] - reconstructed future frame
            cls_emb: [B, D] - CLS-like embedding
        """
        B = img_t.shape[0]
        device = img_t.device

        # Stack frames: [B, 3, 2, H, W]
        video = torch.stack([img_t, img_tk], dim=2)

        # Generate mask
        mask, ids_restore = self.random_masking(B, device)

        # Encode visible patches
        x_vis = self.encoder(video, mask)  # [B, num_visible, D_enc]

        # CLS-like embedding (mean pooling of visible patches)
        cls_emb = self.cls_proj(x_vis.mean(dim=1))  # [B, D_enc]

        # Project to decoder dimension
        x_vis = self.encoder_to_decoder(x_vis)  # [B, num_visible, D_dec]

        # Prepare decoder input
        num_visible = x_vis.shape[1]

        # Position embeddings
        pos_embed = self.decoder_pos_embed.to(device).expand(B, -1, -1)
        pos_vis = pos_embed[~mask].reshape(B, -1, pos_embed.shape[-1])
        pos_mask = pos_embed[mask].reshape(B, -1, pos_embed.shape[-1])

        # Mask tokens
        mask_tokens = self.mask_token.expand(B, self.num_masks, -1)

        # Full sequence: visible + mask tokens
        x_full = torch.cat([x_vis + pos_vis, mask_tokens + pos_mask], dim=1)

        # Decode
        pred = self.decoder(x_full, self.num_masks)  # [B, num_masks, 3*2*16*16]

        # Reconstruct image for visualization
        img_pred = self._reconstruct_image(pred, mask, video)

        return img_pred, cls_emb

    def _reconstruct_image(self, pred, mask, video):
        """
        Reconstruct image from masked predictions.
        """
        B = pred.shape[0]
        device = pred.device

        # Extract img_tk
        img_tk = video[:, :, 1]  # [B, 3, H, W]

        # Patchify
        patches = self._patchify(img_tk)  # [B, 196, 768]

        # pred: [B, num_masked, 1536] (includes both frames)
        # Take second half for img_tk
        patch_pixels = 3 * self.patch_size * self.patch_size
        pred_tk = pred[:, :, patch_pixels:]  # [B, num_masked, 768]

        # Replace masked patches with predictions
        result = patches.clone()
        for b in range(B):
            mask_indices = mask[b].nonzero(as_tuple=True)[0]
            result[b, mask_indices] = pred_tk[b]

        # Unpatchify
        img_pred = self._unpatchify(result)

        return img_pred

    def _patchify(self, img):
        """[B, 3, H, W] -> [B, N, patch_pixels]"""
        B, C, H, W = img.shape
        p = self.patch_size
        h = w = H // p

        patches = img.reshape(B, C, h, p, w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        patches = patches.reshape(B, h * w, p * p * C)
        return patches

    def _unpatchify(self, patches):
        """[B, N, patch_pixels] -> [B, 3, H, W]"""
        B, N, _ = patches.shape
        p = self.patch_size
        h = w = int(N ** 0.5)
        C = 3

        patches = patches.reshape(B, h, w, p, p, C)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = patches.reshape(B, C, h * p, w * p)
        return img

    def compute_loss(self, img_t, img_tk):
        """
        Compute reconstruction loss on masked patches.
        """
        B = img_t.shape[0]
        device = img_t.device

        # Stack frames
        video = torch.stack([img_t, img_tk], dim=2)

        # Generate mask
        mask, ids_restore = self.random_masking(B, device)

        # Encode
        x_vis = self.encoder(video, mask)
        x_vis = self.encoder_to_decoder(x_vis)

        # Position embeddings
        pos_embed = self.decoder_pos_embed.to(device).expand(B, -1, -1)
        pos_vis = pos_embed[~mask].reshape(B, -1, pos_embed.shape[-1])
        pos_mask = pos_embed[mask].reshape(B, -1, pos_embed.shape[-1])

        # Full sequence
        mask_tokens = self.mask_token.expand(B, self.num_masks, -1)
        x_full = torch.cat([x_vis + pos_vis, mask_tokens + pos_mask], dim=1)

        # Decode
        pred = self.decoder(x_full, self.num_masks)

        # Target: patchified frames
        patches_t = self._patchify(img_t)
        patches_tk = self._patchify(img_tk)
        target_patches = torch.cat([patches_t, patches_tk], dim=-1)  # [B, N, 1536]

        # Get masked patches as target
        target = target_patches[mask].reshape(B, self.num_masks, -1)

        # MSE loss
        loss = F.mse_loss(pred, target)

        # Get img_pred for visualization
        img_pred, _ = self.forward(img_t, img_tk)

        return loss, img_pred


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Testing VideoMAEForBridge (Official VideoMAE Architecture)")
    print("=" * 60)

    model = VideoMAEForBridge(model_size="base").to(device)

    img_t = torch.rand(2, 3, 224, 224, device=device)
    img_tk = torch.rand(2, 3, 224, 224, device=device)

    print(f"Input: img_t {img_t.shape}, img_tk {img_tk.shape}")

    # Forward
    img_pred, cls_emb = model(img_t, img_tk)
    print(f"Output: img_pred {img_pred.shape}, cls_emb {cls_emb.shape}")

    # Loss
    loss, _ = model.compute_loss(img_t, img_tk)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print("Backward: OK")

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Parameters: Encoder={encoder_params/1e6:.1f}M, Decoder={decoder_params/1e6:.1f}M, Total={total_params/1e6:.1f}M")

    print("\nVideoMAEForBridge: OK")
