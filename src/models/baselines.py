#!/usr/bin/env python
"""
Baseline Models for Comparison

1. VideoMAEPredictor: Masked autoencoder (reconstruction task)
2. SingleStreamVideoPredictor: Single ViT encoder (future prediction task)

All models have the same interface:
    forward(img_t, img_tk) -> img_pred, cls_emb
    compute_loss(img_t, img_tk) -> loss, img_pred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# VideoMAE: Masked Autoencoder for Video
# =============================================================================

class VideoMAEPredictor(nn.Module):
    """
    VideoMAE-style masked autoencoder.

    Strategy: Mask random patches, reconstruct original image.
    This learns spatial structure but not temporal dynamics.

    Differences from original VideoMAE:
    - Single frame (not video tubes) for simplicity
    - Same decoder structure as Two-Stream for fair comparison

    Input: img_t, img_tk (uses img_tk for reconstruction target)
    Output: img_pred, cls_emb
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        img_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2  # 196

        # Patch embedding (3 channels for RGB)
        self.patch_embed = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Mask token for reconstruction
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder (same structure as Two-Stream for fair comparison)
        self.decoder = VideoMAEDecoder(embed_dim=embed_dim)

        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def random_masking(self, x, mask_ratio):
        """
        Random masking: keep a subset of patches.

        Args:
            x: [B, N, D] - patch embeddings (without CLS)
            mask_ratio: fraction of patches to mask

        Returns:
            x_masked: [B, N_visible, D]
            mask: [B, N] - 0 for kept, 1 for removed
            ids_restore: [B, N] - indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)

        # Sort noise to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Binary mask: 0 for kept, 1 for removed
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, img_t, img_tk):
        """
        Forward pass for VideoMAE.

        Uses img_tk as target (reconstruct future frame).
        This makes it comparable to future prediction task.

        Args:
            img_t: [B, 3, H, W] - current frame (unused, for API compatibility)
            img_tk: [B, 3, H, W] - future frame (reconstruction target)

        Returns:
            img_pred: [B, 3, H, W] - reconstructed image
            cls_emb: [B, D] - CLS embedding
        """
        B = img_tk.shape[0]

        # Patch embedding
        x = self.patch_embed(img_tk)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]

        # Add position embedding (without CLS position)
        x = x + self.pos_embed[:, 1:, :]

        # Random masking
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Add CLS token
        cls_token = self.cls_token.repeat(B, 1, 1)
        cls_token = cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token, x_masked], dim=1)  # [B, 1+N_visible, D]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Extract CLS
        cls_emb = x[:, 0]  # [B, D]
        x_encoded = x[:, 1:]  # [B, N_visible, D]

        # Unshuffle and add mask tokens
        mask_tokens = self.mask_token.repeat(B, self.num_patches - x_encoded.shape[1], 1)
        x_full = torch.cat([x_encoded, mask_tokens], dim=1)  # [B, N, D]
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )

        # Decode
        img_pred = self.decoder(x_full)

        return img_pred, cls_emb

    def compute_loss(self, img_t, img_tk):
        """Compute reconstruction loss on masked patches."""
        img_pred, cls_emb = self.forward(img_t, img_tk)
        loss = F.mse_loss(img_pred, img_tk)
        return loss, img_pred


class VideoMAEDecoder(nn.Module):
    """Simple decoder for VideoMAE (same as Two-Stream decoder without skip/CLS)."""

    def __init__(self, embed_dim: int = 768, out_channels: int = 3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
            ),
        ])

        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, patches):
        """
        Args:
            patches: [B, 196, D]
        Returns:
            img_pred: [B, 3, 224, 224]
        """
        B, N, D = patches.shape
        H = W = int(N ** 0.5)  # 14

        x = self.input_proj(patches)  # [B, 196, 512]
        x = x.permute(0, 2, 1).contiguous().view(B, 512, H, W)

        for block in self.decoder_blocks:
            x = block(x)

        img_pred = torch.sigmoid(self.output_conv(x))
        return img_pred


# =============================================================================
# Single-Stream: Future Prediction with Single ViT
# =============================================================================

class SingleStreamVideoPredictor(nn.Module):
    """
    Single-stream baseline for future prediction.

    Strategy: Concatenate img_t and img_tk as 6-channel input.
    Single ViT encodes both, decoder predicts img_tk.

    This tests whether Two-Stream separation provides benefit.

    Input: img_t, img_tk
    Output: img_pred, cls_emb
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding (6 channels: concat of img_t and img_tk)
        self.patch_embed = nn.Conv2d(
            6, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder (same structure as Two-Stream for fair comparison)
        self.decoder = SingleStreamDecoder(embed_dim=embed_dim)

        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, img_t, img_tk):
        """
        Forward pass for single-stream future prediction.

        Args:
            img_t: [B, 3, H, W] - current frame
            img_tk: [B, 3, H, W] - future frame

        Returns:
            img_pred: [B, 3, H, W] - predicted future frame
            cls_emb: [B, D] - CLS embedding
        """
        B = img_t.shape[0]

        # Concatenate frames (6 channels)
        x = torch.cat([img_t, img_tk], dim=1)  # [B, 6, H, W]

        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]

        # Add CLS token
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Extract CLS and patches
        cls_emb = x[:, 0]  # [B, D]
        patches = x[:, 1:]  # [B, N, D]

        # Decode
        img_pred = self.decoder(patches, img_t)

        return img_pred, cls_emb

    def compute_loss(self, img_t, img_tk):
        """Compute future prediction loss."""
        img_pred, cls_emb = self.forward(img_t, img_tk)
        loss = F.mse_loss(img_pred, img_tk)
        return loss, img_pred


class SingleStreamDecoder(nn.Module):
    """
    Decoder for single-stream (with skip connection from img_t).
    Same structure as Two-Stream decoder but without multi-scale CLS.
    """

    def __init__(self, embed_dim: int = 768, out_channels: int = 3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # Skip connection encoder
        self.skip_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Decoder blocks
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.skip_merge = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, patches, img_t):
        """
        Args:
            patches: [B, 196, D]
            img_t: [B, 3, 224, 224] - for skip connection
        Returns:
            img_pred: [B, 3, 224, 224]
        """
        B, N, D = patches.shape
        H = W = int(N ** 0.5)  # 14

        x = self.input_proj(patches)
        x = x.permute(0, 2, 1).contiguous().view(B, 512, H, W)

        # Skip features
        skip_feat = self.skip_encoder(img_t)  # [B, 64, 56, 56]

        # Decode
        x = self.up1(x)  # [B, 256, 28, 28]
        x = self.up2(x)  # [B, 128, 56, 56]

        # Merge skip
        x = torch.cat([x, skip_feat], dim=1)
        x = self.skip_merge(x)

        x = self.up3(x)  # [B, 64, 112, 112]
        x = self.up4(x)  # [B, 32, 224, 224]

        img_pred = torch.sigmoid(self.output_conv(x))
        return img_pred


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                        choices=["videomae", "singlestream", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    img_t = torch.rand(2, 3, 224, 224, device=device)
    img_tk = torch.rand(2, 3, 224, 224, device=device)

    if args.test in ["videomae", "all"]:
        print("=" * 60)
        print("Testing VideoMAEPredictor")
        print("=" * 60)

        model = VideoMAEPredictor().to(device)
        img_pred, cls_emb = model(img_t, img_tk)

        print(f"Input:  img_t {img_t.shape}, img_tk {img_tk.shape}")
        print(f"Output: img_pred {img_pred.shape}, cls_emb {cls_emb.shape}")
        print(f"Output range: [{img_pred.min():.3f}, {img_pred.max():.3f}]")

        loss, _ = model.compute_loss(img_t, img_tk)
        loss.backward()
        print(f"Loss: {loss.item():.4f}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params/1e6:.1f}M")
        print("VideoMAE: OK\n")

    if args.test in ["singlestream", "all"]:
        print("=" * 60)
        print("Testing SingleStreamVideoPredictor")
        print("=" * 60)

        model = SingleStreamVideoPredictor().to(device)
        img_pred, cls_emb = model(img_t, img_tk)

        print(f"Input:  img_t {img_t.shape}, img_tk {img_tk.shape}")
        print(f"Output: img_pred {img_pred.shape}, cls_emb {cls_emb.shape}")
        print(f"Output range: [{img_pred.min():.3f}, {img_pred.max():.3f}]")

        loss, _ = model.compute_loss(img_t, img_tk)
        loss.backward()
        print(f"Loss: {loss.item():.4f}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params/1e6:.1f}M")
        print("SingleStream: OK\n")

    print("=" * 60)
    print("All baseline tests passed!")
    print("=" * 60)
