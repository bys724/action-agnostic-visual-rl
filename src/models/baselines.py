#!/usr/bin/env python
"""
Baseline Models for Comparison

SingleStreamVideoPredictor: Single ViT encoder (future prediction task)

For VideoMAE, use videomae_wrapper.py (based on official implementation).

All models have the same interface:
    forward(img_t, img_tk) -> img_pred, cls_emb
    compute_loss(img_t, img_tk) -> loss, img_pred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    img_t = torch.rand(2, 3, 224, 224, device=device)
    img_tk = torch.rand(2, 3, 224, 224, device=device)

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
    print("SingleStream: OK")
