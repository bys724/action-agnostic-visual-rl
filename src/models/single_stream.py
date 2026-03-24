"""
Single-Stream Model: Baseline with single ViT encoder.

Architecture:
- Two-stream preprocessing (M + P channels = 8ch)
- Single ViT encoder (no stream separation)
- Future frame prediction task

This serves as a baseline to evaluate the benefit of
two-stream architecture with CLS exchange.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import TwoStreamPreprocessing


class SingleStreamModel(nn.Module):
    """
    Single-stream baseline for future prediction.

    Uses same M/P preprocessing as Two-Stream but processes
    all 8 channels with a single ViT encoder (no separation).

    Args:
        embed_dim: Embedding dimension (default: 768)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        image_size: Input image size (default: 224)
        patch_size: Patch size for ViT (default: 16)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Preprocessing (shared with Two-Stream)
        self.preprocessing = TwoStreamPreprocessing()

        # Patch embedding for 8 channels (M: 3ch + P: 5ch)
        self.patch_embed = nn.Conv2d(
            in_channels=8,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder for future frame prediction
        self.decoder = self._build_decoder(image_size, patch_size)

        self._init_weights()

    def _build_decoder(self, image_size: int, patch_size: int) -> nn.Module:
        """Build U-Net style decoder for image reconstruction."""
        self.num_patches_per_side = image_size // patch_size

        upsampler = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        return upsampler

    def _init_weights(self):
        """Initialize weights following ViT convention."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self, image_current: torch.Tensor, image_future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            image_current: [B, 3, H, W], range [0, 1]
            image_future: [B, 3, H, W], range [0, 1]

        Returns:
            predicted_image: [B, 3, H, W] - predicted future frame
            cls_embedding: [B, embed_dim] - CLS token embedding
        """
        batch_size = image_current.shape[0]

        # 1. Preprocessing: RGB → M/P channels
        m_channel, p_channel = self.preprocessing(image_current, image_future)
        combined_input = torch.cat([m_channel, p_channel], dim=1)  # [B, 9, H, W]

        # 2. Patch embedding
        patches = self.patch_embed(combined_input)  # [B, D, H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, D]

        # 3. Add CLS token and positional encoding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)  # [B, N+1, D]
        tokens = tokens + self.pos_embed

        # 4. Transformer encoding
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)

        # 5. Extract CLS and patch tokens
        cls_embedding = encoded[:, 0]  # [B, D]
        patch_tokens = encoded[:, 1:]  # [B, N, D]

        # 6. Decode to image: [B, N, D] → [B, D, P, P] → upsample
        P = self.num_patches_per_side
        x = patch_tokens.view(-1, P, P, patch_tokens.size(-1))  # [B, P, P, D]
        x = x.permute(0, 3, 1, 2)                               # [B, D, P, P]
        predicted_image = self.decoder(x)

        return predicted_image, cls_embedding

    def compute_loss(
        self, image_current: torch.Tensor, image_future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            image_current: [B, 3, H, W]
            image_future: [B, 3, H, W]

        Returns:
            loss: Scalar loss value
            predicted_image: [B, 3, H, W]
        """
        predicted_image, _ = self.forward(image_current, image_future)
        loss = F.mse_loss(predicted_image, image_future)
        return loss, predicted_image


class SingleStreamEncoder(nn.Module):
    """
    Single-stream encoder for integration with VLA models (e.g., OpenVLA).

    This wraps the trained Single-Stream model to provide only the
    visual encoding capability, suitable for downstream tasks.

    Args:
        checkpoint_path: Path to trained Single-Stream checkpoint (optional)
        embed_dim: Embedding dimension (default: 768)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self._embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Build encoder components only
        self.preprocessing = TwoStreamPreprocessing()
        self.patch_embed = nn.Conv2d(9, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4.0),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Load pretrained weights if provided
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    @property
    def embed_dim(self) -> int:
        """Output embedding dimension (required by VLA interface)."""
        return self._embed_dim

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load encoder weights from trained Single-Stream model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract only encoder weights
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("decoder.")
        }

        self.load_state_dict(encoder_state, strict=False)
        print(f"Loaded encoder weights from: {checkpoint_path}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VLA integration.

        Args:
            pixel_values: [B, 6, H, W] - two RGB images stacked channel-wise
                          (image_current + image_future)

        Returns:
            patch_embeddings: [B, num_patches, embed_dim]
        """
        batch_size = pixel_values.shape[0]

        # Split stacked images
        image_current = pixel_values[:, :3]  # [B, 3, H, W]
        image_future = pixel_values[:, 3:]   # [B, 3, H, W]

        # Preprocess and encode
        m_channel, p_channel = self.preprocessing(image_current, image_future)
        combined_input = torch.cat([m_channel, p_channel], dim=1)

        patches = self.patch_embed(combined_input)
        patches = patches.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = tokens + self.pos_embed

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)

        # Return patch embeddings only (exclude CLS for VLA)
        patch_embeddings = encoded[:, 1:]  # [B, num_patches, embed_dim]

        return patch_embeddings
