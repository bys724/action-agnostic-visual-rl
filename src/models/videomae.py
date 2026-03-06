"""
VideoMAE Model: Masked autoencoder for video representation learning.

Architecture:
- 3D patch embedding for spatio-temporal tokenization
- Masked autoencoding (75% masking ratio)
- Reconstruction loss

Based on official VideoMAE implementation:
https://github.com/MCG-NJU/VideoMAE

Adapted for 2-frame input for fair comparison with other models.
"""

import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add VideoMAE to path for reference components
VIDEOMAE_PATH = Path(__file__).parent.parent.parent / "external" / "VideoMAE"
sys.path.insert(0, str(VIDEOMAE_PATH))

from modeling_finetune import Block, get_sinusoid_encoding_table
from timm.models.layers import trunc_normal_


class PatchEmbed3D(nn.Module):
    """
    3D patch embedding for spatio-temporal tokenization.

    Converts video [B, C, T, H, W] into sequence of patches [B, num_patches, embed_dim].
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_frames: int = 2,
        tubelet_size: int = 2,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames

        # Number of patches
        self.num_patches_spatial = (image_size // patch_size) ** 2  # 196
        self.num_patches_temporal = num_frames // tubelet_size  # 1
        self.num_patches = self.num_patches_spatial * self.num_patches_temporal

        # 3D convolution for spatio-temporal embedding
        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W]

        Returns:
            patches: [B, num_patches, embed_dim]
        """
        patches = self.projection(video)  # [B, D, T', H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, D]
        return patches


class VideoMAEEncoder(nn.Module):
    """
    VideoMAE encoder with masking support.

    Processes only visible (unmasked) patches to reduce computation.
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
        num_frames: int = 2,
        tubelet_size: int = 2,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding (sinusoidal, fixed)
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
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

    def forward(
        self, video: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode visible patches only.

        Args:
            video: [B, C, T, H, W]
            mask: [B, num_patches] - True for masked patches

        Returns:
            visible_embeddings: [B, num_visible, embed_dim]
        """
        # Patch embedding
        patches = self.patch_embed(video)  # [B, num_patches, D]

        # Add positional embedding
        patches = patches + self.pos_embed.type_as(patches).to(patches.device).clone().detach()

        # Select only visible patches
        batch_size, num_patches, channels = patches.shape
        visible_patches = patches[~mask].reshape(batch_size, -1, channels)

        # Transformer encoding
        for block in self.blocks:
            visible_patches = block(visible_patches)

        visible_embeddings = self.norm(visible_patches)
        return visible_embeddings


class VideoMAEDecoder(nn.Module):
    """
    VideoMAE decoder for masked patch reconstruction.

    Predicts pixel values for masked patches only.
    """

    def __init__(
        self,
        patch_size: int = 16,
        num_patches: int = 196,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        tubelet_size: int = 2,
        in_channels: int = 3,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Output dimension: pixels for each masked patch
        self.num_classes = in_channels * tubelet_size * patch_size * patch_size

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

    def forward(
        self, full_sequence: torch.Tensor, num_masked: int
    ) -> torch.Tensor:
        """
        Decode masked patches.

        Args:
            full_sequence: [B, num_patches, embed_dim] - visible + mask tokens
            num_masked: Number of masked patches

        Returns:
            predictions: [B, num_masked, num_classes] - predicted pixels
        """
        for block in self.blocks:
            full_sequence = block(full_sequence)

        full_sequence = self.norm(full_sequence)

        # Predict only masked patches (last num_masked positions)
        predictions = self.head(full_sequence[:, -num_masked:])

        return predictions


class VideoMAEModel(nn.Module):
    """
    Complete VideoMAE model for masked autoencoding.

    Args:
        image_size: Input image size (default: 224)
        patch_size: Patch size (default: 16)
        in_channels: Number of input channels (default: 3)
        encoder_embed_dim: Encoder embedding dimension (default: 768)
        encoder_depth: Encoder depth (default: 12)
        encoder_num_heads: Encoder attention heads (default: 12)
        decoder_embed_dim: Decoder embedding dimension (default: 384)
        decoder_depth: Decoder depth (default: 4)
        decoder_num_heads: Decoder attention heads (default: 6)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        num_frames: Number of frames (default: 2)
        tubelet_size: Temporal patch size (default: 2)
        mask_ratio: Masking ratio (default: 0.75)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_frames: int = 2,
        tubelet_size: int = 2,
        mask_ratio: float = 0.75,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.num_patches = (image_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # Encoder
        self.encoder = VideoMAEEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )

        # Encoder-to-decoder projection
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional embedding
        self.decoder_pos_embed = get_sinusoid_encoding_table(
            self.num_patches, decoder_embed_dim
        )

        # Decoder
        self.decoder = VideoMAEDecoder(
            patch_size=patch_size,
            num_patches=self.num_patches,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        trunc_normal_(self.mask_token, std=0.02)

    def random_masking(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random masking for patches.

        Args:
            batch_size: Batch size
            num_patches: Total number of patches
            device: Device

        Returns:
            mask: [B, num_patches] - True for masked patches
            ids_restore: [B, num_patches] - indices to restore original order
        """
        num_masked = int(self.mask_ratio * num_patches)

        # Random shuffle
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first patches visible, mask the rest
        mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
        mask[:, :num_patches - num_masked] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore

    def forward(
        self, image_current: torch.Tensor, image_future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with random masking.

        Args:
            image_current: [B, 3, H, W]
            image_future: [B, 3, H, W]

        Returns:
            predicted_patches: [B, num_masked, patch_pixels]
            cls_embedding: [B, encoder_embed_dim] - not used but for API consistency
        """
        batch_size = image_current.shape[0]
        device = image_current.device

        # Stack images into video: [B, C, T, H, W]
        video = torch.stack([image_current, image_future], dim=2)

        # Random masking
        mask, ids_restore = self.random_masking(batch_size, self.num_patches, device)

        # Encode visible patches
        visible_embeddings = self.encoder(video, mask)

        # Project to decoder dimension
        visible_embeddings = self.encoder_to_decoder(visible_embeddings)

        # Add mask tokens for masked patches
        num_visible = visible_embeddings.shape[1]
        num_masked = self.num_patches - num_visible

        mask_tokens = self.mask_token.repeat(
            batch_size, num_masked, 1
        )

        # Concatenate visible + mask tokens
        full_sequence = torch.cat([visible_embeddings, mask_tokens], dim=1)

        # Restore original order using ids_restore
        full_sequence = torch.gather(
            full_sequence,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, full_sequence.shape[-1]),
        )

        # Add decoder positional embedding
        full_sequence = full_sequence + self.decoder_pos_embed.type_as(full_sequence).to(device).clone().detach()

        # Decode masked patches
        predicted_patches = self.decoder(full_sequence, num_masked)

        # Return dummy CLS embedding for API consistency
        cls_embedding = visible_embeddings.mean(dim=1)

        return predicted_patches, cls_embedding

    def compute_loss(
        self, image_current: torch.Tensor, image_future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss for masked patches.

        Args:
            image_current: [B, 3, H, W]
            image_future: [B, 3, H, W]

        Returns:
            loss: Scalar loss value
            predicted_patches: Predicted patch pixels
        """
        batch_size = image_current.shape[0]
        device = image_current.device

        # Stack into video
        video = torch.stack([image_current, image_future], dim=2)

        # Random masking
        mask, ids_restore = self.random_masking(batch_size, self.num_patches, device)

        # Forward pass
        predicted_patches, _ = self.forward(image_current, image_future)

        # Extract target pixels for masked patches
        # Patchify video
        video = video.permute(0, 2, 3, 4, 1).contiguous()  # [B, T, H, W, C]
        batch_size, num_frames, height, width, channels = video.shape

        patches_h = height // self.patch_size
        patches_w = width // self.patch_size

        video_patches = video.reshape(
            batch_size,
            num_frames // self.tubelet_size,
            self.tubelet_size,
            patches_h,
            self.patch_size,
            patches_w,
            self.patch_size,
            channels,
        )
        video_patches = video_patches.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        video_patches = video_patches.reshape(
            batch_size, -1, self.tubelet_size * self.patch_size * self.patch_size * channels
        )

        # Select masked patches
        masked_patches_target = video_patches[mask]
        masked_patches_target = masked_patches_target.reshape(
            batch_size, -1, video_patches.shape[-1]
        )

        # Compute MSE loss
        loss = F.mse_loss(predicted_patches, masked_patches_target)

        return loss, predicted_patches


class VideoMAEEncoderForVLA(nn.Module):
    """
    VideoMAE encoder wrapper for VLA integration.

    Provides only encoding capability without masking for downstream tasks.

    Args:
        checkpoint_path: Path to trained checkpoint (optional)
        embed_dim: Embedding dimension (default: 768)
        depth: Transformer depth (default: 12)
        num_heads: Number of heads (default: 12)
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

        # Build encoder (uses the internal VideoMAEEncoder class)
        self.encoder = VideoMAEEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        # Load pretrained weights if provided
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    @property
    def embed_dim(self) -> int:
        """Output embedding dimension."""
        return self._embed_dim

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load encoder weights from trained VideoMAE model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Extract encoder weights
        encoder_state = {
            k: v for k, v in state_dict.items()
            if k.startswith("encoder.")
        }

        self.load_state_dict(encoder_state, strict=False)
        print(f"Loaded encoder weights from: {checkpoint_path}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VLA integration (no masking).

        Args:
            pixel_values: [B, 6, H, W] - two RGB images stacked

        Returns:
            patch_embeddings: [B, num_patches, embed_dim]
        """
        batch_size = pixel_values.shape[0]

        # Split and stack into video
        image_current = pixel_values[:, :3]
        image_future = pixel_values[:, 3:]
        video = torch.stack([image_current, image_future], dim=2)

        # Encode without masking (use all patches)
        mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=video.device)
        patch_embeddings = self.encoder(video, mask)

        return patch_embeddings
