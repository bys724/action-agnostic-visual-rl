"""
VideoMAE Model: Masked autoencoder for video representation learning.

Architecture:
- 3D patch embedding for spatio-temporal tokenization
- Masked autoencoding (75% masking ratio)
- Per-patch normalized reconstruction target
- Reconstruction loss

Faithfully follows the official VideoMAE implementation:
https://github.com/MCG-NJU/VideoMAE

Adapted for 2-frame input (tubelet_size=2, num_frames=2 → 196 spatial patches).
With only 1 temporal token, tube masking = random spatial masking.
"""

import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add VideoMAE to path for reference components
VIDEOMAE_PATH = Path(__file__).parent.parent.parent / "external" / "VideoMAE"
sys.path.insert(0, str(VIDEOMAE_PATH))

from modeling_finetune import Block, get_sinusoid_encoding_table
from timm.models.layers import trunc_normal_ as _trunc_normal_


def trunc_normal_(tensor, mean=0.0, std=1.0):
    """Match official VideoMAE's trunc_normal_ (clamp at [-std, std])."""
    _trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PatchEmbed3D(nn.Module):
    """3D patch embedding: video [B, C, T, H, W] → patches [B, N, D]."""

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
        self.num_patches = (image_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.projection = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.projection(video).flatten(2).transpose(1, 2)


class VideoMAEEncoder(nn.Module):
    """VideoMAE encoder: processes only visible (unmasked) patches."""

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

        self.patch_embed = PatchEmbed3D(
            image_size, patch_size, in_channels, embed_dim, num_frames, tubelet_size,
        )
        num_patches = self.patch_embed.num_patches

        # Sinusoidal positional embedding (fixed, not learnable)
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.0,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, video: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W]
            mask: [B, N] - True for masked patches

        Returns:
            visible_embeddings: [B, N_vis, D]
        """
        patches = self.patch_embed(video)
        patches = patches + self.pos_embed.type_as(patches).to(patches.device).clone().detach()

        B, N, C = patches.shape
        visible = patches[~mask].reshape(B, -1, C)

        for block in self.blocks:
            visible = block(visible)

        return self.norm(visible)


class VideoMAEDecoder(nn.Module):
    """VideoMAE decoder: predicts pixel values for masked patches."""

    def __init__(
        self,
        patch_size: int = 16,
        num_patches: int = 196,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        tubelet_size: int = 2,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = in_channels * tubelet_size * patch_size * patch_size

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.0,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, num_masked: int) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - visible + mask tokens (mask tokens at the end)
            num_masked: Number of mask tokens

        Returns:
            predictions: [B, num_masked, patch_pixels]
        """
        for block in self.blocks:
            x = block(x)

        # Only predict masked patches (last num_masked positions)
        return self.head(self.norm(x[:, -num_masked:]))


class VideoMAEModel(nn.Module):
    """
    VideoMAE model for masked autoencoding.

    Follows official VideoMAE Base configuration:
    - Encoder: ViT-B (768d, 12 layers, 12 heads)
    - Decoder: 384d, 8 layers, 6 heads
    - 75% masking ratio
    - Per-patch normalized reconstruction target

    Args:
        image_size: Input image size (default: 224)
        patch_size: Patch size (default: 16)
        in_channels: Input channels (default: 3)
        encoder_embed_dim: Encoder dim (default: 768)
        encoder_depth: Encoder layers (default: 12)
        encoder_num_heads: Encoder heads (default: 12)
        decoder_embed_dim: Decoder dim (default: 384)
        decoder_depth: Decoder layers (default: 8)
        decoder_num_heads: Decoder heads (default: 6)
        mlp_ratio: MLP ratio (default: 4.0)
        num_frames: Number of frames (default: 2)
        tubelet_size: Temporal patch size (default: 2)
        mask_ratio: Masking ratio (default: 0.75)
        normalize_target: Per-patch normalization for targets (default: True)
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
        decoder_depth: int = 8,
        decoder_num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_frames: int = 2,
        tubelet_size: int = 2,
        mask_ratio: float = 0.75,
        normalize_target: bool = True,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_channels = in_channels
        self.normalize_target = normalize_target
        self.num_patches = (image_size // patch_size) ** 2 * (num_frames // tubelet_size)

        # Encoder
        self.encoder = VideoMAEEncoder(
            image_size=image_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=encoder_embed_dim,
            depth=encoder_depth, num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, num_frames=num_frames, tubelet_size=tubelet_size,
        )

        # Encoder → decoder projection
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional embedding (sinusoidal, shared for vis and mask)
        self.decoder_pos_embed = get_sinusoid_encoding_table(self.num_patches, decoder_embed_dim)

        # Decoder
        self.decoder = VideoMAEDecoder(
            patch_size=patch_size, num_patches=self.num_patches,
            embed_dim=decoder_embed_dim, depth=decoder_depth,
            num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            tubelet_size=tubelet_size, in_channels=in_channels,
        )

        trunc_normal_(self.mask_token, std=0.02)

    def random_masking(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate random mask.

        Returns:
            mask: [B, N] - True for masked patches
        """
        num_masked = int(self.mask_ratio * num_patches)
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        # mask[i] = True if patch i is masked
        mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
        mask[:, :num_patches - num_masked] = False
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def _forward_masked(
        self, video: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with given mask.

        Follows official VideoMAE decoder strategy:
        - Visible tokens + their pos embeddings come first
        - Mask tokens + their pos embeddings come after
        - Decoder predicts the last num_masked positions

        Args:
            video: [B, C, T, H, W]
            mask: [B, N] - True for masked patches

        Returns:
            predicted_patches: [B, num_masked, patch_pixels]
            cls_embedding: [B, encoder_embed_dim]
        """
        B = video.shape[0]
        device = video.device

        # Encode visible patches
        visible_embeddings = self.encoder(video, mask)
        cls_embedding = visible_embeddings.mean(dim=1)

        # Project to decoder dim
        visible_embeddings = self.encoder_to_decoder(visible_embeddings)
        num_visible = visible_embeddings.shape[1]
        num_masked = self.num_patches - num_visible

        # Positional embeddings: split into visible and masked
        pos_embed = self.decoder_pos_embed.expand(B, -1, -1).type_as(video).to(device).clone().detach()
        pos_vis = pos_embed[~mask].reshape(B, -1, pos_embed.shape[-1])
        pos_mask = pos_embed[mask].reshape(B, -1, pos_embed.shape[-1])

        # [visible + pos_vis, mask_token + pos_mask] (official strategy)
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        full_sequence = torch.cat(
            [visible_embeddings + pos_vis, mask_tokens + pos_mask], dim=1
        )

        # Decode (predicts last num_masked positions = the mask tokens)
        predicted_patches = self.decoder(full_sequence, num_masked)

        return predicted_patches, cls_embedding

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
            cls_embedding: [B, encoder_embed_dim]
        """
        B = image_current.shape[0]
        video = torch.stack([image_current, image_future], dim=2)
        mask = self.random_masking(B, self.num_patches, image_current.device)
        return self._forward_masked(video, mask)

    def _patchify(self, video: torch.Tensor) -> torch.Tensor:
        """
        Video [B, C, T, H, W] → patch pixels [B, N, tubelet*patch^2*C].

        Output format: [B, N, (p0 p1 p2) c] where p0=tubelet, p1=p2=patch_size.
        """
        B, C, T, H, W = video.shape
        p0, p1 = self.tubelet_size, self.patch_size
        # Reshape: b c (t p0) (h p1) (w p2) → b (t h w) (p0 p1 p2) c
        video = video.reshape(B, C, T // p0, p0, H // p1, p1, W // p1, p1)
        video = video.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        return video.reshape(B, -1, p0 * p1 * p1 * C)

    def _normalize_target(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Per-patch normalization (official VideoMAE target preprocessing).

        Each patch is normalized by its own mean and std across pixels.
        patches: [B, N, (p0 p1 p2) c] → view as [B, N, pixels, C] → normalize dim=-2

        This forces the model to predict relative structure within each patch
        rather than absolute pixel values.
        """
        # [B, N, pixels*C] → [B, N, pixels, C]
        B, N, D = patches.shape
        pixels_per_patch = D // self.in_channels
        patches = patches.reshape(B, N, pixels_per_patch, self.in_channels)

        # Per-patch normalization across pixel dimension
        mean = patches.mean(dim=-2, keepdim=True)
        var = patches.var(dim=-2, unbiased=True, keepdim=True)
        patches = (patches - mean) / (var.sqrt() + 1e-6)

        return patches.reshape(B, N, D)

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
        B = image_current.shape[0]
        video = torch.stack([image_current, image_future], dim=2)
        mask = self.random_masking(B, self.num_patches, image_current.device)

        # Predict masked patches
        predicted_patches, _ = self._forward_masked(video, mask)

        # Build target: patchify → normalize → select masked
        video_patches = self._patchify(video)
        if self.normalize_target:
            video_patches = self._normalize_target(video_patches)
        target = video_patches[mask].reshape(B, -1, video_patches.shape[-1])

        loss = F.mse_loss(predicted_patches, target)
        return loss, predicted_patches


class VideoMAEEncoderForVLA(nn.Module):
    """
    VideoMAE encoder wrapper for VLA integration.
    Provides encoding without masking for downstream tasks.
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

        self.encoder = VideoMAEEncoder(
            image_size=image_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        )

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load encoder weights from trained VideoMAE model.

        DDP/DataParallel로 저장된 체크포인트의 'module.' prefix를 자동 제거.
        encoder. prefix로 필터링하여 인코더 weight만 추출.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # 'module.' prefix 자동 제거 (DDP/DP wrapped checkpoint 호환)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                          for k, v in state_dict.items()}

        encoder_state = {
            k: v for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        result = self.load_state_dict(encoder_state, strict=False)
        if result.missing_keys:
            print(f"  WARNING: {len(result.missing_keys)} missing encoder keys "
                  f"(first 3: {result.missing_keys[:3]})")
        print(f"Loaded encoder weights from: {checkpoint_path}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 6, H, W] - two RGB images stacked

        Returns:
            patch_embeddings: [B, num_patches, embed_dim]
        """
        B = pixel_values.shape[0]
        image_current = pixel_values[:, :3]
        image_future = pixel_values[:, 3:]
        video = torch.stack([image_current, image_future], dim=2)

        # No masking for inference
        mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=video.device)
        return self.encoder(video, mask)
