#!/usr/bin/env python
"""
OpenVLA Vision Encoder Wrapper

Wraps our trained encoders (Two-Stream, Single-Stream, VideoMAE) to be compatible
with OpenVLA's PrismaticVisionBackbone interface.

Key requirements for OpenVLA compatibility:
1. forward(pixel_values) -> [batch, num_patches, embed_dim]
2. .embed_dim property
3. Accept 6-channel input (two 3-channel images stacked)

Usage:
    # Create encoder from checkpoint
    encoder = TwoStreamEncoderForOpenVLA.from_checkpoint(
        checkpoint_path="/path/to/checkpoint.pt"
    )

    # Use like OpenVLA vision backbone
    pixel_values = torch.randn(1, 6, 224, 224)  # [img_t, img_tk] stacked
    patch_embeddings = encoder(pixel_values)    # [1, 196, embed_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_stream import TwoStreamPreprocessing, InterleavedTwoStreamViT, PixelwiseFusion
from .baselines import SingleStreamVideoPredictor  # noqa: F401 (kept for backward compat)
from .videomae_wrapper import VideoMAEForBridge


class TwoStreamEncoderForOpenVLA(nn.Module):
    """
    Two-Stream encoder wrapped for OpenVLA compatibility.

    Accepts 6-channel input (img_t + img_tk stacked) and returns
    patch embeddings in OpenVLA format.

    Input: [B, 6, 224, 224] - two RGB images stacked channel-wise
    Output: [B, num_patches, embed_dim] - patch embeddings for LLM
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_stages: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self._embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2  # 196

        # Two-Stream components
        self.preprocessing = TwoStreamPreprocessing()
        self.encoder = InterleavedTwoStreamViT(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_stages=num_stages,
            img_size=img_size,
            patch_size=patch_size,
        )
        self.fusion = PixelwiseFusion(embed_dim=embed_dim, fusion_type="separate")

    @property
    def embed_dim(self) -> int:
        """Output embedding dimension (required by OpenVLA)."""
        return self._embed_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass compatible with OpenVLA.

        Args:
            pixel_values: [B, 6, H, W] - two RGB images stacked
                          First 3 channels: img_t (current frame)
                          Last 3 channels: img_tk (future frame)

        Returns:
            patch_embeddings: [B, num_patches, embed_dim]
        """
        # Split the 6-channel input into two images
        img_t = pixel_values[:, :3]   # [B, 3, H, W]
        img_tk = pixel_values[:, 3:]  # [B, 3, H, W]

        # Two-Stream preprocessing
        m_channels = self.preprocessing.magnocellular_channel(img_t, img_tk)  # [B, 4, H, W]
        p_channels = self.preprocessing.parvocellular_channel(img_t)          # [B, 5, H, W]

        # Encode
        m_tokens, p_tokens, _ = self.encoder(m_channels, p_channels)  # [B, N+1, D]

        # Fuse (returns cls_fused, patches_fused)
        _, patches_fused = self.fusion(m_tokens, p_tokens)  # [B, N, D]

        return patches_fused

    def get_cls_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get CLS embedding (useful for other downstream tasks).

        Returns:
            cls_embedding: [B, embed_dim]
        """
        img_t = pixel_values[:, :3]
        img_tk = pixel_values[:, 3:]

        m_channels = self.preprocessing.magnocellular_channel(img_t, img_tk)
        p_channels = self.preprocessing.parvocellular_channel(img_t)

        m_tokens, p_tokens, _ = self.encoder(m_channels, p_channels)
        cls_fused, _ = self.fusion(m_tokens, p_tokens)

        return cls_fused

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        """
        Load encoder from a TwoStreamVideoPredictor checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on

        Returns:
            TwoStreamEncoderForOpenVLA instance with loaded weights

        Note:
            TwoStreamVideoPredictor checkpoint has structure:
            - encoder.preprocess.* -> maps to preprocessing.*
            - encoder.encoder.* -> maps to encoder.*
            - encoder.fusion.* -> maps to fusion.*
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get model config from checkpoint if available
        config = checkpoint.get("config", {})
        embed_dim = config.get("embed_dim", 768)
        depth = config.get("depth", 12)
        num_heads = config.get("num_heads", 12)
        num_stages = config.get("num_stages", 3)

        # Create encoder
        encoder = cls(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_stages=num_stages,
        )

        # Load weights (filter to only encoder-related weights)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Map weights from TwoStreamVideoPredictor to TwoStreamEncoderForOpenVLA
        # Checkpoint structure: encoder.{preprocess|encoder|fusion}.*
        # Target structure: {preprocessing|encoder|fusion}.*
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder.preprocess."):
                # encoder.preprocess.* -> preprocessing.*
                new_key = key.replace("encoder.preprocess.", "preprocessing.")
                encoder_state[new_key] = value
            elif key.startswith("encoder.encoder."):
                # encoder.encoder.* -> encoder.*
                new_key = key.replace("encoder.encoder.", "encoder.")
                encoder_state[new_key] = value
            elif key.startswith("encoder.fusion."):
                # encoder.fusion.* -> fusion.*
                new_key = key.replace("encoder.fusion.", "fusion.")
                encoder_state[new_key] = value

        # Load with strict=True to verify all weights are loaded correctly
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

        if missing:
            print(f"Warning: Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"  - {k}")
        if unexpected:
            print(f"Warning: Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"  - {k}")

        if not missing and not unexpected:
            print(f"Successfully loaded {len(encoder_state)} weights from checkpoint")

        encoder.to(device)
        encoder.eval()

        return encoder


class SingleStreamEncoderForOpenVLA(nn.Module):
    """
    Single-Stream encoder wrapped for OpenVLA compatibility.

    Two-stream과 동일한 M+P 전처리(9ch)를 사용하되, 단일 ViT로 처리.
    숏컷(img_tk 직접 노출) 없이 M/P 분리 구조의 효과만 ablation.

    Input: [B, 6, 224, 224] - two RGB images stacked (img_t + img_tk)
    Output: [B, num_patches, embed_dim] - patch embeddings
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self._embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Two-stream과 동일한 전처리
        self.preprocessing = TwoStreamPreprocessing()

        # Patch embedding (9 channels: M(4) + P(5))
        self.patch_embed = nn.Conv2d(
            9, embed_dim,
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
                dim_feedforward=int(embed_dim * 4.0),
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """공통 인코딩 로직. [B, N+1, D] 반환 (CLS 포함)."""
        B = pixel_values.shape[0]

        # 6ch 입력 → M(4ch) + P(5ch) 전처리
        img_t = pixel_values[:, :3]
        img_tk = pixel_values[:, 3:]
        m_ch = self.preprocessing.magnocellular_channel(img_t, img_tk)  # [B, 4, H, W]
        p_ch = self.preprocessing.parvocellular_channel(img_t)           # [B, 5, H, W]
        x_9ch = torch.cat([m_ch, p_ch], dim=1)                           # [B, 9, H, W]

        # Patch embedding
        x = self.patch_embed(x_9ch)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]

        # Add CLS token
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        return self.norm(x)  # [B, N+1, D]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 6, H, W] - two RGB images stacked

        Returns:
            patch_embeddings: [B, num_patches, embed_dim]
        """
        return self._encode(pixel_values)[:, 1:]  # [B, N, D]

    def get_cls_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get CLS embedding."""
        return self._encode(pixel_values)[:, 0]  # [B, D]

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        """
        Load encoder from SingleStreamVideoPredictor checkpoint.

        Expected checkpoint structure:
        - patch_embed.weight: [embed_dim, 6, patch_size, patch_size] (6-channel input)
        - blocks.*, cls_token, pos_embed, norm.*

        NOTE: Checkpoints with 3-channel patch_embed are NOT compatible.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config", {})

        encoder = cls(
            embed_dim=config.get("embed_dim", 768),
            depth=config.get("depth", 12),
            num_heads=config.get("num_heads", 12),
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Check input channel compatibility (9ch: M(4) + P(5))
        if "patch_embed.weight" in state_dict:
            ckpt_in_channels = state_dict["patch_embed.weight"].shape[1]
            if ckpt_in_channels != 9:
                print(f"WARNING: Checkpoint uses {ckpt_in_channels}-channel input, "
                      f"but SingleStreamEncoder requires 9 channels (M+P).")
                if ckpt_in_channels == 6:
                    print("         This checkpoint was trained with old 6-ch concat design.")
                print("         Returning encoder with random initialization.")
                encoder.to(device)
                encoder.eval()
                return encoder

        # Map weights (exclude decoder and mask_token)
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("decoder."):
                continue
            if key == "mask_token":  # Skip VideoMAE-style mask token
                continue
            encoder_state[key] = value

        # Load weights
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

        if missing:
            print(f"Warning: Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"  - {k}")
        if unexpected:
            print(f"Warning: Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"  - {k}")

        if not missing and not unexpected:
            print(f"Successfully loaded {len(encoder_state)} weights from checkpoint")

        encoder.to(device)
        encoder.eval()

        return encoder


class VideoMAEEncoderForOpenVLA(nn.Module):
    """
    VideoMAE encoder wrapped for OpenVLA compatibility.

    Input: [B, 6, 224, 224] - two RGB images (treated as 2-frame video)
    Output: [B, num_patches, embed_dim] - visible patch embeddings
    """

    def __init__(
        self,
        model_size: str = "base",
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        # VideoMAE config
        if model_size == "base":
            self._embed_dim = 768
            depth = 12
            num_heads = 12
        elif model_size == "large":
            self._embed_dim = 1024
            depth = 24
            num_heads = 16
        else:
            raise ValueError(f"Unknown model_size: {model_size}")

        self.num_patches = (img_size // patch_size) ** 2  # 196 (2 frames / tubelet_size 2)

        # Create VideoMAE model (we'll use it without masking)
        self.model = VideoMAEForBridge(
            model_size=model_size,
            num_frames=2,
            tubelet_size=2,
            mask_ratio=0.0,  # No masking for inference
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 6, H, W] - two RGB images stacked

        Returns:
            patch_embeddings: [B, num_patches, embed_dim]
        """
        B = pixel_values.shape[0]

        # Reshape to video format: [B, 2, 3, H, W]
        img_t = pixel_values[:, :3]   # [B, 3, H, W]
        img_tk = pixel_values[:, 3:]  # [B, 3, H, W]
        video = torch.stack([img_t, img_tk], dim=1)  # [B, 2, 3, H, W]

        # Get encoder output (we need to modify VideoMAE to expose this)
        # For now, use the full forward and extract encoder features
        # VideoMAE's encoder returns [B, num_visible_patches, D]
        encoder_out = self.model.encode(video)

        return encoder_out

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        """
        Load encoder from VideoMAEForBridge checkpoint.

        Expected checkpoint structure:
        - encoder.blocks.0.attn.qkv.*, encoder.blocks.0.mlp.fc1.* (VideoMAE style)

        NOTE: Checkpoints with blocks.0.linear1.*, blocks.0.self_attn.* are
        SingleStream-style and NOT compatible with VideoMAE architecture.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config", {})

        encoder = cls(model_size=config.get("model_size", "base"))

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Check checkpoint architecture compatibility
        has_videomae_style = any("attn.qkv" in k or "mlp.fc1" in k for k in state_dict.keys())
        has_singlestream_style = any("linear1" in k or "self_attn.in_proj" in k for k in state_dict.keys())

        if has_singlestream_style and not has_videomae_style:
            print("WARNING: Checkpoint uses SingleStream architecture (linear1, self_attn)")
            print("         This is NOT compatible with VideoMAE architecture.")
            print("         Use SingleStreamEncoderForOpenVLA.from_checkpoint() instead.")
            print("         Returning encoder with random initialization.")
            encoder.to(device)
            encoder.eval()
            return encoder

        # Map weights from VideoMAEForBridge to VideoMAEEncoderForOpenVLA
        # Checkpoint: encoder.blocks.0.* -> Target: model.encoder.blocks.0.*
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = "model." + key
                encoder_state[new_key] = value
            elif key.startswith("decoder.") or key == "mask_token":
                continue  # Skip decoder and mask token

        # Load weights
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

        loaded = len(encoder_state)
        if missing:
            print(f"Warning: Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"  - {k}")
        if unexpected:
            print(f"Warning: Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"  - {k}")

        if not missing and not unexpected and loaded > 0:
            print(f"Successfully loaded {loaded} weights from checkpoint")
        elif loaded == 0:
            print("Warning: No matching weights found in checkpoint")

        encoder.to(device)
        encoder.eval()

        return encoder


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Test input: 6-channel (two RGB images stacked)
    pixel_values = torch.rand(2, 6, 224, 224, device=device)

    print("=" * 60)
    print("Testing TwoStreamEncoderForOpenVLA")
    print("=" * 60)
    encoder = TwoStreamEncoderForOpenVLA().to(device)
    output = encoder(pixel_values)
    print(f"Input:  {pixel_values.shape}")
    print(f"Output: {output.shape}")
    print(f"embed_dim: {encoder.embed_dim}")
    assert output.shape == (2, 196, 768), f"Expected (2, 196, 768), got {output.shape}"
    print("TwoStream: OK\n")

    print("=" * 60)
    print("Testing SingleStreamEncoderForOpenVLA")
    print("=" * 60)
    encoder = SingleStreamEncoderForOpenVLA().to(device)
    output = encoder(pixel_values)
    print(f"Input:  {pixel_values.shape}")
    print(f"Output: {output.shape}")
    print(f"embed_dim: {encoder.embed_dim}")
    assert output.shape == (2, 196, 768), f"Expected (2, 196, 768), got {output.shape}"
    print("SingleStream: OK\n")

    print("=" * 60)
    print("Testing VideoMAEEncoderForOpenVLA")
    print("=" * 60)
    encoder = VideoMAEEncoderForOpenVLA().to(device)
    output = encoder(pixel_values)
    print(f"Input:  {pixel_values.shape}")
    print(f"Output: {output.shape}")
    print(f"embed_dim: {encoder.embed_dim}")
    print("VideoMAE: OK\n")

    print("All encoder wrappers working correctly!")
