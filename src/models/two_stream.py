"""
Two-Stream Model: Action-agnostic visual representation with M/P channels.

Architecture:
- M Channel (Magnocellular): Temporal change detection
- P Channel (Parvocellular): Spatial structure + color
- Interleaved ViT with CLS token exchange
- Dual reconstruction: CLS bottleneck decodes both current and future frames

CLS bottleneck prevents trivial patch-level shortcuts:
only (m_cls + p_cls) / 2 is passed to both decoders,
forcing the model to encode both temporal states into a single 768-dim vector.

Inspired by biological M/P visual pathways.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import TwoStreamPreprocessing


class CLSExchangeBlock(nn.Module):
    """
    Cross-stream information exchange via CLS token attention.

    Philosophy: CLS tokens summarize each channel's content.
                Self-attention between CLS tokens enables information sharing.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        """
        Exchange information between M and P CLS tokens.

        Args:
            cls_tokens: [B, 2, D] - concatenated M_cls and P_cls

        Returns:
            cls_tokens: [B, 2, D] - updated CLS tokens after exchange
        """
        # Self-attention between M_cls and P_cls
        normed = self.norm1(cls_tokens)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        cls_tokens = cls_tokens + attended

        # MLP
        cls_tokens = cls_tokens + self.mlp(self.norm2(cls_tokens))

        return cls_tokens


class InterleavedTwoStreamViT(nn.Module):
    """
    Interleaved Two-Stream ViT encoder with periodic CLS exchange.

    Structure:
    1. Process each channel with N transformer blocks
    2. Extract CLS tokens → Self-attention (information exchange)
    3. Inject updated CLS back to each stream
    4. Repeat for num_stages

    Args:
        embed_dim: Embedding dimension (default: 768)
        depth: Total number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
        num_stages: Number of CLS exchange stages (default: 3)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        image_size: Input image size (default: 224)
        patch_size: Patch size for ViT (default: 16)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_stages: int = 3,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_stages = num_stages
        self.blocks_per_stage = depth // num_stages

        # Patch embeddings for M (4ch) and P (5ch)
        self.patch_embed_m = nn.Conv2d(
            in_channels=4,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_embed_p = nn.Conv2d(
            in_channels=5,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS tokens
        self.cls_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_p = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_embed_p = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks per stage for each stream
        self.blocks_m = nn.ModuleList()
        self.blocks_p = nn.ModuleList()
        self.cls_exchange = nn.ModuleList()

        for _ in range(num_stages):
            # M channel blocks
            stage_blocks_m = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.blocks_per_stage)
            ])
            self.blocks_m.append(stage_blocks_m)

            # P channel blocks
            stage_blocks_p = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.blocks_per_stage)
            ])
            self.blocks_p.append(stage_blocks_p)

            # CLS exchange block
            self.cls_exchange.append(
                CLSExchangeBlock(embed_dim, num_heads=8, mlp_ratio=mlp_ratio)
            )

        self.norm_m = nn.LayerNorm(embed_dim)
        self.norm_p = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT convention."""
        nn.init.trunc_normal_(self.cls_token_m, std=0.02)
        nn.init.trunc_normal_(self.cls_token_p, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_m, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_p, std=0.02)

    def forward(
        self, m_channel: torch.Tensor, p_channel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode M and P channels with interleaved CLS exchange.

        Args:
            m_channel: [B, 4, H, W] - temporal change channel
            p_channel: [B, 5, H, W] - spatial structure channel

        Returns:
            m_tokens: [B, N+1, D] - M stream tokens (CLS + patches)
            p_tokens: [B, N+1, D] - P stream tokens (CLS + patches)
        """
        batch_size = m_channel.shape[0]

        # 1. Patch embedding
        m_patches = self.patch_embed_m(m_channel)  # [B, D, H', W']
        p_patches = self.patch_embed_p(p_channel)

        m_patches = m_patches.flatten(2).transpose(1, 2)  # [B, N, D]
        p_patches = p_patches.flatten(2).transpose(1, 2)

        # 2. Add CLS tokens
        m_cls = self.cls_token_m.expand(batch_size, -1, -1)
        p_cls = self.cls_token_p.expand(batch_size, -1, -1)

        m_tokens = torch.cat([m_cls, m_patches], dim=1)  # [B, N+1, D]
        p_tokens = torch.cat([p_cls, p_patches], dim=1)

        # 3. Add positional embeddings
        m_tokens = m_tokens + self.pos_embed_m
        p_tokens = p_tokens + self.pos_embed_p

        # 4. Interleaved processing with CLS exchange
        for stage_idx in range(self.num_stages):
            # Process each stream with transformer blocks
            for block_m in self.blocks_m[stage_idx]:
                m_tokens = block_m(m_tokens)

            for block_p in self.blocks_p[stage_idx]:
                p_tokens = block_p(p_tokens)

            # Extract CLS tokens
            m_cls = m_tokens[:, 0:1]  # [B, 1, D]
            p_cls = p_tokens[:, 0:1]

            # Exchange information between CLS tokens
            cls_combined = torch.cat([m_cls, p_cls], dim=1)  # [B, 2, D]
            cls_exchanged = self.cls_exchange[stage_idx](cls_combined)

            # Inject updated CLS back
            m_tokens = torch.cat([cls_exchanged[:, 0:1], m_tokens[:, 1:]], dim=1)
            p_tokens = torch.cat([cls_exchanged[:, 1:2], p_tokens[:, 1:]], dim=1)

        # 5. Final normalization
        m_tokens = self.norm_m(m_tokens)
        p_tokens = self.norm_p(p_tokens)

        return m_tokens, p_tokens


class PixelwiseFusion(nn.Module):
    """
    Fuse M and P patch tokens into unified representation.

    Args:
        embed_dim: Embedding dimension
        fusion_type: "separate" keeps M/P separate, "merged" combines them
    """

    def __init__(self, embed_dim: int = 768, fusion_type: str = "separate"):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "merged":
            self.fusion = nn.Linear(embed_dim * 2, embed_dim)

    def forward(
        self, m_tokens: torch.Tensor, p_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse M and P tokens.

        Args:
            m_tokens: [B, N+1, D]
            p_tokens: [B, N+1, D]

        Returns:
            cls_embedding: [B, D] - combined CLS token
            patch_embeddings: [B, N, D or 2D] - fused patches
        """
        # Extract CLS and patches
        m_cls = m_tokens[:, 0]  # [B, D]
        p_cls = p_tokens[:, 0]
        m_patches = m_tokens[:, 1:]  # [B, N, D]
        p_patches = p_tokens[:, 1:]

        # Fuse CLS tokens (always averaged)
        cls_embedding = (m_cls + p_cls) / 2

        # Fuse patches
        if self.fusion_type == "separate":
            # Concatenate M and P patches
            patch_embeddings = torch.cat([m_patches, p_patches], dim=2)  # [B, N, 2D]
        elif self.fusion_type == "merged":
            # Project concatenated patches to embed_dim
            patch_concat = torch.cat([m_patches, p_patches], dim=2)
            patch_embeddings = self.fusion(patch_concat)  # [B, N, D]
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        return cls_embedding, patch_embeddings


class CLSDecoder(nn.Module):
    """
    Decode CLS embedding to image via spatial expansion + upsampling.

    CLS [B, D] → expand to patch grid [B, D, P, P] → upsample to [B, 3, H, W].
    This forces the CLS bottleneck: reconstruction must succeed from a single
    global vector, preventing trivial patch-level shortcuts.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        P = image_size // patch_size  # 14 for 224/16
        self.num_patches_per_side = P

        # Upsample 14×14 → 224×224
        # (CLS는 단순 repeat으로 공간 확장 — learned expand는 115M 파라미터 낭비이고
        #  CNN이 공간 구조를 직접 생성하도록 강제하는 것이 bottleneck 효과에 더 부합)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=4, stride=2, padding=1),
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

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_embedding: [B, D]

        Returns:
            [B, 3, H, W]
        """
        P = self.num_patches_per_side
        x = cls_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, P, P)  # [B, D, P, P]
        return self.upsampler(x)


class TwoStreamModel(nn.Module):
    """
    Complete Two-Stream model with dual CLS-bottleneck reconstruction.

    Pipeline:
    1. Preprocessing: RGB → M/P channels
    2. Encoder: Interleaved ViT with CLS exchange
    3. CLS fusion: (m_cls + p_cls) / 2  →  single global vector [B, D]
    4. Dual decoding: decoder_current → image_t, decoder_future → image_tk

    Args:
        embed_dim: Embedding dimension (default: 768)
        depth: Total transformer depth (default: 12)
        num_heads: Number of attention heads (default: 12)
        num_stages: Number of CLS exchange stages (default: 3)
        image_size: Input image size (default: 224)
        patch_size: Patch size (default: 16)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_stages: int = 3,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self.preprocessing = TwoStreamPreprocessing()
        self.encoder = InterleavedTwoStreamViT(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_stages=num_stages,
            mlp_ratio=mlp_ratio,
            image_size=image_size,
            patch_size=patch_size,
        )
        # Two decoders share the same fused CLS bottleneck.
        # decoder_current reconstructs image_t, decoder_future reconstructs image_tk.
        # Both losses force the CLS to encode a compact representation of both frames.
        self.decoder_current = CLSDecoder(embed_dim=embed_dim, image_size=image_size, patch_size=patch_size)
        self.decoder_future = CLSDecoder(embed_dim=embed_dim, image_size=image_size, patch_size=patch_size)

    def forward(
        self, image_current: torch.Tensor, image_future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            image_current: [B, 3, H, W], range [0, 1]
            image_future: [B, 3, H, W], range [0, 1]

        Returns:
            pred_current: [B, 3, H, W] - reconstructed current frame
            pred_future: [B, 3, H, W] - predicted future frame
            cls_embedding: [B, embed_dim] - fused CLS (M+P averaged)
        """
        # 1. Preprocessing
        m_channel, p_channel = self.preprocessing(image_current, image_future)

        # 2. Encoding with CLS exchange
        m_tokens, p_tokens = self.encoder(m_channel, p_channel)

        # 3. CLS fusion (bottleneck): only global summaries go to decoder
        cls_embedding = (m_tokens[:, 0] + p_tokens[:, 0]) / 2  # [B, D]

        # 4. Dual decoding from the same CLS bottleneck
        pred_current = self.decoder_current(cls_embedding)
        pred_future = self.decoder_future(cls_embedding)

        return pred_current, pred_future, cls_embedding

    def compute_loss(
        self, image_current: torch.Tensor, image_future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dual reconstruction loss (current + future).

        Returns:
            loss: Scalar combined loss
            pred_future: [B, 3, H, W] for visualization
        """
        pred_current, pred_future, _ = self.forward(image_current, image_future)
        loss = (F.mse_loss(pred_current, image_current) + F.mse_loss(pred_future, image_future)) / 2
        return loss, pred_future


class TwoStreamEncoder(nn.Module):
    """
    Two-Stream encoder for integration with VLA models.

    This wraps the trained Two-Stream model to provide only
    the visual encoding capability for downstream tasks.

    Args:
        checkpoint_path: Path to trained checkpoint (optional)
        embed_dim: Embedding dimension (default: 768)
        depth: Transformer depth (default: 12)
        num_heads: Number of heads (default: 12)
        num_stages: CLS exchange stages (default: 3)
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_stages: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self._embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Build encoder components
        self.preprocessing = TwoStreamPreprocessing()
        self.encoder = InterleavedTwoStreamViT(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_stages=num_stages,
            image_size=image_size,
            patch_size=patch_size,
        )
        self.fusion = PixelwiseFusion(embed_dim=embed_dim, fusion_type="separate")

        # Load pretrained weights if provided
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    @property
    def embed_dim(self) -> int:
        """Output embedding dimension (required by VLA interface)."""
        return self._embed_dim * 2  # Returns 2D due to separate fusion

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load encoder weights from trained Two-Stream model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Extract encoder weights only (exclude both decoders)
        encoder_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("decoder_current.") and not k.startswith("decoder_future.")
        }

        self.load_state_dict(encoder_state, strict=False)
        print(f"Loaded encoder weights from: {checkpoint_path}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VLA integration.

        Args:
            pixel_values: [B, 6, H, W] - two RGB images stacked
                          (image_current + image_future)

        Returns:
            patch_embeddings: [B, num_patches, 2*embed_dim]
        """
        # Split stacked images
        image_current = pixel_values[:, :3]
        image_future = pixel_values[:, 3:]

        # Encode
        m_channel, p_channel = self.preprocessing(image_current, image_future)
        m_tokens, p_tokens = self.encoder(m_channel, p_channel)
        _, patch_embeddings = self.fusion(m_tokens, p_tokens)

        return patch_embeddings
