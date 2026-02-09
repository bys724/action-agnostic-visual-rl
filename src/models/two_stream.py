#!/usr/bin/env python
"""
Two-Stream Vision Model for Action-Agnostic Behavior Representation

Progressive implementation:
1. Two-Stream Preprocessing (M/P channels)
2. Two-Stream ViT Encoder
3. Pixel-wise Fusion
4. Video Prediction Decoder
5. Training Pipeline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# =============================================================================
# Step 1: Two-Stream Preprocessing
# =============================================================================

class TwoStreamPreprocessing(nn.Module):
    """
    Biological-inspired image preprocessing.

    M Channel (Magnocellular/Dorsal): Temporal change detection
        - Input: img_prev, img_curr
        - Output: [ΔL, ΔR, ΔG, ΔB] (4 channels)
        - Absolute range preservation [-1, 1]

    P Channel (Parvocellular/Ventral): Spatial structure + color
        - Input: img_curr (or img_prev)
        - Output: [∂x, ∂y, R, G, B] (5 channels)
        - Luminance normalized, magnitude normalized gradients
    """

    def __init__(self):
        super().__init__()

        # Luminance weights (BT.709 standard, fixed)
        # NOTE: 학습 가능하게 시도했으나 w/w.sum() 정규화의 backward에서
        #       gradient 불안정 발생. detach, 낮은 lr 등 시도했으나 해결 안됨.
        #       BT.709은 인간 시각에 최적화된 값이므로 고정 사용.
        self.register_buffer('lum_weights', torch.tensor([0.2126, 0.7152, 0.0722]))

        # Sobel filters for spatial gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # Register as buffers (not parameters, but moves with device)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_luminance(self, img):
        """Compute luminance using BT.709 weights."""
        # img: [B, 3, H, W], range [0, 1]
        # Ensure weights are on the same device as input (for DataParallel)
        w = self.lum_weights.to(img.device).view(1, 3, 1, 1)
        lum = (img * w).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        return lum

    def magnocellular_channel(self, img_prev, img_curr):
        """
        M Channel: Temporal change (Dorsal pathway)

        Returns:
            [B, 4, H, W]: [ΔL, ΔR, ΔG, ΔB], range [-1, 1]
        """
        # Luminance change
        lum_prev = self.compute_luminance(img_prev)
        lum_curr = self.compute_luminance(img_curr)
        delta_L = torch.clamp(lum_curr - lum_prev, -1, 1)

        # RGB channel changes
        delta_R = torch.clamp(img_curr[:, 0:1] - img_prev[:, 0:1], -1, 1)
        delta_G = torch.clamp(img_curr[:, 1:2] - img_prev[:, 1:2], -1, 1)
        delta_B = torch.clamp(img_curr[:, 2:3] - img_prev[:, 2:3], -1, 1)

        return torch.cat([delta_L, delta_R, delta_G, delta_B], dim=1)

    def parvocellular_channel(self, img):
        """
        P Channel: Spatial structure + color (Ventral pathway)

        Returns:
            [B, 5, H, W]: [∂x, ∂y, R, G, B]
        """
        # 1. Compute luminance
        lum = self.compute_luminance(img)

        # 2. Per-image normalization (lighting invariance)
        mean = lum.mean(dim=(2, 3), keepdim=True)
        std = lum.std(dim=(2, 3), keepdim=True) + 1e-6
        lum_norm = torch.clamp((lum - mean) / std, -3, 3)

        # 3. Sobel gradients (ensure filters are on same device for DataParallel)
        grad_x = F.conv2d(lum_norm, self.sobel_x.to(lum_norm.device), padding=1)
        grad_y = F.conv2d(lum_norm, self.sobel_y.to(lum_norm.device), padding=1)

        # 4. Magnitude normalization (preserve direction + sharpness)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        max_mag = magnitude.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)

        grad_x_norm = grad_x / max_mag
        grad_y_norm = grad_y / max_mag

        # 5. RGB from current frame
        R, G, B = img[:, 0:1], img[:, 1:2], img[:, 2:3]

        return torch.cat([grad_x_norm, grad_y_norm, R, G, B], dim=1)

    def forward(self, img_prev, img_curr):
        """
        Process consecutive frames into M and P channels.

        Args:
            img_prev: [B, 3, H, W], range [0, 1]
            img_curr: [B, 3, H, W], range [0, 1]

        Returns:
            m_channels: [B, 4, H, W] - temporal change
            p_channels: [B, 5, H, W] - spatial structure + color
        """
        m_channels = self.magnocellular_channel(img_prev, img_curr)
        p_channels = self.parvocellular_channel(img_curr)

        return m_channels, p_channels


# =============================================================================
# Step 2: Two-Stream ViT Encoder
# =============================================================================

class TwoStreamViTEncoder(nn.Module):
    """
    Two separate ViT encoders for M and P channels.

    M Encoder: 4-channel input (temporal change)
    P Encoder: 5-channel input (spatial + color)

    Both output CLS token + patch tokens.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        embed_dim: int = 768,
        pretrained: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Create base ViT models (we'll replace patch_embed)
        self.vit_m = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        self.vit_p = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Get patch size from model
        patch_size = self.vit_m.patch_embed.patch_size[0]
        img_size = self.vit_m.patch_embed.img_size[0]

        # Replace patch embeddings for custom channels
        # M channel: 4 input channels
        self.vit_m.patch_embed = nn.Conv2d(
            4, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # P channel: 5 input channels
        self.vit_p.patch_embed = nn.Conv2d(
            5, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2

    def forward_vit(self, vit, x):
        """Forward through a single ViT, returning CLS + patches."""
        # Patch embedding
        x = vit.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]

        # Add CLS token (use repeat for DataParallel compatibility)
        cls_token = vit.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]

        # Add positional embedding
        x = x + vit.pos_embed

        # Transformer blocks
        x = vit.blocks(x)
        x = vit.norm(x)

        return x  # [B, N+1, D]

    def forward(self, m_channels, p_channels):
        """
        Encode M and P channels separately.

        Args:
            m_channels: [B, 4, H, W] - from preprocessing
            p_channels: [B, 5, H, W] - from preprocessing

        Returns:
            m_tokens: [B, N+1, D] - CLS + patch tokens for M
            p_tokens: [B, N+1, D] - CLS + patch tokens for P
        """
        m_tokens = self.forward_vit(self.vit_m, m_channels)
        p_tokens = self.forward_vit(self.vit_p, p_channels)

        return m_tokens, p_tokens


# =============================================================================
# Step 2.5: Interleaved Two-Stream ViT (CLS Exchange)
# =============================================================================

class CLSExchangeBlock(nn.Module):
    """
    CLS tokens 간 정보 교환을 위한 Transformer block.

    철학: CLS는 각 채널의 상황 요약.
         CLS끼리 self-attention → 요약 정보 교환.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, cls_tokens):
        """
        Args:
            cls_tokens: [B, 2, D] - M_cls와 P_cls 결합

        Returns:
            cls_tokens: [B, 2, D] - 정보 교환된 CLS tokens
        """
        # Self-attention (M_cls ↔ P_cls 상호작용)
        x = self.norm1(cls_tokens)
        x = cls_tokens + self.attn(x, x, x, need_weights=False)[0]

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class InterleavedTwoStreamViT(nn.Module):
    """
    Interleaved Two-Stream ViT with CLS Exchange.

    구조:
    1. 각 채널 N개 블록 처리 → CLS + patches
    2. CLS 추출 → CLS끼리 self-attention (정보 교환)
    3. 교환된 CLS를 원래 채널에 주입
    4. 1-3 반복 (num_stages 번)

    철학: CLS = 채널의 상황 요약
         CLS 교환 = 요약 정보 공유
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_stages: int = 3,  # CLS 교환 횟수
        mlp_ratio: float = 4.0,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_stages = num_stages
        self.blocks_per_stage = depth // num_stages  # 각 stage당 블록 수

        # Patch embeddings (M: 4ch, P: 5ch)
        self.patch_embed_m = nn.Conv2d(4, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embed_p = nn.Conv2d(5, embed_dim, kernel_size=patch_size, stride=patch_size)

        # CLS tokens
        self.cls_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_p = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_embed_p = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks for each channel (per stage)
        self.blocks_m = nn.ModuleList()
        self.blocks_p = nn.ModuleList()
        self.cls_exchange = nn.ModuleList()

        for stage in range(num_stages):
            # M channel blocks
            stage_blocks_m = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    activation='gelu',
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
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.blocks_per_stage)
            ])
            self.blocks_p.append(stage_blocks_p)

            # CLS exchange block (마지막 stage 제외)
            if stage < num_stages - 1:
                self.cls_exchange.append(CLSExchangeBlock(embed_dim, num_heads, mlp_ratio))

        # Final layer norms
        self.norm_m = nn.LayerNorm(embed_dim)
        self.norm_p = nn.LayerNorm(embed_dim)

        # Initialize
        nn.init.trunc_normal_(self.cls_token_m, std=0.02)
        nn.init.trunc_normal_(self.cls_token_p, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_m, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_p, std=0.02)

    def forward(self, m_channels, p_channels):
        """
        Args:
            m_channels: [B, 4, H, W]
            p_channels: [B, 5, H, W]

        Returns:
            m_tokens: [B, N+1, D]
            p_tokens: [B, N+1, D]
            p_cls_intermediates: dict with 'stage1', 'stage2', 'final' CLS tokens [B, D]
        """
        B = m_channels.shape[0]
        device = m_channels.device  # For DataParallel compatibility

        # Patch embedding (contiguous() for DataParallel memory alignment)
        m_patches = self.patch_embed_m(m_channels)  # [B, D, H', W']
        m_patches = m_patches.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]

        p_patches = self.patch_embed_p(p_channels)
        p_patches = p_patches.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]

        # Add CLS token (use repeat instead of expand for DataParallel compatibility)
        # expand() shares memory which can cause device issues with DataParallel
        cls_m = self.cls_token_m.repeat(B, 1, 1)  # [B, 1, D]
        cls_p = self.cls_token_p.repeat(B, 1, 1)

        m_tokens = torch.cat([cls_m, m_patches], dim=1)  # [B, N+1, D]
        p_tokens = torch.cat([cls_p, p_patches], dim=1)

        # Add position embedding (broadcast is safe, no memory sharing issues)
        m_tokens = m_tokens + self.pos_embed_m
        p_tokens = p_tokens + self.pos_embed_p

        # Store intermediate P CLS tokens for decoder
        p_cls_intermediates = {}

        # Interleaved processing
        for stage in range(self.num_stages):
            # Process M channel blocks
            for block in self.blocks_m[stage]:
                m_tokens = block(m_tokens)

            # Process P channel blocks
            for block in self.blocks_p[stage]:
                p_tokens = block(p_tokens)

            # Save intermediate P CLS (after block processing, before exchange)
            if stage == 0:
                p_cls_intermediates['stage1'] = p_tokens[:, 0].clone()  # [B, D]
            elif stage == 1:
                p_cls_intermediates['stage2'] = p_tokens[:, 0].clone()  # [B, D]

            # CLS exchange (마지막 stage 제외)
            if stage < self.num_stages - 1:
                # Extract CLS tokens
                m_cls = m_tokens[:, 0:1]  # [B, 1, D]
                p_cls = p_tokens[:, 0:1]  # [B, 1, D]

                # Combine and exchange
                cls_combined = torch.cat([m_cls, p_cls], dim=1)  # [B, 2, D]
                cls_exchanged = self.cls_exchange[stage](cls_combined)  # [B, 2, D]

                # Inject back
                m_tokens = torch.cat([cls_exchanged[:, 0:1], m_tokens[:, 1:]], dim=1)
                p_tokens = torch.cat([cls_exchanged[:, 1:2], p_tokens[:, 1:]], dim=1)

        # Final norm
        m_tokens = self.norm_m(m_tokens)
        p_tokens = self.norm_p(p_tokens)

        # Save final P CLS (after norm)
        p_cls_intermediates['final'] = p_tokens[:, 0].clone()  # [B, D]

        return m_tokens, p_tokens, p_cls_intermediates


# =============================================================================
# Step 3: Pixel-wise Fusion
# =============================================================================

class PixelwiseFusion(nn.Module):
    """
    Pixel-wise channel fusion preserving spatial structure.

    Key insight: avg(patches) loses spatial info critical for manipulation.
    Instead, we fuse M+P at each spatial location.

    CLS fusion:   FC([m_cls; p_cls]) -> [B, D]
    Patch fusion: FC([m_patch; p_patch]) per pixel -> [B, N, D]
    """

    def __init__(self, embed_dim: int = 768, fusion_type: str = "separate"):
        """
        Args:
            embed_dim: Dimension of ViT embeddings
            fusion_type: "separate" (different FC for CLS and patches) or
                        "shared" (same FC for all tokens)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type

        if fusion_type == "separate":
            self.fc_cls = nn.Linear(2 * embed_dim, embed_dim)
            self.fc_patch = nn.Linear(2 * embed_dim, embed_dim)
        elif fusion_type == "shared":
            self.fc_fusion = nn.Linear(2 * embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, m_tokens, p_tokens):
        """
        Fuse M and P tokens while preserving spatial structure.

        Args:
            m_tokens: [B, N+1, D] - CLS + patches from M encoder
            p_tokens: [B, N+1, D] - CLS + patches from P encoder

        Returns:
            cls_fused: [B, D] - fused CLS token
            patches_fused: [B, N, D] - fused patch tokens (spatial preserved)
        """
        # Split CLS and patches
        m_cls = m_tokens[:, 0]       # [B, D]
        m_patches = m_tokens[:, 1:]  # [B, N, D]
        p_cls = p_tokens[:, 0]
        p_patches = p_tokens[:, 1:]

        if self.fusion_type == "separate":
            # CLS fusion
            cls_concat = torch.cat([m_cls, p_cls], dim=-1)  # [B, 2D]
            cls_fused = self.fc_cls(cls_concat)  # [B, D]

            # Pixel-wise patch fusion
            patches_concat = torch.cat([m_patches, p_patches], dim=-1)  # [B, N, 2D]
            patches_fused = self.fc_patch(patches_concat)  # [B, N, D]

        elif self.fusion_type == "shared":
            # All tokens use same fusion
            cls_concat = torch.cat([m_cls, p_cls], dim=-1)
            cls_fused = self.fc_fusion(cls_concat)

            patches_concat = torch.cat([m_patches, p_patches], dim=-1)
            patches_fused = self.fc_fusion(patches_concat)

        return cls_fused, patches_fused


# =============================================================================
# Step 4: Combined Encoder (Preprocessing + ViT + Fusion)
# =============================================================================

class TwoStreamEncoder(nn.Module):
    """
    Complete two-stream encoder: Preprocessing -> ViT -> Fusion

    Input: Two consecutive RGB frames
    Output: Fused CLS token + spatial patch embeddings

    encoder_type:
        - "separate": 분리된 ViT (timm 기반, pretrained 가능)
        - "interleaved": CLS 교환 ViT (custom, 처음부터 학습)
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        embed_dim: int = 768,
        pretrained: bool = True,
        fusion_type: str = "separate",
        encoder_type: str = "interleaved",  # "separate" or "interleaved"
        num_stages: int = 3,  # interleaved only: CLS 교환 횟수
        depth: int = 12,  # interleaved only: 총 블록 수
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.preprocess = TwoStreamPreprocessing()

        if encoder_type == "separate":
            # 분리된 ViT (timm pretrained)
            self.encoder = TwoStreamViTEncoder(
                model_name=model_name,
                embed_dim=embed_dim,
                pretrained=pretrained,
            )
        elif encoder_type == "interleaved":
            # CLS 교환 ViT (custom)
            self.encoder = InterleavedTwoStreamViT(
                embed_dim=embed_dim,
                depth=depth,
                num_heads=embed_dim // 64,  # 768 -> 12 heads
                num_stages=num_stages,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.fusion = PixelwiseFusion(
            embed_dim=embed_dim,
            fusion_type=fusion_type,
        )

    def forward(self, img_prev, img_curr):
        """
        End-to-end encoding of consecutive frames.

        Args:
            img_prev: [B, 3, H, W], range [0, 1]
            img_curr: [B, 3, H, W], range [0, 1]

        Returns:
            cls_fused: [B, D] - global change representation
            patches_fused: [B, N, D] - spatial change representation
            p_cls_intermediates: dict with 'stage1', 'stage2', 'final' CLS [B, D]
                                 (only for interleaved encoder, None otherwise)
        """
        # Step 1: Preprocessing
        m_ch, p_ch = self.preprocess(img_prev, img_curr)

        # Step 2: Encode
        p_cls_intermediates = None
        if self.encoder_type == "interleaved":
            m_tokens, p_tokens, p_cls_intermediates = self.encoder(m_ch, p_ch)
        else:
            # Separate encoder doesn't support intermediate CLS yet
            m_tokens, p_tokens = self.encoder(m_ch, p_ch)

        # Step 3: Fuse
        cls_fused, patches_fused = self.fusion(m_tokens, p_tokens)

        return cls_fused, patches_fused, p_cls_intermediates


# =============================================================================
# Step 5: Video Prediction Decoder (U-Net style)
# =============================================================================

class VideoDecoder(nn.Module):
    """
    Video prediction decoder with skip connections and multi-scale CLS injection.

    Architecture:
    - Skip connection from img_t (downsampled to 56x56)
    - Multi-scale CLS injection via FiLM-like modulation
    - 4 upsampling levels: 14→28→56→112→224

    CLS Injection Strategy (coarse-to-fine):
    - Level 1 (14→28): final CLS (abstract/high-level)
    - Level 2 (28→56): stage2 CLS (mid-level) + skip from img_t
    - Level 3 (56→112): stage1 CLS (concrete/low-level)
    - Level 4 (112→224): No CLS (detail refinement)

    Input:
        patches: [B, 196, D] (14x14 spatial tokens from encoder)
        img_t: [B, 3, 224, 224] (current frame for skip connection)
        p_cls_intermediates: dict with 'stage1', 'stage2', 'final' CLS [B, D]
    Output:
        img_pred: [B, 3, 224, 224]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        out_channels: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Project embedding dimension to decoder channels
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # Skip connection encoder: img_t (224x224) -> features (56x56)
        self.skip_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),   # 224→112
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 112→56
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # CLS projection layers (embed_dim -> channel dim for FiLM modulation)
        self.cls_proj_1 = nn.Linear(embed_dim, 256)  # for level 1 output
        self.cls_proj_2 = nn.Linear(embed_dim, 128)  # for level 2 output
        self.cls_proj_3 = nn.Linear(embed_dim, 64)   # for level 3 output

        # Upsampling blocks
        # Level 1: 14x14 -> 28x28 (512 -> 256)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Level 2: 28x28 -> 56x56 (256 -> 128)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        # Skip merge: concat(128 + 64) -> 128
        self.skip_merge = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Level 3: 56x56 -> 112x112 (128 -> 64)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Level 4: 112x112 -> 224x224 (64 -> 32, no CLS)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Final output
        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 1),
        )

    def inject_cls(self, feature_map, cls_token, proj_layer):
        """
        FiLM-like CLS injection (additive modulation).

        Args:
            feature_map: [B, C, H, W]
            cls_token: [B, D] - CLS embedding
            proj_layer: nn.Linear(D, C)

        Returns:
            modulated_feature: [B, C, H, W]
        """
        B, C, H, W = feature_map.shape
        cls_proj = proj_layer(cls_token)  # [B, C]
        cls_proj = cls_proj.view(B, C, 1, 1)  # [B, C, 1, 1]
        return feature_map + cls_proj

    def forward(self, patches, img_t=None, p_cls_intermediates=None):
        """
        Generate future frame from patch embeddings with skip and CLS injection.

        Args:
            patches: [B, 196, D] - patch embeddings from encoder (14x14 tokens)
            img_t: [B, 3, 224, 224] - current frame for skip connection (optional)
            p_cls_intermediates: dict with 'stage1', 'stage2', 'final' CLS [B, D] (optional)

        Returns:
            img_pred: [B, 3, 224, 224] - predicted future frame
        """
        B, N, D = patches.shape
        H = W = int(N ** 0.5)  # 14

        # Project to decoder dimension
        x = self.input_proj(patches)  # [B, 196, 512]
        x = x.permute(0, 2, 1).contiguous().view(B, 512, H, W)  # [B, 512, 14, 14]

        # Compute skip features from img_t (-> 56x56)
        skip_feat = None
        if img_t is not None:
            skip_feat = self.skip_encoder(img_t)  # [B, 64, 56, 56]

        # Level 1: 14 -> 28 (+ final CLS)
        x = self.up1(x)  # [B, 256, 28, 28]
        if p_cls_intermediates is not None and 'final' in p_cls_intermediates:
            x = self.inject_cls(x, p_cls_intermediates['final'], self.cls_proj_1)

        # Level 2: 28 -> 56 (+ stage2 CLS)
        x = self.up2(x)  # [B, 128, 56, 56]
        if p_cls_intermediates is not None and 'stage2' in p_cls_intermediates:
            x = self.inject_cls(x, p_cls_intermediates['stage2'], self.cls_proj_2)

        # Merge skip connection at 56x56
        if skip_feat is not None:
            x = torch.cat([x, skip_feat], dim=1)  # [B, 192, 56, 56]
            x = self.skip_merge(x)  # [B, 128, 56, 56]

        # Level 3: 56 -> 112 (+ stage1 CLS)
        x = self.up3(x)  # [B, 64, 112, 112]
        if p_cls_intermediates is not None and 'stage1' in p_cls_intermediates:
            x = self.inject_cls(x, p_cls_intermediates['stage1'], self.cls_proj_3)

        # Level 4: 112 -> 224 (no CLS, detail refinement)
        x = self.up4(x)  # [B, 32, 224, 224]

        # Final output
        img_pred = torch.sigmoid(self.output_conv(x))

        return img_pred


# =============================================================================
# Step 6: Complete Video Prediction Model
# =============================================================================

class TwoStreamVideoPredictor(nn.Module):
    """
    Complete model for video prediction pre-training.

    Pipeline:
    1. Two-Stream Encoder: (img_t, img_t+k) -> cls_emb, patches
    2. Video Decoder: patches -> img_pred
    3. Loss: MSE(img_pred, img_t+k)

    NOTE: 디코더는 이전 이미지(img_t) 없이 순수하게 임베딩만으로 이미지 생성.
          이렇게 하면 인코더가 더 완전한 표현을 학습하도록 강제됨.

    encoder_type:
        - "separate": 분리된 ViT (timm pretrained 가능)
        - "interleaved": CLS 교환 ViT (처음부터 학습, 권장)
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        embed_dim: int = 768,
        pretrained: bool = True,
        encoder_type: str = "interleaved",
        num_stages: int = 3,
        depth: int = 12,
    ):
        super().__init__()

        self.encoder = TwoStreamEncoder(
            model_name=model_name,
            embed_dim=embed_dim,
            pretrained=pretrained,
            encoder_type=encoder_type,
            num_stages=num_stages,
            depth=depth,
        )

        self.decoder = VideoDecoder(
            embed_dim=embed_dim,
        )

    def forward(self, img_t, img_tk):
        """
        Forward pass for video prediction.

        Args:
            img_t: [B, 3, H, W] - current frame
            img_tk: [B, 3, H, W] - future frame (k steps ahead)

        Returns:
            img_pred: [B, 3, H, W] - predicted future frame
            cls_emb: [B, D] - CLS embedding (for downstream tasks)
        """
        # Encode: two frames -> embeddings + intermediate CLS
        cls_emb, patches, p_cls_intermediates = self.encoder(img_t, img_tk)

        # Decode: patches + img_t skip + multi-scale CLS -> future frame
        img_pred = self.decoder(patches, img_t, p_cls_intermediates)

        return img_pred, cls_emb

    def compute_loss(self, img_t, img_tk):
        """
        Compute reconstruction loss.

        Args:
            img_t: Current frame
            img_tk: Target future frame

        Returns:
            loss: MSE reconstruction loss
            img_pred: Predicted frame (for visualization)
        """
        img_pred, cls_emb = self.forward(img_t, img_tk)
        loss = F.mse_loss(img_pred, img_tk)
        return loss, img_pred


# =============================================================================
# Step 7: Training Pipeline
# =============================================================================

class MultiGapVideoDataset(torch.utils.data.Dataset):
    """
    Dataset for video prediction training with multi-gap sampling.

    Multi-Gap Strategy:
    - 다양한 프레임 간격(gap)의 페어를 샘플링
    - 샘플링 확률: 완만한 지수 감쇠 (모든 간격이 어느정도 샘플링됨)
    - Loss 가중치: 급격한 지수 감쇠 (작은 간격에 집중)

    Expected data format:
    - video_paths: list of video file paths or frame directories
    - Each video provides pairs (frame_t, frame_t+k)
    """

    def __init__(
        self,
        video_paths,
        max_gap: int = 10,
        img_size: int = 224,
        transform=None,
        sample_decay: float = 0.3,  # 완만한 감쇠 (샘플링 확률)
        loss_decay: float = 0.7,    # 급격한 감쇠 (loss 가중치)
    ):
        """
        Args:
            video_paths: List of video paths (placeholder for actual implementation)
            max_gap: Maximum frame gap (1 to max_gap)
            img_size: Output image size
            transform: Optional augmentation transforms
            sample_decay: Exponential decay rate for sampling probability (smaller = more uniform)
            loss_decay: Exponential decay rate for loss weight (larger = more focus on small gaps)
        """
        self.video_paths = video_paths
        self.max_gap = max_gap
        self.img_size = img_size
        self.transform = transform
        self.sample_decay = sample_decay
        self.loss_decay = loss_decay

        # Compute sampling probabilities (완만한 감쇠)
        # p(k) ∝ exp(-sample_decay * (k-1))
        gaps = np.arange(1, max_gap + 1)
        raw_probs = np.exp(-sample_decay * (gaps - 1))
        self.sample_probs = raw_probs / raw_probs.sum()  # Normalize to sum=1

        # Compute loss weights (급격한 감쇠)
        # w(k) = exp(-loss_decay * (k-1))
        self.loss_weights = np.exp(-loss_decay * (gaps - 1))

        # For demo: create synthetic data
        self._demo_mode = len(video_paths) == 0

        # Log configuration
        print(f"MultiGapVideoDataset initialized:")
        print(f"  Max gap: {max_gap}")
        print(f"  Sample probs (gap 1-{max_gap}): {self.sample_probs.round(3)}")
        print(f"  Loss weights (gap 1-{max_gap}): {self.loss_weights.round(3)}")

    def get_loss_weight(self, gap: int) -> float:
        """Get loss weight for a given gap."""
        if gap < 1 or gap > self.max_gap:
            return 1.0
        return self.loss_weights[gap - 1]

    def __len__(self):
        if self._demo_mode:
            return 1000  # Synthetic dataset size
        return len(self.video_paths) * 100  # ~100 pairs per video

    def __getitem__(self, idx):
        # Sample gap according to probability distribution
        gap = np.random.choice(
            np.arange(1, self.max_gap + 1),
            p=self.sample_probs
        )

        if self._demo_mode:
            # Generate synthetic video frames
            # 간격에 비례하는 변화량으로 시뮬레이션
            img_t = torch.rand(3, self.img_size, self.img_size)
            # gap에 비례한 변화 (gap이 클수록 변화가 큼)
            noise_scale = 0.05 * gap
            noise = torch.randn_like(img_t) * noise_scale
            img_tk = torch.clamp(img_t + noise, 0, 1)
            return img_t, img_tk, gap

        # Real implementation would load actual video frames
        # video = load_video(self.video_paths[idx // 100])
        # t = random.randint(0, len(video) - gap - 1)
        # img_t = video[t]
        # img_tk = video[t + gap]
        raise NotImplementedError("Real video loading not yet implemented")


class EgoDexDataset(torch.utils.data.Dataset):
    """
    EgoDex Dataset for video prediction training.

    데이터 구조:
    - egodex_root/
        - test/ or part1/, part2/, ...
            - task_name/
                - 0.mp4, 0.hdf5
                - 1.mp4, 1.hdf5
                - ...

    우리는 영상(mp4)만 사용하고, 손 포즈 데이터(hdf5)는 사용하지 않음.
    (Action-Agnostic 학습을 위해)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "test",
        max_gap: int = 10,
        img_size: int = 224,
        sample_decay: float = 0.3,
        loss_decay: float = 0.7,
        max_videos: int = None,  # 디버깅용: 최대 비디오 수 제한
        cache_frames: bool = False,  # 메모리에 프레임 캐싱 (작은 데이터셋용)
    ):
        """
        Args:
            data_root: EgoDex 데이터 루트 경로
            split: "test" 또는 "train" (part1-5)
            max_gap: 최대 프레임 간격
            img_size: 출력 이미지 크기
            sample_decay: 샘플링 확률 감쇠율
            loss_decay: Loss 가중치 감쇠율
            max_videos: 디버깅용 최대 비디오 수
            cache_frames: 프레임 캐싱 여부
        """
        import cv2
        from pathlib import Path

        self.data_root = Path(data_root)
        self.split = split
        self.max_gap = max_gap
        self.img_size = img_size
        self.cache_frames = cache_frames

        # Multi-gap 설정
        gaps = np.arange(1, max_gap + 1)
        raw_probs = np.exp(-sample_decay * (gaps - 1))
        self.sample_probs = raw_probs / raw_probs.sum()
        self.loss_weights = np.exp(-loss_decay * (gaps - 1))

        # 비디오 파일 탐색
        if split == "test":
            split_dir = self.data_root / "test"
        else:
            # train split: part1, part2, ... 또는 단일 폴더
            split_dir = self.data_root / split

        self.video_paths = sorted(split_dir.glob("**/*.mp4"))
        if max_videos:
            self.video_paths = self.video_paths[:max_videos]

        # 각 비디오의 프레임 수 캐싱 (첫 로드 시)
        self.video_info = {}  # {path: num_frames}
        self._frame_cache = {}  # {(path, frame_idx): tensor}

        print(f"EgoDexDataset initialized:")
        print(f"  Data root: {data_root}")
        print(f"  Split: {split}")
        print(f"  Videos found: {len(self.video_paths)}")
        print(f"  Max gap: {max_gap}")
        print(f"  Sample probs: {self.sample_probs.round(3)}")
        print(f"  Loss weights: {self.loss_weights.round(3)}")

    def _get_video_info(self, video_path):
        """비디오의 프레임 수를 반환 (캐싱됨)."""
        import cv2

        if video_path not in self.video_info:
            cap = cv2.VideoCapture(str(video_path))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.video_info[video_path] = num_frames
        return self.video_info[video_path]

    def _load_frame(self, video_path, frame_idx):
        """비디오에서 특정 프레임 로드."""
        import cv2

        cache_key = (video_path, frame_idx)
        if self.cache_frames and cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")

        # BGR -> RGB, resize, normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = torch.from_numpy(frame).float() / 255.0  # [H, W, C]
        frame = frame.permute(2, 0, 1)  # [C, H, W]

        if self.cache_frames:
            self._frame_cache[cache_key] = frame

        return frame

    def get_loss_weight(self, gap: int) -> float:
        """Get loss weight for a given gap."""
        if gap < 1 or gap > self.max_gap:
            return 1.0
        return self.loss_weights[gap - 1]

    def __len__(self):
        # 각 비디오당 여러 샘플 (대략 프레임수 / 30)
        return len(self.video_paths) * 100

    def __getitem__(self, idx):
        # 비디오 선택 (순환)
        video_idx = idx % len(self.video_paths)
        video_path = self.video_paths[video_idx]

        # 프레임 수 확인
        num_frames = self._get_video_info(video_path)

        # Gap 샘플링
        gap = np.random.choice(
            np.arange(1, self.max_gap + 1),
            p=self.sample_probs
        )

        # 시작 프레임 샘플링 (gap 고려)
        max_start = max(0, num_frames - gap - 1)
        if max_start <= 0:
            # 비디오가 너무 짧으면 gap 줄이기
            gap = 1
            max_start = max(0, num_frames - 2)

        frame_t = np.random.randint(0, max_start + 1)
        frame_tk = frame_t + gap

        # 프레임 로드
        img_t = self._load_frame(video_path, frame_t)
        img_tk = self._load_frame(video_path, frame_tk)

        return img_t, img_tk, gap


class BridgeDataset(torch.utils.data.Dataset):
    """
    Bridge V2 Dataset for video prediction training.

    데이터 구조:
    - bridge_root/raw/bridge_data_v2/
        - datacol1_toykitchen1/
            - many_skills/07/2023-.../
                - raw/traj_group0/traj{N}/images0/
                    - im_0.jpg, im_1.jpg, ...

    이미지 시퀀스 기반 (mp4가 아님)
    """

    def __init__(
        self,
        data_root: str,
        max_gap: int = 10,
        img_size: int = 224,
        sample_decay: float = 0.3,
        loss_decay: float = 0.7,
        max_trajectories: int = None,
    ):
        """
        Args:
            data_root: Bridge V2 데이터 루트 경로
            max_gap: 최대 프레임 간격 (권장: 10, multi-scale temporal learning)
            img_size: 출력 이미지 크기
            sample_decay: 샘플링 확률 감쇠율
            loss_decay: Loss 가중치 감쇠율
            max_trajectories: 디버깅용 최대 trajectory 수
        """
        from pathlib import Path

        self.data_root = Path(data_root)
        self.max_gap = max_gap
        self.img_size = img_size

        # Multi-gap 설정
        gaps = np.arange(1, max_gap + 1)
        raw_probs = np.exp(-sample_decay * (gaps - 1))
        self.sample_probs = raw_probs / raw_probs.sum()
        self.loss_weights = np.exp(-loss_decay * (gaps - 1))

        # Trajectory 폴더 탐색 (images0 폴더가 있는 곳)
        self.trajectories = []
        search_root = self.data_root / "raw" / "bridge_data_v2"
        if not search_root.exists():
            search_root = self.data_root  # fallback

        for images_dir in search_root.glob("**/images0"):
            # 이미지 파일들 확인
            images = sorted(images_dir.glob("im_*.jpg"))
            if len(images) >= 2:
                self.trajectories.append({
                    'dir': images_dir,
                    'num_frames': len(images),
                })

        if max_trajectories:
            self.trajectories = self.trajectories[:max_trajectories]

        print(f"BridgeDataset initialized:")
        print(f"  Data root: {data_root}")
        print(f"  Trajectories found: {len(self.trajectories)}")
        print(f"  Max gap: {max_gap}")
        print(f"  Sample probs: {self.sample_probs.round(3)}")
        print(f"  Loss weights: {self.loss_weights.round(3)}")

    def _load_frame(self, images_dir, frame_idx):
        """이미지 파일 로드."""
        import cv2
        from pathlib import Path

        img_path = images_dir / f"im_{frame_idx}.jpg"
        if not img_path.exists():
            raise ValueError(f"Image not found: {img_path}")

        frame = cv2.imread(str(img_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # [C, H, W]

        return frame

    def get_loss_weight(self, gap: int) -> float:
        """Get loss weight for a given gap."""
        if gap < 1 or gap > self.max_gap:
            return 1.0
        return self.loss_weights[gap - 1]

    def __len__(self):
        return len(self.trajectories) * 50  # 각 trajectory당 50 샘플

    def __getitem__(self, idx):
        # Trajectory 선택
        traj_idx = idx % len(self.trajectories)
        traj = self.trajectories[traj_idx]

        # Gap 샘플링
        gap = np.random.choice(
            np.arange(1, self.max_gap + 1),
            p=self.sample_probs
        )

        # 시작 프레임 샘플링
        num_frames = traj['num_frames']
        max_start = max(0, num_frames - gap - 1)
        if max_start <= 0:
            gap = 1
            max_start = max(0, num_frames - 2)

        frame_t = np.random.randint(0, max_start + 1)
        frame_tk = frame_t + gap

        # 프레임 로드
        img_t = self._load_frame(traj['dir'], frame_t)
        img_tk = self._load_frame(traj['dir'], frame_tk)

        return img_t, img_tk, gap


# Backward compatibility alias
VideoDataset = MultiGapVideoDataset


@torch.no_grad()
def evaluate(model, eval_dataset, device, batch_size=8, num_samples=500):
    """
    Evaluate model on validation/test dataset.

    Args:
        model: TwoStreamVideoPredictor (or DataParallel wrapped)
        eval_dataset: EgoDexDataset or similar
        device: Device
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate (for speed)

    Returns:
        dict: {'loss': float, 'weighted_loss': float}
    """
    model.eval()

    # Handle DataParallel: get the underlying model for forward pass
    # DataParallel handles device placement automatically
    eval_model = model

    # 샘플 수 제한
    eval_size = min(num_samples, len(eval_dataset))
    indices = np.random.choice(len(eval_dataset), eval_size, replace=False)

    total_loss = 0
    total_weighted_loss = 0
    num_batches = 0
    gap_counts = {}

    # 수동 배치 처리 (빠른 평가용)
    for i in range(0, eval_size, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = [eval_dataset[idx] for idx in batch_indices]

        img_t = torch.stack([d[0] for d in batch_data]).to(device)
        img_tk = torch.stack([d[1] for d in batch_data]).to(device)
        gaps = np.array([d[2] for d in batch_data])

        # Forward
        img_pred, _ = model(img_t, img_tk)
        per_sample_loss = F.mse_loss(img_pred, img_tk, reduction='none')
        per_sample_loss = per_sample_loss.mean(dim=(1, 2, 3))

        # Gap-dependent weights
        if hasattr(eval_dataset, 'get_loss_weight'):
            weights = torch.tensor(
                [eval_dataset.get_loss_weight(int(g)) for g in gaps],
                device=device, dtype=per_sample_loss.dtype
            )
        else:
            weights = torch.ones_like(per_sample_loss)

        weighted_loss = (per_sample_loss * weights).mean()
        unweighted_loss = per_sample_loss.mean()

        total_loss += unweighted_loss.item()
        total_weighted_loss += weighted_loss.item()
        num_batches += 1

        for g in gaps:
            gap_counts[int(g)] = gap_counts.get(int(g), 0) + 1

    avg_loss = total_loss / num_batches
    avg_weighted = total_weighted_loss / num_batches

    model.train()

    return {
        'loss': avg_loss,
        'weighted_loss': avg_weighted,
        'gap_distribution': gap_counts,
    }


def train_epoch(model, dataloader, optimizer, device, epoch, dataset=None):
    """
    Train for one epoch with multi-gap weighted loss.

    Args:
        model: TwoStreamVideoPredictor
        dataloader: DataLoader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number
        dataset: MultiGapVideoDataset (for loss weight lookup)
    """
    model.train()
    total_loss = 0
    total_weighted_loss = 0
    num_batches = 0
    gap_counts = {}

    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch (img_t, img_tk, gap) or (img_t, img_tk)
        if len(batch) == 3:
            img_t, img_tk, gaps = batch
            gaps = gaps.numpy()  # [B]
        else:
            img_t, img_tk = batch
            gaps = np.ones(img_t.shape[0])  # Default gap=1

        img_t = img_t.to(device)
        img_tk = img_tk.to(device)

        optimizer.zero_grad()

        # Compute per-sample loss
        img_pred, change_emb = model(img_t, img_tk)
        per_sample_loss = F.mse_loss(img_pred, img_tk, reduction='none')
        per_sample_loss = per_sample_loss.mean(dim=(1, 2, 3))  # [B]

        # Apply gap-dependent weights
        if dataset is not None and hasattr(dataset, 'get_loss_weight'):
            weights = torch.tensor(
                [dataset.get_loss_weight(int(g)) for g in gaps],
                device=device, dtype=per_sample_loss.dtype
            )
        else:
            weights = torch.ones_like(per_sample_loss)

        # Weighted loss
        weighted_loss = (per_sample_loss * weights).mean()
        unweighted_loss = per_sample_loss.mean()

        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += unweighted_loss.item()
        total_weighted_loss += weighted_loss.item()
        num_batches += 1

        # Track gap distribution
        for g in gaps:
            gap_counts[int(g)] = gap_counts.get(int(g), 0) + 1

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {unweighted_loss.item():.4f}, "
                  f"Weighted: {weighted_loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_weighted = total_weighted_loss / num_batches

    # Print gap distribution
    total_samples = sum(gap_counts.values())
    gap_dist = {k: f"{v/total_samples*100:.1f}%" for k, v in sorted(gap_counts.items())}
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Weighted = {avg_weighted:.4f}")
    print(f"  Gap distribution: {gap_dist}")

    return avg_loss


def train(
    model,
    train_dataset,
    num_epochs=10,
    batch_size=8,
    lr=1e-4,
    device="cuda",
    checkpoint_dir=None,
    save_interval=None,
    eval_dataset=None,
    eval_interval=1,
    resume_from=None,
    multi_gpu=True,
):
    """
    Main training loop with periodic evaluation and checkpointing.

    Args:
        model: TwoStreamVideoPredictor
        train_dataset: BridgeDataset or EgoDexDataset (for training)
        num_epochs: Number of training epochs
        batch_size: Batch size (per GPU if multi_gpu=True)
        lr: Learning rate
        device: Device to train on
        checkpoint_dir: Base directory for checkpoints (auto-creates timestamped subfolder)
        save_interval: Save checkpoint every N epochs (None = only save best)
        eval_dataset: EgoDexDataset (for evaluation, optional)
        eval_interval: Evaluate every N epochs
        resume_from: Path to checkpoint to resume training from
        multi_gpu: Use DataParallel if multiple GPUs available
    """
    from datetime import datetime
    from pathlib import Path
    import json

    # Multi-GPU setup
    use_multi_gpu = multi_gpu and torch.cuda.device_count() > 1
    if use_multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        # Adjust batch size for multi-GPU (batch_size is per-GPU)
        effective_batch_size = batch_size * torch.cuda.device_count()
        print(f"  Per-GPU batch size: {batch_size}")
        print(f"  Effective batch size: {effective_batch_size}")
    else:
        model = model.to(device)
        effective_batch_size = batch_size

    # Setup checkpoint directory with timestamp
    run_dir = None
    if checkpoint_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(checkpoint_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint directory: {run_dir}")

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=8 if use_multi_gpu else 4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_eval_loss = float('inf')
    history = {'train_loss': [], 'eval_loss': []}

    if resume_from:
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        # Handle DataParallel: load into model.module if wrapped
        model_to_load = model.module if use_multi_gpu else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'history' in checkpoint:
            history = checkpoint['history']
        if 'best_eval_loss' in checkpoint:
            best_eval_loss = checkpoint['best_eval_loss']
        print(f"  Resumed from epoch {checkpoint['epoch']}")

    print(f"\nTraining for {num_epochs} epochs (starting from epoch {start_epoch})")
    print(f"  Train dataset: {len(train_dataset)} samples")
    if eval_dataset:
        print(f"  Eval dataset: {len(eval_dataset)} samples (every {eval_interval} epochs)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    if save_interval:
        print(f"  Save interval: every {save_interval} epochs")
    print()

    # Save config
    if run_dir:
        config = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'eval_interval': eval_interval,
            'save_interval': save_interval,
            'train_dataset_size': len(train_dataset),
            'eval_dataset_size': len(eval_dataset) if eval_dataset else 0,
            'start_time': datetime.now().isoformat(),
        }
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    for epoch in range(start_epoch, num_epochs + 1):
        # Train
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch, dataset=train_dataset)
        scheduler.step()
        history['train_loss'].append(avg_loss)

        # Evaluate
        eval_loss = None
        if eval_dataset and epoch % eval_interval == 0:
            eval_result = evaluate(model, eval_dataset, device, batch_size=batch_size)
            eval_loss = eval_result['loss']
            history['eval_loss'].append(eval_loss)
            print(f"  [Eval] Loss: {eval_loss:.4f}, Weighted: {eval_result['weighted_loss']:.4f}")

        # Save checkpoint
        if run_dir:
            # Handle DataParallel: save model.module.state_dict() if wrapped
            model_to_save = model.module if use_multi_gpu else model
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'eval_loss': eval_loss,
                'best_eval_loss': best_eval_loss,
                'history': history,
            }

            # Save periodic checkpoint
            if save_interval and epoch % save_interval == 0:
                ckpt_path = run_dir / f'checkpoint_epoch{epoch:04d}.pt'
                torch.save(checkpoint_data, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path.name}")

            # Save best model
            current_loss = eval_loss if eval_loss is not None else avg_loss
            if current_loss < best_eval_loss:
                best_eval_loss = current_loss
                checkpoint_data['best_eval_loss'] = best_eval_loss
                best_path = run_dir / 'best_model.pt'
                torch.save(checkpoint_data, best_path)
                print(f"  Saved best model (loss: {current_loss:.4f})")

            # Always save latest (for resume)
            latest_path = run_dir / 'latest.pt'
            torch.save(checkpoint_data, latest_path)

    # Save final history
    if run_dir:
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining completed. Results saved to {run_dir}")

    return model, history


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="full",
                        choices=["preprocess", "encoder", "decoder", "full", "interleaved", "train"])
    parser.add_argument("--encoder-type", type=str, default="interleaved",
                        choices=["separate", "interleaved"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    if args.test == "preprocess":
        # =================================================================
        # Test Preprocessing only
        # =================================================================
        print("=" * 60)
        print("Testing Two-Stream Preprocessing")
        print("=" * 60)

        preprocess = TwoStreamPreprocessing().to(device)
        img_prev = torch.rand(2, 3, 224, 224, device=device)
        img_curr = torch.rand(2, 3, 224, 224, device=device)

        m_ch, p_ch = preprocess(img_prev, img_curr)

        print(f"Input:  img_prev {img_prev.shape}, img_curr {img_curr.shape}")
        print(f"Output: M {m_ch.shape}, P {p_ch.shape}")
        print(f"M range: [{m_ch.min():.2f}, {m_ch.max():.2f}]")
        print(f"P range: [{p_ch.min():.2f}, {p_ch.max():.2f}]")

        # Static test
        m_static, _ = preprocess(img_prev, img_prev)
        print(f"Static M (should be ~0): {m_static.abs().mean():.6f}")
        print("\nPreprocessing: OK")

    elif args.test == "encoder":
        # =================================================================
        # Test Encoder (Preprocessing + ViT + Fusion)
        # =================================================================
        print("=" * 60)
        print(f"Testing TwoStreamEncoder (encoder_type={args.encoder_type})")
        print("=" * 60)

        encoder = TwoStreamEncoder(
            model_name="vit_base_patch16_224",
            embed_dim=768,
            pretrained=(args.encoder_type == "separate"),
            encoder_type=args.encoder_type,
            num_stages=3,
            depth=12,
        ).to(device)

        img_prev = torch.rand(2, 3, 224, 224, device=device)
        img_curr = torch.rand(2, 3, 224, 224, device=device)

        cls_out, patches_out, p_cls_intermediates = encoder(img_prev, img_curr)

        print(f"Input:  img_prev {img_prev.shape}, img_curr {img_curr.shape}")
        print(f"Output: cls {cls_out.shape}, patches {patches_out.shape}")
        if p_cls_intermediates is not None:
            print(f"Intermediate CLS: {list(p_cls_intermediates.keys())}")
            for k, v in p_cls_intermediates.items():
                print(f"  {k}: {v.shape}")

        # Gradient check
        loss = cls_out.sum() + patches_out.sum()
        loss.backward()
        print(f"Gradient: OK")

        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"Parameters: {total_params/1e6:.1f}M")
        print("\nTwoStreamEncoder: OK")

    elif args.test == "decoder":
        # =================================================================
        # Test Decoder only (with new interface)
        # =================================================================
        print("=" * 60)
        print("Testing VideoDecoder (with skip + multi-scale CLS)")
        print("=" * 60)

        decoder = VideoDecoder(embed_dim=768).to(device)

        # Simulate encoder outputs
        patches = torch.rand(2, 196, 768, device=device)  # 14x14 patches
        img_t = torch.rand(2, 3, 224, 224, device=device)
        p_cls_intermediates = {
            'stage1': torch.rand(2, 768, device=device),
            'stage2': torch.rand(2, 768, device=device),
            'final': torch.rand(2, 768, device=device),
        }

        img_pred = decoder(patches, img_t, p_cls_intermediates)

        print(f"Input:  patches {patches.shape}, img_t {img_t.shape}")
        print(f"        p_cls_intermediates: {list(p_cls_intermediates.keys())}")
        print(f"Output: img_pred {img_pred.shape}")
        print(f"Output range: [{img_pred.min():.3f}, {img_pred.max():.3f}]")

        loss = F.mse_loss(img_pred, img_t)
        loss.backward()
        print(f"Loss: {loss.item():.4f}")

        total_params = sum(p.numel() for p in decoder.parameters())
        print(f"Parameters: {total_params/1e6:.1f}M")
        print("\nVideoDecoder: OK")

    elif args.test == "interleaved":
        # =================================================================
        # Test Interleaved Two-Stream ViT (CLS Exchange)
        # =================================================================
        print("=" * 60)
        print("Testing InterleavedTwoStreamViT (CLS Exchange)")
        print("=" * 60)

        # Create interleaved encoder
        encoder = InterleavedTwoStreamViT(
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_stages=3,  # 3번의 CLS 교환
        ).to(device)

        # Preprocessing
        preprocess = TwoStreamPreprocessing().to(device)
        img_prev = torch.rand(2, 3, 224, 224, device=device)
        img_curr = torch.rand(2, 3, 224, 224, device=device)

        m_ch, p_ch = preprocess(img_prev, img_curr)
        print(f"Preprocessed: M {m_ch.shape}, P {p_ch.shape}")

        # Encode
        m_tokens, p_tokens, p_cls_intermediates = encoder(m_ch, p_ch)
        print(f"Encoded: m_tokens {m_tokens.shape}, p_tokens {p_tokens.shape}")
        print(f"Intermediate CLS: {list(p_cls_intermediates.keys())}")
        for k, v in p_cls_intermediates.items():
            print(f"  {k}: {v.shape}")

        # Structure info
        print(f"\nArchitecture:")
        print(f"  Total depth: 12 blocks")
        print(f"  Num stages: 3")
        print(f"  Blocks per stage: 4")
        print(f"  CLS exchanges: 2 (between stages)")

        # Gradient check
        loss = m_tokens.sum() + p_tokens.sum()
        loss.backward()
        print(f"\nGradient: OK")

        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"Parameters: {total_params/1e6:.1f}M")
        print("\nInterleavedTwoStreamViT: OK")

    elif args.test == "train":
        # =================================================================
        # Test Training Pipeline (with smaller model for memory)
        # =================================================================
        print("=" * 60)
        print("Testing Training Pipeline")
        print("=" * 60)

        # Use smaller ViT for training test
        model = TwoStreamVideoPredictor(
            model_name="vit_tiny_patch16_224",  # 5.7M params vs 86M
            embed_dim=192,  # Smaller embedding
            pretrained=True,
        )

        # Create synthetic dataset
        dataset = VideoDataset(video_paths=[], img_size=224)
        print(f"Dataset size: {len(dataset)}")

        # Train
        model = train(
            model=model,
            train_dataset=dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-4,
            device=device,
        )

        print("\nTraining: OK")

    else:  # full
        # =================================================================
        # Test Complete TwoStreamVideoPredictor
        # =================================================================
        print("=" * 60)
        print("Testing TwoStreamVideoPredictor (Full Model)")
        print("=" * 60)

        model = TwoStreamVideoPredictor(
            model_name="vit_base_patch16_224",
            embed_dim=768,
            pretrained=True,
        ).to(device)

        img_t = torch.rand(2, 3, 224, 224, device=device)
        img_tk = torch.rand(2, 3, 224, 224, device=device)

        # Forward
        img_pred, change_emb = model(img_t, img_tk)

        print(f"Input:  img_t {img_t.shape}, img_tk {img_tk.shape}")
        print(f"Output: img_pred {img_pred.shape}, change_emb {change_emb.shape}")

        # Loss
        loss, _ = model.compute_loss(img_t, img_tk)
        print(f"MSE Loss: {loss.item():.4f}")

        # Backward
        loss.backward()
        print(f"Gradient: OK")

        # Parameters
        total = sum(p.numel() for p in model.parameters())
        enc = sum(p.numel() for p in model.encoder.parameters())
        dec = sum(p.numel() for p in model.decoder.parameters())
        print(f"\nParameters: Encoder={enc/1e6:.1f}M, Decoder={dec/1e6:.1f}M, Total={total/1e6:.1f}M")

        print("\nTwoStreamVideoPredictor: OK")
        print("\nArchitecture:")
        print("  img_t, img_tk [B,3,224,224]")
        print("       ↓ Two-Stream Preprocessing")
        print("  M [B,4,224,224], P [B,5,224,224]")
        print("       ↓ Two-Stream ViT Encoder")
        print("  m_tokens, p_tokens [B,197,768]")
        print("       ↓ Pixel-wise Fusion")
        print("  cls [B,768], patches [B,196,768]")
        print("       ↓ U-Net Decoder (+ img_t)")
        print("  img_pred [B,3,224,224]")
