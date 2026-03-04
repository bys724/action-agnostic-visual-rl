"""
Two-Stream Interleaved ViT

M채널(움직임)과 P채널(형상)을 분리 처리하되,
CLS 토큰 간 Cross-Attention으로 정보를 주기적으로 교환하는 구조

Architecture:
- M-ViT: 움직임 전문화 (4 blocks × 3 stages)
- P-ViT: 형상 전문화 (4 blocks × 3 stages)
- Cross-Attention: CLS 간 양방향 attention (2 blocks per stage)
- Residual: CLS 교환 시 정보 보존
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from .two_stream_preprocessing import TwoStreamPreprocessing


@dataclass
class TwoStreamViTConfig:
    """Two-Stream ViT 설정"""
    # Image
    img_size: int = 224
    patch_size: int = 14

    # ViT dimensions
    embed_dim: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Stage structure
    blocks_per_stage: int = 4   # 교환 전 블록 수
    num_stages: int = 3         # 교환 횟수
    # total_depth = blocks_per_stage × num_stages = 12

    # Cross-attention
    cross_attn_blocks: int = 2
    cross_attn_heads: int = 4

    # Output
    output_dim: int = 512


class PatchEmbed(nn.Module):
    """이미지를 패치 임베딩으로 변환"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 4,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            tokens: [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class CrossAttention(nn.Module):
    """Cross-Attention (Query가 다른 소스를 attend)"""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D]
            context: [B, N_kv, D]
        Returns:
            output: [B, N_q, D]
        """
        B, N_q, D = query.shape
        N_kv = context.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    """Feed-Forward Network"""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Standard ViT Block"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CLSCrossAttentionBlock(nn.Module):
    """CLS 토큰 간 양방향 Cross-Attention Block"""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        # M → P attention
        self.norm_m = nn.LayerNorm(dim)
        self.cross_m_to_p = CrossAttention(dim, num_heads, dropout)

        # P → M attention
        self.norm_p = nn.LayerNorm(dim)
        self.cross_p_to_m = CrossAttention(dim, num_heads, dropout)

        # MLPs
        self.norm_m2 = nn.LayerNorm(dim)
        self.mlp_m = MLP(dim, mlp_ratio, dropout)
        self.norm_p2 = nn.LayerNorm(dim)
        self.mlp_p = MLP(dim, mlp_ratio, dropout)

    def forward(
        self,
        cls_m: torch.Tensor,
        cls_p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_m: [B, 1, D] M채널 CLS
            cls_p: [B, 1, D] P채널 CLS
        Returns:
            (cls_m_new, cls_p_new): 업데이트된 CLS 토큰들
        """
        # Bidirectional cross-attention with residual
        cls_m_attn = cls_m + self.cross_m_to_p(self.norm_m(cls_m), self.norm_p(cls_p))
        cls_p_attn = cls_p + self.cross_p_to_m(self.norm_p(cls_p), self.norm_m(cls_m))

        # MLP with residual
        cls_m_out = cls_m_attn + self.mlp_m(self.norm_m2(cls_m_attn))
        cls_p_out = cls_p_attn + self.mlp_p(self.norm_p2(cls_p_attn))

        return cls_m_out, cls_p_out


class ChannelViT(nn.Module):
    """단일 채널용 ViT (M 또는 P)"""

    def __init__(
        self,
        in_channels: int,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 512,
        num_heads: int = 8,
        blocks_per_stage: int = 4,
        num_stages: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.blocks_per_stage = blocks_per_stage

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token & position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks (stages로 분리)
        self.stages = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(blocks_per_stage)
            ])
            for _ in range(num_stages)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_stage(
        self,
        x: torch.Tensor,
        stage_idx: int
    ) -> torch.Tensor:
        """특정 stage만 실행"""
        for block in self.stages[stage_idx]:
            x = block(x)
        return x

    def prepare_tokens(self, img: torch.Tensor) -> torch.Tensor:
        """이미지 → 토큰 준비 (CLS + patches + pos)"""
        B = img.shape[0]

        # Patch embedding
        x = self.patch_embed(img)  # [B, N, D]

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, N+1, D]

        # Add position embedding
        x = x + self.pos_embed

        return x

    def get_cls(self, x: torch.Tensor) -> torch.Tensor:
        """토큰에서 CLS 추출"""
        return x[:, :1, :]  # [B, 1, D]

    def replace_cls(self, x: torch.Tensor, new_cls: torch.Tensor) -> torch.Tensor:
        """CLS 토큰 교체"""
        return torch.cat([new_cls, x[:, 1:, :]], dim=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """전체 forward (교환 없이)"""
        x = self.prepare_tokens(img)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self.norm(x)
        return x[:, 0]  # CLS만 반환


class TwoStreamInterleavedViT(nn.Module):
    """
    Two-Stream Interleaved ViT

    M채널과 P채널을 분리 처리하면서 CLS 간 정보 교환

    Flow:
    1. Preprocessing: RGB pair → M(4ch) + P(2ch)
    2. Patch Embed: 각 채널별 패치 임베딩
    3. Interleaved Processing:
       - Stage N: M-ViT blocks + P-ViT blocks
       - Cross-Attention: CLS_M ↔ CLS_P (with residual)
       - Repeat for num_stages
    4. Fusion: 두 CLS를 하나로 통합
    """

    def __init__(self, config: Optional[TwoStreamViTConfig] = None):
        super().__init__()
        self.config = config or TwoStreamViTConfig()
        c = self.config

        # Two-Stream Preprocessing
        self.preprocessing = TwoStreamPreprocessing(trainable_weights=True)

        # M-channel ViT (4 channels: ΔL + RGB)
        self.m_vit = ChannelViT(
            in_channels=4,
            img_size=c.img_size,
            patch_size=c.patch_size,
            embed_dim=c.embed_dim,
            num_heads=c.num_heads,
            blocks_per_stage=c.blocks_per_stage,
            num_stages=c.num_stages,
            mlp_ratio=c.mlp_ratio,
            dropout=c.dropout,
        )

        # P-channel ViT (2 channels: ∂L/∂x, ∂L/∂y)
        self.p_vit = ChannelViT(
            in_channels=2,
            img_size=c.img_size,
            patch_size=c.patch_size,
            embed_dim=c.embed_dim,
            num_heads=c.num_heads,
            blocks_per_stage=c.blocks_per_stage,
            num_stages=c.num_stages,
            mlp_ratio=c.mlp_ratio,
            dropout=c.dropout,
        )

        # Cross-Attention blocks (per stage)
        self.cross_attn_stages = nn.ModuleList([
            nn.ModuleList([
                CLSCrossAttentionBlock(
                    c.embed_dim, c.cross_attn_heads, c.mlp_ratio, c.dropout
                )
                for _ in range(c.cross_attn_blocks)
            ])
            for _ in range(c.num_stages)
        ])

        # Final norms
        self.m_norm = nn.LayerNorm(c.embed_dim)
        self.p_norm = nn.LayerNorm(c.embed_dim)

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(c.embed_dim * 2, c.embed_dim),
            nn.LayerNorm(c.embed_dim),
            nn.GELU(),
            nn.Linear(c.embed_dim, c.output_dim),
        )

    def forward(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            img_prev: [B, 3, H, W] 이전 프레임
            img_curr: [B, 3, H, W] 현재 프레임
            return_intermediates: True면 중간 CLS들도 반환

        Returns:
            embedding: [B, output_dim]
        """
        # 1. Preprocessing
        m_feat, p_feat = self.preprocessing(img_prev, img_curr, return_separate=True)
        # m_feat: [B, 4, H, W], p_feat: [B, 2, H, W]

        # 2. Prepare tokens
        m_tokens = self.m_vit.prepare_tokens(m_feat)  # [B, N+1, D]
        p_tokens = self.p_vit.prepare_tokens(p_feat)  # [B, N+1, D]

        intermediates = [] if return_intermediates else None

        # 3. Interleaved processing
        for stage_idx in range(self.config.num_stages):
            # Run ViT blocks for this stage
            m_tokens = self.m_vit.forward_stage(m_tokens, stage_idx)
            p_tokens = self.p_vit.forward_stage(p_tokens, stage_idx)

            # Extract CLS tokens
            cls_m = self.m_vit.get_cls(m_tokens)  # [B, 1, D]
            cls_p = self.p_vit.get_cls(p_tokens)  # [B, 1, D]

            # Cross-attention between CLS tokens
            for cross_block in self.cross_attn_stages[stage_idx]:
                cls_m, cls_p = cross_block(cls_m, cls_p)

            # Replace CLS with updated ones (residual already applied in cross_block)
            m_tokens = self.m_vit.replace_cls(m_tokens, cls_m)
            p_tokens = self.p_vit.replace_cls(p_tokens, cls_p)

            if return_intermediates:
                intermediates.append((cls_m.squeeze(1), cls_p.squeeze(1)))

        # 4. Final norm
        cls_m_final = self.m_norm(m_tokens[:, 0])  # [B, D]
        cls_p_final = self.p_norm(p_tokens[:, 0])  # [B, D]

        # 5. Fusion
        combined = torch.cat([cls_m_final, cls_p_final], dim=-1)  # [B, 2D]
        output = self.fusion(combined)  # [B, output_dim]

        if return_intermediates:
            return output, intermediates
        return output

    def get_separate_cls(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """분리된 CLS 반환 (분석용)"""
        m_feat, p_feat = self.preprocessing(img_prev, img_curr, return_separate=True)

        m_tokens = self.m_vit.prepare_tokens(m_feat)
        p_tokens = self.p_vit.prepare_tokens(p_feat)

        for stage_idx in range(self.config.num_stages):
            m_tokens = self.m_vit.forward_stage(m_tokens, stage_idx)
            p_tokens = self.p_vit.forward_stage(p_tokens, stage_idx)

            cls_m = self.m_vit.get_cls(m_tokens)
            cls_p = self.p_vit.get_cls(p_tokens)

            for cross_block in self.cross_attn_stages[stage_idx]:
                cls_m, cls_p = cross_block(cls_m, cls_p)

            m_tokens = self.m_vit.replace_cls(m_tokens, cls_m)
            p_tokens = self.p_vit.replace_cls(p_tokens, cls_p)

        cls_m_final = self.m_norm(m_tokens[:, 0])
        cls_p_final = self.p_norm(p_tokens[:, 0])

        return cls_m_final, cls_p_final


def count_parameters(model: nn.Module) -> int:
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 간단한 테스트
    config = TwoStreamViTConfig()
    model = TwoStreamInterleavedViT(config)

    print(f"Config: {config}")
    print(f"Total parameters: {count_parameters(model):,}")

    # Forward test
    img_prev = torch.randn(2, 3, 224, 224)
    img_curr = torch.randn(2, 3, 224, 224)

    output = model(img_prev, img_curr)
    print(f"Input: {img_prev.shape}, {img_curr.shape}")
    print(f"Output: {output.shape}")
