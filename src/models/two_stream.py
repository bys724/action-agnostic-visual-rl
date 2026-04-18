"""
Two-Stream Model: Action-agnostic visual representation with M/P channels.

Architecture:
- M Channel (Magnocellular): Temporal change detection (pixel diff + magnitude)
- P Channel (Parvocellular): Spatial structure + color (RGB + Sobel)
- Interleaved ViT with CLS token exchange
- Per-stream future prediction: each stream uses own patches + fused CLS to predict future

M channel (3ch): ΔL + Sobel(ΔL) — 밝기 변화 + 변화의 경계 형상
P channel (5ch): Sobel + RGB — 현재 형상 + 외형

Key design: M and P streams are kept separate in the decoder.
Each decoder receives only its own stream's patch tokens + fused CLS as global context.
Cross-stream info flows ONLY through CLS exchange in the encoder.
This prevents trivial patch-level shortcuts (M+P spatial alignment)
while giving decoders enough spatial information for meaningful reconstruction.

Inspired by biological M/P visual pathways.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import TwoStreamPreprocessing


# ============================================================================
# 2D Rotary Position Embedding (RoPE)
# ============================================================================

def build_2d_rope_freqs(num_patches_per_side: int, dim: int, theta: float = 10000.0):
    """2D RoPE 주파수 테이블 생성.

    패치의 (row, col) 좌표를 dim의 절반씩 나눠 인코딩.
    Returns: [N, dim] complex frequencies (N = num_patches_per_side^2)
    """
    half_dim = dim // 2
    freqs_row = 1.0 / (theta ** (torch.arange(0, half_dim // 2, dtype=torch.float32) / (half_dim // 2)))
    freqs_col = 1.0 / (theta ** (torch.arange(0, half_dim // 2, dtype=torch.float32) / (half_dim // 2)))

    rows = torch.arange(num_patches_per_side, dtype=torch.float32)
    cols = torch.arange(num_patches_per_side, dtype=torch.float32)

    # [H, W, half_dim//2] for row and col separately
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
    grid_r = grid_r.reshape(-1)  # [N]
    grid_c = grid_c.reshape(-1)  # [N]

    # outer product: [N, half_dim//2]
    angles_r = torch.outer(grid_r, freqs_row)
    angles_c = torch.outer(grid_c, freqs_col)

    # concat row and col angles: [N, half_dim]
    angles = torch.cat([angles_r, angles_c], dim=-1)

    # complex form: cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # [N, half_dim]
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, has_cls: bool = True) -> torch.Tensor:
    """RoPE를 attention의 Q, K에 적용.

    Args:
        x: [B, N(+1), H, D_head] — CLS가 있으면 idx 0이 CLS
        freqs_cis: [N, D_head//2] — 패치 위치별 complex frequencies
        has_cls: True면 idx 0(CLS)은 RoPE 적용 안 함

    Returns: [B, N(+1), H, D_head]
    """
    if has_cls:
        cls_tok = x[:, :1]
        patches = x[:, 1:]
    else:
        patches = x

    B, N, H, D = patches.shape

    # [B, N, H, D] → [B, N, H, D//2, 2] → complex
    patches_c = patches.float().reshape(B, N, H, D // 2, 2)
    patches_c = torch.view_as_complex(patches_c)  # [B, N, H, D//2]

    # freqs_cis: [N, D//2] → broadcast: [1, N, 1, D//2]
    freqs = freqs_cis[:N].unsqueeze(0).unsqueeze(2).to(patches_c.device)
    patches_c = patches_c * freqs

    # complex → real
    patches = torch.view_as_real(patches_c).reshape(B, N, H, D).type_as(x)

    if has_cls:
        return torch.cat([cls_tok, patches], dim=1)
    return patches


# ============================================================================
# Custom Transformer Block with RoPE
# ============================================================================

class TransformerBlock(nn.Module):
    """RoPE 지원 Transformer block (Pre-norm, ViT style)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            freqs_cis: [N_patches, D_head//2] — None이면 RoPE 없이 동작
        """
        # Self-attention with RoPE
        h = self.norm1(x)
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [B, N, H, D_head]

        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis, has_cls=True)
            k = apply_rope(k, freqs_cis, has_cls=True)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [B, H, N, D_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)  # [B, H, N, D_head]
        attn = attn.transpose(1, 2).reshape(B, N, D)  # [B, N, D]

        x = x + self.proj(attn)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


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
        use_ape: bool = False,
        num_p_cls: int = 1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_stages = num_stages
        self.blocks_per_stage = depth // num_stages
        self.use_ape = use_ape
        self.num_p_cls = num_p_cls

        # Patch embeddings for M (3ch: ΔL + Sobel(ΔL)) and P (5ch: Sobel + RGB)
        self.patch_embed_m = nn.Conv2d(
            in_channels=3,
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

        # CLS tokens — M은 1개, P는 num_p_cls개 (v7-big에서 bg/motion 분리용)
        self.cls_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_p = nn.Parameter(torch.zeros(1, num_p_cls, embed_dim))

        num_patches = (image_size // patch_size) ** 2
        num_patches_per_side = image_size // patch_size

        if use_ape:
            # APE path: learnable positional embedding (CLS + patches)
            # RoPE buffer는 만들지 않음. block에는 freqs_cis=None 전달
            self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_embed_p = nn.Parameter(
                torch.zeros(1, num_patches + num_p_cls, embed_dim)
            )
            self.freqs_cis = None
        else:
            # 2D RoPE frequencies (not a parameter, just a buffer)
            head_dim = embed_dim // num_heads
            freqs_cis = build_2d_rope_freqs(num_patches_per_side, head_dim)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Transformer blocks per stage for each stream
        self.blocks_m = nn.ModuleList()
        self.blocks_p = nn.ModuleList()
        self.cls_exchange = nn.ModuleList()

        for _ in range(num_stages):
            stage_blocks_m = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio)
                for _ in range(self.blocks_per_stage)
            ])
            self.blocks_m.append(stage_blocks_m)

            stage_blocks_p = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio)
                for _ in range(self.blocks_per_stage)
            ])
            self.blocks_p.append(stage_blocks_p)

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
        if self.use_ape:
            nn.init.trunc_normal_(self.pos_embed_m, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_p, std=0.02)

    def forward(
        self, m_channel: torch.Tensor, p_channel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode M and P channels with interleaved CLS exchange.

        Args:
            m_channel: [B, 3, H, W] - temporal change channel
            p_channel: [B, 5, H, W] - spatial structure channel

        Returns:
            m_tokens: [B, N+1, D] - M stream tokens (CLS + patches)
            p_tokens: [B, N+1, D] - P stream tokens (CLS + patches)
        """
        batch_size = m_channel.shape[0]
        K = self.num_p_cls  # P-side CLS token 개수

        # 1. Patch embedding
        m_patches = self.patch_embed_m(m_channel)  # [B, D, H', W']
        p_patches = self.patch_embed_p(p_channel)

        m_patches = m_patches.flatten(2).transpose(1, 2)  # [B, N, D]
        p_patches = p_patches.flatten(2).transpose(1, 2)

        # 2. Add CLS tokens  (M: 1개, P: K개)
        m_cls = self.cls_token_m.expand(batch_size, -1, -1)  # [B, 1, D]
        p_cls = self.cls_token_p.expand(batch_size, -1, -1)  # [B, K, D]

        m_tokens = torch.cat([m_cls, m_patches], dim=1)  # [B, N+1, D]
        p_tokens = torch.cat([p_cls, p_patches], dim=1)  # [B, N+K, D]

        # 3. Positional encoding: APE (broadcast add) or RoPE (attention 내부)
        if self.use_ape:
            m_tokens = m_tokens + self.pos_embed_m
            p_tokens = p_tokens + self.pos_embed_p
            freqs = None
        else:
            freqs = self.freqs_cis

        # 4. Interleaved processing with CLS exchange (1+K tokens)
        for stage_idx in range(self.num_stages):
            for block_m in self.blocks_m[stage_idx]:
                m_tokens = block_m(m_tokens, freqs_cis=freqs)

            for block_p in self.blocks_p[stage_idx]:
                p_tokens = block_p(p_tokens, freqs_cis=freqs)

            # Extract CLS tokens  (M: 1, P: K)
            m_cls = m_tokens[:, 0:1]     # [B, 1, D]
            p_cls = p_tokens[:, 0:K]     # [B, K, D]

            # Exchange between all CLS tokens  (1+K tokens self-attention)
            cls_combined = torch.cat([m_cls, p_cls], dim=1)  # [B, 1+K, D]
            cls_exchanged = self.cls_exchange[stage_idx](cls_combined)

            # Inject updated CLS back
            m_tokens = torch.cat([cls_exchanged[:, 0:1], m_tokens[:, 1:]], dim=1)
            p_tokens = torch.cat([cls_exchanged[:, 1:1+K], p_tokens[:, K:]], dim=1)

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


class PatchDecoder(nn.Module):
    """
    Decode patch tokens + fused CLS into a predicted image.

    Architecture (MAE-style):
    1. Prepend fused CLS to patch tokens → [B, N+1, D]
    2. Lightweight transformer decoder (4 layers)
    3. Linear projection: patch embeddings → pixel patches
    4. Reshape to image

    Each stream's decoder only sees its own patches + the shared fused CLS.
    Cross-stream information is embedded in the CLS via encoder exchange.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        image_size: int = 224,
        patch_size: int = 16,
        num_context: int = 1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size  # 14
        num_patches = self.num_patches_per_side ** 2  # 196
        self.num_context = num_context  # context CLS tokens 개수

        # Project encoder dim → decoder dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Decoder positional embedding (context CLS들 + patches)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + num_context, decoder_embed_dim)
        )

        # Lightweight transformer decoder
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Predict pixels: each patch → patch_size^2 * 3 channels
        self.pred_head = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(
        self, patch_tokens: torch.Tensor, cls_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, N, D] - single stream's patch embeddings
            cls_embedding: [B, D] (num_context=1) or [B, K, D] (num_context=K)
                           — encoder로부터의 context CLS 토큰(들)

        Returns:
            [B, 3, H, W] - predicted image
        """
        # 0. Normalize cls_embedding shape → [B, K, D]
        if cls_embedding.dim() == 2:
            cls_embedding = cls_embedding.unsqueeze(1)  # [B, 1, D]
        assert cls_embedding.shape[1] == self.num_context, (
            f"Expected {self.num_context} context tokens, got {cls_embedding.shape[1]}"
        )

        # 1. Project to decoder dim
        patches = self.decoder_embed(patch_tokens)   # [B, N, d]
        cls = self.decoder_embed(cls_embedding)      # [B, K, d]

        # 2. Prepend context CLS tokens + add positional embedding
        tokens = torch.cat([cls, patches], dim=1)  # [B, K+N, d]
        tokens = tokens + self.decoder_pos_embed

        # 3. Transformer decoder
        for block in self.decoder_blocks:
            tokens = block(tokens)
        tokens = self.decoder_norm(tokens)

        # 4. Predict pixels from patch tokens (skip context CLS들)
        patch_preds = self.pred_head(tokens[:, self.num_context:])  # [B, N, ps²*3]

        # 5. Reshape to image
        P = self.num_patches_per_side
        ps = self.patch_size
        # [B, N, ps*ps*3] → [B, P, P, ps, ps, 3] → [B, 3, H, W]
        img = patch_preds.reshape(-1, P, P, ps, ps, 3)
        img = img.permute(0, 5, 1, 3, 2, 4)  # [B, 3, P, ps, P, ps]
        img = img.reshape(-1, 3, P * ps, P * ps)  # [B, 3, H, W]

        return torch.sigmoid(img)


class TwoStreamModel(nn.Module):
    """
    Two-Stream model with per-stream future prediction.

    Pipeline:
    1. Preprocessing: RGB pair → M channel (change) + P channel (appearance)
    2. Encoder: Interleaved ViT, CLS exchange between streams
       - mask_ratio > 0: MAE-style, visible patches만 인코더 처리
    3. CLS fusion: (m_cls + p_cls) / 2 → global context [B, D]
    4. Decoder: mask token 삽입 후 전체 패치로 future frame 복원
    5. Loss: future prediction + (optional) masked reconstruction auxiliary

    Cross-stream info flows only through CLS exchange.
    Each decoder sees only its own patches, preventing trivial M+P shortcuts.
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
        mask_ratio: float = 0.0,
        mask_ratio_p: float = None,
        use_ape: bool = False,
        rotation_aug: bool = False,
        independent_rotation_prob: float = 0.1,
        v7_big_mode: bool = False,
        sigma: float = 0.03,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        # P stream은 자기 입력만으로 복원이 쉬워 표현이 shallow해지는 문제가 있음.
        # P의 masking을 M보다 높게 설정하면 CLS exchange 의존도가 높아져 표현 품질 개선.
        self.mask_ratio_p = mask_ratio_p if mask_ratio_p is not None else mask_ratio
        self.rotation_aug = rotation_aug
        self.independent_rotation_prob = independent_rotation_prob
        self.num_patches = (image_size // patch_size) ** 2
        self.v7_big_mode = v7_big_mode
        self.sigma = sigma

        # v7-big: CLS_P가 bg/motion 2개로 분리됨, M decoder도 2개
        num_p_cls = 2 if v7_big_mode else 1

        self.preprocessing = TwoStreamPreprocessing()
        self.encoder = InterleavedTwoStreamViT(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_stages=num_stages,
            mlp_ratio=mlp_ratio,
            image_size=image_size,
            patch_size=patch_size,
            use_ape=use_ape,
            num_p_cls=num_p_cls,
        )
        if v7_big_mode:
            # M decoder 2개: 각각 CLS_P_bg / CLS_P_motion context만 사용 (CLS_M 제외).
            # 설계 철학: cross-stream 정보는 오직 CLS exchange를 통해서만 흐름.
            # CLS_M은 exchange에서 CLS_P_bg/motion을 shaping하는 역할로 소진되고,
            # 그 이후 decoder에서는 사용되지 않음 (pure Option 3).
            self.decoder_m_bg = PatchDecoder(
                embed_dim=embed_dim, image_size=image_size,
                patch_size=patch_size, num_context=1,
            )
            self.decoder_m_motion = PatchDecoder(
                embed_dim=embed_dim, image_size=image_size,
                patch_size=patch_size, num_context=1,
            )
            # [v7-big] P decoder 제거 — CLS_P는 M decoder의 context로만 사용.
            #          필요 시 복원: self.decoder_p = PatchDecoder(...)
        else:
            self.decoder_m = PatchDecoder(embed_dim=embed_dim, image_size=image_size, patch_size=patch_size)
            self.decoder_p = PatchDecoder(embed_dim=embed_dim, image_size=image_size, patch_size=patch_size)

        # MAE mask tokens (stream별 독립)
        # v7-big에서는 P stream의 full patch 복원이 필요 없음(decoder_p 없음) → mask_token_p 생성 안 함.
        # 생성하면 DDP가 "gradient 없는 파라미터"로 간주해 에러.
        if mask_ratio > 0:
            self.mask_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.mask_token_m, std=0.02)
            if not v7_big_mode:
                self.mask_token_p = nn.Parameter(torch.zeros(1, 1, embed_dim))
                nn.init.trunc_normal_(self.mask_token_p, std=0.02)

    def _random_mask(self, B: int, device: torch.device, ratio: float = None) -> torch.Tensor:
        """독립 랜덤 마스크 생성. Returns: [B, N] bool, True=masked."""
        N = self.num_patches
        r = ratio if ratio is not None else self.mask_ratio
        num_masked = int(r * N)
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :N - num_masked], False)
        return mask

    def _apply_mask(self, tokens: torch.Tensor, mask: torch.Tensor, num_cls: int = 1):
        """마스킹: CLS(앞 num_cls개) 보존, 나머지 중 visible만 추출 + ids_keep 반환.

        Args:
            tokens: [B, N+num_cls, D] (CLS들 + patches)
            mask: [B, N] bool, True=masked
            num_cls: 보존할 CLS 토큰 개수 (M stream: 1, v7-big P stream: 2)

        Returns:
            visible_tokens: [B, N_vis+num_cls, D]
            ids_keep: [N_vis] visible patch의 원래 position index (RoPE gather용)
        """
        cls = tokens[:, :num_cls]       # [B, num_cls, D]
        patches = tokens[:, num_cls:]   # [B, N, D]
        B, N, D = patches.shape

        # visible patches만 추출
        visible = patches[~mask].reshape(B, -1, D)  # [B, N_vis, D]

        # visible patch의 원래 position index (batch 내 mask 동일하므로 [0] 사용)
        ids_keep = (~mask[0]).nonzero(as_tuple=True)[0]  # [N_vis]

        return torch.cat([cls, visible], dim=1), ids_keep

    def _restore_with_mask_tokens(
        self, visible_tokens: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor,
        num_cls: int = 1,
    ) -> torch.Tensor:
        """Visible embeddings + mask tokens → 원래 순서의 전체 패치 복원.

        Args:
            visible_tokens: [B, N_vis+num_cls, D] (CLS들 + visible)
            mask: [B, N] bool, True=masked
            mask_token: [1, 1, D]
            num_cls: CLS 토큰 개수

        Returns:
            full_patches: [B, N, D] (CLS 제외, 원래 patch 순서)
        """
        visible_patches = visible_tokens[:, num_cls:]  # [B, N_vis, D]
        B, _, D = visible_patches.shape
        N = mask.shape[1]

        # 전체 패치를 mask token으로 초기화
        full = mask_token.expand(B, N, -1).clone()
        # visible 위치에 실제 embedding 삽입
        full[~mask] = visible_patches.reshape(-1, D)
        return full

    def forward(
        self, image_current: torch.Tensor, image_future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            image_current: [B, 3, H, W], range [0, 1]
            image_future: [B, 3, H, W], range [0, 1]

        Returns:
            pred_m: [B, 3, H, W] - M stream's future prediction
            pred_p: [B, 3, H, W] - P stream's future prediction
            cls_embedding: [B, embed_dim] - fused CLS (downstream representation)
        """
        # 1. Preprocessing
        m_channel, p_channel = self.preprocessing(image_current, image_future)

        if self.mask_ratio > 0 and self.training:
            return self._forward_masked(m_channel, p_channel)
        else:
            return self._forward_full(m_channel, p_channel)

    def _forward_full(
        self, m_channel: torch.Tensor, p_channel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """마스킹 없는 forward (eval 또는 mask_ratio=0)."""
        m_tokens, p_tokens = self.encoder(m_channel, p_channel)

        if self.v7_big_mode:
            # v7-big: M patches + (CLS_P_bg 또는 CLS_P_motion) 만. CLS_M은 decoder에 안 들어감.
            p_cls_bg = p_tokens[:, 0:1]       # [B, 1, D]
            p_cls_motion = p_tokens[:, 1:2]   # [B, 1, D]
            m_patches = m_tokens[:, 1:]       # [B, N, D]

            pred_bg = self.decoder_m_bg(m_patches, p_cls_bg)
            pred_motion = self.decoder_m_motion(m_patches, p_cls_motion)

            # Downstream representation: 3개 CLS 평균 (probing에서 별도 mode로 세분화 가능)
            cls_embedding = (m_tokens[:, 0] + p_tokens[:, 0] + p_tokens[:, 1]) / 3
            return pred_bg, pred_motion, cls_embedding

        # Legacy path (v4~v6)
        cls_embedding = (m_tokens[:, 0] + p_tokens[:, 0]) / 2
        m_patches = m_tokens[:, 1:]
        p_patches = p_tokens[:, 1:]
        pred_m = self.decoder_m(m_patches, cls_embedding)
        pred_p = self.decoder_p(p_patches, cls_embedding)
        return pred_m, pred_p, cls_embedding

    def _forward_masked(
        self, m_channel: torch.Tensor, p_channel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MAE-style masked forward (training only)."""
        B = m_channel.shape[0]
        device = m_channel.device
        K = self.encoder.num_p_cls  # P-side CLS 토큰 개수 (v7-big: 2, legacy: 1)

        # M/P 독립 마스크 생성
        mask_m = self._random_mask(B, device, ratio=self.mask_ratio)    # [B, N]
        mask_p = self._random_mask(B, device, ratio=self.mask_ratio_p)  # [B, N]

        # 인코더: patch embed + pos embed까지는 전체, 그 후 마스킹
        enc = self.encoder
        batch_size = B

        # Patch embedding
        m_patches = enc.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        p_patches = enc.patch_embed_p(p_channel).flatten(2).transpose(1, 2)

        # CLS + patches  (P는 K개 CLS)
        m_cls = enc.cls_token_m.expand(batch_size, -1, -1)   # [B, 1, D]
        p_cls = enc.cls_token_p.expand(batch_size, -1, -1)   # [B, K, D]
        m_tokens = torch.cat([m_cls, m_patches], dim=1)
        p_tokens = torch.cat([p_cls, p_patches], dim=1)

        # APE path: masking 전에 pos_embed를 더함 (standard MAE 방식)
        if enc.use_ape:
            m_tokens = m_tokens + enc.pos_embed_m
            p_tokens = p_tokens + enc.pos_embed_p

        # 마스킹 적용 (CLS 보존) + visible position index 반환
        m_tokens, ids_keep_m = self._apply_mask(m_tokens, mask_m, num_cls=1)
        p_tokens, ids_keep_p = self._apply_mask(p_tokens, mask_p, num_cls=K)

        # RoPE path: visible patch 위치의 freqs만 gather. APE이면 freqs=None
        if enc.use_ape:
            freqs_m = freqs_p = None
        else:
            freqs_m = enc.freqs_cis[ids_keep_m]  # [N_vis_m, D_head//2]
            freqs_p = enc.freqs_cis[ids_keep_p]  # [N_vis_p, D_head//2]

        # Interleaved processing with CLS exchange (1+K tokens)
        for stage_idx in range(enc.num_stages):
            for block_m in enc.blocks_m[stage_idx]:
                m_tokens = block_m(m_tokens, freqs_cis=freqs_m)
            for block_p in enc.blocks_p[stage_idx]:
                p_tokens = block_p(p_tokens, freqs_cis=freqs_p)

            m_cls_tok = m_tokens[:, 0:1]      # [B, 1, D]
            p_cls_tok = p_tokens[:, 0:K]      # [B, K, D]
            cls_combined = torch.cat([m_cls_tok, p_cls_tok], dim=1)  # [B, 1+K, D]
            cls_exchanged = enc.cls_exchange[stage_idx](cls_combined)
            m_tokens = torch.cat([cls_exchanged[:, 0:1], m_tokens[:, 1:]], dim=1)
            p_tokens = torch.cat([cls_exchanged[:, 1:1+K], p_tokens[:, K:]], dim=1)

        m_tokens = enc.norm_m(m_tokens)
        p_tokens = enc.norm_p(p_tokens)

        # Mask token 삽입 → 원래 순서의 전체 패치 복원 (M만)
        m_full_patches = self._restore_with_mask_tokens(
            m_tokens, mask_m, self.mask_token_m, num_cls=1)

        if self.v7_big_mode:
            # v7-big: decoder는 M patches만 복원, context는 CLS_P_bg/motion 각 1개.
            # p_full_patches 복원 불필요 → mask_token_p도 생성 안 함.
            p_cls_bg = p_tokens[:, 0:1]             # [B, 1, D]
            p_cls_motion = p_tokens[:, 1:2]         # [B, 1, D]

            pred_bg = self.decoder_m_bg(m_full_patches, p_cls_bg)
            pred_motion = self.decoder_m_motion(m_full_patches, p_cls_motion)

            cls_embedding = (m_tokens[:, 0] + p_tokens[:, 0] + p_tokens[:, 1]) / 3
            return pred_bg, pred_motion, cls_embedding

        # Legacy path (v4~v6): P 복원도 필요
        p_full_patches = self._restore_with_mask_tokens(
            p_tokens, mask_p, self.mask_token_p, num_cls=K)
        cls_embedding = (m_tokens[:, 0] + p_tokens[:, 0]) / 2
        pred_m = self.decoder_m(m_full_patches, cls_embedding)
        pred_p = self.decoder_p(p_full_patches, cls_embedding)
        return pred_m, pred_p, cls_embedding

    def _apply_rotation_aug(
        self, image_current: torch.Tensor, image_future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-time rotation augmentation으로 position prior overfit 방지.

        90%: 동일 회전 (두 프레임 같은 각도, motion 보존 + position prior 깨기)
        10%: 독립 회전 (각 프레임 다른 각도, 극단적 viewpoint 변화 대응)
        """
        if torch.rand(1).item() < self.independent_rotation_prob:
            # 독립 회전: 각 프레임 별도 각도
            k_curr = torch.randint(0, 4, (1,)).item()
            k_fut = torch.randint(0, 4, (1,)).item()
        else:
            # 동일 회전: 두 프레임 같은 각도 (0/90/180/270 균등)
            k_curr = torch.randint(0, 4, (1,)).item()
            k_fut = k_curr

        if k_curr != 0:
            image_current = torch.rot90(image_current, k_curr, dims=(2, 3))
        if k_fut != 0:
            image_future = torch.rot90(image_future, k_fut, dims=(2, 3))

        return image_current, image_future

    def compute_loss(
        self, image_current: torch.Tensor, image_future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Both streams predict future frame independently.

        Returns:
            loss: MSE(M_pred, future) + MSE(P_pred, future)  (legacy)
                  or w_bg·MSE(pred_bg, future) + w_motion·MSE(pred_motion, future)  (v7-big)
            pred: [B, 3, H, W] - visualization prediction (legacy: pred_p, v7-big: pred_motion)
        """
        if self.rotation_aug and self.training:
            image_current, image_future = self._apply_rotation_aug(
                image_current, image_future
            )

        out1, out2, _ = self.forward(image_current, image_future)

        if self.v7_big_mode:
            # out1 = pred_bg, out2 = pred_motion
            # |ΔL| 기반 pixel-wise weighting
            with torch.no_grad():
                lum_prev = (image_current * self.preprocessing.luminance_weights
                            .to(image_current.device).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
                lum_curr = (image_future * self.preprocessing.luminance_weights
                            .to(image_future.device).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
                delta_mag = (lum_curr - lum_prev).abs()  # [B, 1, H, W]
                w_bg = torch.exp(-(delta_mag / self.sigma) ** 2)   # [B, 1, H, W]
                w_motion = 1.0 - w_bg

            sq_bg = (out1 - image_future) ** 2          # [B, 3, H, W]
            sq_motion = (out2 - image_future) ** 2

            # Weighted mean (broadcasting [B,1,H,W] × [B,3,H,W])
            # 분모: w.sum() × 3 (RGB 채널 수)
            loss_bg = (w_bg * sq_bg).sum() / (w_bg.sum() * 3 + 1e-8)
            loss_motion = (w_motion * sq_motion).sum() / (w_motion.sum() * 3 + 1e-8)
            loss = loss_bg + loss_motion
            return loss, out2  # pred_motion for visualization

        # Legacy path (v4~v6)
        loss_m = F.mse_loss(out1, image_future)
        loss_p = F.mse_loss(out2, image_future)
        loss = loss_m + loss_p
        return loss, out2


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

        # Auto-detect use_ape from checkpoint (pos_embed_m 존재 → APE)
        use_ape = False
        if checkpoint_path:
            _ckpt = torch.load(checkpoint_path, map_location="cpu")
            _sd = _ckpt.get("model_state_dict", _ckpt)
            use_ape = any("encoder.pos_embed_m" in k for k in _sd)

        # Build encoder components
        self.preprocessing = TwoStreamPreprocessing()
        self.encoder = InterleavedTwoStreamViT(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_stages=num_stages,
            image_size=image_size,
            patch_size=patch_size,
            use_ape=use_ape,
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
        """Load encoder weights from trained Two-Stream model.

        DDP/DataParallel로 저장된 체크포인트의 'module.' prefix를 자동 제거.
        decoder weights는 제외 (인코더만 사용).
        키 매칭 결과를 출력하여 silent fail 방지.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # 'module.' prefix 자동 제거 (DDP/DP wrapped checkpoint 호환)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                          for k, v in state_dict.items()}

        # Extract encoder weights only (exclude decoders)
        encoder_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("decoder_m.") and not k.startswith("decoder_p.")
        }

        result = self.load_state_dict(encoder_state, strict=False)
        if result.missing_keys:
            print(f"  WARNING: {len(result.missing_keys)} missing encoder keys "
                  f"(first 3: {result.missing_keys[:3]})")
        # unexpected_keys는 보통 decoder/SSIM 등 — 정상이라 silent
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
