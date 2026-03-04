"""
Behavior Encoder

Task-conditioned Visual Behavior Representation을 학습하는 인코더
2-Frame 입력으로 Speed-invariant behavior embedding 생성

Architecture:
- Two-Stream Preprocessing (M/P channels)
- DINO+SigLIP Partially Shared ViT
- Task-conditioned Cross-Attention

References:
- 논문 - Action-Agnostic Visual Behavior Representation.md
- Two-Stream Image Preprocessing.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .two_stream_preprocessing import TwoStreamPreprocessing, TwoStreamViTAdapter


@dataclass
class BehaviorEncoderConfig:
    """Behavior Encoder 설정"""
    # Image settings
    img_size: int = 224
    patch_size: int = 14

    # ViT architecture
    embed_dim: int = 768
    num_heads: int = 12
    num_shared_layers: int = 6  # Early layers (공유)
    num_branch_layers: int = 6  # Late layers (분리)
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Task conditioning
    text_embed_dim: int = 512  # Language model output dim
    use_task_conditioning: bool = True

    # Training
    use_momentum_encoder: bool = True
    momentum: float = 0.999

    # Output
    output_dim: int = 768
    use_cls_token: bool = True


class MultiHeadAttention(nn.Module):
    """Multi-Head Self/Cross Attention"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
        kv_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.is_cross_attention = is_cross_attention

        kv_dim = kv_dim or embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] query tokens
            context: [B, M, D_kv] key/value tokens (for cross-attention)
            attention_mask: [B, N, M] attention mask

        Returns:
            output: [B, N, D]
        """
        B, N, _ = x.shape

        # Query from x
        q = self.q_proj(x)

        # Key/Value from context (cross-attention) or x (self-attention)
        kv_input = context if self.is_cross_attention and context is not None else x
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Standard Transformer Block"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cross_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(embed_dim)
            self.cross_attn = MultiHeadAttention(
                embed_dim, num_heads, dropout,
                is_cross_attention=True,
                kv_dim=cross_attention_dim or embed_dim
            )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens
            context: [B, M, D_ctx] context for cross-attention

        Returns:
            output: [B, N, D]
        """
        # Self-attention
        x = x + self.attn(self.norm1(x))

        # Cross-attention (if enabled)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class PartiallySharedViT(nn.Module):
    """
    DINO + SigLIP Partially Shared Architecture

    Early layers (0-6): 공유 - 기본적인 visual features
    Late layers (6-12): 분리
      - DINO branch: Spatial features (self-supervised)
      - SigLIP branch: Semantic features (contrastive with text)

    총 ~300M params (vs Prismatic 700M)
    """

    def __init__(self, config: BehaviorEncoderConfig):
        super().__init__()
        self.config = config

        # Two-Stream Preprocessing
        self.preprocessing = TwoStreamPreprocessing(trainable_weights=True)
        self.patch_embed = TwoStreamViTAdapter(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            in_channels=6,  # M(4) + P(2) channels
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Position embeddings
        num_patches = (config.img_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim)
        )

        # Shared early layers
        self.shared_layers = nn.ModuleList([
            TransformerBlock(
                config.embed_dim, config.num_heads,
                config.mlp_ratio, config.dropout
            )
            for _ in range(config.num_shared_layers)
        ])

        # DINO branch (spatial features)
        self.dino_layers = nn.ModuleList([
            TransformerBlock(
                config.embed_dim, config.num_heads,
                config.mlp_ratio, config.dropout
            )
            for _ in range(config.num_branch_layers)
        ])
        self.dino_norm = nn.LayerNorm(config.embed_dim)

        # SigLIP branch (semantic features) with task-conditioning
        self.siglip_layers = nn.ModuleList([
            TransformerBlock(
                config.embed_dim, config.num_heads,
                config.mlp_ratio, config.dropout,
                use_cross_attention=config.use_task_conditioning,
                cross_attention_dim=config.text_embed_dim,
            )
            for _ in range(config.num_branch_layers)
        ])
        self.siglip_norm = nn.LayerNorm(config.embed_dim)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
        )

        # Output projection
        self.output_proj = nn.Linear(config.embed_dim, config.output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_shared(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shared early layers forward pass

        Args:
            img_prev: [B, 3, H, W]
            img_curr: [B, 3, H, W]

        Returns:
            x: [B, N+1, D] tokens after shared layers
        """
        B = img_prev.shape[0]

        # Two-Stream preprocessing
        features = self.preprocessing(img_prev, img_curr)  # [B, 6, H, W]

        # Patch embedding
        x = self.patch_embed(features)  # [B, N, D]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]

        # Add position embeddings
        x = x + self.pos_embed

        # Shared layers
        for layer in self.shared_layers:
            x = layer(x)

        return x

    def forward_dino(self, x: torch.Tensor) -> torch.Tensor:
        """
        DINO branch for spatial features

        Args:
            x: [B, N+1, D] tokens from shared layers

        Returns:
            dino_features: [B, D] CLS token features
        """
        for layer in self.dino_layers:
            x = layer(x)
        x = self.dino_norm(x)
        return x[:, 0]  # CLS token

    def forward_siglip(
        self,
        x: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        SigLIP branch for semantic features (task-conditioned)

        Args:
            x: [B, N+1, D] tokens from shared layers
            task_embedding: [B, M, D_text] task description embedding

        Returns:
            siglip_features: [B, D] CLS token features
        """
        for layer in self.siglip_layers:
            x = layer(x, context=task_embedding)
        x = self.siglip_norm(x)
        return x[:, 0]  # CLS token

    def forward(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
        return_both_branches: bool = False,
    ) -> torch.Tensor:
        """
        Full forward pass

        Args:
            img_prev: [B, 3, H, W]
            img_curr: [B, 3, H, W]
            task_embedding: [B, M, D_text] task description
            return_both_branches: if True, return (dino_feat, siglip_feat)

        Returns:
            behavior_embedding: [B, output_dim]
        """
        # Shared layers
        shared_features = self.forward_shared(img_prev, img_curr)

        # Branch processing
        dino_feat = self.forward_dino(shared_features)
        siglip_feat = self.forward_siglip(shared_features, task_embedding)

        if return_both_branches:
            return dino_feat, siglip_feat

        # Feature fusion
        combined = torch.cat([dino_feat, siglip_feat], dim=-1)
        fused = self.fusion(combined)

        # Output projection
        output = self.output_proj(fused)

        return output


class BehaviorEncoder(nn.Module):
    """
    Task-conditioned Visual Behavior Encoder

    Full model with:
    - Two-Stream Preprocessing
    - Partially Shared ViT (DINO + SigLIP)
    - Task conditioning via cross-attention
    - Optional momentum encoder for self-supervised learning
    """

    def __init__(self, config: Optional[BehaviorEncoderConfig] = None):
        super().__init__()

        self.config = config or BehaviorEncoderConfig()

        # Main encoder
        self.encoder = PartiallySharedViT(self.config)

        # Momentum encoder (for self-supervised learning)
        self.momentum_encoder = None
        if self.config.use_momentum_encoder:
            self._create_momentum_encoder()

        # Projector for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(self.config.output_dim, self.config.output_dim),
            nn.GELU(),
            nn.Linear(self.config.output_dim, self.config.output_dim),
        )

    def _create_momentum_encoder(self):
        """Create momentum encoder (EMA)"""
        import copy
        self.momentum_encoder = copy.deepcopy(self.encoder)
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_momentum_encoder(self):
        """Update momentum encoder via EMA"""
        if self.momentum_encoder is None:
            return

        momentum = self.config.momentum
        for param, momentum_param in zip(
            self.encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            momentum_param.data = (
                momentum * momentum_param.data +
                (1 - momentum) * param.data
            )

    def forward(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            img_prev: [B, 3, H, W] previous frame
            img_curr: [B, 3, H, W] current frame
            task_embedding: [B, M, D_text] task description

        Returns:
            Dict with:
                - "behavior_embedding": [B, D] main embedding
                - "projected": [B, D] projected embedding (for contrastive)
        """
        # Main encoder
        behavior_embedding = self.encoder(img_prev, img_curr, task_embedding)

        # Projection for contrastive learning
        projected = self.projector(behavior_embedding)

        return {
            "behavior_embedding": behavior_embedding,
            "projected": projected,
        }

    def forward_with_momentum(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with momentum encoder (for self-supervised training)

        Returns both student and teacher embeddings
        """
        # Student (online encoder)
        student_out = self.forward(img_prev, img_curr, task_embedding)

        # Teacher (momentum encoder)
        if self.momentum_encoder is not None:
            with torch.no_grad():
                teacher_embedding = self.momentum_encoder(
                    img_prev, img_curr, task_embedding
                )
        else:
            teacher_embedding = None

        return {
            **student_out,
            "teacher_embedding": teacher_embedding,
        }

    def get_preprocessing_output(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
    ) -> torch.Tensor:
        """Get Two-Stream preprocessing output (for visualization)"""
        return self.encoder.preprocessing(img_prev, img_curr)


class TextEncoder(nn.Module):
    """
    Task Description Encoder

    CLIP 또는 SigLIP text encoder 기반으로 task description embedding 생성
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        output_dim: int = 512,
        freeze: bool = True,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.freeze = freeze

        try:
            from transformers import CLIPTextModel, CLIPTokenizer

            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name)

            if freeze:
                for param in self.text_model.parameters():
                    param.requires_grad = False

            self.text_dim = self.text_model.config.hidden_size

        except ImportError:
            print("Warning: transformers not available, using random embeddings")
            self.tokenizer = None
            self.text_model = None
            self.text_dim = output_dim

        # Project to output_dim if needed
        if self.text_dim != output_dim:
            self.proj = nn.Linear(self.text_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        text: str | list[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode task description

        Args:
            text: task description string or list of strings
            device: target device

        Returns:
            text_embedding: [B, seq_len, output_dim]
        """
        if isinstance(text, str):
            text = [text]

        if self.text_model is None:
            # Fallback: random embeddings
            B = len(text)
            return torch.randn(B, 1, self.output_dim, device=device)

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.text_model(**inputs)

        # Get all token embeddings (not just CLS)
        text_embedding = outputs.last_hidden_state  # [B, seq_len, D]

        # Project
        text_embedding = self.proj(text_embedding)

        return text_embedding
