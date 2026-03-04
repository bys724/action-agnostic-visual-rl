"""
Action Head module for LIBERO evaluation.

This module is extracted from scripts/finetune_libero.py for reuse
in custom encoder evaluation.
"""

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """
    Simple action prediction head.

    Takes visual embeddings and predicts 7D action.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        action_dim: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, embed_dim] - visual embedding

        Returns:
            action: [B, action_dim]
        """
        return self.mlp(x)


class EncoderWithActionHead(nn.Module):
    """
    Vision encoder with action prediction head for LIBERO evaluation.

    This combines a pre-trained encoder with a fine-tuned action head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 768,
        action_dim: int = 7,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        self.action_head = ActionHead(embed_dim=embed_dim, action_dim=action_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 6, H, W] - two stacked RGB images

        Returns:
            action: [B, 7] - predicted action
        """
        # Encode
        patch_embeddings = self.encoder(pixel_values)  # [B, num_patches, D]

        # Mean pooling over patches
        visual_embedding = patch_embeddings.mean(dim=1)  # [B, D]

        # Predict action
        action = self.action_head(visual_embedding)  # [B, 7]

        return action

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        encoder_type: str = "two-stream",
        device: str = "cuda",
    ) -> "EncoderWithActionHead":
        """
        Load EncoderWithActionHead from a fine-tuning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (.pt file from finetune_libero.py)
            encoder_type: Type of encoder ("two-stream", "single-stream", "videomae")
            device: Device to load model on

        Returns:
            Loaded model
        """
        from .openvla_encoder import (
            TwoStreamEncoderForOpenVLA,
            SingleStreamEncoderForOpenVLA,
            VideoMAEEncoderForOpenVLA,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config", {})

        # Create encoder based on type
        if encoder_type == "two-stream":
            encoder = TwoStreamEncoderForOpenVLA()
        elif encoder_type == "single-stream":
            encoder = SingleStreamEncoderForOpenVLA()
        elif encoder_type == "videomae":
            encoder = VideoMAEEncoderForOpenVLA()
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        embed_dim = config.get("encoder_embed_dim", encoder.embed_dim)

        # Create model
        model = cls(
            encoder=encoder,
            embed_dim=embed_dim,
            freeze_encoder=True,  # Always freeze for inference
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        return model
