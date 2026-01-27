"""
커스텀 VLA (Vision-Language-Action) 모델

Action-agnostic visual representation을 활용한 로봇 정책 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .visual_encoder import VisualEncoder, MomentumEncoder


class ActionDecoder(nn.Module):
    """
    액션 디코더

    시각 특징으로부터 로봇 액션을 예측
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        action_dim: int = 7,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, action_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 시각 특징 [B, D] 또는 [B, T, D]

        Returns:
            액션 [B, A] 또는 [B, T, A]
        """
        return self.decoder(x)


class CustomVLA(nn.Module):
    """
    커스텀 VLA 모델

    Architecture:
    1. Visual Encoder (DINOv2/CLIP 기반, action-agnostic)
    2. (Optional) Language Encoder
    3. Action Decoder

    학습 모드:
    - pretrain: Action-agnostic visual representation 학습 (self-supervised)
    - finetune: Action prediction 학습 (supervised)
    """

    def __init__(
        self,
        visual_backbone: str = "dinov2",
        visual_model_size: str = "base",
        freeze_visual: bool = False,
        partial_finetune_layers: Optional[int] = None,
        hidden_dim: int = 768,
        action_dim: int = 7,
        use_momentum_encoder: bool = True,
        momentum: float = 0.999,
    ):
        super().__init__()

        # Visual Encoder
        self.visual_encoder = VisualEncoder(
            backbone=visual_backbone,
            model_size=visual_model_size,
            freeze_backbone=freeze_visual,
            partial_finetune=partial_finetune_layers,
            output_dim=hidden_dim
        )

        # Momentum Encoder (for self-supervised learning)
        self.use_momentum = use_momentum_encoder
        if use_momentum_encoder:
            self.momentum_encoder = MomentumEncoder(
                self.visual_encoder,
                momentum=momentum
            )

        # Action Decoder
        self.action_decoder = ActionDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            action_dim=action_dim
        )

        # Loss functions
        self.action_loss_fn = nn.MSELoss(reduction='none')
        self.contrastive_temp = 0.1

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            images: [B, T, C, H, W] 또는 [B, C, H, W]
            return_features: 특징 벡터도 반환할지

        Returns:
            {"actions": ..., "features": ...}
        """
        # Visual encoding
        features = self.visual_encoder(images)

        # Action prediction
        actions = self.action_decoder(features)

        output = {"actions": actions}
        if return_features:
            output["features"] = features

        return output

    def compute_loss(
        self,
        images: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mode: str = "finetune"
    ) -> torch.Tensor:
        """
        손실 계산

        Args:
            images: [B, T, C, H, W]
            target_actions: [B, T, A]
            mask: [B, T] 유효한 timestep 마스크
            mode: "pretrain" (self-supervised) 또는 "finetune" (supervised)

        Returns:
            손실 값
        """
        if mode == "pretrain":
            return self._compute_pretrain_loss(images)
        else:
            return self._compute_finetune_loss(images, target_actions, mask)

    def _compute_finetune_loss(
        self,
        images: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Supervised action prediction loss"""
        output = self.forward(images)
        pred_actions = output["actions"]

        # Action loss
        loss = self.action_loss_fn(pred_actions, target_actions)

        # 마스크 적용
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    def _compute_pretrain_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        Self-supervised pre-training loss

        DINO 스타일의 self-distillation loss
        """
        if not self.use_momentum:
            raise ValueError("Momentum encoder required for pretraining")

        # Student와 Teacher 특징 추출
        student_features, teacher_features = self.momentum_encoder(images)

        # Contrastive loss (SimCLR 스타일)
        student_norm = F.normalize(student_features, dim=-1)
        teacher_norm = F.normalize(teacher_features, dim=-1)

        # 유사도 행렬
        similarity = torch.matmul(student_norm, teacher_norm.T) / self.contrastive_temp

        # Labels: 대각선이 positive
        labels = torch.arange(similarity.shape[0], device=similarity.device)

        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)

        # Momentum encoder 업데이트
        self.momentum_encoder.update_momentum_encoder()

        return loss

    def predict_action(self, image: torch.Tensor) -> torch.Tensor:
        """
        단일 이미지에서 액션 예측 (추론용)

        Args:
            image: [C, H, W] 또는 [1, C, H, W]

        Returns:
            액션 [A]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            output = self.forward(image)

        return output["actions"].squeeze(0)
