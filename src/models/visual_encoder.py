"""
시각 인코더

Action-agnostic visual representation 학습을 위한 인코더
DINOv2, CLIP 등 pre-trained 모델 기반
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class VisualEncoder(nn.Module):
    """
    시각 인코더

    Pre-trained vision model (DINOv2, CLIP)을 기반으로
    action-agnostic visual representation을 학습

    Args:
        backbone: 백본 모델 ("dinov2", "clip", "resnet")
        freeze_backbone: 백본 동결 여부
        partial_finetune: 부분 fine-tuning (마지막 N 레이어만)
    """

    def __init__(
        self,
        backbone: Literal["dinov2", "clip", "resnet"] = "dinov2",
        model_size: str = "base",
        freeze_backbone: bool = False,
        partial_finetune: Optional[int] = None,
        output_dim: int = 768,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone

        # 백본 로드
        if backbone == "dinov2":
            self.backbone = self._load_dinov2(model_size)
            backbone_dim = 768 if model_size == "base" else 1024
        elif backbone == "clip":
            self.backbone = self._load_clip(model_size)
            backbone_dim = 768 if model_size == "base" else 1024
        elif backbone == "resnet":
            self.backbone = self._load_resnet()
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        elif partial_finetune is not None:
            self._partial_freeze(partial_finetune)

    def _load_dinov2(self, model_size: str):
        """DINOv2 로드"""
        model_name = f"dinov2_vit{model_size[0]}14"  # e.g., dinov2_vitb14
        try:
            model = torch.hub.load("facebookresearch/dinov2", model_name)
            print(f"Loaded DINOv2: {model_name}")
            return model
        except Exception as e:
            print(f"Failed to load DINOv2: {e}")
            # Fallback: 간단한 ViT
            return self._create_simple_vit()

    def _load_clip(self, model_size: str):
        """CLIP 비전 인코더 로드"""
        try:
            import clip
            model_name = f"ViT-B/16" if model_size == "base" else "ViT-L/14"
            model, _ = clip.load(model_name, device="cpu")
            print(f"Loaded CLIP: {model_name}")
            return model.visual
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            return self._create_simple_vit()

    def _load_resnet(self):
        """ResNet50 로드"""
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # 마지막 FC 레이어 제거
        return nn.Sequential(*list(model.children())[:-1])

    def _create_simple_vit(self):
        """Fallback용 간단한 ViT"""
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()
        return model

    def _freeze_backbone(self):
        """백본 전체 동결"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _partial_freeze(self, num_unfrozen_layers: int):
        """부분 동결 (마지막 N 레이어만 학습)"""
        # 모든 파라미터 동결
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 마지막 N 레이어 해제 (DINOv2/ViT 구조 가정)
        if hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-num_unfrozen_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 이미지 텐서 [B, C, H, W] 또는 [B, T, C, H, W]

        Returns:
            특징 벡터 [B, D] 또는 [B, T, D]
        """
        # 시퀀스 입력 처리
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self._encode(x)
            features = features.view(B, T, -1)
        else:
            features = self._encode(x)

        return self.projection(features)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """단일 이미지 인코딩"""
        if self.backbone_name == "dinov2":
            return self.backbone(x)
        elif self.backbone_name == "clip":
            return self.backbone(x).float()
        else:
            return self.backbone(x).squeeze(-1).squeeze(-1)


class MomentumEncoder(nn.Module):
    """
    EMA 기반 Momentum Encoder

    Self-supervised learning에서 안정적인 타겟 생성을 위해 사용
    (DINO, BYOL 스타일)
    """

    def __init__(
        self,
        encoder: VisualEncoder,
        momentum: float = 0.999
    ):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum

        # Momentum encoder 생성 (파라미터 복사)
        self.momentum_encoder = self._create_momentum_encoder()

    def _create_momentum_encoder(self) -> VisualEncoder:
        """Momentum encoder 생성"""
        import copy
        momentum_encoder = copy.deepcopy(self.encoder)
        for param in momentum_encoder.parameters():
            param.requires_grad = False
        return momentum_encoder

    @torch.no_grad()
    def update_momentum_encoder(self):
        """EMA 업데이트"""
        for param, momentum_param in zip(
            self.encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            momentum_param.data = (
                self.momentum * momentum_param.data +
                (1 - self.momentum) * param.data
            )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns:
            (student_features, teacher_features)
        """
        student_features = self.encoder(x)

        with torch.no_grad():
            teacher_features = self.momentum_encoder(x)

        return student_features, teacher_features
