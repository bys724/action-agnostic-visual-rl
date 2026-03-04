"""
Two-Stream Image Preprocessing

생물학적 시각 시스템의 Magnocellular/Parvocellular 경로를 모델링한 전처리 방법
연속된 이미지로부터 시간적 변화(M채널)와 공간적 구조(P채널)를 분리 추출

References:
- Two-Stream Image Preprocessing.md
- Two Visual Pathways.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TwoStreamPreprocessing(nn.Module):
    """
    Two-Stream Image Preprocessing Module

    M채널 (4ch): 시간적 밝기 변화 + RGB 색상 정보
    P채널 (2ch): 공간적 edge/형태 정보

    총 6채널 출력으로 시간·공간·색상 통합 표현

    Args:
        init_lum_weights: 초기 luminance weights [R, G, B]
        trainable_weights: luminance weights 학습 여부
    """

    def __init__(
        self,
        init_lum_weights: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
        trainable_weights: bool = True,
    ):
        super().__init__()

        # Luminance weights (M채널과 P채널에서 공유)
        # BT.709 표준값으로 초기화
        self.lum_weights = nn.Parameter(
            torch.tensor(init_lum_weights, dtype=torch.float32),
            requires_grad=trainable_weights
        )

        # Sobel kernels for P-channel (고정)
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _compute_luminance(self, img: torch.Tensor) -> torch.Tensor:
        """
        RGB → Luminance 변환 (trainable weights)

        Args:
            img: [B, 3, H, W], range [0, 1]

        Returns:
            luminance: [B, 1, H, W]
        """
        # Softmax로 weights 합이 1이 되도록 보장
        w = torch.softmax(self.lum_weights, dim=0).view(1, 3, 1, 1)
        luminance = (img * w).sum(dim=1, keepdim=True)
        return luminance

    def magnocellular_channel(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor
    ) -> torch.Tensor:
        """
        M채널: 절대적 밝기 변화 + RGB 색상 정보

        시간적 변화를 감지하는 채널
        - ΔL: 밝기 변화 (움직임 방향, 조명 변화)
        - RGB: 색상 정보 보존 (물체 정체성)

        Args:
            img_prev: [B, 3, H, W], range [0, 1], 이전 프레임
            img_curr: [B, 3, H, W], range [0, 1], 현재 프레임

        Returns:
            m_output: [B, 4, H, W] - [ΔL, R(t-1), G(t-1), B(t-1)]
        """
        # 1. Luminance 계산
        lum_prev = self._compute_luminance(img_prev)  # [B, 1, H, W]
        lum_curr = self._compute_luminance(img_curr)  # [B, 1, H, W]

        # 2. 시간 미분 (절대적 변화량 보존)
        delta_L = lum_curr - lum_prev  # 범위 [-1, 1]
        delta_L = torch.clamp(delta_L, -1, 1)  # outlier 제거

        # 3. RGB 색상 정보 (이전 프레임 기준)
        rgb_prev = img_prev  # [B, 3, H, W]

        # 4. 통합 출력
        m_output = torch.cat([delta_L, rgb_prev], dim=1)  # [B, 4, H, W]

        return m_output

    def parvocellular_channel(self, img: torch.Tensor) -> torch.Tensor:
        """
        P채널: 상대적 대비 (형태/edge 정보)

        공간적 구조를 인코딩하는 채널
        - 조명 불변적 edge 검출
        - 방향(비율) + 선명도(강도) 보존

        Args:
            img: [B, 3, H, W], range [0, 1]

        Returns:
            p_output: [B, 2, H, W] - [∂L/∂x, ∂L/∂y], range [-1, 1]
        """
        # 1. Luminance 계산 (M채널과 동일한 weights)
        lum = self._compute_luminance(img)  # [B, 1, H, W]

        # 2. 이미지별 normalize (조명 불변성)
        mean = lum.mean(dim=(2, 3), keepdim=True)
        std = lum.std(dim=(2, 3), keepdim=True) + 1e-6
        lum_norm = (lum - mean) / std
        lum_norm = torch.clamp(lum_norm, -3, 3)  # outlier 제거

        # 3. Sobel 필터 적용 (공간 미분)
        grad_x = F.conv2d(lum_norm, self.sobel_x, padding=1)  # [B, 1, H, W]
        grad_y = F.conv2d(lum_norm, self.sobel_y, padding=1)  # [B, 1, H, W]

        # 4. Gradient magnitude 계산
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # [B, 1, H, W]

        # 5. Max magnitude로 normalize (방향 + 선명도 보존)
        max_magnitude = magnitude.amax(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
        max_magnitude = torch.clamp(max_magnitude, min=1e-6)  # zero division 방지

        grad_x_norm = grad_x / max_magnitude
        grad_y_norm = grad_y / max_magnitude

        # 6. 출력
        p_output = torch.cat([grad_x_norm, grad_y_norm], dim=1)  # [B, 2, H, W]

        return p_output

    def forward(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        return_separate: bool = False
    ) -> torch.Tensor:
        """
        Two-Stream 전처리 수행

        Args:
            img_prev: [B, 3, H, W], range [0, 1], 이전 프레임 (t)
            img_curr: [B, 3, H, W], range [0, 1], 현재 프레임 (t+1)
            return_separate: True면 M, P 채널을 따로 반환

        Returns:
            combined: [B, 6, H, W] - [ΔL, R, G, B, ∂L/∂x, ∂L/∂y]
            또는 (m_output, p_output) if return_separate=True
        """
        # M채널: 시간적 변화 + 색상
        m_output = self.magnocellular_channel(img_prev, img_curr)  # [B, 4, H, W]

        # P채널: 공간적 구조 (이전 프레임 기준)
        p_output = self.parvocellular_channel(img_prev)  # [B, 2, H, W]

        if return_separate:
            return m_output, p_output

        # 통합 출력
        combined = torch.cat([m_output, p_output], dim=1)  # [B, 6, H, W]

        return combined

    def get_learned_weights(self) -> torch.Tensor:
        """학습된 luminance weights 반환 (디버깅용)"""
        return torch.softmax(self.lum_weights, dim=0)


class TwoStreamViTAdapter(nn.Module):
    """
    Two-Stream 전처리 출력을 ViT 입력 형태로 변환

    6채널 입력을 ViT의 patch embedding에 맞게 조정

    Args:
        img_size: 입력 이미지 크기
        patch_size: ViT patch 크기
        embed_dim: ViT embedding dimension
        in_channels: 입력 채널 수 (기본 6)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 768,
        in_channels: int = 6,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 6채널 → embed_dim으로 변환하는 patch embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 6, H, W] - Two-Stream 전처리 출력

        Returns:
            tokens: [B, num_patches, embed_dim]
        """
        # Patch embedding
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]

        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Normalize
        x = self.norm(x)

        return x


class TwoStreamProcessor(nn.Module):
    """
    Two-Stream 전처리 + ViT Adapter 통합 모듈

    연속된 이미지를 받아 ViT 입력 토큰으로 변환

    Args:
        img_size: 입력 이미지 크기
        patch_size: ViT patch 크기
        embed_dim: ViT embedding dimension
        trainable_lum_weights: luminance weights 학습 여부
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 768,
        trainable_lum_weights: bool = True,
    ):
        super().__init__()

        self.preprocessing = TwoStreamPreprocessing(
            trainable_weights=trainable_lum_weights
        )

        self.adapter = TwoStreamViTAdapter(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=6
        )

    def forward(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            img_prev: [B, 3, H, W], range [0, 1]
            img_curr: [B, 3, H, W], range [0, 1]

        Returns:
            tokens: [B, num_patches, embed_dim]
        """
        # Two-Stream 전처리
        features = self.preprocessing(img_prev, img_curr)  # [B, 6, H, W]

        # ViT 토큰으로 변환
        tokens = self.adapter(features)  # [B, num_patches, embed_dim]

        return tokens

    def get_preprocessing_output(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor
    ) -> torch.Tensor:
        """전처리 출력만 반환 (시각화용)"""
        return self.preprocessing(img_prev, img_curr)
