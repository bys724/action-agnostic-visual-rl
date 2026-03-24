"""
Two-Stream Preprocessing: Biological-inspired M/P channels.

M Channel (Magnocellular/Dorsal): Temporal change detection
    - Input: image_previous, image_current
    - Output: [ΔL, Sobel_x(ΔL), Sobel_y(ΔL)] (3 channels)
    - ΔL: raw luminance difference [-1, 1] (정규화 없음, 절대 스케일 유지)
    - Sobel(ΔL): 변화 영역의 경계 형상 (magnitude 정규화)

P Channel (Parvocellular/Ventral): Spatial structure + color
    - Input: image_current
    - Output: [∂x, ∂y, R, G, B] (5 channels)
    - Luminance z-score normalized, magnitude normalized gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStreamPreprocessing(nn.Module):
    """
    Biological-inspired two-stream preprocessing.

    Separates visual input into:
    - M channel: Temporal dynamics (motion/change)
    - P channel: Spatial structure (shape/color)
    """

    def __init__(self):
        super().__init__()

        # Luminance weights (BT.709 standard, fixed)
        # NOTE: Attempted learnable weights but gradient instability occurred
        #       BT.709 is optimized for human vision, so kept fixed
        self.register_buffer(
            "luminance_weights", torch.tensor([0.2126, 0.7152, 0.0722])
        )

        # Sobel filters for spatial gradients
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def compute_luminance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute luminance using BT.709 standard weights.

        Args:
            image: [B, 3, H, W], range [0, 1]

        Returns:
            luminance: [B, 1, H, W]
        """
        weights = self.luminance_weights.to(image.device).view(1, 3, 1, 1)
        luminance = (image * weights).sum(dim=1, keepdim=True)
        return luminance

    def compute_m_channel(
        self, image_previous: torch.Tensor, image_current: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute M channel: Temporal change (motion pathway).

        ΔL은 정규화 없이 절대 스케일 유지 — 변화의 크기 자체가 정보.
        Sobel(ΔL)은 변화 영역의 경계 형상을 포착.

        Args:
            image_previous: [B, 3, H, W], range [0, 1]
            image_current: [B, 3, H, W], range [0, 1]

        Returns:
            m_channel: [B, 3, H, W], [ΔL, Sobel_x(ΔL), Sobel_y(ΔL)]
        """
        # 1. Luminance change (no normalization — absolute scale is meaningful)
        lum_prev = self.compute_luminance(image_previous)
        lum_curr = self.compute_luminance(image_current)
        delta_l = lum_curr - lum_prev  # [-1, 1]

        # 2. Sobel on ΔL — edges of the change map (변화 영역의 경계)
        grad_x = F.conv2d(
            delta_l, self.sobel_x.to(delta_l.device), padding=1
        )
        grad_y = F.conv2d(
            delta_l, self.sobel_y.to(delta_l.device), padding=1
        )

        # 3. Magnitude normalization (same as P channel Sobel)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        max_magnitude = magnitude.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        grad_x_normalized = grad_x / max_magnitude
        grad_y_normalized = grad_y / max_magnitude

        return torch.cat([delta_l, grad_x_normalized, grad_y_normalized], dim=1)

    def compute_p_channel(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute P channel: Spatial structure + color (form pathway).

        Args:
            image: [B, 3, H, W], range [0, 1]

        Returns:
            p_channel: [B, 5, H, W], [∂x, ∂y, R, G, B]
        """
        # 1. Compute luminance
        luminance = self.compute_luminance(image)

        # 2. Per-image normalization (lighting invariance)
        mean = luminance.mean(dim=(2, 3), keepdim=True)
        std = luminance.std(dim=(2, 3), keepdim=True) + 1e-6
        luminance_normalized = torch.clamp((luminance - mean) / std, -3, 3)

        # 3. Sobel gradients (ensure device compatibility for DataParallel)
        grad_x = F.conv2d(
            luminance_normalized,
            self.sobel_x.to(luminance_normalized.device),
            padding=1,
        )
        grad_y = F.conv2d(
            luminance_normalized,
            self.sobel_y.to(luminance_normalized.device),
            padding=1,
        )

        # 4. Magnitude normalization (preserve direction + sharpness)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        max_magnitude = magnitude.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)

        grad_x_normalized = grad_x / max_magnitude
        grad_y_normalized = grad_y / max_magnitude

        # 5. Extract RGB channels
        r_channel = image[:, 0:1]
        g_channel = image[:, 1:2]
        b_channel = image[:, 2:3]

        return torch.cat(
            [grad_x_normalized, grad_y_normalized, r_channel, g_channel, b_channel],
            dim=1,
        )

    def forward(
        self, image_previous: torch.Tensor, image_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process consecutive frames into M and P channels.

        Args:
            image_previous: [B, 3, H, W], range [0, 1]
            image_current: [B, 3, H, W], range [0, 1]

        Returns:
            m_channel: [B, 3, H, W] - temporal change (ΔL + Sobel(ΔL))
            p_channel: [B, 5, H, W] - spatial structure + color
        """
        m_channel = self.compute_m_channel(image_previous, image_current)
        # Use previous frame to avoid future leakage
        p_channel = self.compute_p_channel(image_previous)

        return m_channel, p_channel
