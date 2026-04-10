"""
비디오 프레임 데이터셋 베이스 클래스.

공통 로직:
- Multi-gap temporal sampling (uniform 또는 exponential decay)
- Gap 기반 loss weighting (uniform 또는 exponential decay)
- 사전 추출된 프레임(frame_XXXXXX.jpg) 로딩 + synchronized spatial crop
- 프레임 수 캐싱

서브클래스는 _scan_frame_dirs()만 구현하면 됨.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VideoFrameDataset(ABC, Dataset):
    """사전 추출된 프레임(256x256) 기반 비디오 데이터셋."""

    def __init__(
        self,
        data_root: str,
        max_gap: int = 30,
        img_size: int = 224,
        sample_decay: float = 0.0,
        loss_decay: float = 0.0,
        samples_per_video: int = 100,
        train: bool = True,
        max_videos: Optional[int] = None,
        cache_frames: bool = False,
        sample_dist: str = "auto",
        sample_center: int = None,
    ):
        self.data_root = Path(data_root)
        self.max_gap = max_gap
        self.img_size = img_size
        self.samples_per_video = samples_per_video
        self.cache_frames = cache_frames

        # Spatial transform: 256x256 → 224x224
        if train:
            self.spatial_transform = transforms.RandomCrop(img_size)
        else:
            self.spatial_transform = transforms.CenterCrop(img_size)

        # Gap sampling 확률
        # sample_dist:
        #   "auto": sample_decay 값으로 결정 (기존 호환)
        #   "uniform": 균일
        #   "linear": gap에 비례 (큰 gap 선호)
        #   "exp": exponential decay (sample_decay 양수면 작은 gap 선호)
        #   "triangular": center에서 최대, 양쪽으로 선형 감소
        gaps = np.arange(1, max_gap + 1)

        if sample_dist == "auto":
            if sample_decay > 0:
                sample_dist = "exp"
            elif sample_decay < 0:
                sample_dist = "linear"
            else:
                sample_dist = "uniform"

        if sample_dist == "exp":
            raw_probs = np.exp(-abs(sample_decay) * (gaps - 1))
        elif sample_dist == "linear":
            raw_probs = gaps.astype(float)
        elif sample_dist == "triangular":
            center = sample_center if sample_center is not None else max_gap // 2
            # |gap - center|에 역비례, 경계에서 최소
            raw_probs = np.maximum(1.0 - np.abs(gaps - center) / max(center, max_gap - center), 0.01)
        else:  # uniform
            raw_probs = np.ones_like(gaps, dtype=float)
        self.sample_probs = raw_probs / raw_probs.sum()

        # Loss weights (decay=0 → uniform weight=1.0)
        if loss_decay > 0:
            self.loss_weights = np.exp(-loss_decay * (gaps - 1))
        else:
            self.loss_weights = np.ones_like(gaps, dtype=float)

        # 프레임 디렉토리 탐색 (서브클래스 구현)
        self.frame_dirs = self._scan_frame_dirs()
        if max_videos:
            self.frame_dirs = self.frame_dirs[:max_videos]

        # 캐시
        self._video_info = {}  # {path: num_frames}
        self._frame_cache = {}  # {(path, idx): tensor}

    @abstractmethod
    def _scan_frame_dirs(self) -> List[Path]:
        """프레임 디렉토리 목록 반환. 각 디렉토리에 frame_XXXXXX.jpg 파일들이 있어야 함."""
        ...

    def _get_num_frames(self, frame_dir: Path) -> int:
        """프레임 수 반환 (캐싱)."""
        if frame_dir not in self._video_info:
            self._video_info[frame_dir] = len(list(frame_dir.glob("frame_*.jpg")))
        return self._video_info[frame_dir]

    def _load_image(self, path: Path) -> torch.Tensor:
        """JPEG → [C, H, W] float32 텐서."""
        img = Image.open(path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() / 255.0
        return img.permute(2, 0, 1)

    def _load_frame_pair(
        self, frame_dir: Path, idx1: int, idx2: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """두 프레임을 로드하고 동일한 spatial crop 적용."""
        path1 = frame_dir / f"frame_{idx1:06d}.jpg"
        path2 = frame_dir / f"frame_{idx2:06d}.jpg"

        # 캐시 확인
        if self.cache_frames:
            key1, key2 = (frame_dir, idx1), (frame_dir, idx2)
            if key1 in self._frame_cache and key2 in self._frame_cache:
                return self._frame_cache[key1], self._frame_cache[key2]

        img1 = self._load_image(path1)
        img2 = self._load_image(path2)

        # 프레임 쌍에 독립적인 random crop 적용
        # → pixel-level 정렬을 깨서 더 높은 수준의 표현 학습 유도
        if isinstance(self.spatial_transform, transforms.RandomCrop):
            params1 = transforms.RandomCrop.get_params(
                img1, (self.img_size, self.img_size)
            )
            params2 = transforms.RandomCrop.get_params(
                img2, (self.img_size, self.img_size)
            )
            img1 = transforms.functional.crop(img1, *params1)
            img2 = transforms.functional.crop(img2, *params2)
        else:
            img1 = self.spatial_transform(img1)
            img2 = self.spatial_transform(img2)

        if self.cache_frames:
            self._frame_cache[(frame_dir, idx1)] = img1
            self._frame_cache[(frame_dir, idx2)] = img2

        return img1, img2

    def get_loss_weight(self, gap: int) -> float:
        """Gap에 대한 loss 가중치."""
        if gap < 1 or gap > self.max_gap:
            return 1.0
        return self.loss_weights[gap - 1]

    def __len__(self) -> int:
        return len(self.frame_dirs) * self.samples_per_video

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # 손상된 프레임 파일 대비: 실패 시 랜덤 다른 샘플로 fallback
        for _retry in range(5):
            try:
                video_idx = idx % len(self.frame_dirs)
                frame_dir = self.frame_dirs[video_idx]
                num_frames = self._get_num_frames(frame_dir)

                gap = np.random.choice(
                    np.arange(1, self.max_gap + 1), p=self.sample_probs
                )

                max_start = max(0, num_frames - gap - 1)
                if max_start <= 0:
                    gap = 1
                    max_start = max(0, num_frames - 2)

                frame_t = np.random.randint(0, max_start + 1)
                frame_tk = frame_t + gap

                img_t, img_tk = self._load_frame_pair(frame_dir, frame_t, frame_tk)
                return img_t, img_tk, gap
            except Exception:
                # 손상 파일 → 다른 비디오에서 재시도
                idx = np.random.randint(0, len(self))
        # 5회 연속 실패 시 (극히 드묾) 검은 이미지 반환
        fallback = torch.zeros(3, self.img_size, self.img_size)
        return fallback, fallback, 1
