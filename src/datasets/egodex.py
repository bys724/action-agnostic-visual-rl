"""
EgoDex Dataset for video prediction training.

데이터 구조 (프레임 추출 버전):
- frames_root/
    - task_name/
        - video_name/
            - frame_000000.jpg
            - frame_000001.jpg
            - ...

프레임은 미리 추출되어 JPEG로 저장됨 (256x256 센터크롭, 95% quality).
학습 시 RandomCrop(224), 평가 시 CenterCrop(224) 적용.
VideoCapture 대신 이미지 파일 직접 로드로 학습 속도 대폭 개선.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EgoDexDataset(Dataset):
    """EgoDex Dataset for video prediction training."""

    def __init__(
        self,
        data_root: str,
        split: str = "test",
        max_gap: int = 10,
        img_size: int = 224,
        sample_decay: float = 0.3,
        loss_decay: float = 0.7,
        max_videos: Optional[int] = None,
        cache_frames: bool = False,
        train: bool = True,
    ):
        """
        Initialize EgoDex Dataset.

        Args:
            data_root: 추출된 프레임 루트 경로 (e.g. /workspace/data/egodex_frames)
            split: "test" 또는 "train" (part1-5)
            max_gap: 최대 프레임 간격
            img_size: 출력 이미지 크기 (프레임은 256x256으로 추출, 학습 시 RandomCrop)
            sample_decay: 샘플링 확률 감쇠율
            loss_decay: Loss 가중치 감쇠율
            max_videos: 디버깅용 최대 비디오 수
            cache_frames: 프레임 캐싱 여부
            train: True면 RandomCrop, False면 CenterCrop
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_gap = max_gap
        self.img_size = img_size
        self.cache_frames = cache_frames

        # 학습 시 RandomCrop, 평가 시 CenterCrop (프레임은 256x256으로 추출됨)
        if train:
            self.spatial_transform = transforms.RandomCrop(img_size)
        else:
            self.spatial_transform = transforms.CenterCrop(img_size)

        # Multi-gap 설정
        gaps = np.arange(1, max_gap + 1)
        raw_probs = np.exp(-sample_decay * (gaps - 1))
        self.sample_probs = raw_probs / raw_probs.sum()
        self.loss_weights = np.exp(-loss_decay * (gaps - 1))

        # 프레임 디렉토리 탐색 (task_name/video_name/ 구조)
        self.frame_dirs = []
        for task_dir in sorted(self.data_root.glob("*")):
            if not task_dir.is_dir():
                continue
            for video_dir in sorted(task_dir.glob("*")):
                if video_dir.is_dir():
                    self.frame_dirs.append(video_dir)

        if max_videos:
            self.frame_dirs = self.frame_dirs[:max_videos]

        # 각 비디오의 프레임 수 캐싱 (첫 로드 시)
        self.video_info = {}  # {path: num_frames}
        self._frame_cache = {}  # {(path, frame_idx): tensor}

        print(f"EgoDexDataset initialized:")
        print(f"  Data root: {data_root}")
        print(f"  Split: {split}")
        print(f"  Frame directories found: {len(self.frame_dirs)}")
        print(f"  Max gap: {max_gap}")
        print(f"  Sample probs: {self.sample_probs.round(3)}")
        print(f"  Loss weights: {self.loss_weights.round(3)}")

    def _get_video_info(self, frame_dir: Path) -> int:
        """프레임 디렉토리의 프레임 수를 반환 (캐싱됨)."""
        if frame_dir not in self.video_info:
            # 프레임 파일 개수 카운트
            num_frames = len(list(frame_dir.glob("frame_*.jpg")))
            self.video_info[frame_dir] = num_frames
        return self.video_info[frame_dir]

    def _load_frame_pair(
        self, frame_dir: Path, frame_idx1: int, frame_idx2: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        프레임 디렉토리에서 두 프레임을 로드 (JPEG).

        Args:
            frame_dir: 프레임이 저장된 디렉토리
            frame_idx1: 첫 번째 프레임 인덱스
            frame_idx2: 두 번째 프레임 인덱스

        Returns:
            (img1, img2): 두 개의 프레임 텐서 [C, H, W]
        """
        # 캐시 확인
        cache_key1 = (frame_dir, frame_idx1)
        cache_key2 = (frame_dir, frame_idx2)

        if self.cache_frames:
            if cache_key1 in self._frame_cache and cache_key2 in self._frame_cache:
                return self._frame_cache[cache_key1], self._frame_cache[cache_key2]

        # JPEG 파일 경로
        frame_path1 = frame_dir / f"frame_{frame_idx1:06d}.jpg"
        frame_path2 = frame_dir / f"frame_{frame_idx2:06d}.jpg"

        if not frame_path1.exists():
            raise ValueError(f"Frame not found: {frame_path1}")
        if not frame_path2.exists():
            raise ValueError(f"Frame not found: {frame_path2}")

        # 프레임 로드 및 전처리 (프레임 쌍에 동일한 crop 적용)
        def load_as_tensor(path: Path) -> torch.Tensor:
            img = Image.open(path).convert("RGB")
            img = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, C]
            img = img.permute(2, 0, 1)  # [C, H, W]
            return img

        img1 = load_as_tensor(frame_path1)
        img2 = load_as_tensor(frame_path2)

        # 프레임 쌍에 동일한 crop 적용 (temporal consistency 유지)
        if isinstance(self.spatial_transform, transforms.RandomCrop):
            crop_params = transforms.RandomCrop.get_params(img1, (self.img_size, self.img_size))
            img1 = transforms.functional.crop(img1, *crop_params)
            img2 = transforms.functional.crop(img2, *crop_params)
        else:
            img1 = self.spatial_transform(img1)
            img2 = self.spatial_transform(img2)

        # 캐싱
        if self.cache_frames:
            self._frame_cache[cache_key1] = img1
            self._frame_cache[cache_key2] = img2

        return img1, img2

    def _load_frame(self, frame_dir: Path, frame_idx: int) -> torch.Tensor:
        """프레임 디렉토리에서 특정 프레임 로드 (단일 프레임, spatial_transform 적용)."""
        cache_key = (frame_dir, frame_idx)
        if self.cache_frames and cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
        if not frame_path.exists():
            raise ValueError(f"Frame not found: {frame_path}")

        img = Image.open(frame_path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, C]
        img = img.permute(2, 0, 1)  # [C, H, W]
        img = self.spatial_transform(img)

        if self.cache_frames:
            self._frame_cache[cache_key] = img

        return img

    def get_loss_weight(self, gap: int) -> float:
        """Get loss weight for a given gap."""
        if gap < 1 or gap > self.max_gap:
            return 1.0
        return self.loss_weights[gap - 1]

    def __len__(self) -> int:
        # 각 비디오당 여러 샘플 (대략 프레임수 / 30)
        return len(self.frame_dirs) * 100

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # 비디오 선택 (순환)
        video_idx = idx % len(self.frame_dirs)
        frame_dir = self.frame_dirs[video_idx]

        # 프레임 수 확인
        num_frames = self._get_video_info(frame_dir)

        # Gap 샘플링
        gap = np.random.choice(np.arange(1, self.max_gap + 1), p=self.sample_probs)

        # 시작 프레임 샘플링 (gap 고려)
        max_start = max(0, num_frames - gap - 1)
        if max_start <= 0:
            # 비디오가 너무 짧으면 gap 줄이기
            gap = 1
            max_start = max(0, num_frames - 2)

        frame_t = np.random.randint(0, max_start + 1)
        frame_tk = frame_t + gap

        # 프레임 로드 (JPEG 직접 로드)
        img_t, img_tk = self._load_frame_pair(frame_dir, frame_t, frame_tk)

        return img_t, img_tk, gap
