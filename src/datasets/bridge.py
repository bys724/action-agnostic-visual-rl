"""
Bridge V2 Dataset for video prediction training.

데이터 구조:
- bridge_root/raw/bridge_data_v2/
    - datacol1_toykitchen1/
        - many_skills/07/2023-.../
            - raw/traj_group0/traj{N}/images0/
                - im_0.jpg, im_1.jpg, ...

이미지 시퀀스 기반 (mp4가 아님)
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BridgeDataset(Dataset):
    """Bridge V2 Dataset for video prediction training."""

    def __init__(
        self,
        data_root: str,
        max_gap: int = 10,
        img_size: int = 224,
        sample_decay: float = 0.3,
        loss_decay: float = 0.7,
        max_trajectories: Optional[int] = None,
    ):
        """
        Initialize Bridge V2 Dataset.

        Args:
            data_root: Bridge V2 데이터 루트 경로
            max_gap: 최대 프레임 간격 (권장: 10, multi-scale temporal learning)
            img_size: 출력 이미지 크기
            sample_decay: 샘플링 확률 감쇠율
            loss_decay: Loss 가중치 감쇠율
            max_trajectories: 디버깅용 최대 trajectory 수
        """
        self.data_root = Path(data_root)
        self.max_gap = max_gap
        self.img_size = img_size

        # Multi-gap 설정
        gaps = np.arange(1, max_gap + 1)
        raw_probs = np.exp(-sample_decay * (gaps - 1))
        self.sample_probs = raw_probs / raw_probs.sum()
        self.loss_weights = np.exp(-loss_decay * (gaps - 1))

        # Trajectory 폴더 탐색 (images0 폴더가 있는 곳)
        self.trajectories = []
        search_root = self.data_root / "raw" / "bridge_data_v2"
        if not search_root.exists():
            search_root = self.data_root  # fallback

        for images_dir in search_root.glob("**/images0"):
            # 이미지 파일들 확인
            images = sorted(images_dir.glob("im_*.jpg"))
            if len(images) >= 2:
                self.trajectories.append(
                    {
                        "dir": images_dir,
                        "num_frames": len(images),
                    }
                )

        if max_trajectories:
            self.trajectories = self.trajectories[:max_trajectories]

        print(f"BridgeDataset initialized:")
        print(f"  Data root: {data_root}")
        print(f"  Trajectories found: {len(self.trajectories)}")
        print(f"  Max gap: {max_gap}")
        print(f"  Sample probs: {self.sample_probs.round(3)}")
        print(f"  Loss weights: {self.loss_weights.round(3)}")

    def _load_frame(self, images_dir: Path, frame_idx: int) -> torch.Tensor:
        """이미지 파일 로드."""
        img_path = images_dir / f"im_{frame_idx}.jpg"
        if not img_path.exists():
            raise ValueError(f"Image not found: {img_path}")

        frame = cv2.imread(str(img_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # [C, H, W]

        return frame

    def get_loss_weight(self, gap: int) -> float:
        """Get loss weight for a given gap."""
        if gap < 1 or gap > self.max_gap:
            return 1.0
        return self.loss_weights[gap - 1]

    def __len__(self) -> int:
        return len(self.trajectories) * 50  # 각 trajectory당 50 샘플

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Trajectory 선택
        traj_idx = idx % len(self.trajectories)
        traj = self.trajectories[traj_idx]

        # Gap 샘플링
        gap = np.random.choice(np.arange(1, self.max_gap + 1), p=self.sample_probs)

        # 시작 프레임 샘플링
        num_frames = traj["num_frames"]
        max_start = max(0, num_frames - gap - 1)
        if max_start <= 0:
            gap = 1
            max_start = max(0, num_frames - 2)

        frame_t = np.random.randint(0, max_start + 1)
        frame_tk = frame_t + gap

        # 프레임 로드
        img_t = self._load_frame(traj["dir"], frame_t)
        img_tk = self._load_frame(traj["dir"], frame_tk)

        return img_t, img_tk, gap
