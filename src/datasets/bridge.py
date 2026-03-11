"""
Bridge V2 Dataset.

프레임 구조 (사전 추출 후):
  frames_root/
    traj_XXXXXXXX/
      frame_000000.jpg, frame_000001.jpg, ...

프레임은 리사이즈(480x640 → 256x256, crop 없음)로 사전 추출됨.
"""

from pathlib import Path
from typing import List, Optional

from .base import VideoFrameDataset


class BridgeDataset(VideoFrameDataset):
    """Bridge V2 프레임 데이터셋. traj_* 1단계 디렉토리 구조."""

    def __init__(
        self,
        data_root: str,
        max_gap: int = 10,
        img_size: int = 224,
        sample_decay: float = 0.3,
        loss_decay: float = 0.7,
        max_videos: Optional[int] = None,
        cache_frames: bool = False,
        train: bool = True,
    ):
        super().__init__(
            data_root=data_root,
            max_gap=max_gap,
            img_size=img_size,
            sample_decay=sample_decay,
            loss_decay=loss_decay,
            samples_per_video=50,
            train=train,
            max_videos=max_videos,
            cache_frames=cache_frames,
        )

        print(f"BridgeDataset: {len(self.frame_dirs)} trajectories")

    def _scan_frame_dirs(self) -> List[Path]:
        """traj_*/ 디렉토리 탐색."""
        return sorted(
            d for d in self.data_root.glob("traj_*") if d.is_dir()
        )
