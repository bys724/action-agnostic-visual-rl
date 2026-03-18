"""
DROID Dataset.

프레임 구조 (사전 추출 후):
  frames_root/
    ext1/
      ep_XXXXXX/
        frame_000000.jpg, frame_000001.jpg, ...
    ext2/
      ep_XXXXXX/
        frame_000000.jpg, ...
    wrist/
      ep_XXXXXX/
        frame_000000.jpg, ...

프레임은 리사이즈(180x320 → 256x256, crop 없음)로 사전 추출됨.
카메라별로 독립된 데이터셋으로 사용하거나, 여러 카메라를 합쳐서 사용 가능.
"""

from pathlib import Path
from typing import List, Optional

from .base import VideoFrameDataset


class DROIDDataset(VideoFrameDataset):
    """DROID 프레임 데이터셋. cam/ep_XXXXXX 2단계 디렉토리 구조."""

    def __init__(
        self,
        data_root: str,
        cameras: Optional[List[str]] = None,
        max_gap: int = 10,
        img_size: int = 224,
        sample_decay: float = 0.3,
        loss_decay: float = 0.7,
        max_videos: Optional[int] = None,
        cache_frames: bool = False,
        train: bool = True,
    ):
        self.cameras = cameras or ["ext1", "ext2"]
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

        print(f"DROIDDataset: {len(self.frame_dirs)} episodes, cameras={self.cameras}")

    def _scan_frame_dirs(self) -> List[Path]:
        """cam/ep_*/ 디렉토리 탐색. 지정된 카메라에서 에피소드 수집."""
        dirs = []
        for cam in self.cameras:
            cam_dir = self.data_root / cam
            if cam_dir.is_dir():
                dirs.extend(sorted(d for d in cam_dir.glob("ep_*") if d.is_dir()))
        return dirs
