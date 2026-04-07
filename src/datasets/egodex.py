"""
EgoDex Dataset.

프레임 구조:
  frames_root/
    task_name/
      video_name/
        frame_000000.jpg, frame_000001.jpg, ...

프레임은 센터크롭(1080x1080) → 256x256으로 사전 추출됨.
"""

from pathlib import Path
from typing import List, Optional

from .base import VideoFrameDataset


class EgoDexDataset(VideoFrameDataset):
    """EgoDex 프레임 데이터셋. task/video 2단계 디렉토리 구조."""

    def __init__(
        self,
        data_root: str,
        split: str = "test",
        max_gap: int = 30,
        img_size: int = 224,
        sample_decay: float = 0.0,
        loss_decay: float = 0.0,
        max_videos: Optional[int] = None,
        cache_frames: bool = False,
        train: bool = True,
        composition: bool = False,
        sample_dist: str = "auto",
        sample_center: int = None,
    ):
        self.split = split
        super().__init__(
            data_root=data_root,
            max_gap=max_gap,
            img_size=img_size,
            sample_decay=sample_decay,
            loss_decay=loss_decay,
            samples_per_video=100,
            train=train,
            max_videos=max_videos,
            cache_frames=cache_frames,
            composition=composition,
            sample_dist=sample_dist,
            sample_center=sample_center,
        )

        print(f"EgoDexDataset: {len(self.frame_dirs)} videos, split={split}")

    def _scan_frame_dirs(self) -> List[Path]:
        """task_name/video_name/ 구조 탐색.

        data_root 구조에 따라 split 처리:
        - data_root/task_name/video_name/ → split 무시 (단일 split 디렉토리를 직접 지정한 경우)
        - data_root/split/task_name/video_name/ → split 하위에서 탐색
        """
        # split 하위 디렉토리가 있으면 거기서 탐색
        split_dir = self.data_root / self.split
        scan_root = split_dir if split_dir.is_dir() else self.data_root

        dirs = []
        for task_dir in sorted(scan_root.glob("*")):
            if not task_dir.is_dir():
                continue
            for video_dir in sorted(task_dir.glob("*")):
                if video_dir.is_dir():
                    dirs.append(video_dir)
        return dirs
