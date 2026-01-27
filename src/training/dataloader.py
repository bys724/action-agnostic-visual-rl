"""
데이터셋 로더

BridgeData V2와 DROID 데이터셋을 PyTorch DataLoader로 제공
"""

import os
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class BridgeDataset(Dataset):
    """
    BridgeData V2 데이터셋 로더

    데이터 구조 (압축 해제 후):
    demos_8_17/
    ├── {scene_name}/
    │   ├── {trajectory_id}/
    │   │   ├── lang.txt
    │   │   ├── obs_dict.pkl
    │   │   └── images{camera_id}/
    │   │       ├── im_{step}.jpg
    │   │       └── ...
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        max_seq_len: int = 100,
        transform=None
    ):
        """
        Args:
            data_dir: BridgeData V2 루트 디렉토리
            split: train/val/test
            image_size: 이미지 리사이즈 크기
            max_seq_len: 최대 시퀀스 길이
            transform: 이미지 변환
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.transform = transform

        # Trajectory 목록 로드
        self.trajectories = self._load_trajectory_list()

    def _load_trajectory_list(self) -> List[Path]:
        """모든 trajectory 경로 수집"""
        trajectories = []

        # demos 디렉토리 탐색
        demos_dir = self.data_dir / "demos_8_17"
        if not demos_dir.exists():
            # 압축 해제 전이면 빈 리스트 반환
            print(f"Warning: {demos_dir} not found. Extract the zip file first.")
            return trajectories

        for scene_dir in demos_dir.iterdir():
            if scene_dir.is_dir():
                for traj_dir in scene_dir.iterdir():
                    if traj_dir.is_dir() and (traj_dir / "obs_dict.pkl").exists():
                        trajectories.append(traj_dir)

        print(f"Found {len(trajectories)} trajectories")
        return trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_path = self.trajectories[idx]

        # Language instruction 로드
        lang_file = traj_path / "lang.txt"
        if lang_file.exists():
            with open(lang_file, 'r') as f:
                instruction = f.read().strip()
        else:
            instruction = ""

        # Observation dict 로드
        obs_file = traj_path / "obs_dict.pkl"
        with open(obs_file, 'rb') as f:
            obs_dict = pickle.load(f)

        # 이미지 로드
        images = []
        image_dir = traj_path / "images0"  # 기본 카메라
        if image_dir.exists():
            image_files = sorted(image_dir.glob("im_*.jpg"))[:self.max_seq_len]
            for img_file in image_files:
                img = Image.open(img_file).convert("RGB")
                img = img.resize(self.image_size)
                if self.transform:
                    img = self.transform(img)
                else:
                    img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                images.append(img)

        if images:
            images = torch.stack(images)
        else:
            images = torch.zeros(1, 3, *self.image_size)

        # Actions 추출
        actions = obs_dict.get("actions", np.zeros((len(images), 7)))
        actions = torch.tensor(actions[:len(images)], dtype=torch.float32)

        return {
            "images": images,
            "actions": actions,
            "instruction": instruction,
            "trajectory_path": str(traj_path)
        }


class TrajectoryDataset(Dataset):
    """
    수집된 trajectory 데이터셋 로더 (collect_trajectories.py 출력용)

    데이터 구조 (.pkl):
    {
        "observations": [...],
        "actions": [...],
        "rewards": [...],
        "instruction": "...",
        "success": True/False
    }
    """

    def __init__(
        self,
        data_dir: str,
        success_only: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.success_only = success_only
        self.image_size = image_size
        self.transform = transform

        self.trajectories = self._load_trajectories()

    def _load_trajectories(self) -> List[Dict]:
        """모든 trajectory 파일 로드"""
        trajectories = []

        for pkl_file in self.data_dir.glob("*.pkl"):
            with open(pkl_file, 'rb') as f:
                traj = pickle.load(f)
                if self.success_only and not traj.get("success", False):
                    continue
                traj["file_path"] = str(pkl_file)
                trajectories.append(traj)

        print(f"Loaded {len(trajectories)} trajectories")
        return trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]

        # 이미지 처리
        images = []
        for obs in traj["observations"]:
            if isinstance(obs, np.ndarray):
                img = Image.fromarray(obs)
                img = img.resize(self.image_size)
                if self.transform:
                    img = self.transform(img)
                else:
                    img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                images.append(img)

        images = torch.stack(images) if images else torch.zeros(1, 3, *self.image_size)
        actions = torch.tensor(traj["actions"], dtype=torch.float32)

        return {
            "images": images,
            "actions": actions,
            "instruction": traj.get("instruction", ""),
            "rewards": torch.tensor(traj.get("rewards", []), dtype=torch.float32)
        }


def create_dataloader(
    dataset_type: str,
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    데이터로더 생성 헬퍼

    Args:
        dataset_type: "bridge" 또는 "trajectory"
        data_dir: 데이터 디렉토리
        batch_size: 배치 크기
        num_workers: 워커 수
        **kwargs: 데이터셋 추가 인자
    """
    if dataset_type == "bridge":
        dataset = BridgeDataset(data_dir, **kwargs)
    elif dataset_type == "trajectory":
        dataset = TrajectoryDataset(data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=variable_length_collate
    )


def variable_length_collate(batch: List[Dict]) -> Dict[str, Any]:
    """가변 길이 시퀀스를 위한 collate 함수"""
    # 최대 시퀀스 길이 찾기
    max_len = max(item["images"].shape[0] for item in batch)

    # 패딩
    padded_images = []
    padded_actions = []
    masks = []

    for item in batch:
        seq_len = item["images"].shape[0]
        pad_len = max_len - seq_len

        # 이미지 패딩
        if pad_len > 0:
            pad_imgs = torch.zeros(pad_len, *item["images"].shape[1:])
            padded_images.append(torch.cat([item["images"], pad_imgs], dim=0))
            pad_acts = torch.zeros(pad_len, item["actions"].shape[-1])
            padded_actions.append(torch.cat([item["actions"], pad_acts], dim=0))
        else:
            padded_images.append(item["images"])
            padded_actions.append(item["actions"])

        # 마스크 생성
        mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        masks.append(mask)

    return {
        "images": torch.stack(padded_images),
        "actions": torch.stack(padded_actions),
        "mask": torch.stack(masks),
        "instructions": [item["instruction"] for item in batch]
    }
