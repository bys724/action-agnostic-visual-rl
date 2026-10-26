"""
LIBERO 시연 프레임 쌍 데이터셋 — alpha §4-h (C1-DN 의 LIBERO 도메인 추가학습).

- 소스 = 공식 LIBERO hdf5 (`data/demo_N/obs/{agentview_rgb, eye_in_hand_rgb}` · 128×128 uint8 · 비압축).
- 전처리 = alpha 평가(`alpha_readout.py:ValueData._x`)와 동일: [0,1] raw · bilinear 224 (align_corners=False)
  · ImageNet 정규화 없음 · crop 없음 (train↔inference parity · eval_protocols §0).
- 샘플 = (카메라, 에피소드, t) 전 프레임 균등 · 간격 gap ~ sample_probs (1..max_gap) · 끝을 넘으면 t 재추출.
  반환 형식 = EgoDex pair 모드와 같음 (img_t, img_tk, gap).
- 에피소드 길이 색인은 GPFS 메타데이터 스캔이 느려 json 으로 캐시 (`index_cache`).
"""

import json
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CAM_KEYS = ("agentview_rgb", "eye_in_hand_rgb")


def build_index(root: Path, suites: List[str]) -> List[list]:
    """[(hdf5 경로, demo 키, T), ...] — suite 순 · 파일 이름 순 · demo 번호 순."""
    rows = []
    for s in suites:
        for p in sorted((root / s).glob("*.hdf5")):
            with h5py.File(p, "r") as f:
                keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                rows += [[str(p), k, int(f["data"][k]["actions"].shape[0])] for k in keys]
    return rows


class LiberoPairDataset(Dataset):
    def __init__(self, data_root: str, suites: List[str], max_gap: int = 20, img_size: int = 224,
                 sample_probs: Optional[np.ndarray] = None, index_cache: Optional[str] = None):
        root = Path(data_root)
        cache = Path(index_cache) if index_cache else root / f"pair_index_{'_'.join(suites)}.json"
        if cache.exists():
            rows = json.loads(cache.read_text())
        else:
            rows = build_index(root, suites)
            cache.write_text(json.dumps(rows))
        self.rows = rows
        self.T = np.array([r[2] for r in rows])
        self.start = np.concatenate([[0], np.cumsum(self.T)[:-1]])
        self.n_frames = int(self.T.sum())
        self.max_gap, self.img_size = max_gap, img_size
        self.sample_probs = (np.full(max_gap, 1.0 / max_gap) if sample_probs is None else sample_probs)
        self._h5 = {}  # 워커별 파일 핸들 (fork 뒤 lazy open)
        print(f"LiberoPairDataset: suites={suites} episodes={len(rows)} frames/cam={self.n_frames} "
              f"cams={len(CAM_KEYS)} max_gap={max_gap}")

    def __len__(self) -> int:
        return self.n_frames * len(CAM_KEYS)

    def _obs(self, i):
        path, key, _ = self.rows[i]
        if path not in self._h5:
            self._h5[path] = h5py.File(path, "r")
        return self._h5[path]["data"][key]["obs"]

    def _prep(self, x: np.ndarray) -> torch.Tensor:  # (2,128,128,3) uint8 → (2,3,224,224) [0,1]
        t = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)
        return F.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

    def __getitem__(self, idx: int):
        cam, g = divmod(idx, self.n_frames)
        ep = int(np.searchsorted(self.start, g, side="right") - 1)
        T = int(self.T[ep])
        gap = int(np.random.choice(np.arange(1, self.max_gap + 1), p=self.sample_probs))
        gap = min(gap, T - 1)
        t = g - int(self.start[ep])
        if t + gap >= T:
            t = np.random.randint(0, T - gap)
        x = self._obs(ep)[CAM_KEYS[cam]][[t, t + gap]]
        img = self._prep(x)
        return img[0], img[1], gap
