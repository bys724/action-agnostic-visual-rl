"""
LIBERO 시연 프레임 쌍 데이터셋 — alpha §4-h (C1-DN 의 LIBERO 도메인 추가학습).

- 소스 = 공식 LIBERO hdf5 (`data/demo_N/obs/{agentview_rgb, eye_in_hand_rgb}` · 128×128 uint8) 를
  **한 번 추출한 원시 uint8 파일**(카메라별 `frames_<key>_<cam>.u8` · 프레임 연속 · 색인 json).
  hdf5 핸들을 워커에 유지하면 시연 그룹 메타데이터 캐시가 샘플당 ≈ 0.55 MB 씩 무한 증가 →
  OOM (42638341 · 10-26). 매번 열고 닫기는 메모리는 평평하나 ≈ 45 ms/샘플(GPFS open 비용).
  → 원시 파일 + os.pread (mmap 매핑 없음 · 워커 힙 증가 없음).
- 전처리 = alpha 평가(`alpha_readout.py:ValueData._x`)와 동일: [0,1] raw · bilinear 224 (align_corners=False)
  · ImageNet 정규화 없음 · crop 없음 (train↔inference parity · eval_protocols §0).
- 샘플 = (카메라, 에피소드, t) 전 프레임 균등 · 간격 gap ~ sample_probs (1..max_gap) · 끝을 넘으면 t 재추출.
  반환 형식 = EgoDex pair 모드와 같음 (img_t, img_tk, gap).

추출 (컴퓨트 노드 · 1 회):
    python -m src.datasets.libero --root /proj/external_group/mrg/datasets/libero \
        --suites libero_object,libero_spatial,libero_10,libero_90
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CAMS = {"agentview": "agentview_rgb", "wrist": "eye_in_hand_rgb"}
HW = 128
FRAME_BYTES = HW * HW * 3


def _key(suites: List[str]) -> str:
    return "_".join(suites)


def extract(root: Path, suites: List[str]) -> None:
    """[(hdf5 경로, demo 키, T), ...] 색인 + 카메라별 원시 프레임 파일. 순서 = suite · 파일 이름 · demo 번호."""
    import h5py
    key = _key(suites)
    rows, outs = [], {c: open(root / f"frames_{key}_{c}.u8.tmp", "wb") for c in CAMS}
    for s in suites:
        for p in sorted((root / s).glob("*.hdf5")):
            with h5py.File(p, "r") as f:
                for k in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1])):
                    obs = f["data"][k]["obs"]
                    T = int(obs[CAMS["agentview"]].shape[0])
                    for c, h in CAMS.items():
                        x = obs[h][:]
                        assert x.shape == (T, HW, HW, 3) and x.dtype == np.uint8, (p, k, c, x.shape)
                        outs[c].write(x.tobytes())
                    rows.append([str(p), k, T])
            print(f"{s}/{p.name}: {len(rows)} demos", flush=True)
    for c, fh in outs.items():
        fh.close()
        os.replace(root / f"frames_{key}_{c}.u8.tmp", root / f"frames_{key}_{c}.u8")
    (root / f"frames_{key}_index.json").write_text(json.dumps(rows))
    print(f"done: {len(rows)} demos, {sum(r[2] for r in rows)} frames/cam")


class LiberoPairDataset(Dataset):
    def __init__(self, data_root: str, suites: List[str], max_gap: int = 20, img_size: int = 224,
                 sample_probs: Optional[np.ndarray] = None):
        root, key = Path(data_root), _key(suites)
        idx = root / f"frames_{key}_index.json"
        assert idx.exists(), f"추출 먼저: python -m src.datasets.libero --root {root} --suites {','.join(suites)}"
        self.rows = json.loads(idx.read_text())
        self.T = np.array([r[2] for r in self.rows])
        self.start = np.concatenate([[0], np.cumsum(self.T)[:-1]])
        self.n_frames = int(self.T.sum())
        self.paths = [str(root / f"frames_{key}_{c}.u8") for c in CAMS]
        for p in self.paths:
            assert os.path.getsize(p) == self.n_frames * FRAME_BYTES, p
        self.max_gap, self.img_size = max_gap, img_size
        self.sample_probs = (np.full(max_gap, 1.0 / max_gap) if sample_probs is None else sample_probs)
        self._fd = None  # 워커별 fd (fork 뒤 lazy open)
        print(f"LiberoPairDataset: suites={suites} episodes={len(self.rows)} frames/cam={self.n_frames} "
              f"cams={len(CAMS)} max_gap={max_gap}")

    def __len__(self) -> int:
        return self.n_frames * len(CAMS)

    def _read(self, cam: int, g: int) -> np.ndarray:
        if self._fd is None:
            self._fd = [os.open(p, os.O_RDONLY) for p in self.paths]
        b = os.pread(self._fd[cam], FRAME_BYTES, g * FRAME_BYTES)
        return np.frombuffer(b, np.uint8).reshape(HW, HW, 3)

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
        s = int(self.start[ep])
        img = self._prep(np.stack([self._read(cam, s + t), self._read(cam, s + t + gap)]))
        return img[0], img[1], gap


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/proj/external_group/mrg/datasets/libero")
    ap.add_argument("--suites", default="libero_object,libero_spatial,libero_10,libero_90")
    a = ap.parse_args()
    extract(Path(a.root), [s.strip() for s in a.suites.split(",")])
