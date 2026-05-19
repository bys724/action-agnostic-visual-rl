"""CALVIN dataset loader (§C10 action probing).

CALVIN (Mees et al. 2021) tabletop language-conditioned manipulation.
- 4 splits: A, B, C, D (task_ABCD_D 다운로드 시 모두 포함)
- Per-frame .npz: episode_XXXXXXX.npz
- Episode boundaries: ep_start_end_ids.npy (N, 2) [start_id, end_id) per episode
- 30Hz continuous control

Used by `scripts/eval/probe_action_calvin.py` (LIBERO probing과 동일 protocol).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


# CALVIN per-frame .npz keys (확정: dataset/README.md)
# rgb_static: (200, 200, 3) uint8       — paper main view (default)
# rgb_gripper: (84, 84, 3) uint8        — wrist view (sub-analysis only, §C12와 별개)
# actions: (7,) float32                 — abs TCP pose (x,y,z,euler_x,euler_y,euler_z,gripper)
# rel_actions: (7,) float32             — relative, normalized/clipped to (-1, 1)
# robot_obs: (15,) float32              — ee_pos(3) + ee_euler(3) + gripper_width(1) + joint(7) + flag(1)
# scene_obs: (24,) float32              — scene state (object poses 등)
FRAME_FILENAME_FMT = "episode_{:07d}.npz"
EP_INDEX_FILE = "ep_start_end_ids.npy"


def load_episode_index(split_dir: Path) -> np.ndarray:
    """Load (N, 2) episode boundary array [start_id, end_id]."""
    path = split_dir / EP_INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing CALVIN episode index: {path}")
    return np.load(path)


def load_frame(split_dir: Path, frame_id: int, keys: Tuple[str, ...] = ("rgb_static", "rel_actions")) -> dict:
    """Load single frame .npz. keys filter: 메모리 절약."""
    path = split_dir / FRAME_FILENAME_FMT.format(frame_id)
    with np.load(path) as f:
        return {k: np.asarray(f[k]) for k in keys}


def load_episode_frames(
    split_dir: Path,
    start_id: int,
    end_id: int,
    view: str = "rgb_static",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all frames in an episode.

    Returns:
        frames: (T, H, W, 3) uint8
        actions: (T, 7) float32   — rel_actions (delta, normalized -1~1)
    """
    T = end_id - start_id + 1
    frames = []
    actions = []
    for fid in range(start_id, end_id + 1):
        d = load_frame(split_dir, fid, keys=(view, "rel_actions"))
        frames.append(d[view])
        actions.append(d["rel_actions"])
    return np.stack(frames), np.stack(actions, dtype=np.float32)


def list_episodes(split_dir: Path) -> List[Tuple[int, int]]:
    """Return list of (start_id, end_id) tuples."""
    ep_arr = load_episode_index(split_dir)
    return [(int(s), int(e)) for s, e in ep_arr]


def calvin_action_target(actions: np.ndarray, t: int, k: int) -> np.ndarray:
    """gap=k action target. CALVIN rel_actions는 이미 frame-to-frame delta이므로
    [t, t+k) 구간의 누적 위치/각도 변화 + t+k-1 시점의 gripper.

    NOTE: LIBERO와 다른 점 — LIBERO는 pose-derived target (ee_pos[t+k]-ee_pos[t])이지만
    CALVIN은 robot_obs 슬라이스 가정이 위험하므로 안전하게 rel_actions 누적 사용.
    Pos/euler dims는 sum (small delta linear approx), gripper는 t+k-1 시점 값.

    Returns (7,): cumulative pos delta(3) + cumulative euler delta(3) + gripper at t+k-1(1)
    """
    if t + k > actions.shape[0]:
        raise ValueError(f"t+k={t+k} exceeds episode length {actions.shape[0]}")
    cum_pos_euler = actions[t:t+k, :6].sum(axis=0)  # (6,)
    gripper = actions[t + k - 1, 6:7]               # (1,)
    return np.concatenate([cum_pos_euler, gripper], dtype=np.float32)
