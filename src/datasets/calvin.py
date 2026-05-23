"""CALVIN dataset loader (§C10 action probing).

CALVIN (Mees et al. 2021) tabletop language-conditioned manipulation.
- 4 splits: A, B, C, D (task_ABCD_D 다운로드 시 모두 포함)
- Per-frame .npz: episode_XXXXXXX.npz
- Episode boundaries: ep_start_end_ids.npy (N, 2) [start_id, end_id) per episode
- 30Hz continuous control

Per-frame .npz keys (CALVIN dataset/README.md):
- rgb_static:  (200, 200, 3) uint8       — paper main view
- rgb_gripper: (84, 84, 3) uint8         — wrist view (sub-analysis only)
- robot_obs:   (15,) float32             — [ee_pos(3), ee_euler(3), gripper_width(1),
                                              joint_pos(7), gripper_action(1)]
- actions:     (7,) float32              — abs TCP pose (x,y,z,euler_x/y/z,gripper)
- rel_actions: (7,) float32              — relative, normalized [-1, 1]

**Target (pose-derived, LIBERO/EgoDex와 동일 protocol)**:
- target_pos    = ee_pos[t+k] - ee_pos[t]                                       (3,)
- target_rotvec = (R.from_euler(ee_euler[t]).inv() * R.from_euler(ee_euler[t+k])).as_rotvec()  (3,)
- target_gripper = actions[t+k-1, 6:7]                                          (1,)
→ 7-DoF target, scale 단위 m + rad
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


FRAME_FILENAME_FMT = "episode_{:07d}.npz"
EP_INDEX_FILE = "ep_start_end_ids.npy"


def load_episode_index(split_dir: Path) -> np.ndarray:
    """Load (N, 2) episode boundary array [start_id, end_id]."""
    path = split_dir / EP_INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing CALVIN episode index: {path}")
    return np.load(path)


def load_frame(split_dir: Path, frame_id: int,
               keys: Tuple[str, ...] = ("rgb_static", "robot_obs", "actions")) -> dict:
    """Load single frame .npz. keys filter: 메모리 절약."""
    path = split_dir / FRAME_FILENAME_FMT.format(frame_id)
    with np.load(path) as f:
        return {k: np.asarray(f[k]) for k in keys}


def load_episode_frames(
    split_dir: Path,
    start_id: int,
    end_id: int,
    view: str = "rgb_static",
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load frames + robot_obs + actions in an episode (with optional stride).

    CALVIN episode 길이가 매우 큼 (validation 평균 24k frames). stride>1 적용
    필수 — 메모리 폭증 방지. stride=10 → CALVIN 30Hz가 effective 3Hz로 sub-sample.

    Returns:
        frames:    (T_eff, H, W, 3) uint8     where T_eff = ceil((end_id-start_id+1) / stride)
        robot_obs: (T_eff, 15) float32        — TCP pose + gripper + joints
        actions:   (T_eff, 7) float32         — abs TCP pose actions (마지막 dim = gripper binary)
    """
    frames, robot_obs, actions = [], [], []
    for fid in range(start_id, end_id + 1, stride):
        d = load_frame(split_dir, fid, keys=(view, "robot_obs", "actions"))
        frames.append(d[view])
        robot_obs.append(d["robot_obs"])
        actions.append(d["actions"])
    return (
        np.stack(frames),
        np.stack(robot_obs, dtype=np.float32),
        np.stack(actions, dtype=np.float32),
    )


def list_episodes(split_dir: Path) -> List[Tuple[int, int]]:
    """Return list of (start_id, end_id) tuples."""
    ep_arr = load_episode_index(split_dir)
    return [(int(s), int(e)) for s, e in ep_arr]


def calvin_action_target(
    robot_obs: np.ndarray,
    actions: np.ndarray,
    t: int,
    k: int,
) -> np.ndarray:
    """Pose-derived 7-DoF target (LIBERO/EgoDex 일관성).

    Args:
        robot_obs: (T, 15) — ee_pos[:3], ee_euler[3:6], ...
        actions:   (T, 7)  — last dim = gripper binary (-1/+1)
        t, k: frame indices in strided-space

    Returns:
        7-DoF: pos Δ(3, meters) + rotvec Δ(3, rad) + gripper binary(1)
    """
    if t + k >= robot_obs.shape[0]:
        raise ValueError(f"t+k={t+k} exceeds episode length {robot_obs.shape[0]}")
    target_pos = robot_obs[t + k, :3] - robot_obs[t, :3]
    r1 = R.from_euler("xyz", robot_obs[t, 3:6])
    r2 = R.from_euler("xyz", robot_obs[t + k, 3:6])
    target_rotvec = (r1.inv() * r2).as_rotvec()
    target_gripper = actions[t + k - 1, 6:7]
    return np.concatenate([target_pos, target_rotvec, target_gripper], dtype=np.float32)
