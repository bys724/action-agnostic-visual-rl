"""CALVIN dataset loader -- segment-based sampling for action probing.

CALVIN (Mees et al. 2021): tabletop, language-conditioned manipulation.
- 4 environment splits: A, B, C, D (all included in the task_ABCD_D download).
- Per-frame archives: episode_XXXXXXX.npz
- Per-segment language annotation: lang_annotations/auto_lang_ann.npy

Segment-based sampling:
    Episode-level sampling draws random frame pairs across a multi-task chain, so
    roughly 35-40% of pairs fall outside any single task (idle / transition frames),
    which makes motion hard to anchor. To match the other manipulation benchmarks
    fairly, we sample frame pairs only within a single annotated task segment.

Annotation layout (lang_annotations/auto_lang_ann.npy):
    - d['info']['indx']:     list of (start_frame, end_frame) -- task segment bounds
    - d['language']['task']: list of task-name strings (34 classes)
    - d['language']['ann']:  list of natural-language instruction strings
    - segment length: 34-65 frames (1.13-2.17 s @ 30 Hz)

Target (pose-derived, 7-DoF; identical convention across the manipulation benchmarks):
    - target_pos     = robot_obs[t+k, :3] - robot_obs[t, :3]
    - target_rotvec  = (R.from_euler(ee_euler[t]).inv() * R.from_euler(ee_euler[t+k])).as_rotvec()
    - target_gripper = actions[t+k-1, 6:7]
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


FRAME_FILENAME_FMT = "episode_{:07d}.npz"
ANN_PATH_RELATIVE = "lang_annotations/auto_lang_ann.npy"


def load_segments(split_dir: Path) -> List[Tuple[int, int, str]]:
    """Load language-annotated task segments.

    Returns:
        list of (start_frame, end_frame, task_name) for every segment.
        (validation ~1k segments, training ~23k segments)
    """
    ann_path = split_dir / ANN_PATH_RELATIVE
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing CALVIN language annotation: {ann_path}")
    d = np.load(ann_path, allow_pickle=True).item()
    indx = d["info"]["indx"]            # list of (start, end)
    tasks = d["language"]["task"]       # list of str
    out = []
    for (s, e), task in zip(indx, tasks):
        out.append((int(s), int(e), str(task)))
    return out


def load_frame(split_dir: Path, frame_id: int,
               keys: Tuple[str, ...] = ("rgb_static", "robot_obs", "actions")) -> dict:
    """Load a single frame archive; `keys` filters the returned fields to save memory."""
    path = split_dir / FRAME_FILENAME_FMT.format(frame_id)
    with np.load(path) as f:
        return {k: np.asarray(f[k]) for k in keys}


def load_segment_frames(
    split_dir: Path,
    start_id: int,
    end_id: int,
    view: str = "rgb_static",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all frames in a task segment.

    Segments are at most ~65 frames, so no striding is needed (fits in memory).

    Returns:
        frames:    (T, H, W, 3) uint8       where T = end_id - start_id + 1
        robot_obs: (T, 15) float32          -- TCP pose + gripper + joints
        actions:   (T, 7) float32           -- abs TCP pose + binary gripper
    """
    frames, robot_obs, actions = [], [], []
    for fid in range(start_id, end_id + 1):
        d = load_frame(split_dir, fid, keys=(view, "robot_obs", "actions"))
        frames.append(d[view])
        robot_obs.append(d["robot_obs"])
        actions.append(d["actions"])
    return (
        np.stack(frames),
        np.stack(robot_obs, dtype=np.float32),
        np.stack(actions, dtype=np.float32),
    )


def calvin_action_target(
    robot_obs: np.ndarray,
    actions: np.ndarray,
    t: int,
    k: int,
) -> np.ndarray:
    """Pose-derived 7-DoF target (same convention as the other manipulation benchmarks).

    Args:
        robot_obs: (T, 15) -- ee_pos[:3], ee_euler[3:6], ...
        actions:   (T, 7)  -- last dim = gripper binary (-1/+1)
        t, k: frame indices within the segment

    Returns:
        7-DoF: pos delta (3, m) + rotvec delta (3, rad) + gripper binary (1)
    """
    if t + k >= robot_obs.shape[0]:
        raise ValueError(f"t+k={t+k} exceeds segment length {robot_obs.shape[0]}")
    target_pos = robot_obs[t + k, :3] - robot_obs[t, :3]
    r1 = R.from_euler("xyz", robot_obs[t, 3:6])
    r2 = R.from_euler("xyz", robot_obs[t + k, 3:6])
    target_rotvec = (r1.inv() * r2).as_rotvec()
    target_gripper = actions[t + k - 1, 6:7]
    return np.concatenate([target_pos, target_rotvec, target_gripper], dtype=np.float32)
