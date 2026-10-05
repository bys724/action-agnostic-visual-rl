"""CALVIN dataset loader — segment-based sampling (§C10 action probing).

CALVIN (Mees et al. 2021) tabletop language-conditioned manipulation.
- 4 splits: A, B, C, D (task_ABCD_D 다운로드 시 모두 포함)
- Per-frame .npz: episode_XXXXXXX.npz
- Per-segment language annotation: lang_annotations/auto_lang_ann.npy

**Segment-based sampling (2026-05-26 update)**:
이전 episode-level sampling은 multi-task chain 안에서 random pair 추출 → frame
약 35-40%가 task 외 (idle/transition) → motion encoding이 anchor 잡기 어려움.
LIBERO/EgoDex와 fair 비교 위해 **단일 task segment 안 frame pair만** sampling.

Annotation 구조 (lang_annotations/auto_lang_ann.npy):
- d['info']['indx']: list of (start_frame, end_frame) — 각 task segment 경계
- d['language']['task']: list of task name str (34종)
- d['language']['ann']: list of natural language instruction str
- segment 길이: min 34, max 65 frame (1.13~2.17s @ 30Hz)

**Target (pose-derived, LIBERO/EgoDex와 동일)**:
- target_pos    = robot_obs[t+k, :3] - robot_obs[t, :3]
- target_rotvec = (R.from_euler(ee_euler[t]).inv() * R.from_euler(ee_euler[t+k])).as_rotvec()
- target_gripper = actions[t+k-1, 6:7]
→ 7-DoF
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
        list of (start_frame, end_frame, task_name) — 모든 segment.
        validation: ~1k segments, training: ~23k segments.
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


def load_scene_ranges(split_dir: Path) -> dict:
    """training/scene_info.npy → {"A": (start, end), ...} (프레임 id 구간, 양끝 포함).

    task_ABCD_D: training = A·B·C·D 네 환경(탁자 텍스처·고정 물체 배치 상이), validation = D
    (CALVIN 공식 split 정의 — validation/엔 scene_info.npy 없음).
    """
    d = np.load(split_dir / "scene_info.npy", allow_pickle=True).item()
    return {k.replace("calvin_scene_", ""): (int(a), int(b)) for k, (a, b) in d.items()}


def filter_segments_by_scene(segments, scene_ranges: dict, scenes) -> list:
    """세그먼트 (start, end)가 지정 환경 구간 안에 통째로 들어가는 것만 (factor_shift_plan §2)."""
    keep = [scene_ranges[sc] for sc in scenes]
    return [seg for seg in segments if any(a <= seg[0] and seg[1] <= b for a, b in keep)]


def load_frame(split_dir: Path, frame_id: int,
               keys: Tuple[str, ...] = ("rgb_static", "robot_obs", "actions")) -> dict:
    """Load single frame .npz. keys filter: 메모리 절약."""
    path = split_dir / FRAME_FILENAME_FMT.format(frame_id)
    with np.load(path) as f:
        return {k: np.asarray(f[k]) for k in keys}


def load_segment_frames(
    split_dir: Path,
    start_id: int,
    end_id: int,
    view: str = "rgb_static",
    with_scene: bool = False,
):
    """Load all frames in a task segment.

    Segment 길이 max 65 frame이라 stride 불필요 (메모리 OK).

    Returns:
        frames:    (T, H, W, 3) uint8       where T = end_id - start_id + 1
        robot_obs: (T, 15) float32          — TCP pose + gripper + joints
        actions:   (T, 7) float32           — abs TCP pose + binary gripper
        scene_obs: (T, 24) float32          — with_scene=True일 때만 4번째로 반환
    """
    keys = (view, "robot_obs", "actions") + (("scene_obs",) if with_scene else ())
    cols = {k: [] for k in keys}
    for fid in range(start_id, end_id + 1):
        d = load_frame(split_dir, fid, keys=keys)
        for k in keys:
            cols[k].append(d[k])
    out = (
        np.stack(cols[view]),
        np.stack(cols["robot_obs"], dtype=np.float32),
        np.stack(cols["actions"], dtype=np.float32),
    )
    return out + (np.stack(cols["scene_obs"], dtype=np.float32),) if with_scene else out


# scene_obs 24-d = 서랍·슬라이더·버튼·스위치·전구·초록불(6) + 빨강/파랑/분홍 블록 각 (xyz, euler) 6.
# 블록 xyz 인덱스 = 6:9 / 12:15 / 18:21 (10-04 확인: lift_blue_block 세그먼트에서 12:15만 변화, z≈0.46 탁자면)
SCENE_BLOCK_POS = np.r_[6:9, 12:15, 18:21]


def calvin_scene_target(scene_obs: np.ndarray, t: int) -> np.ndarray:
    """프레임 t의 블록 3개 위치 9-d (factor_shift_plan §3-a, P 외형 판독 타깃)."""
    return scene_obs[t, SCENE_BLOCK_POS].astype(np.float32)


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
        t, k: frame indices within segment

    Returns:
        7-DoF: pos Δ(3, m) + rotvec Δ(3, rad) + gripper binary(1)
    """
    if t + k >= robot_obs.shape[0]:
        raise ValueError(f"t+k={t+k} exceeds segment length {robot_obs.shape[0]}")
    target_pos = robot_obs[t + k, :3] - robot_obs[t, :3]
    r1 = R.from_euler("xyz", robot_obs[t, 3:6])
    r2 = R.from_euler("xyz", robot_obs[t + k, 3:6])
    target_rotvec = (r1.inv() * r2).as_rotvec()
    target_gripper = actions[t + k - 1, 6:7]
    return np.concatenate([target_pos, target_rotvec, target_gripper], dtype=np.float32)
