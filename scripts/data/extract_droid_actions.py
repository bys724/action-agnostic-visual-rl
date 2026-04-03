#!/usr/bin/env python3
"""
DROID TFRecord에서 에피소드별 action을 npy로 추출.

프레임 추출(extract_droid_frames.py)과 동일한 에피소드 순서:
  ep_000000, ep_000001, ... → action_000000.npy, action_000001.npy, ...

각 npy: [T, 7] — 7-DoF action (cartesian_position 6D + gripper 1D)

출력 구조:
  output_dir/
    ep_000000/action.npy  (T, 7)
    ep_000001/action.npy
    ...

Usage:
    python scripts/data/extract_droid_actions.py \
        --droid-dir /mnt/data/droid/1.0.1 \
        --output-dir /mnt/data/droid_frames/ext1 \
        --max-episodes 1000
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np


def extract_actions(droid_dir: str, output_dir: str, max_episodes: int = None):
    """TFRecord에서 action 추출하여 에피소드 디렉토리에 저장."""
    import tensorflow as tf

    droid_path = Path(droid_dir)
    output_path = Path(output_dir)

    # TFRecord shards 찾기
    shards = sorted(droid_path.glob("droid_101-train.tfrecord-*"))
    if not shards:
        print(f"No TFRecord shards found in {droid_dir}")
        sys.exit(1)
    print(f"Found {len(shards)} shards")

    ep_idx = 0
    saved = 0
    skipped = 0

    for shard_path in shards:
        ds = tf.data.TFRecordDataset(str(shard_path))

        for raw in ds:
            if max_episodes and ep_idx >= max_episodes:
                break

            ep_name = f"ep_{ep_idx:06d}"
            ep_dir = output_path / ep_name
            action_path = ep_dir / "action.npy"

            # 이미 추출된 에피소드 스킵
            if action_path.exists():
                ep_idx += 1
                skipped += 1
                continue

            example = tf.train.Example()
            example.ParseFromString(raw.numpy())
            features = example.features.feature

            # Action 추출: 7-DoF (cartesian_position 6D + gripper_position 1D)
            n_steps = len(features['steps/is_first'].int64_list.value)

            cart_pos = np.array(features['steps/action_dict/cartesian_position'].float_list.value)
            cart_pos = cart_pos.reshape(n_steps, 6)

            gripper = np.array(features['steps/action_dict/gripper_position'].float_list.value)
            gripper = gripper.reshape(n_steps, 1)

            action = np.concatenate([cart_pos, gripper], axis=1)  # [T, 7]

            # 저장
            ep_dir.mkdir(parents=True, exist_ok=True)
            np.save(action_path, action.astype(np.float32))
            saved += 1

            if saved % 1000 == 0:
                print(f"  Extracted {saved} episodes (ep_{ep_idx:06d})")

            ep_idx += 1

        if max_episodes and ep_idx >= max_episodes:
            break

    print(f"\nDone: {saved} saved, {skipped} skipped, {ep_idx} total episodes")


def main():
    parser = argparse.ArgumentParser(description="Extract DROID actions from TFRecord")
    parser.add_argument("--droid-dir", type=str, default="/mnt/data/droid/1.0.1")
    parser.add_argument("--output-dir", type=str, default="/mnt/data/droid_frames/ext1",
                        help="Same directory as extracted frames (saves action.npy per episode)")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    extract_actions(args.droid_dir, args.output_dir, args.max_episodes)


if __name__ == "__main__":
    main()
