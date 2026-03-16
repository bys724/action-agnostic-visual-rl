#!/usr/bin/env python3
"""
DROID 데이터셋(TFRecord/RLDS)에서 프레임 추출.

카메라 3개 모두 추출:
  - exterior_image_1_left (외부 카메라 1)
  - exterior_image_2_left (외부 카메라 2)
  - wrist_image_left (손목 카메라)

원본 180x320 (16:9) → 256x256 리사이즈 (crop 없이 전체 장면 보존).

출력 구조:
  output_dir/
    ext1/
      ep_000000/
        frame_000000.jpg, frame_000001.jpg, ...
    ext2/
      ep_000000/
        frame_000000.jpg, frame_000001.jpg, ...
    wrist/
      ep_000000/
        frame_000000.jpg, frame_000001.jpg, ...
"""

import argparse
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np


# 카메라 이름 매핑
CAMERAS = {
    "ext1": "exterior_image_1_left",
    "ext2": "exterior_image_2_left",
    "wrist": "wrist_image_left",
}


def extract_episode(ep_idx, episode, output_dir, img_size):
    """단일 에피소드에서 모든 카메라의 프레임 추출."""
    ep_name = f"ep_{ep_idx:06d}"

    # 이미 추출된 에피소드 스킵 (ext1 기준으로 체크)
    check_dir = output_dir / "ext1" / ep_name
    if check_dir.exists() and len(list(check_dir.glob("*.jpg"))) > 0:
        return ep_idx, True, "already exists", 0

    frame_count = 0
    for cam_short, cam_key in CAMERAS.items():
        ep_dir = output_dir / cam_short / ep_name
        ep_dir.mkdir(parents=True, exist_ok=True)

    for step_idx, step in enumerate(episode["steps"]):
        obs = step["observation"]
        for cam_short, cam_key in CAMERAS.items():
            img_rgb = obs[cam_key].numpy()  # (180, 320, 3) RGB
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, (img_size, img_size))

            frame_path = output_dir / cam_short / ep_name / f"frame_{step_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        frame_count += 1

    return ep_idx, True, f"{frame_count} frames", frame_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from DROID TFRecord dataset")
    parser.add_argument("--droid-dir", type=str, default="/mnt/data/droid/1.0.1",
                        help="DROID RLDS directory (default: /mnt/data/droid/1.0.1)")
    parser.add_argument("--output-dir", type=str, default="/mnt/data/droid_frames",
                        help="Output directory for frames")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Output image size (default: 256)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start episode index (for resume)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max episodes to process (default: all)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt")

    args = parser.parse_args()

    # Lazy import (느린 TF 초기화를 argparse 후에)
    import tensorflow_datasets as tfds
    from pathlib import Path

    droid_dir = args.droid_dir
    output_dir = Path(args.output_dir)

    # 데이터셋 로드
    print(f"Loading DROID from {droid_dir}...")
    builder = tfds.builder_from_directory(droid_dir)
    ds = builder.as_dataset(split="train")

    # 에피소드 수 확인
    info = builder.info
    total_episodes = info.splits["train"].num_examples
    print(f"Total episodes: {total_episodes:,}")
    print(f"Output: {output_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Cameras: {', '.join(CAMERAS.keys())}")
    print(f"Mode: resize (no crop)")

    if args.start > 0:
        print(f"Starting from episode {args.start}")
    if args.limit:
        print(f"Limit: {args.limit} episodes")

    if not args.yes:
        response = input(f"\nExtract frames? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled.")
            sys.exit(0)

    # 출력 디렉토리 생성
    for cam in CAMERAS:
        (output_dir / cam).mkdir(parents=True, exist_ok=True)

    # 추출
    success_count = 0
    fail_count = 0
    total_frames = 0
    start_time = time.time()

    end_idx = args.start + args.limit if args.limit else total_episodes

    for ep_idx, episode in enumerate(ds):
        if ep_idx < args.start:
            continue
        if ep_idx >= end_idx:
            break

        try:
            idx, ok, msg, frames = extract_episode(ep_idx, episode, output_dir, args.img_size)
            if ok:
                success_count += 1
                total_frames += frames
            else:
                fail_count += 1
                print(f"  FAILED ep_{ep_idx:06d}: {msg}")
        except Exception as e:
            fail_count += 1
            print(f"  ERROR ep_{ep_idx:06d}: {e}")

        # 진행 상황 출력 (100개마다)
        done = success_count + fail_count
        if done % 100 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = done / elapsed if elapsed > 0 else 0
            remaining = (end_idx - args.start - done) / eps_per_sec if eps_per_sec > 0 else 0
            print(f"  [{done:,}/{end_idx - args.start:,}] "
                  f"{eps_per_sec:.1f} ep/s, "
                  f"~{remaining/3600:.1f}h remaining, "
                  f"frames={total_frames:,}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Extraction complete! ({elapsed/3600:.1f}h)")
    print(f"{'='*60}")
    print(f"Success: {success_count:,}")
    print(f"Failed: {fail_count:,}")
    print(f"Total frames: {total_frames:,}")
    print(f"Output: {output_dir}")

    # 디스크 사용량
    import subprocess
    result = subprocess.run(["du", "-sh", str(output_dir)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Total size: {result.stdout.split()[0]}")


if __name__ == "__main__":
    main()
