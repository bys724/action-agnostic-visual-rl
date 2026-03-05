#!/usr/bin/env python3
"""
Bridge V2 이미지 시퀀스를 256x256으로 리사이즈하여 저장.
병렬 처리로 빠르게 변환.

Bridge V2 원본 구조:
  bridge_root/raw/bridge_data_v2/
    datacol1_toykitchen1/
      many_skills/07/2023-.../
        raw/traj_group0/traj0/images0/
          im_0.jpg, im_1.jpg, ...

출력 구조:
  output_dir/
    traj_XXXXXXXX/   (trajectory 해시로 고유 식별)
      frame_000000.jpg, frame_000001.jpg, ...

전처리 방식:
  - 비율 무시 리사이즈 (480x640 → 256x256)
  - 로봇 팔이 이미지 전반에 걸쳐 움직이므로 crop 대신 전체 장면 보존
  - EgoDex(센터크롭)와 다른 방식 — 데이터 특성에 따른 의도적 선택
"""

import argparse
import hashlib
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
from tqdm import tqdm


def process_trajectory(args):
    """단일 trajectory의 이미지 시퀀스를 리사이즈하여 저장."""
    images_dir, output_dir, img_size = args

    # trajectory 고유 식별자 (경로 해시)
    traj_id = hashlib.md5(str(images_dir).encode()).hexdigest()[:8]
    frames_dir = output_dir / f"traj_{traj_id}"

    # 이미 처리되었으면 스킵
    if frames_dir.exists() and len(list(frames_dir.glob("*.jpg"))) > 0:
        return images_dir, True, "already exists"

    # 원본 이미지 목록
    src_images = sorted(images_dir.glob("im_*.jpg"))
    if len(src_images) < 2:
        return images_dir, False, "too few frames"

    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, src_path in enumerate(src_images):
            frame = cv2.imread(str(src_path))
            if frame is None:
                return images_dir, False, f"failed to read {src_path.name}"

            # 비율 무시 리사이즈 (전체 장면 보존)
            frame = cv2.resize(frame, (img_size, img_size))

            dst_path = frames_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(dst_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return images_dir, True, f"{len(src_images)} frames"

    except Exception as e:
        return images_dir, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Resize Bridge V2 image sequences to 256x256"
    )
    parser.add_argument(
        "--bridge-root", type=str, required=True,
        help="Bridge V2 root (contains raw/bridge_data_v2/)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for resized frames",
    )
    parser.add_argument(
        "--img-size", type=int, default=256,
        help="Output image size (default: 256)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    bridge_root = Path(args.bridge_root)
    output_dir = Path(args.output_dir)

    # images0 디렉토리 탐색
    search_root = bridge_root / "raw" / "bridge_data_v2"
    if not search_root.exists():
        search_root = bridge_root

    print(f"Scanning {search_root} for trajectories...")
    traj_dirs = sorted(search_root.glob("**/images0"))
    # 최소 2프레임 이상인 것만
    traj_dirs = [d for d in traj_dirs if len(list(d.glob("im_*.jpg"))) >= 2]

    print(f"\nFound {len(traj_dirs):,} trajectories")
    print(f"Output: {output_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Mode: resize (no crop, preserves full scene)")

    num_workers = args.num_workers or cpu_count()
    print(f"Workers: {num_workers}")

    if not args.yes:
        response = input(f"\nProcess {len(traj_dirs):,} trajectories? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled.")
            sys.exit(0)

    process_args = [(d, output_dir, args.img_size) for d in traj_dirs]

    print("\nProcessing...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_trajectory, process_args),
            total=len(traj_dirs),
            desc="Progress",
        ))

    success_count = 0
    fail_count = 0
    for traj_dir, success, message in results:
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"FAILED: {traj_dir} - {message}")

    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"{'='*60}")
    print(f"Success: {success_count:,}")
    print(f"Failed: {fail_count:,}")
    print(f"Output: {output_dir}")

    import subprocess
    result = subprocess.run(["du", "-sh", str(output_dir)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Total size: {result.stdout.split()[0]}")


if __name__ == "__main__":
    main()
