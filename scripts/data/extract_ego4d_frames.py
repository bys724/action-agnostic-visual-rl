#!/usr/bin/env python3
"""
Ego4D mp4 → 256x256 JPEG frames

Input  : /proj/external_group/mrg/datasets/ego4d/v2/full_scale/<uuid>.mp4
Output : <output_dir>/<uuid>/frame_000000.jpg, frame_000001.jpg, ...

처리:
  - cv2.VideoCapture로 모든 frame 디코드 (30 fps 원본 그대로 — subsample 없음)
  - 짧은 변 기준 센터 크롭 → 256x256 리사이즈 (EgoDex와 동일 spec)
  - 144 worker 병렬 (multiprocessing.Pool)
  - 출력 디렉토리에 1개 이상 jpg 있으면 skip (resume)
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2


def extract_video_frames(args):
    """단일 mp4 → JPEG 시퀀스. (video_path, output_video_dir, img_size) 입력."""
    video_path, output_video_dir, img_size = args

    if output_video_dir.exists() and any(output_video_dir.glob("*.jpg")):
        return video_path.name, True, "already extracted", 0

    output_video_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return video_path.name, False, "cv2 open failed", 0

    extracted = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if h != w:
                s = min(h, w)
                y = (h - s) // 2
                x = (w - s) // 2
                frame = frame[y : y + s, x : x + s]
            frame = cv2.resize(frame, (img_size, img_size))

            out_path = output_video_dir / f"frame_{extracted:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1
    except Exception as e:
        cap.release()
        return video_path.name, False, f"exception: {e}", extracted
    finally:
        cap.release()

    if extracted == 0:
        return video_path.name, False, "no frames decoded", 0

    return video_path.name, True, f"{extracted} frames", extracted


def main():
    parser = argparse.ArgumentParser(description="Extract frames from Ego4D mp4 videos")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/proj/external_group/mrg/datasets/ego4d/v2/full_scale",
        help="Directory containing Ego4D mp4 videos (flat).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/proj/external_group/mrg/datasets/ego4d/frames",
        help="Output root. Each video → <output>/<uuid>/frame_*.jpg",
    )
    parser.add_argument("--img-size", type=int, default=256, help="Output resolution (square)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers (default: SLURM_CPUS_PER_TASK or all CPUs)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N videos (sanity)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    if not video_dir.exists():
        print(f"ERROR: video dir not found: {video_dir}", file=sys.stderr)
        sys.exit(1)

    video_paths = sorted(video_dir.glob("*.mp4"))
    if args.limit is not None:
        video_paths = video_paths[: args.limit]

    if not video_paths:
        print(f"ERROR: no mp4 files in {video_dir}", file=sys.stderr)
        sys.exit(1)

    if args.num_workers is None:
        import os
        args.num_workers = int(
            os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
        )

    print(f"Video dir   : {video_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Image size  : {args.img_size}x{args.img_size}")
    print(f"Videos found: {len(video_paths):,}")
    print(f"Workers     : {args.num_workers}")
    if args.limit is not None:
        print(f"Limit       : {args.limit} (sanity mode)")

    if not args.yes:
        resp = input(f"\nExtract from {len(video_paths):,} videos? [y/N]: ")
        if resp.lower() != "y":
            print("Cancelled.")
            sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    pool_args = [
        (vp, output_dir / vp.stem, args.img_size) for vp in video_paths
    ]

    print("\nExtracting...")
    t0 = time.time()
    success = 0
    failed = 0
    total_frames = 0

    with Pool(args.num_workers) as pool:
        for i, (name, ok, msg, n_frames) in enumerate(
            pool.imap_unordered(extract_video_frames, pool_args), start=1
        ):
            if ok:
                success += 1
                total_frames += n_frames
            else:
                failed += 1
                print(f"  FAILED {name}: {msg}", file=sys.stderr)

            # 진행 로그: 50개마다 + 마지막
            if i % 50 == 0 or i == len(pool_args):
                elapsed = time.time() - t0
                vps = i / elapsed if elapsed > 0 else 0
                eta_h = (len(pool_args) - i) / vps / 3600 if vps > 0 else 0
                print(
                    f"  [{i:,}/{len(pool_args):,}] {vps:.2f} vid/s, "
                    f"~{eta_h:.1f}h remaining, frames={total_frames:,}"
                )

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("Extraction complete")
    print("=" * 60)
    print(f"Elapsed     : {elapsed / 3600:.2f}h ({elapsed:.0f}s)")
    print(f"Success     : {success:,}")
    print(f"Failed      : {failed:,}")
    print(f"Total frames: {total_frames:,}")
    print(f"Output      : {output_dir}")

    if failed > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
