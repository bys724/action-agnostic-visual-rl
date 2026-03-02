#!/usr/bin/env python3
"""
EgoDex 비디오에서 프레임 추출
병렬 처리로 빠르게 추출
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys


def extract_video_frames(args):
    """단일 비디오에서 프레임 추출"""
    video_path, output_dir, img_size = args

    # 출력 디렉토리 생성 (비디오 이름으로)
    video_name = video_path.stem
    task_name = video_path.parent.name
    frames_dir = output_dir / task_name / video_name

    # 이미 추출되었으면 스킵
    if frames_dir.exists() and len(list(frames_dir.glob("*.jpg"))) > 0:
        return video_path, True, "already exists"

    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return video_path, False, "failed to open"

        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 리사이즈
            frame = cv2.resize(frame, (img_size, img_size))

            # 저장
            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            frame_idx += 1
            extracted += 1

        cap.release()

        return video_path, True, f"{extracted} frames"

    except Exception as e:
        return video_path, False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Extract frames from EgoDex videos')
    parser.add_argument('--egodex-root', type=str, required=True,
                        help='EgoDex root directory')
    parser.add_argument('--split', type=str, default='part1',
                        help='Split to process (part1, part2, etc.)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for frames')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')

    args = parser.parse_args()

    egodex_root = Path(args.egodex_root)
    split_dir = egodex_root / args.split
    output_dir = Path(args.output_dir)

    if not split_dir.exists():
        print(f"Error: {split_dir} does not exist!")
        sys.exit(1)

    # 비디오 파일 찾기
    print(f"Scanning {split_dir} for videos...")
    video_paths = sorted(split_dir.glob("**/*.mp4"))

    print(f"\nFound {len(video_paths):,} videos")
    print(f"Output: {output_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}")

    num_workers = args.num_workers or cpu_count()
    print(f"Workers: {num_workers}")

    # 확인
    response = input(f"\nExtract frames from {len(video_paths):,} videos? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    # 병렬 추출
    extract_args = [(vp, output_dir, args.img_size) for vp in video_paths]

    print("\nExtracting frames...")
    success_count = 0
    fail_count = 0

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_video_frames, extract_args),
            total=len(video_paths),
            desc="Progress"
        ))

    # 결과 집계
    for video_path, success, message in results:
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"FAILED: {video_path} - {message}")

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"{'='*60}")
    print(f"Success: {success_count:,}")
    print(f"Failed: {fail_count:,}")
    print(f"Output: {output_dir}")

    # 디스크 사용량
    import subprocess
    result = subprocess.run(['du', '-sh', str(output_dir)], capture_output=True, text=True)
    if result.returncode == 0:
        size = result.stdout.split()[0]
        print(f"Total size: {size}")


if __name__ == '__main__':
    main()
