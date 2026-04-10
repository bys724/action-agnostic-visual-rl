#!/usr/bin/env python3
"""
EgoDex 프레임 무결성 검증 — 손상된 JPG 파일 탐지.

multiprocessing.Pool로 병렬 검증. PIL.Image.open + verify()로 디코딩 가능 여부 확인.
손상 파일 목록을 stdout + 파일로 출력 (삭제/재추출은 별도 수행).

Usage:
    python scripts/data/validate_frames.py \
        --frames-root /proj/external_group/mrg/datasets/egodex/frames \
        --splits part1,part2,part3,part4,part5,test \
        --num-workers 144 \
        --output /proj/external_group/mrg/logs/corrupted_frames.txt
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path


def validate_chunk(args):
    """파일 목록을 받아 PIL로 검증. 손상된 파일 경로 반환."""
    file_paths, = args
    corrupted = []
    from PIL import Image
    for p in file_paths:
        try:
            with Image.open(p) as img:
                img.verify()  # 디코딩 없이 헤더+구조 검증 (빠름)
        except Exception:
            corrupted.append(str(p))
    return corrupted


def main():
    parser = argparse.ArgumentParser(description="Validate EgoDex frame JPGs")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--splits", default="part1,part2,part3,part4,part5,test")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--output", default=None, help="손상 파일 목록 저장 경로")
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.frames_root)
    splits = [s.strip() for s in args.splits.split(",")]

    # 전체 JPG 파일 목록 수집
    print(f"Scanning {root} for JPG files (splits: {splits})...")
    all_files = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  WARNING: {split_dir} not found, skipping")
            continue
        # glob은 느릴 수 있지만 1회만 실행
        jpgs = sorted(split_dir.rglob("*.jpg"))
        print(f"  {split}: {len(jpgs):,} files")
        all_files.extend(jpgs)

    n_total = len(all_files)
    print(f"\nTotal: {n_total:,} files to validate")
    print(f"Workers: {args.num_workers}, chunk size: {args.chunk_size}")

    # chunk 분배
    chunks = [all_files[i:i + args.chunk_size]
              for i in range(0, n_total, args.chunk_size)]
    tasks = [(chunk,) for chunk in chunks]

    t0 = time.time()
    all_corrupted = []
    checked = 0

    with Pool(args.num_workers) as pool:
        for result in pool.imap_unordered(validate_chunk, tasks):
            all_corrupted.extend(result)
            checked += args.chunk_size
            if checked % (args.chunk_size * 50) == 0 or checked >= n_total:
                elapsed = time.time() - t0
                rate = checked / elapsed if elapsed > 0 else 0
                print(f"  {min(checked, n_total):,} / {n_total:,} "
                      f"({100*min(checked,n_total)/n_total:.1f}%) "
                      f"{rate:.0f} files/s  "
                      f"corrupted so far: {len(all_corrupted)}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Validation complete: {n_total:,} files in {elapsed:.1f}s")
    print(f"Corrupted: {len(all_corrupted):,}")

    if all_corrupted:
        print(f"\nCorrupted files:")
        for f in all_corrupted:
            print(f"  {f}")

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as fout:
                fout.write("\n".join(all_corrupted) + "\n")
            print(f"\nSaved to: {args.output}")
    else:
        print("\nNo corrupted files found!")


if __name__ == "__main__":
    main()
