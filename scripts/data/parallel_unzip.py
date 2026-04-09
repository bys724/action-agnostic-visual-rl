#!/usr/bin/env python3
"""
병렬 zip 압축 해제 — 144 worker로 zip entry를 chunk 단위 분산 추출.

표준 `unzip`은 단일 스레드라 144 CPU 노드에서 1 코어만 사용 → 병목.
이 스크립트는 Python multiprocessing.Pool로 zip entry를 worker에 분배하여
디스크 I/O 한계까지 병렬화한다.

Usage:
    python parallel_unzip.py --zip path/to/file.zip --output-dir path/to/dest \\
        --num-workers 144
"""

import argparse
import os
import sys
import time
import zipfile
from multiprocessing import Pool


def extract_chunk(task):
    """한 worker가 zip을 열고 entry list를 순차 추출.
    부모 디렉토리는 메인 프로세스에서 사전 생성되어 있어야 함 (race 방지)."""
    zip_path, entry_names, output_dir = task
    with zipfile.ZipFile(zip_path) as zf:
        for name in entry_names:
            zf.extract(name, output_dir)
    return len(entry_names)


def pre_create_dirs(zip_path, output_dir):
    """zip 내 모든 entry의 부모 디렉토리를 메인 프로세스에서 사전 생성.
    zipfile.extract()는 내부적으로 os.makedirs(exist_ok=False)를 호출해서
    multiprocessing worker들이 동시에 같은 dir 생성 시도 시 race 발생."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    dirs = set()
    for name in names:
        # entry가 dir 자체인 경우
        if name.endswith("/"):
            dirs.add(os.path.join(output_dir, name.rstrip("/")))
        else:
            parent = os.path.dirname(name)
            if parent:
                dirs.add(os.path.join(output_dir, parent))
    for d in sorted(dirs):
        os.makedirs(d, exist_ok=True)
    return names


def main():
    parser = argparse.ArgumentParser(description="Parallel zip extraction")
    parser.add_argument("--zip", required=True, help="zip file path")
    parser.add_argument("--output-dir", required=True, help="extraction destination")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="entries per worker task (default: 50)")
    args = parser.parse_args()

    if not os.path.isfile(args.zip):
        print(f"ERROR: zip file not found: {args.zip}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading central directory from {args.zip} ...")
    print(f"Pre-creating parent directories (race-free worker setup)...")
    names = pre_create_dirs(args.zip, args.output_dir)
    n_total = len(names)
    print(f"  {n_total:,} entries")

    chunks = [names[i:i + args.chunk_size] for i in range(0, n_total, args.chunk_size)]
    print(f"  {len(chunks):,} chunks of size {args.chunk_size}")
    print(f"  workers: {args.num_workers}")

    tasks = [(args.zip, chunk, args.output_dir) for chunk in chunks]

    t0 = time.time()
    done = 0
    with Pool(args.num_workers) as pool:
        for n in pool.imap_unordered(extract_chunk, tasks):
            done += n
            if done % (max(args.chunk_size, 1) * 20) == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else float("inf")
                print(f"  {done:,} / {n_total:,} ({100*done/n_total:5.1f}%)  "
                      f"{rate:6.1f} entries/s  ETA {eta/60:5.1f} min", flush=True)
    elapsed = time.time() - t0
    print(f"Done: {n_total:,} entries in {elapsed:.1f} s ({n_total/elapsed:.1f} entries/s)")


if __name__ == "__main__":
    main()
