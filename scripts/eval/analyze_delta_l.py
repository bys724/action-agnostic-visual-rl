#!/usr/bin/env python3
"""|ΔL| 분포 분석 — v7-big의 Gaussian weighting σ 결정용.

샘플링한 EgoDex 프레임 쌍에서 ΔL(BT.709 luminance 차) 계산,
pixel-level magnitude 히스토그램을 보고 σ 후보를 제안.

Usage:
    python scripts/eval/analyze_delta_l.py \\
        --frames-root /proj/external_group/mrg/datasets/egodex/frames \\
        --split part1 --num-pairs 500 --gap 10
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# BT.709 luminance weights
LUMINANCE_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def load_frame(path):
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # [3, H, W], [0, 1]
    ])(img)
    return t


def compute_delta_l(img_prev, img_curr):
    """|luminance(curr) - luminance(prev)|."""
    lum_prev = (img_prev * LUMINANCE_WEIGHTS).sum(dim=1, keepdim=True)
    lum_curr = (img_curr * LUMINANCE_WEIGHTS).sum(dim=1, keepdim=True)
    return (lum_curr - lum_prev).abs()  # [B, 1, H, W]


def sample_pairs(frames_root, split, num_pairs, gap):
    """Random sample (prev, curr) frame pairs."""
    split_dir = Path(frames_root) / split
    # Gather all episode dirs with enough frames
    episodes = []
    for task_dir in split_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for ep_dir in task_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            frames = sorted(ep_dir.glob("frame_*.jpg"))
            if len(frames) >= gap + 1:
                episodes.append((ep_dir, len(frames)))

    print(f"Found {len(episodes)} episodes in {split_dir}")
    random.shuffle(episodes)

    pairs = []
    for ep_dir, n_frames in episodes:
        if len(pairs) >= num_pairs:
            break
        t = random.randint(0, n_frames - gap - 1)
        prev_path = ep_dir / f"frame_{t:06d}.jpg"
        curr_path = ep_dir / f"frame_{t + gap:06d}.jpg"
        pairs.append((prev_path, curr_path))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", default="/proj/external_group/mrg/datasets/egodex/frames")
    parser.add_argument("--split", default="part1")
    parser.add_argument("--num-pairs", type=int, default=500)
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="docs/architecture/delta_l_histogram.png")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Sampling {args.num_pairs} pairs with gap={args.gap} from {args.split}...")
    pairs = sample_pairs(args.frames_root, args.split, args.num_pairs, args.gap)
    print(f"Got {len(pairs)} pairs")

    # Compute |ΔL| for all pairs, flatten to single tensor
    all_delta = []
    for i, (prev_p, curr_p) in enumerate(pairs):
        img_prev = load_frame(prev_p).unsqueeze(0)
        img_curr = load_frame(curr_p).unsqueeze(0)
        delta = compute_delta_l(img_prev, img_curr).squeeze().numpy()  # [H, W]
        all_delta.append(delta.flatten())
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(pairs)}")

    all_delta = np.concatenate(all_delta)
    print(f"\nTotal pixels analyzed: {len(all_delta):,}")

    # Distribution stats
    print("\n=== |ΔL| pixel-level distribution ===")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        v = np.percentile(all_delta, p)
        print(f"  p{p:2d}: {v:.4f}")
    print(f"  mean: {all_delta.mean():.4f}")
    print(f"  std:  {all_delta.std():.4f}")
    print(f"  max:  {all_delta.max():.4f}")

    # Coverage analysis: for each candidate σ, compute fraction of pixels
    # with w_bg > 0.5, i.e., exp(-(|ΔL|/σ)²) > 0.5 → |ΔL| < σ·sqrt(ln2) ≈ 0.832σ
    print("\n=== σ candidate coverage (fraction of pixels with w_bg > 0.5) ===")
    print("   σ      |  coverage  |  interpretation")
    print("   ---    |  --------  |  ---")
    for sigma in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        thresh = sigma * np.sqrt(np.log(2))
        coverage = (all_delta < thresh).mean()
        interp = "static-heavy" if coverage > 0.85 else \
                 "balanced" if 0.5 < coverage <= 0.85 else \
                 "motion-heavy" if coverage <= 0.5 else "?"
        print(f"   {sigma:.2f}   |   {coverage:6.1%}   |  {interp}")

    # Histogram plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(all_delta, bins=200, range=(0, 0.3), color="steelblue", edgecolor="black")
        axes[0].set_xlabel("|ΔL| (pixel-wise)")
        axes[0].set_ylabel("count")
        axes[0].set_title("pixel-level |ΔL| distribution (linear, clipped at 0.3)")
        axes[0].set_yscale("log")

        # w_bg curves for candidate σ
        delta_range = np.linspace(0, 0.3, 500)
        for sigma in [0.03, 0.05, 0.10]:
            w_bg = np.exp(-(delta_range / sigma) ** 2)
            axes[1].plot(delta_range, w_bg, label=f"σ={sigma:.2f}")
        axes[1].set_xlabel("|ΔL|")
        axes[1].set_ylabel("w_bg = exp(-(|ΔL|/σ)²)")
        axes[1].set_title("weight functions for candidate σ")
        axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="w=0.5 threshold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=120, bbox_inches="tight")
        print(f"\nSaved plot: {args.output}")
    except ImportError:
        print("\n[skipping plot — matplotlib not available]")


if __name__ == "__main__":
    main()
