#!/usr/bin/env python
"""정확히-0 ΔL 비율 진단 (refinement_floor_plan §9 R2-4). GPU 불필요.

데이터셋마다 무작위 프레임 쌍 N개 → probe와 같은 입력 경로(uint8 → [0,1] → 224 bilinear →
compute_m_channel의 ΔL) → 픽셀 단위 `ΔL == 0` 비율 · 16×16 패치 "전부 0" 비율 · |ΔL| < 1/255 비율.
gap은 1프레임과 1초(probe 정규 gap: CALVIN 30 · LIBERO 20 · EgoDex 30) 두 가지.
EgoDex = part1 실제 영상 쌍(사전학습 Case B 입력과 같은 종류).
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.eval.probe_action_libero import preprocess_frames
from src.datasets.calvin import load_segment_frames, load_segments
from src.models.common.preprocessing import TwoStreamPreprocessing

PREP = TwoStreamPreprocessing(use_sobel=False)


def stats(prev_u8, curr_u8):
    """(N,H,W,3) uint8 쌍 → (exact0 픽셀 비율, 전부-0 패치 비율, |ΔL|<1/255 비율)."""
    dl = PREP.compute_m_channel(preprocess_frames(prev_u8, 224), preprocess_frames(curr_u8, 224))[:, :1]
    zero = dl == 0
    patch_all0 = F.avg_pool2d(zero.float(), 16) == 1.0
    return float(zero.float().mean()), float(patch_all0.float().mean()), float((dl.abs() < 1 / 255).float().mean())


def pairs_calvin(root, n, gap, rng):
    d = Path(root) / "validation"
    segs = [s for s in load_segments(d) if s[1] - s[0] + 1 > gap]
    P, C = [], []
    for i in rng.choice(len(segs), n):
        s, e, _ = segs[i]
        fr, _, _ = load_segment_frames(d, s, e, view="rgb_static")
        t = rng.integers(0, len(fr) - gap)
        P.append(fr[t]); C.append(fr[t + gap])
    return np.stack(P), np.stack(C)


def pairs_libero(root, suite, n, gap, rng):
    files = sorted((Path(root) / suite).glob("*.hdf5"))
    P, C = [], []
    for _ in range(n):
        with h5py.File(files[rng.integers(len(files))], "r") as f:
            demos = [k for k in f["data"] if k.startswith("demo_")]
            fr = f[f"data/{demos[rng.integers(len(demos))]}/obs/agentview_rgb"]
            t = rng.integers(0, fr.shape[0] - gap)
            P.append(fr[t]); C.append(fr[t + gap])
    return np.stack(P), np.stack(C)


def pairs_egodex(root, n, gap, rng):
    vids = [v for v in Path(root).glob("*/*") if v.is_dir()]
    P, C = [], []
    while len(P) < n:
        fs = sorted(vids[rng.integers(len(vids))].glob("frame_*.jpg"))
        if len(fs) <= gap:
            continue
        t = rng.integers(0, len(fs) - gap)
        P.append(np.asarray(Image.open(fs[t]).convert("RGB")))
        C.append(np.asarray(Image.open(fs[t + gap]).convert("RGB")))
    return np.stack(P), np.stack(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calvin-root", default="/proj/external_group/mrg/datasets/calvin/task_ABCD_D")
    ap.add_argument("--libero-root", default="/proj/external_group/mrg/datasets/libero")
    ap.add_argument("--egodex-root", default="/proj/external_group/mrg/datasets/egodex/frames/part1")
    a = ap.parse_args()
    jobs = [("calvin", 30, lambda g, r: pairs_calvin(a.calvin_root, a.n, g, r))]
    jobs += [(s, 20, lambda g, r, s=s: pairs_libero(a.libero_root, s, a.n, g, r))
             for s in ("libero_spatial", "libero_object", "libero_goal")]
    jobs += [("egodex_part1", 30, lambda g, r: pairs_egodex(a.egodex_root, a.n, g, r))]
    print(f"{'dataset':<16}{'gap':>5}{'exact0_px':>11}{'all0_patch':>12}{'lt1/255_px':>12}   (n={a.n} pairs, seed={a.seed})")
    for name, g1s, fn in jobs:
        for gap in (1, g1s):
            z, p, s = stats(*fn(gap, np.random.default_rng(a.seed)))
            print(f"{name:<16}{gap:>5}{z:>11.4f}{p:>12.4f}{s:>12.4f}", flush=True)


if __name__ == "__main__":
    main()
