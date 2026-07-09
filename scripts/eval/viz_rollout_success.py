#!/usr/bin/env python3
"""LIBERO BC-T rollout 성공 케이스 → suite별 fastest/slowest **PCA-overlay** GIF.

모델(1 TAG) 하위 모든 result JSON을 스캔해, suite 단위로 성공(non-errored) 중
**가장 빠른**(min steps_to_done)·**가장 오래 걸린**(max) 2 케이스를 선정 → 그 에피소드
rollout mp4 프레임에 인코더 patch feature의 PCA-RGB overlay(= scripts/viz/pca_overlay.py
방식)를 입혀 GIF로 저장. best/worst-case 실행 대비.

선정은 suite 전체(task·seed 통합) 전역 min/max. overlay 인코더는 --encoder로 지정.
산출 folder 구조:
    <out>/<encoder>/<suite>/{fastest,slowest}_task{NN}_seed{S}_ep{E}_{steps}st.gif

산출은 재생성 가능한 가시화 덤프 → gitignored `scratch/viz/`에 저장 (paper_artifacts는
확정 자료만; artifacts 재편 2026-07-09). 컨테이너(libero-eval) 안에서 실행 —
비디오·모델·imageio가 거기 있음:
    docker exec -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=0 libero-eval \
      python3 scripts/eval/viz_rollout_success.py \
        --results-dir data/libero/results/siglip_v3_t50 \
        --videos-dir  data/libero/videos/siglip_v3_t50 \
        --encoder siglip
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import imageio.v2 as imageio
import numpy as np
import torch

# pca_overlay의 patch추출·PCA·overlay·gif 로직 재사용 (단일 출처)
from scripts.viz.pca_overlay import (
    ENCODER_BUILDERS, preprocess_frames, fit_pca_to_rgb,
    overlay_rgb_on_frame, write_gif,
)

SEED_RE = re.compile(r"_seed(\d+)_\d{8}")


def pick_cases(results_dir: pathlib.Path):
    """suite → {'fastest': meta, 'slowest': meta}. 성공(non-errored) 중 전역 min/max steps."""
    cases: dict[str, dict[str, dict]] = {}
    for jp in sorted(results_dir.glob("*.json")):
        data = json.loads(jp.read_text())
        md = data.get("metadata", {})
        suite = md.get("task_suite", data.get("task_suite", "unknown"))
        ck = md.get("checkpoint", "")
        m = SEED_RE.search(ck)
        seed = m.group(1) if m else "?"
        video_subdir = pathlib.Path(ck).parent.name if ck else ""
        for tr in data.get("task_results", []):
            tid = tr["task_id"]
            for ep in tr.get("episode_records", []):
                if not ep.get("success") or ep.get("errored"):
                    continue
                steps = ep.get("steps_to_done")
                if steps is None:
                    continue
                meta = {
                    "suite": suite, "task_id": tid, "seed": seed,
                    "ep_id": ep["ep_id"], "steps": steps,
                    "video_subdir": video_subdir,
                    "task_description": tr.get("task_description", ""),
                }
                c = cases.setdefault(suite, {})
                if "fastest" not in c or steps < c["fastest"]["steps"]:
                    c["fastest"] = meta
                if "slowest" not in c or steps > c["slowest"]["steps"]:
                    c["slowest"] = meta
    return cases


def read_mp4(mp4: pathlib.Path, max_frames: int) -> np.ndarray:
    frames = imageio.mimread(str(mp4), memtest=False)
    if not frames:
        raise RuntimeError("no frames")
    if len(frames) > max_frames:
        stride = int(np.ceil(len(frames) / max_frames))
        frames = frames[::stride]
    return np.stack([np.asarray(f)[..., :3] for f in frames]).astype(np.uint8)


def make_overlay_gif(frames_uint8, encoder, device, v15_ckpt, vmae_ckpt,
                     gif: pathlib.Path, alpha: float, grayscale: bool, fps: int):
    ft = preprocess_frames(frames_uint8)                          # (T,3,224,224)
    patches, grid = ENCODER_BUILDERS[encoder](ft, device, v15_ckpt, vmae_ckpt)
    rgb_per_frame, _ = fit_pca_to_rgb(patches, grid)              # per-episode PCA fit
    overlays = [overlay_rgb_on_frame(frames_uint8[t], rgb_per_frame[t],
                                     alpha=alpha, grayscale_base=grayscale)
                for t in range(len(frames_uint8))]
    gif.parent.mkdir(parents=True, exist_ok=True)
    write_gif(overlays, gif, fps=fps)
    return len(overlays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--encoder", required=True,
                    help=f"overlay 인코더 ({'/'.join(ENCODER_BUILDERS)}). 출력 folder 이름도 겸함")
    ap.add_argument("--out", default="scratch/viz/rollout_success")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--color-base", action="store_true",
                    help="원본 RGB 위 overlay (기본=grayscale base, pca_overlay v4 방식)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--v15-ckpt", default=None)
    ap.add_argument("--videomae-ckpt", default=None)
    args = ap.parse_args()

    if args.encoder not in ENCODER_BUILDERS:
        raise SystemExit(f"unknown encoder {args.encoder}; choices={list(ENCODER_BUILDERS)}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = pathlib.Path(args.results_dir)
    videos_dir = pathlib.Path(args.videos_dir)
    out_root = pathlib.Path(args.out) / args.encoder

    cases = pick_cases(results_dir)
    n_cells = sum(len(v) for v in cases.values())
    print(f"[{args.encoder}] {len(cases)} suites × fastest/slowest = {n_cells} cells | device={device}")

    ok = miss = 0
    for suite in sorted(cases):
        for label in ("fastest", "slowest"):
            r = cases[suite].get(label)
            if r is None:
                continue
            tid = r["task_id"]
            mp4 = videos_dir / r["video_subdir"] / f"task{tid}_ep{r['ep_id']}_success.mp4"
            gif = out_root / suite / (
                f"{label}_task{tid:02d}_seed{r['seed']}_ep{r['ep_id']}_{r['steps']}st.gif")
            if not mp4.exists():
                print(f"  [MISS] {suite} {label} task{tid}: {mp4}")
                miss += 1
                continue
            frames = read_mp4(mp4, args.max_frames)
            n = make_overlay_gif(frames, args.encoder, device, args.v15_ckpt,
                                 args.videomae_ckpt, gif, args.alpha,
                                 grayscale=not args.color_base, fps=args.fps)
            print(f"  [OK] {suite:14s} {label:7s} task{tid:2d} seed{r['seed']} "
                  f"ep{r['ep_id']:2d} {r['steps']:3d}st → {gif.name} ({n}f)")
            ok += 1
    print(f"done: {ok} gifs, {miss} missing → {out_root}")


if __name__ == "__main__":
    main()
