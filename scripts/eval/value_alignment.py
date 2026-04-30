#!/usr/bin/env python
"""Trajectory-Level Value Alignment (VIP-inspired, Phase 2.5).

LIBERO demo trajectory에서 frozen encoder로 frame-wise embedding 추출 후
V(t) = cos_sim(e_t, e_T) → Spearman ρ(t, V(t)) 계산.

5 encoder × 3 LIBERO suite (spatial/object/goal) × ~1500 trajectory.
Frozen encoder 재활용, 학습 없음.

기준:
- 길이 분포 상위 percentile 초과 trajectory 제외 (default: 95th).
- View: agentview_rgb (3rd-person, task progress 가장 명확).
- v11은 A+B+D' (probing champion mode) 사용 — plan 준수.
- 그 외 encoder는 기존 build_adapter() 출력 그대로.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal")
SUPPORTED_ENCODERS = (
    "two-stream-v11",
    "videomae-ours",
    "dinov2",
    "siglip",
    "vc1",
)


# ─────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ─────────────────────────────────────────────────────────────────────────

def list_demos(hdf5_path: str) -> List[str]:
    with h5py.File(hdf5_path, "r") as f:
        return sorted([k for k in f["data"].keys() if k.startswith("demo_")])


def load_demo_frames(hdf5_path: str, demo_key: str, view: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        return np.asarray(f[f"data/{demo_key}/obs/{view}"])  # (T, H, W, 3) uint8


def collect_lengths(hdf5_paths: List[str], view: str) -> np.ndarray:
    lens = []
    for p in hdf5_paths:
        with h5py.File(p, "r") as f:
            for k in f["data"].keys():
                if not k.startswith("demo_"):
                    continue
                lens.append(f[f"data/{k}/obs/{view}"].shape[0])
    return np.array(lens, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────
# Encoder: standard adapters (videomae-ours / dinov2 / siglip / vc1)
# ─────────────────────────────────────────────────────────────────────────

def build_standard_encoder(encoder_type: str, checkpoint: str | None, device: torch.device):
    from src.encoders.adapters import build_adapter
    adapter = build_adapter(encoder_type, checkpoint_path=checkpoint).to(device)
    adapter.eval()
    return adapter


@torch.no_grad()
def embed_via_adapter(adapter, frames_uint8: np.ndarray, device: torch.device) -> np.ndarray:
    """frames: (T, H, W, 3) uint8 → (T, embed_dim) np.ndarray.

    Adapter는 (1, T, 3, H, W) [0,1] float 입력 받고 (1, T, embed_dim) 반환.
    어댑터 내부에서 T>1 path가 prev=obs[0,0,1,...,T-1] 자동 구성 (videomae/v11 공통).
    """
    T = frames_uint8.shape[0]
    x = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != adapter.img_size:
        x = F.interpolate(
            x, size=(adapter.img_size, adapter.img_size),
            mode="bilinear", align_corners=False,
        )
    adapter.reset()
    obs = x.unsqueeze(0).to(device, non_blocking=True)  # (1, T, 3, H, W)
    out = adapter(obs).squeeze(0).cpu().numpy()  # (T, D)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Encoder: v11 with A+B+D' mode (plan-compliant probing champion)
# ─────────────────────────────────────────────────────────────────────────

def build_v11_encoder(checkpoint: str, p_depth: int, m_depth: int, device: torch.device):
    from scripts.eval.probe_action_v11 import (
        _full_forward_with_d_prime,
        load_v11_model,
    )
    model = load_v11_model(checkpoint, p_depth=p_depth, m_depth=m_depth, device=str(device))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, _full_forward_with_d_prime


@torch.no_grad()
def embed_v11(
    model, full_forward_fn, frames_uint8: np.ndarray, device: torch.device,
    mode: str = "abd_prime", img_size: int = 224, chunk: int = 32,
) -> np.ndarray:
    """v11 frame-wise embedding per mode.

    각 timestep에서 (img_prev, img_curr) pair forward (t=0은 (img[0], img[0])).
    Mode:
      - "abd_prime"    : A ⊕ B ⊕ D'  (probing champion, 2304-d)
      - "b_only"       : B           (P encoder patch_mean, 768-d)
      - "d_prime_only" : D'          (P state after motion-routing, 768-d)
    """
    T, H, W, _ = frames_uint8.shape
    x = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    img_curr = x  # (T, 3, H, W)
    img_prev = torch.cat([x[:1], x[:-1]], dim=0)  # t=0은 self-pair, t>=1은 prev frame

    embeds = []
    for s in range(0, T, chunk):
        e_curr = img_curr[s:s+chunk].to(device, non_blocking=True)
        e_prev = img_prev[s:s+chunk].to(device, non_blocking=True)
        out = full_forward_fn(model, e_prev, e_curr)
        if mode == "abd_prime":
            A = out["m_encoded"][:, 1:].mean(dim=1)
            B = out["p_encoded"][:, 1:].mean(dim=1)
            D_prime = out["p_state_routing"][:, 1:].mean(dim=1)
            tok = torch.cat([A, B, D_prime], dim=-1)
        elif mode == "b_only":
            tok = out["p_encoded"][:, 1:].mean(dim=1)
        elif mode == "d_prime_only":
            tok = out["p_state_routing"][:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Unknown v11 mode: {mode}")
        embeds.append(tok.cpu())
    return torch.cat(embeds, dim=0).numpy()


# ─────────────────────────────────────────────────────────────────────────
# Spearman ρ per trajectory
# ─────────────────────────────────────────────────────────────────────────

def value_alignment_rho(emb: np.ndarray, min_T: int = 5) -> float | None:
    """emb: (T, D) → Spearman ρ(t, V(t)) where V(t) = cos_sim(e_t, e_T)."""
    T = emb.shape[0]
    if T < min_T:
        return None
    e_T = emb[-1]
    e_T_norm = np.linalg.norm(e_T) + 1e-12
    emb_norm = np.linalg.norm(emb, axis=1) + 1e-12
    V = (emb @ e_T) / (emb_norm * e_T_norm)
    rho, _ = spearmanr(np.arange(T), V)
    if np.isnan(rho):
        return None
    return float(rho)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=SUPPORTED_ENCODERS)
    parser.add_argument("--checkpoint", default=None,
                        help="Encoder ckpt path (필요 시: v11/videomae-ours)")
    parser.add_argument("--task-suite", required=True, choices=LIBERO_SUITES)
    parser.add_argument("--data-root", default="/proj/external_group/mrg/datasets/libero")
    parser.add_argument("--view", default="agentview_rgb",
                        choices=["agentview_rgb", "eye_in_hand_rgb"])
    parser.add_argument("--max-length-percentile", type=float, default=95.0,
                        help="이 percentile 초과 길이 trajectory 제외")
    parser.add_argument("--max-length", type=int, default=None,
                        help="절대 길이 cutoff (지정 시 percentile 무시)")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="None이면 전체 task")
    parser.add_argument("--max-demos-per-task", type=int, default=None,
                        help="Sanity test용. None이면 전체.")
    parser.add_argument("--last-k-fractions", type=float, nargs="+", default=[1.0],
                        help="Trailing window 비율 sweep. 1.0=full, 0.3=마지막 30%. "
                             "v11처럼 motion-specific encoder의 골 근접 영역 가설 검증용.")
    parser.add_argument("--min-T", type=int, default=5,
                        help="ρ 계산 최소 frame 수 (window 적용 후)")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda")
    # v11-only options
    parser.add_argument("--v11-p-depth", type=int, default=12)
    parser.add_argument("--v11-m-depth", type=int, default=6)
    parser.add_argument("--v11-mode", default="abd_prime",
                        choices=["abd_prime", "b_only", "d_prime_only"],
                        help="v11 embedding composition: A⊕B⊕D' (default), B only, or D' only")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[value_alignment] encoder={args.encoder} suite={args.task_suite} device={device}")

    # ── Benchmark / HDF5 paths ───────────────────────────────────────────
    sys.path.insert(0, str(PROJECT_ROOT / "external/LIBERO"))
    from libero.libero.benchmark import get_benchmark
    bm = get_benchmark(args.task_suite)(0)
    n_tasks = bm.n_tasks
    task_ids = args.task_ids or list(range(n_tasks))
    suite_dir = os.path.join(args.data_root, args.task_suite)
    hdf5_paths = [
        os.path.join(suite_dir, bm.get_task_demonstration(i).split("/")[-1])
        for i in task_ids
    ]
    print(f"  tasks: {len(task_ids)}/{n_tasks}, data_root={suite_dir}")

    # ── Length distribution → cutoff ─────────────────────────────────────
    lens = collect_lengths(hdf5_paths, view=args.view)
    print(f"  length stats (n={len(lens)}): min={lens.min()} max={lens.max()} "
          f"mean={lens.mean():.1f} P50={np.percentile(lens,50):.0f} "
          f"P90={np.percentile(lens,90):.0f} P95={np.percentile(lens,95):.0f} "
          f"P99={np.percentile(lens,99):.0f}")
    if args.max_length is not None:
        cutoff = float(args.max_length)
    else:
        cutoff = float(np.percentile(lens, args.max_length_percentile))
    n_kept = int((lens <= cutoff).sum())
    print(f"  cutoff={cutoff:.0f} → kept {n_kept}/{len(lens)} "
          f"({100*n_kept/len(lens):.1f}%)")

    # ── Encoder ──────────────────────────────────────────────────────────
    if args.encoder == "two-stream-v11":
        if args.checkpoint is None:
            raise ValueError("v11 encoder requires --checkpoint")
        model, fwd = build_v11_encoder(
            args.checkpoint, args.v11_p_depth, args.v11_m_depth, device,
        )

        def embed_one(frames):
            return embed_v11(model, fwd, frames, device, mode=args.v11_mode)
    else:
        adapter = build_standard_encoder(args.encoder, args.checkpoint, device)

        def embed_one(frames):
            return embed_via_adapter(adapter, frames, device)

    # ── Iterate demos → embed → ρ (per fraction) ────────────────────────
    fractions = sorted(set(args.last_k_fractions), reverse=True)
    rows = []
    rhos_per_frac = {f: [] for f in fractions}
    skipped_long = 0
    skipped_short = {f: 0 for f in fractions}
    t_start = time.time()

    for tid, hdf5_path in zip(task_ids, hdf5_paths):
        demos = list_demos(hdf5_path)
        if args.max_demos_per_task is not None:
            demos = demos[: args.max_demos_per_task]
        for d in demos:
            with h5py.File(hdf5_path, "r") as f:
                T_full = f[f"data/{d}/obs/{args.view}"].shape[0]
            if T_full > cutoff:
                skipped_long += 1
                continue
            frames = load_demo_frames(hdf5_path, d, args.view)
            emb_full = embed_one(frames)  # (T, D)
            for frac in fractions:
                k = max(int(round(len(emb_full) * frac)), args.min_T)
                k = min(k, len(emb_full))
                emb_sub = emb_full[-k:]
                rho = value_alignment_rho(emb_sub, min_T=args.min_T)
                if rho is None:
                    skipped_short[frac] += 1
                    continue
                rhos_per_frac[frac].append(rho)
                rows.append({
                    "task_id": tid, "demo": d, "T_full": T_full,
                    "fraction": frac, "T_used": k, "rho": rho,
                })

    elapsed = time.time() - t_start
    print(f"\n[value_alignment] done in {elapsed:.1f}s "
          f"(skipped_long={skipped_long})")
    summary_per_frac = {}
    for frac in fractions:
        arr = np.array(rhos_per_frac[frac]) if rhos_per_frac[frac] else np.array([np.nan])
        summary_per_frac[frac] = {
            "n_valid": len(rhos_per_frac[frac]),
            "skipped_short": skipped_short[frac],
            "rho_mean": float(arr.mean()),
            "rho_std": float(arr.std()),
            "rho_median": float(np.median(arr)),
            "rho_min": float(arr.min()),
            "rho_max": float(arr.max()),
        }
        print(f"  frac={frac:.2f}  valid={len(rhos_per_frac[frac]):3d}  "
              f"skipped_short={skipped_short[frac]:3d}  "
              f"ρ mean={arr.mean():+.4f} ± {arr.std():.4f}  "
              f"median={np.median(arr):+.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["task_id", "demo", "T_full", "fraction", "T_used", "rho"],
        )
        w.writeheader()
        w.writerows(rows)
    summary_path = args.output_csv.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "encoder": args.encoder,
            "checkpoint": args.checkpoint,
            "task_suite": args.task_suite,
            "view": args.view,
            "n_tasks": len(task_ids),
            "n_total_trajectories": int(len(lens)),
            "n_kept_after_length_cutoff": n_kept,
            "length_cutoff": cutoff,
            "length_percentile": args.max_length_percentile if args.max_length is None else None,
            "fractions": fractions,
            "per_fraction": {str(f): summary_per_frac[f] for f in fractions},
            "v11_mode": args.v11_mode if args.encoder == "two-stream-v11" else None,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"  CSV: {args.output_csv}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
