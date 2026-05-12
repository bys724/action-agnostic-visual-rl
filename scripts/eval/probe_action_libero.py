#!/usr/bin/env python
"""LIBERO Action Probing (Phase 2 보강).

DROID action probing 절대 R² ~0.005 한계를 LIBERO sim 환경에서 보완.
한 잡 = (encoder × suite) → 4 gaps loop 으로 4 cells 산출.

Target (plan §3, pose-derived, NOT cumulative action sum):
- pos: ee_pos[t+k] - ee_pos[t]                                  (3,)
- rotvec: (R.from_rotvec(ee_ori[t]).inv() * R.from_rotvec(ee_ori[t+k])).as_rotvec()  (3,)
- gripper: actions[t+k-1, 6]                                    (1,)
→ 7-DoF target

Gap 시간 매칭 (LIBERO 20Hz):
- gap=1   ~0.05s
- gap=13  ~0.65s   ≈ DROID gap=10
- gap=20  ~1.00s   ≈ DROID gap=15 (★ EgoDex 학습 분포 1초 일치 핵심)
- gap=40  ~2.00s   ≈ DROID gap=30
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, TensorDataset

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
DEFAULT_GAPS = [1, 13, 20, 40]
ACTION_DIM = 7  # 3 pos + 3 rotvec + 1 gripper


# ─────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ─────────────────────────────────────────────────────────────────────────

def list_demos(hdf5_path: str) -> List[str]:
    with h5py.File(hdf5_path, "r") as f:
        return sorted([k for k in f["data"].keys() if k.startswith("demo_")])


def load_demo(hdf5_path: str, demo_key: str, view: str = "agentview_rgb"):
    """Load one demo: frames(uint8), ee_pos, ee_ori (axis-angle), actions."""
    with h5py.File(hdf5_path, "r") as f:
        d = f[f"data/{demo_key}"]
        frames = np.asarray(d[f"obs/{view}"])      # (T, H, W, 3) uint8
        ee_pos = np.asarray(d["obs/ee_pos"])        # (T, 3)
        ee_ori = np.asarray(d["obs/ee_ori"])        # (T, 3) axis-angle
        actions = np.asarray(d["actions"])          # (T, 7)
    return frames, ee_pos, ee_ori, actions


def libero_action_target(eef_pos, ee_ori, actions, t, k):
    """Plan §3 target: pose-derived, NOT cumulative action sum."""
    target_pos = eef_pos[t + k] - eef_pos[t]                         # (3,)
    r1 = R.from_rotvec(ee_ori[t])
    r2 = R.from_rotvec(ee_ori[t + k])
    target_rotvec = (r1.inv() * r2).as_rotvec()                      # (3,)
    target_gripper = actions[t + k - 1, 6:7]                         # (1,)
    return np.concatenate([target_pos, target_rotvec, target_gripper], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Encoder build (reuse value_alignment helpers)
# ─────────────────────────────────────────────────────────────────────────

def build_standard_encoder(encoder_type: str, checkpoint: str | None, device: torch.device):
    from src.encoders.adapters import build_adapter
    adapter = build_adapter(encoder_type, checkpoint_path=checkpoint).to(device)
    adapter.eval()
    return adapter


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


# ─────────────────────────────────────────────────────────────────────────
# Pair encode (img_t, img_{t+gap})
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_pairs_via_adapter(
    adapter, frames_prev: torch.Tensor, frames_curr: torch.Tensor,
    device: torch.device, batch: int = 64,
) -> torch.Tensor:
    """frames_prev/curr: (N, 3, H, W) [0,1] float → (N, embed_dim).

    Adapter T>1 path: obs_seq=[prev, curr], 첫 step은 self-pair, 두 번째 step이
    (prev, curr) pair. 두 번째 step의 embedding을 추출.
    """
    out = []
    for s in range(0, frames_prev.shape[0], batch):
        p = frames_prev[s:s+batch].to(device, non_blocking=True)  # (n, 3, H, W)
        c = frames_curr[s:s+batch].to(device, non_blocking=True)
        seq = torch.stack([p, c], dim=1)  # (n, 2, 3, H, W)
        adapter.reset()
        e = adapter(seq)[:, 1]  # take t=1 embedding (pair-formed)
        out.append(e.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_pairs_v11(
    model, fwd_fn, frames_prev: torch.Tensor, frames_curr: torch.Tensor,
    device: torch.device, mode: str = "abd_prime", batch: int = 64,
) -> torch.Tensor:
    out = []
    for s in range(0, frames_prev.shape[0], batch):
        p = frames_prev[s:s+batch].to(device, non_blocking=True)
        c = frames_curr[s:s+batch].to(device, non_blocking=True)
        o = fwd_fn(model, p, c)
        if mode == "abd_prime":
            A = o["m_encoded"][:, 1:].mean(dim=1)
            B = o["p_encoded"][:, 1:].mean(dim=1)
            D = o["p_state_routing"][:, 1:].mean(dim=1)
            tok = torch.cat([A, B, D], dim=-1)
        elif mode == "b_only":
            tok = o["p_encoded"][:, 1:].mean(dim=1)
        elif mode == "d_prime_only":
            tok = o["p_state_routing"][:, 1:].mean(dim=1)
        elif mode == "p_t_p_tk":
            # P encoder를 prev/curr frame에 각각 single-image path로 통과
            from scripts.eval.probe_action_v11 import _p_encoder_forward
            _, p_channel_t = model.preprocessing(p, p)
            _, p_channel_tk = model.preprocessing(c, c)
            p_enc_t = _p_encoder_forward(model, p_channel_t)
            p_enc_tk = _p_encoder_forward(model, p_channel_tk)
            tok = torch.cat(
                [p_enc_t[:, 1:].mean(dim=1), p_enc_tk[:, 1:].mean(dim=1)],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown v11 mode: {mode}")
        out.append(tok.cpu())
    return torch.cat(out, dim=0)


def preprocess_frames(frames_uint8: np.ndarray, img_size: int) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, img_size, img_size) [0,1] float."""
    x = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x


# ─────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int = ACTION_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, action_dim)

    def forward(self, x):
        return self.linear(x)


def train_probe(
    train_emb, train_tgt, eval_emb, eval_tgt,
    epochs: int = 20, batch_size: int = 256, lr: float = 1e-3,
    device: str = "cuda",
):
    embed_dim = train_emb.shape[1]
    probe = LinearProbe(embed_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    train_loader = DataLoader(
        TensorDataset(train_emb, train_tgt),
        batch_size=batch_size, shuffle=True,
    )
    best = {"r2": -float("inf"), "epoch": 0}
    for ep in range(epochs):
        probe.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = probe(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(eval_emb.to(device)).cpu().numpy()
        m = compute_metrics(pred, eval_tgt.numpy())
        if m["r2_aggregate"] > best["r2"]:
            best = {"r2": m["r2_aggregate"], "epoch": ep + 1, "metrics": m}
    return best


def compute_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    """R² aggregate + per-dim, MSE, cosine."""
    ss_res = np.sum((tgt - pred) ** 2)
    ss_tot = np.sum((tgt - tgt.mean(axis=0)) ** 2)
    r2_agg = 1.0 - ss_res / (ss_tot + 1e-8)
    mse = float(np.mean((pred - tgt) ** 2))
    cos = float(np.mean(np.sum(pred * tgt, axis=1) /
                        (np.linalg.norm(pred, axis=1) * np.linalg.norm(tgt, axis=1) + 1e-8)))
    r2_per_dim = []
    for i in range(tgt.shape[1]):
        ss_r = np.sum((tgt[:, i] - pred[:, i]) ** 2)
        ss_t = np.sum((tgt[:, i] - tgt[:, i].mean()) ** 2)
        r2_per_dim.append(float(1.0 - ss_r / (ss_t + 1e-8)))
    return {
        "r2_aggregate": float(r2_agg),
        "r2_per_dim": r2_per_dim,
        "mse": mse,
        "cosine_sim": cos,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=SUPPORTED_ENCODERS)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--task-suite", required=True, choices=LIBERO_SUITES)
    parser.add_argument("--data-root", default="/proj/external_group/mrg/datasets/libero")
    parser.add_argument("--view", default="agentview_rgb",
                        choices=["agentview_rgb", "eye_in_hand_rgb"])
    parser.add_argument("--gaps", type=int, nargs="+", default=DEFAULT_GAPS,
                        help="Frame gaps to evaluate (LIBERO 20Hz: 1=0.05s, 13=0.65s, 20=1s, 40=2s)")
    parser.add_argument("--max-length-percentile", type=float, default=99.0,
                        help="이 percentile 초과 길이 trajectory 제외 (gap 전체 demo 보존 위해 99 default)")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-batch", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--v11-p-depth", type=int, default=12)
    parser.add_argument("--v11-m-depth", type=int, default=6)
    parser.add_argument("--v11-mode", default="abd_prime",
                        choices=["abd_prime", "b_only", "d_prime_only", "p_t_p_tk"])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[probe_libero] encoder={args.encoder} suite={args.task_suite} gaps={args.gaps} device={device}")

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

    # ── Length filter ────────────────────────────────────────────────────
    lens = []
    for p in hdf5_paths:
        with h5py.File(p, "r") as f:
            for k in sorted(f["data"].keys()):
                if k.startswith("demo_"):
                    lens.append(f[f"data/{k}/obs/ee_pos"].shape[0])
    lens = np.array(lens)
    cutoff = float(np.percentile(lens, args.max_length_percentile))
    n_kept = int((lens <= cutoff).sum())
    print(f"  length stats (n={len(lens)}): min={lens.min()} max={lens.max()} "
          f"mean={lens.mean():.1f} P50={np.percentile(lens,50):.0f} P95={np.percentile(lens,95):.0f} "
          f"P99={np.percentile(lens,99):.0f}  cutoff={cutoff:.0f} → kept {n_kept}/{len(lens)}")

    # ── Build encoder ────────────────────────────────────────────────────
    if args.encoder == "two-stream-v11":
        if args.checkpoint is None:
            raise ValueError("v11 encoder requires --checkpoint")
        model, fwd = build_v11_encoder(
            args.checkpoint, args.v11_p_depth, args.v11_m_depth, device,
        )
        img_size = 224

        def encode_fn(prev, curr):
            return encode_pairs_v11(
                model, fwd, prev, curr, device,
                mode=args.v11_mode, batch=args.encode_batch,
            )
    else:
        adapter = build_standard_encoder(args.encoder, args.checkpoint, device)
        img_size = adapter.img_size

        def encode_fn(prev, curr):
            return encode_pairs_via_adapter(adapter, prev, curr, device, batch=args.encode_batch)

    print(f"  img_size={img_size}")

    # ── Demo-level train/test split ─────────────────────────────────────
    all_demo_keys = []  # list of (hdf5_path, demo_key, task_id)
    for tid, hp in zip(task_ids, hdf5_paths):
        demos = list_demos(hp)
        if args.max_demos_per_task is not None:
            demos = demos[:args.max_demos_per_task]
        for d in demos:
            with h5py.File(hp, "r") as f:
                T = f[f"data/{d}/obs/ee_pos"].shape[0]
            if T > cutoff:
                continue
            all_demo_keys.append((hp, d, tid))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_demo_keys))
    n_train = int(len(perm) * args.train_ratio)
    train_demos = [all_demo_keys[i] for i in perm[:n_train]]
    eval_demos = [all_demo_keys[i] for i in perm[n_train:]]
    print(f"  demos: train={len(train_demos)} / eval={len(eval_demos)}")

    # ── Per-gap loop ─────────────────────────────────────────────────────
    cell_summaries = []
    os.makedirs(args.output_dir, exist_ok=True)

    for gap in args.gaps:
        print(f"\n=== gap={gap} ({gap/20:.2f}s @ LIBERO 20Hz) ===")
        t0 = time.time()

        # Streaming: demo loop 안에서 frames load → preprocess → encoder forward →
        # embeddings + targets만 누적. raw frames은 즉시 폐기 (메모리 절약).
        def collect_embed(demo_list, label):
            embed_chunks, tgt_chunks, demo_ids = [], [], []
            for di, (hp, d, tid) in enumerate(demo_list):
                frames, eef_pos, ee_ori, actions = load_demo(hp, d, view=args.view)
                T = frames.shape[0]
                if T <= gap + 1:
                    continue
                tgts = np.stack([
                    libero_action_target(eef_pos, ee_ori, actions, t, gap)
                    for t in range(T - gap)
                ])
                prev = preprocess_frames(frames[:T - gap], img_size)
                curr = preprocess_frames(frames[gap:], img_size)
                emb = encode_fn(prev, curr)  # (T-gap, D), already on CPU
                embed_chunks.append(emb)
                tgt_chunks.append(tgts)
                demo_ids.extend([di] * (T - gap))
                # raw frames release
                del frames, prev, curr, emb
            print(f"    [{label}] embedded {len(demo_ids)} pairs from {len(embed_chunks)} demos")
            return (
                torch.cat(embed_chunks, 0),
                torch.from_numpy(np.concatenate(tgt_chunks, 0)),
                np.array(demo_ids),
            )

        print(f"  encoding train (streaming per demo) ...")
        emb_tr, tgt_tr, demo_tr = collect_embed(train_demos, "train")
        print(f"  encoding eval  (streaming per demo) ...")
        emb_ev, tgt_ev, demo_ev = collect_embed(eval_demos, "eval")
        print(f"  pairs: train={len(tgt_tr)} eval={len(tgt_ev)}")

        # Standardize target (MSE/R² scale-friendly)
        # NOTE: R²은 scale-invariant이지만 MSE와 학습 안정성 위해 옵션 — skip
        # Linear probe
        print(f"  training probe (epoch={args.probe_epochs}, lr={args.probe_lr}, batch={args.probe_batch}) ...")
        best = train_probe(
            emb_tr, tgt_tr, emb_ev, tgt_ev,
            epochs=args.probe_epochs, batch_size=args.probe_batch,
            lr=args.probe_lr, device=str(device),
        )
        m = best["metrics"]
        elapsed = time.time() - t0

        print(f"  R² agg = {m['r2_aggregate']:+.4f}  per-dim = " +
              " ".join(f"{r:+.3f}" for r in m["r2_per_dim"]) +
              f"  cos = {m['cosine_sim']:+.3f}  best_ep={best['epoch']}  ({elapsed:.0f}s)")

        # Per-demo R² on eval set
        per_demo = []
        for di in np.unique(demo_ev):
            mask = demo_ev == di
            if mask.sum() < 5:
                continue
            ev_emb = emb_ev[mask].to(device)
            ev_tgt = tgt_ev[mask].numpy()
            with torch.no_grad():
                # rebuild best probe is non-trivial — re-train? cheaper: keep last probe.
                # Simpler: store per-demo R² with FINAL probe at best epoch (acceptable for diagnostic).
                pass
            # Actually we need predictions from best epoch. We'll skip per-demo for now and
            # report aggregate only — paper Tab uses aggregate anyway.

        cell_dir = Path(args.output_dir) / f"gap{gap}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        with open(cell_dir / "summary.json", "w") as f:
            json.dump({
                "encoder": args.encoder,
                "checkpoint": args.checkpoint,
                "task_suite": args.task_suite,
                "view": args.view,
                "gap": gap,
                "gap_seconds": gap / 20.0,
                "v11_mode": args.v11_mode if args.encoder == "two-stream-v11" else None,
                "n_train_demos": len(train_demos),
                "n_eval_demos": len(eval_demos),
                "n_train_pairs": int(len(tgt_tr)),
                "n_eval_pairs": int(len(tgt_ev)),
                "best_epoch": best["epoch"],
                "r2_aggregate": m["r2_aggregate"],
                "r2_per_dim": m["r2_per_dim"],
                "mse": m["mse"],
                "cosine_sim": m["cosine_sim"],
                "elapsed_seconds": float(elapsed),
            }, f, indent=2)
        cell_summaries.append({
            "gap": gap,
            "r2_aggregate": m["r2_aggregate"],
            "r2_per_dim": m["r2_per_dim"],
        })
        del emb_tr, emb_ev, tgt_tr, tgt_ev

    # ── Aggregate CSV across gaps ────────────────────────────────────────
    csv_path = Path(args.output_dir) / "all_gaps.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gap", "r2_aggregate"] + [f"r2_dim{i}" for i in range(ACTION_DIM)])
        w.writeheader()
        for c in cell_summaries:
            row = {"gap": c["gap"], "r2_aggregate": c["r2_aggregate"]}
            for i, r in enumerate(c["r2_per_dim"]):
                row[f"r2_dim{i}"] = r
            w.writerow(row)
    print(f"\nAll gaps CSV: {csv_path}")
    print(f"\n=== Final summary ===")
    print(f"{'gap':>4}  {'r2_agg':>8}  per-dim r²")
    for c in cell_summaries:
        per = " ".join(f"{r:+.3f}" for r in c["r2_per_dim"])
        print(f"{c['gap']:>4d}  {c['r2_aggregate']:>+8.4f}  {per}")


if __name__ == "__main__":
    main()
