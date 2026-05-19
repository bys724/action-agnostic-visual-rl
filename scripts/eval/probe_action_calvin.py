#!/usr/bin/env python
"""CALVIN Action Probing (§C10 DROID 보강).

DROID image-only R²≈0 한계를 cleaner sim 환경 (CALVIN tabletop) 으로 보완.
한 잡 = (encoder × split) → 4 gaps loop → 4 cells.

CALVIN 30Hz, target = rel_actions cumulative delta (pos 3 + euler 3) + gripper at t+k-1.
Gaps (CALVIN 30Hz 기준):
- gap=1   ~0.033s
- gap=15  ~0.500s
- gap=30  ~1.000s   ≈ LIBERO gap=20 ≈ EgoDex 학습 분포 1초
- gap=60  ~2.000s

probe_action_libero.py 구조 그대로 — 차이는 dataset loader (calvin.py) 만.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.calvin import (
    calvin_action_target,
    list_episodes,
    load_episode_frames,
)
# probe_action_libero에서 재사용 (DRY)
from scripts.eval.probe_action_libero import (
    SUPPORTED_ENCODERS,
    build_standard_encoder,
    build_v11_encoder,
    compute_metrics,
    encode_pairs_v11,
    encode_pairs_via_adapter,
    preprocess_frames,
    train_probe,
)

CALVIN_SPLITS = ("training", "validation")  # task_ABCD_D 압축 풀면 양쪽 모두
DEFAULT_GAPS = [1, 15, 30, 60]   # 30Hz → 0.03/0.5/1.0/2.0s
ACTION_DIM = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=SUPPORTED_ENCODERS)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default="/proj/external_group/mrg/datasets/calvin/task_ABCD_D",
                        help="압축 풀린 CALVIN 루트 (training/, validation/ 하위에 episode .npz)")
    parser.add_argument("--split", default="training", choices=list(CALVIN_SPLITS),
                        help="training (수많은 ep) 또는 validation (소수, sanity용)")
    parser.add_argument("--view", default="rgb_static",
                        choices=["rgb_static", "rgb_gripper"],
                        help="rgb_static (200x200, paper main) | rgb_gripper (84x84, sub)")
    parser.add_argument("--gaps", type=int, nargs="+", default=DEFAULT_GAPS,
                        help="Frame gaps (CALVIN 30Hz: 1=0.03s, 15=0.5s, 30=1s, 60=2s)")
    parser.add_argument("--max-episodes", type=int, default=200,
                        help="Subsample episodes (full = 수천 ep, 너무 많음)")
    parser.add_argument("--max-frames-per-episode", type=int, default=None,
                        help="Optional truncation for very long episodes")
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
    parser.add_argument("--videomae-mode", default="paired",
                        choices=["paired", "p_t_p_tk"])
    parser.add_argument("--v11-mode", default="p_t_p_tk",
                        choices=["abd_prime", "b_only", "d_prime_only", "p_t_p_tk"])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[probe_calvin] encoder={args.encoder} split={args.split} gaps={args.gaps} device={device}")

    split_dir = Path(args.data_root) / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"CALVIN split dir not found: {split_dir}")

    eps = list_episodes(split_dir)
    print(f"  total episodes in {args.split}: {len(eps)}")
    if args.max_episodes and len(eps) > args.max_episodes:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(eps))[:args.max_episodes]
        eps = [eps[i] for i in idx]
        print(f"  subsampled to {len(eps)} episodes")

    # ── Build encoder ────────────────────────────────────────────────────
    if args.encoder == "two-stream-v11":
        if args.checkpoint is None:
            raise ValueError("v11 encoder requires --checkpoint")
        model, fwd = build_v11_encoder(
            args.checkpoint, args.v11_p_depth, args.v11_m_depth, device,
        )
        img_size = 224

        def encode_fn(prev, curr):
            return encode_pairs_v11(model, fwd, prev, curr, device,
                                    mode=args.v11_mode, batch=args.encode_batch)
    else:
        adapter_kwargs = {}
        if args.encoder == "videomae-ours":
            adapter_kwargs["mode"] = args.videomae_mode
        adapter = build_standard_encoder(args.encoder, args.checkpoint, device, **adapter_kwargs)
        img_size = adapter.img_size

        def encode_fn(prev, curr):
            return encode_pairs_via_adapter(adapter, prev, curr, device, batch=args.encode_batch)

    print(f"  img_size={img_size}")

    # ── Episode-level train/eval split ──────────────────────────────────
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(eps))
    n_train = int(len(perm) * args.train_ratio)
    train_eps = [eps[i] for i in perm[:n_train]]
    eval_eps = [eps[i] for i in perm[n_train:]]
    print(f"  episodes: train={len(train_eps)} / eval={len(eval_eps)}")

    # ── Per-gap loop ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    for gap in args.gaps:
        print(f"\n=== gap={gap} ({gap/30:.2f}s @ CALVIN 30Hz) ===")
        t0 = time.time()

        def collect_embed(ep_list, label):
            embed_chunks, tgt_chunks, ep_ids = [], [], []
            for ei, (s, e) in enumerate(ep_list):
                frames, actions = load_episode_frames(split_dir, s, e, view=args.view)
                if args.max_frames_per_episode:
                    frames = frames[:args.max_frames_per_episode]
                    actions = actions[:args.max_frames_per_episode]
                T = frames.shape[0]
                if T <= gap + 1:
                    continue
                tgts = np.stack([
                    calvin_action_target(actions, t, gap) for t in range(T - gap)
                ])
                prev = preprocess_frames(frames[:T - gap], img_size)
                curr = preprocess_frames(frames[gap:], img_size)
                emb = encode_fn(prev, curr)  # (T-gap, D)
                embed_chunks.append(emb)
                tgt_chunks.append(tgts)
                ep_ids.extend([ei] * (T - gap))
                del frames, prev, curr, emb
            print(f"    [{label}] embedded {len(ep_ids)} pairs from {len(embed_chunks)} episodes")
            return (
                torch.cat(embed_chunks, 0),
                torch.from_numpy(np.concatenate(tgt_chunks, 0)),
                np.array(ep_ids),
            )

        print("  encoding train ...")
        emb_tr, tgt_tr, _ = collect_embed(train_eps, "train")
        print("  encoding eval ...")
        emb_ev, tgt_ev, _ = collect_embed(eval_eps, "eval")
        print(f"  pairs: train={len(tgt_tr)} eval={len(tgt_ev)}")

        print(f"  training probe (epoch={args.probe_epochs}, lr={args.probe_lr}) ...")
        best = train_probe(emb_tr, tgt_tr, emb_ev, tgt_ev,
                           epochs=args.probe_epochs, batch_size=args.probe_batch,
                           lr=args.probe_lr, device=str(device))
        m = best["metrics"]
        elapsed = time.time() - t0
        print(f"  R² agg = {m['r2_aggregate']:+.4f}  per-dim = " +
              " ".join(f"{r:+.3f}" for r in m["r2_per_dim"]) +
              f"  cos = {m['cosine_sim']:+.3f}  best_ep={best['epoch']}  ({elapsed:.0f}s)")

        cell_dir = Path(args.output_dir) / f"gap{gap}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        with open(cell_dir / "summary.json", "w") as f:
            json.dump({
                "encoder": args.encoder,
                "checkpoint": args.checkpoint,
                "split": args.split,
                "view": args.view,
                "gap": gap,
                "gap_seconds": gap / 30.0,
                "v11_mode": args.v11_mode if args.encoder == "two-stream-v11" else None,
                "n_train_episodes": len(train_eps),
                "n_eval_episodes": len(eval_eps),
                "n_train_pairs": int(len(tgt_tr)),
                "n_eval_pairs": int(len(tgt_ev)),
                "best_epoch": best["epoch"],
                **m,
            }, f, indent=2)

    # all_gaps.csv
    csv_path = Path(args.output_dir) / "all_gaps.csv"
    with open(csv_path, "w") as f:
        f.write("gap,r2_aggregate," + ",".join(f"r2_dim{i}" for i in range(ACTION_DIM)) + "\n")
        for gap in args.gaps:
            sj = json.load(open(Path(args.output_dir) / f"gap{gap}" / "summary.json"))
            r2pd = ",".join(f"{r:.4f}" for r in sj["r2_per_dim"])
            f.write(f"{gap},{sj['r2_aggregate']:.4f},{r2pd}\n")
    print(f"\n=== Done. Summary CSV: {csv_path} ===")


if __name__ == "__main__":
    main()
