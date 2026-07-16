#!/usr/bin/env python
"""CALVIN action probing.

Complements the near-zero image-only R^2 on a static real-robot dataset with a
cleaner simulation environment (CALVIN tabletop). One job = (encoder x split);
the inner loop sweeps 4 temporal gaps to produce 4 cells.

CALVIN runs at 30 Hz. Gaps (raw 30 Hz):
    - gap=1   ~0.033 s
    - gap=15  ~0.500 s
    - gap=30  ~1.000 s   (matches the ~1 s temporal scale of the pretraining data)
    - gap=60  ~2.000 s

Structurally identical to probe_action_libero.py -- the only difference is the
dataset loader (calvin.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the supplement root importable so the `model` and `eval` packages resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # supplement root (contains model/, eval/)

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eval.datasets.calvin import (
    calvin_action_target,
    load_segments,
    load_segment_frames,
)
# Reused from the LIBERO probe (DRY).
from eval.probe_action_libero import (
    SUPPORTED_ENCODERS,
    build_comp_mae_encoder,
    build_standard_encoder,
    build_videomae_token_encoder,
    compute_metrics,
    encode_pairs_comp_mae,
    encode_pairs_via_adapter,
    encode_pairs_videomae_vla,
    preprocess_frames,
    train_probe,
)

CALVIN_SPLITS = ("training", "validation")  # both present after unpacking task_ABCD_D
DEFAULT_GAPS = [10, 20, 30, 45]  # segment-based raw 30Hz: 0.33/0.67/1.00/1.50s
ACTION_DIM = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=SUPPORTED_ENCODERS)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=None, required=False,
                        help="Path to the unpacked CALVIN root, i.e. <CALVIN_DATA_ROOT> "
                             "with training/ and validation/ subfolders of episode .npz. Required to run.")
    parser.add_argument("--split", default="training", choices=list(CALVIN_SPLITS),
                        help="single-folder self-contained 80:20 split (legacy)")
    parser.add_argument("--cross-folder", action="store_true",
                        help="If set: train the probe on training/ segments and evaluate R^2 on "
                             "validation/ segments (true OOD test).")
    parser.add_argument("--view", default="rgb_static",
                        choices=["rgb_static", "rgb_gripper"],
                        help="rgb_static (200x200, main) | rgb_gripper (84x84, sub)")
    parser.add_argument("--gaps", type=int, nargs="+", default=DEFAULT_GAPS,
                        help="Frame gaps (CALVIN 30Hz: 1=0.03s, 15=0.5s, 30=1s, 60=2s)")
    parser.add_argument("--max-episodes", type=int, default=200,
                        help="Subsample episodes (full = thousands of episodes, too many)")
    parser.add_argument("--max-frames-per-episode", type=int, default=None,
                        help="Optional truncation for very long episodes")
    parser.add_argument("--frame-stride", type=int, default=10,
                        help="CALVIN 30Hz -> stride=10 = 3Hz effective sampling. "
                             "Episodes average ~13k frames -> ~1.3k after stride=10. "
                             "--gaps apply on the strided frame index (gap=3 @ stride=10 = 1.0s).")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-batch", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--videomae-mode", default="paired",
                        choices=["paired", "p_t_p_tk"])
    # CoMP-MAE / VideoMAE readout axis.
    parser.add_argument("--readout", default="mean", choices=["mean", "attentive"],
                        help="mean = patch-mean concat (LinearProbe) / attentive = per-stream query pool (AttentivePoolProbe)")
    parser.add_argument("--stream-mode", default="p_t_p_tk", choices=["p_t_p_tk", "p_t_m"],
                        help="comp-mae 2-stream: p_t_p_tk(appearance) / p_t_m(P(t)+M motion)")
    parser.add_argument("--videomae-encoder", default="adapter", choices=["adapter", "vla"],
                        help="adapter = BC-T adapter (mean only) / vla = token-level encoder (mean+attentive self-consistent)")
    parser.add_argument("--probe-weight-decay", type=float, default=0.0,
                        help="AdamW weight decay (suppresses attentive P-appearance overfit). default 0")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[probe_calvin] encoder={args.encoder} split={args.split} gaps={args.gaps} device={device}")

    # Segment-based sampling: a random pair from a CALVIN multi-task chain can straddle
    # a task boundary, which would be unfair vs. LIBERO. Sampling frame pairs only within
    # the same task segment gives every encoder the same fair condition.
    if args.cross_folder:
        # Main: train on training/ segments + evaluate on validation/ segments (true OOD).
        train_split_dir = Path(args.data_root) / "training"
        eval_split_dir = Path(args.data_root) / "validation"
        if not train_split_dir.exists() or not eval_split_dir.exists():
            raise FileNotFoundError(f"Cross-folder: need both training/ and validation/ under {args.data_root}")
        train_segments_all = load_segments(train_split_dir)
        eval_segments_all = load_segments(eval_split_dir)
        print(f"  CROSS-FOLDER mode: train segments={len(train_segments_all)} (training/) | "
              f"eval segments={len(eval_segments_all)} (validation/, OOD)")
        if args.max_episodes and len(train_segments_all) > args.max_episodes:
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(len(train_segments_all))[:args.max_episodes]
            train_segments_all = [train_segments_all[i] for i in idx]
            print(f"  train sub-sampled to {len(train_segments_all)} segments (max_episodes cap)")
        # Main matrix: eval uses all validation segments.
    else:
        # Legacy: 80:20 self-contained split within a single folder.
        split_dir = Path(args.data_root) / args.split
        if not split_dir.exists():
            raise FileNotFoundError(f"CALVIN split dir not found: {split_dir}")
        segments = load_segments(split_dir)
        print(f"  total task segments in {args.split}: {len(segments)}")
        if args.max_episodes and len(segments) > args.max_episodes:
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(len(segments))[:args.max_episodes]
            segments = [segments[i] for i in idx]
            print(f"  subsampled to {len(segments)} segments")

    # ── Build encoder ────────────────────────────────────────────────────
    n_streams = 1
    if args.encoder == "comp-mae":
        model = build_comp_mae_encoder(args.checkpoint, device)
        img_size = 224
        n_streams = 2

        def encode_fn(prev, curr):
            return encode_pairs_comp_mae(model, prev, curr, device,
                                      mode=args.stream_mode, readout=args.readout,
                                      batch=args.encode_batch)
    elif args.encoder == "videomae-ours" and args.videomae_encoder == "vla":
        # Baseline path -- loader is stubbed in this release.
        model = build_videomae_token_encoder(args.checkpoint, device)
        img_size = 224
        n_streams = 2

        def encode_fn(prev, curr):
            return encode_pairs_videomae_vla(model, prev, curr, device,
                                             readout=args.readout, batch=args.encode_batch)
    else:
        # Remaining baselines route through build_standard_encoder (stubbed in this release).
        if args.readout == "attentive":
            raise ValueError(f"attentive readout supports comp-mae/videomae(vla) only (encoder={args.encoder})")
        adapter_kwargs = {}
        if args.encoder == "videomae-ours":
            adapter_kwargs["mode"] = args.videomae_mode
        adapter = build_standard_encoder(args.encoder, args.checkpoint, device, **adapter_kwargs)
        img_size = adapter.img_size

        def encode_fn(prev, curr):
            return encode_pairs_via_adapter(adapter, prev, curr, device, batch=args.encode_batch)

    print(f"  img_size={img_size}  readout={args.readout}  n_streams={n_streams}")

    # ── Segment-level train/eval split ───────────────────────────────────
    if args.cross_folder:
        train_segs = train_segments_all
        eval_segs = eval_segments_all
        train_dir = train_split_dir
        eval_dir = eval_split_dir
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(segments))
        n_train = int(len(perm) * args.train_ratio)
        train_segs = [segments[i] for i in perm[:n_train]]
        eval_segs = [segments[i] for i in perm[n_train:]]
        train_dir = split_dir
        eval_dir = split_dir
    print(f"  segments: train={len(train_segs)} / eval={len(eval_segs)}")

    # ── Per-gap loop ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    # CALVIN raw 30Hz, gap = raw frame interval. gap=30 = 1.0s
    print(f"  segment-based sampling (raw 30Hz, no stride)")

    for gap in args.gaps:
        seconds = gap / 30.0
        print(f"\n=== gap={gap} ({seconds:.2f}s @ raw 30Hz) ===")
        t0 = time.time()

        def collect_embed(seg_list, label, base_dir):
            embed_chunks, tgt_chunks, ep_ids = [], [], []
            for ei, (s, e, task) in enumerate(seg_list):
                frames, robot_obs, actions = load_segment_frames(
                    base_dir, s, e, view=args.view,
                )
                T = frames.shape[0]
                if T <= gap + 1:
                    continue
                tgts = np.stack([
                    calvin_action_target(robot_obs, actions, t, gap)
                    for t in range(T - gap)
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
        emb_tr, tgt_tr, _ = collect_embed(train_segs, "train", train_dir)
        print("  encoding eval ...")
        emb_ev, tgt_ev, _ = collect_embed(eval_segs, "eval", eval_dir)
        print(f"  pairs: train={len(tgt_tr)} eval={len(tgt_ev)}")

        print(f"  training probe (epoch={args.probe_epochs}, lr={args.probe_lr}) ...")
        best = train_probe(emb_tr, tgt_tr, emb_ev, tgt_ev,
                           epochs=args.probe_epochs, batch_size=args.probe_batch,
                           lr=args.probe_lr, device=str(device),
                           readout=args.readout, n_streams=n_streams,
                           weight_decay=args.probe_weight_decay)
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
                "readout": args.readout,
                "stream_mode": args.stream_mode if args.encoder == "comp-mae" else None,
                "n_train_episodes": len(train_segs),
                "n_eval_episodes": len(eval_segs),
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
