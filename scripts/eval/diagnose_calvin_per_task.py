#!/usr/bin/env python
"""Case 1 diagnostic: per-task R² breakdown for CALVIN cross-folder OOD probing.

5 encoder × probe (training/ 2000 segs) × eval (validation/ 1087 segs)
→ per-task R² (34 unique tasks)

목적: v15가 모든 task에서 일관 약한지, 또는 specific task에서만 약한지 확인.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.calvin import calvin_action_target, load_segment_frames, load_segments
from scripts.eval.probe_action_libero import (
    build_standard_encoder, build_v11_encoder,
    encode_pairs_v11, encode_pairs_via_adapter,
    preprocess_frames,
)

DATA_ROOT = Path("/proj/external_group/mrg/datasets/calvin/task_ABCD_D")
GAP = 30
N_TRAIN_SEG = 2000
SEED = 42
DEVICE = "cuda"
V15_CKPT = "/proj/external_group/mrg/checkpoints/two_stream_v15/20260511_045319/latest.pt"
VIDEOMAE_CKPT = "/proj/external_group/mrg/checkpoints/videomae/20260415_012017/best_model.pt"

ENCODER_CONFIGS = [
    ("two-stream-v11", V15_CKPT, "v15"),
    ("videomae-ours", VIDEOMAE_CKPT, "videomae-ours"),
    ("dinov2", None, "dinov2"),
    ("siglip", None, "siglip"),
    ("vc1", None, "vc1"),
]


def build_encoder(enc_type, ckpt):
    if enc_type == "two-stream-v11":
        model, fwd = build_v11_encoder(ckpt, 12, 6, DEVICE)
        return (lambda p, c: encode_pairs_v11(model, fwd, p, c, DEVICE, mode="p_t_p_tk", batch=64)), 224
    kwargs = {"mode": "paired"} if enc_type == "videomae-ours" else {}
    adapter = build_standard_encoder(enc_type, ckpt, DEVICE, **kwargs)
    return (lambda p, c: encode_pairs_via_adapter(adapter, p, c, DEVICE, batch=64)), adapter.img_size


def collect(seg_list, base_dir, img_size, encode_fn):
    embs, tgts, ep_ids = [], [], []
    for ei, (s, e, _) in enumerate(seg_list):
        frames, robot_obs, actions = load_segment_frames(base_dir, s, e, view="rgb_static")
        T = frames.shape[0]
        if T <= GAP + 1:
            continue
        tg = np.stack([calvin_action_target(robot_obs, actions, t, GAP) for t in range(T - GAP)])
        prev = preprocess_frames(frames[:T - GAP], img_size)
        curr = preprocess_frames(frames[GAP:], img_size)
        emb = encode_fn(prev, curr)
        embs.append(emb); tgts.append(tg); ep_ids.extend([ei] * (T - GAP))
    return torch.cat(embs), torch.from_numpy(np.concatenate(tgts)), np.array(ep_ids)


def main():
    val_segs = load_segments(DATA_ROOT / "validation")
    task_to_seg_idx = defaultdict(list)
    for i, (_, _, task) in enumerate(val_segs):
        task_to_seg_idx[task].append(i)
    print(f"validation: {len(val_segs)} segments, {len(task_to_seg_idx)} unique tasks", flush=True)

    train_segs_all = load_segments(DATA_ROOT / "training")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train_segs_all))[:N_TRAIN_SEG]
    train_segs = [train_segs_all[i] for i in idx]
    print(f"training: {N_TRAIN_SEG} sub-sample segments", flush=True)

    results = {}
    for enc_type, ckpt, label in ENCODER_CONFIGS:
        print(f"\n=== {label} ===", flush=True)
        encode_fn, img_size = build_encoder(enc_type, ckpt)
        emb_tr, tgt_tr, _ = collect(train_segs, DATA_ROOT / "training", img_size, encode_fn)
        print(f"  train: {len(tgt_tr)} pairs", flush=True)
        emb_ev, tgt_ev, ep_ev = collect(val_segs, DATA_ROOT / "validation", img_size, encode_fn)
        print(f"  eval:  {len(tgt_ev)} pairs across {len(np.unique(ep_ev))} segments", flush=True)

        probe = nn.Linear(emb_tr.shape[1], 7).to(DEVICE)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        loader = DataLoader(TensorDataset(emb_tr, tgt_tr), batch_size=256, shuffle=True)
        best = {"r2": -float("inf"), "pred": None}
        for ep in range(20):
            probe.train()
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = F.mse_loss(probe(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                pred = probe(emb_ev.to(DEVICE)).cpu().numpy()
            tgt_np = tgt_ev.numpy()
            ss_r = np.sum((tgt_np - pred) ** 2)
            ss_t = np.sum((tgt_np - tgt_np.mean(0)) ** 2)
            r2 = 1 - ss_r / (ss_t + 1e-8)
            if r2 > best["r2"]:
                best = {"r2": r2, "pred": pred}
        print(f"  R² overall: {best['r2']:+.4f}", flush=True)

        per_task = {}
        tgt_np = tgt_ev.numpy()
        for task, seg_idx in task_to_seg_idx.items():
            mask = np.isin(ep_ev, seg_idx)
            if mask.sum() < 10:
                continue
            t_t = tgt_np[mask]; t_p = best["pred"][mask]
            ss_r = np.sum((t_t - t_p) ** 2)
            ss_t = np.sum((t_t - t_t.mean(0)) ** 2)
            per_task[task] = {"r2": float(1 - ss_r / (ss_t + 1e-8)), "n_pairs": int(mask.sum())}
        results[label] = {"r2_overall": float(best["r2"]), "per_task": per_task}

    out = Path("paper_artifacts/calvin_action_probing/_diagnostic/case1_per_task_r2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
