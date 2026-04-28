#!/usr/bin/env python
"""LIBERO 공식 BC-Transformer 정책 + 우리 인코더 어댑터 학습 driver.

학습은 클러스터에서 실행, 평가/rollout은 로컬 워크스테이션에서 (별도 driver).
이 스크립트는 BC ckpt만 생성. 시뮬레이터 미사용.

Usage:
    python scripts/eval/finetune_libero_bct.py \\
        --encoder two-stream-v11 \\
        --checkpoint /proj/.../two_stream_v11/.../checkpoint_epoch0044.pt \\
        --task-suite libero_spatial --epochs 50 --batch-size 32 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LIBERO_REPO = _PROJECT_ROOT / "external" / "LIBERO"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_LIBERO_REPO))

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.utils import control_seed, get_task_embs

from src.policies.bc_transformer_adapted import AdaptedBCTransformerPolicy


# ============================================================================
# Config builder (LIBERO BC-T config schema 호환)
# ============================================================================

def build_cfg(args, shape_meta) -> OmegaConf:
    """LIBERO BCTransformerPolicy + 우리 어댑터 호환 config."""
    embed_size = args.embed_size
    cfg_dict = {
        "encoder": {
            "type": args.encoder,
            "checkpoint": args.checkpoint,
            "adapter_kwargs": {},
        },
        "policy": {
            "policy_type": "AdaptedBCTransformerPolicy",
            "embed_size": embed_size,
            "extra_num_layers": 0,
            "extra_hidden_size": 128,
            "transformer_input_size": None,
            "transformer_num_layers": 4,
            "transformer_num_heads": 6,
            "transformer_head_output_size": 64,
            "transformer_mlp_hidden_size": 256,
            "transformer_dropout": 0.1,
            "transformer_max_seq_len": args.seq_len,
            "language_encoder": {
                "network": "MLPEncoder",
                "network_kwargs": {
                    "input_size": 768,  # CLIP/BERT default
                    "hidden_size": 128,
                    "num_layers": 1,
                    "output_size": embed_size,
                },
            },
            "temporal_position_encoding": {
                "network": "SinusoidalPositionEncoding",
                "network_kwargs": {"input_size": embed_size, "inv_freq_factor": 10},
            },
            "policy_head": {
                "network": "GMMHead",
                "network_kwargs": {
                    "hidden_size": 1024,
                    "num_layers": 2,
                    "min_std": 0.0001,
                    "num_modes": 5,
                    "activation": "softplus",
                    "low_eval_noise": False,
                },
                "loss_kwargs": {"loss_coef": 1.0},
            },
        },
        "data": {
            "use_joint": False,
            "use_gripper": True,
            "use_ee": True,
            "seq_len": args.seq_len,
            "obs": {
                "modality": {
                    "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
                    "depth": [],
                    "low_dim": ["gripper_states", "ee_pos", "ee_ori"],
                },
            },
            "task_group_size": 1,
            "task_order_index": 0,
        },
        "shape_meta": shape_meta,
        "task_embedding_format": "clip",
        "task_embedding_one_hot_offset": 1,
        "device": "cuda",
        "seed": args.seed,
    }
    return OmegaConf.create(cfg_dict)


# ============================================================================
# Image preprocessing (LIBERO obs 128 → encoder native size)
# ============================================================================

def resize_obs_inplace(batch: dict, image_keys: list, target_size: int):
    """batch["obs"][k]: (B, T, C, H, W) → (B, T, C, target, target)."""
    obs = batch["obs"]
    for k in image_keys:
        x = obs[k]
        B, T, C, H, W = x.shape
        if H == target_size and W == target_size:
            continue
        # uint8 → float [0,1]은 robomimic이 처리. 여기선 float 가정
        x_flat = x.reshape(B * T, C, H, W)
        x_resized = F.interpolate(
            x_flat, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
        obs[k] = x_resized.reshape(B, T, C, target_size, target_size)


# ============================================================================
# Training loop
# ============================================================================

def train_one_epoch(policy, loader, optimizer, device, image_keys, img_size, log_every=50):
    policy.train()
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        # Move to device
        for k in batch["obs"]:
            batch["obs"][k] = batch["obs"][k].to(device, non_blocking=True)
        batch["actions"] = batch["actions"].to(device, non_blocking=True)
        if "task_emb" in batch:
            batch["task_emb"] = batch["task_emb"].to(device, non_blocking=True)

        # Resize obs to encoder native size
        resize_obs_inplace(batch, image_keys, img_size)

        # Forward (returns GMM dist via policy_head)
        dist = policy(batch)
        # GMM negative log-likelihood
        loss = -dist.log_prob(batch["actions"]).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0,
        )
        optimizer.step()

        total += loss.item()
        n += 1

        if i % log_every == 0:
            print(f"  step {i:5d} | loss {loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def evaluate(policy, loader, device, image_keys, img_size):
    policy.eval()
    total = 0.0
    n = 0
    for batch in loader:
        for k in batch["obs"]:
            batch["obs"][k] = batch["obs"][k].to(device, non_blocking=True)
        batch["actions"] = batch["actions"].to(device, non_blocking=True)
        if "task_emb" in batch:
            batch["task_emb"] = batch["task_emb"].to(device, non_blocking=True)
        resize_obs_inplace(batch, image_keys, img_size)
        dist = policy(batch)
        loss = -dist.log_prob(batch["actions"]).mean()
        total += loss.item()
        n += 1
    return total / max(n, 1)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Encoder
    parser.add_argument("--encoder", type=str, required=True,
                        choices=["two-stream-v11", "videomae-ours",
                                 "dinov2", "siglip", "vc1", "vjepa2-1"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Encoder ckpt (V-JEPA/SigLIP/DINOv2은 None 가능)")
    parser.add_argument("--p-depth", type=int, default=12)
    parser.add_argument("--m-depth", type=int, default=6)

    # Data
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal",
                                 "libero_10", "libero_90"])
    parser.add_argument("--data-root", type=str,
                        default="/proj/external_group/mrg/datasets/libero")
    parser.add_argument("--bddl-folder", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=10,
                        help="V-JEPA용 25, 그 외 10")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed-size", type=int, default=64,
                        help="BC-T embed_size (LIBERO default 64)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()

    control_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Encoder: {args.encoder}")
    print(f"Task suite: {args.task_suite} | seq_len: {args.seq_len}")

    # ── 1. Benchmark + datasets (LIBERO 공식) ─────────────────────────────
    bm_cls = get_benchmark(args.task_suite)
    benchmark = bm_cls(0)  # task_order_index=0
    n_tasks = benchmark.n_tasks
    print(f"Loaded benchmark {args.task_suite} with {n_tasks} tasks")

    # Build cfg with placeholder shape_meta first; we get real shape_meta below
    # Load each task's HDF5 dataset
    folder = args.data_root or get_libero_path("datasets")

    manip_datasets = []
    descriptions = []
    shape_meta = None

    obs_modality = {
        "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
        "depth": [],
        "low_dim": ["gripper_states", "ee_pos", "ee_ori"],
    }

    for i in range(n_tasks):
        ds, sm = get_dataset(
            dataset_path=os.path.join(folder, benchmark.get_task_demonstration(i)),
            obs_modality=obs_modality,
            initialize_obs_utils=(i == 0),
            seq_len=args.seq_len,
        )
        if shape_meta is None:
            shape_meta = sm
        manip_datasets.append(ds)
        descriptions.append(benchmark.get_task(i).language)

    # ── 2. Task embeddings (CLIP) ─────────────────────────────────────────
    cfg_emb = OmegaConf.create({
        "task_embedding_format": "clip",
        "task_embedding_one_hot_offset": 1,
        "device": str(device),
    })
    task_embs = get_task_embs(cfg_emb, descriptions)

    # ── 3. Wrap each dataset with task_emb ────────────────────────────────
    wrapped = [
        SequenceVLDataset(ds, emb) for ds, emb in zip(manip_datasets, task_embs)
    ]
    full_ds = ConcatDataset(wrapped)
    print(f"Total sequences: {len(full_ds)}")

    # Train/eval split
    eval_size = int(len(full_ds) * args.eval_split)
    train_size = len(full_ds) - eval_size
    train_ds, eval_ds = torch.utils.data.random_split(
        full_ds, [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # ── 4. Build policy ──────────────────────────────────────────────────
    cfg = build_cfg(args, shape_meta)
    policy = AdaptedBCTransformerPolicy(cfg, shape_meta).to(device)

    image_keys = list(cfg.data.obs.modality.rgb)
    img_size = policy.adapter.img_size
    print(f"Adapter native img_size: {img_size}")

    total_params = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy params: {total_params/1e6:.1f}M total | {trainable/1e6:.1f}M trainable")

    # ── 5. Optimizer + scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # ── 6. Output dir ────────────────────────────────────────────────────
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            f"/proj/external_group/mrg/checkpoints/libero_bct/"
            f"{args.encoder}_{args.task_suite}_seed{args.seed}_{ts}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")
    OmegaConf.save(cfg, os.path.join(args.output_dir, "config.yaml"))

    # ── 7. Training loop ─────────────────────────────────────────────────
    best_eval_loss = float("inf")
    history = {"train": [], "eval": []}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            policy, train_loader, optimizer, device, image_keys, img_size,
        )
        eval_loss = evaluate(policy, eval_loader, device, image_keys, img_size)
        scheduler.step()

        history["train"].append(train_loss)
        history["eval"].append(eval_loss)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train {train_loss:.4f} | eval {eval_loss:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save({
                "epoch": epoch, "policy_state_dict": policy.state_dict(),
                "eval_loss": eval_loss, "config": OmegaConf.to_container(cfg),
            }, os.path.join(args.output_dir, "best.pt"))

        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch, "policy_state_dict": policy.state_dict(),
                "history": history, "config": OmegaConf.to_container(cfg),
            }, os.path.join(args.output_dir, f"epoch_{epoch}.pt"))

    torch.save({
        "epoch": args.epochs, "policy_state_dict": policy.state_dict(),
        "history": history, "config": OmegaConf.to_container(cfg),
    }, os.path.join(args.output_dir, "final.pt"))

    print(f"\nDone. Best eval loss: {best_eval_loss:.4f}")
    print(f"Checkpoints: {args.output_dir}")


if __name__ == "__main__":
    main()
