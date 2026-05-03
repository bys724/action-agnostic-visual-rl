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
# robomimic 0.x는 deprecated np.bool 사용 → numpy 1.20+ 호환을 위한 monkeypatch
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
# libero는 conda env site-packages에서 import (modules/는 setup 시 복사됨)

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.utils import control_seed

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
        "train": {"use_augmentation": args.use_augmentation},
        "policy": {
            "policy_type": "AdaptedBCTransformerPolicy",
            "embed_size": embed_size,
            "extra_num_layers": 0,
            "extra_hidden_size": 128,
            # V3: ColorJitter + Translation 활성. LIBERO 표준 (DataAugGroup이 dim=1
            # 시점 concat 후 단일 random 적용 → 시점/카메라 일관성 자동 보장).
            "color_aug": {
                "network": "ImgColorJitterAug" if args.use_augmentation else "IdentityAug",
                "network_kwargs": {
                    "input_shape": [3, args.img_size, args.img_size],
                    "brightness": 0.3,
                    "contrast": 0.3,
                    "saturation": 0.3,
                    "hue": 0.3,
                    "epsilon": 0.05,
                } if args.use_augmentation else {},
            },
            "translation_aug": {
                "network": "TranslationAug" if args.use_augmentation else "IdentityAug",
                "network_kwargs": {
                    # input_shape는 BasePolicy.__init__이 shape_meta에서 자동 주입.
                    # translation=4 (LIBERO 공식 default)
                    "translation": 4,
                } if args.use_augmentation else {},
            },
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
                    "input_size": 512,  # CLIP ViT-B/32 text features
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
            "use_joint": True,
            "use_gripper": True,
            "use_ee": False,  # ExtraModalityTokens는 'ee_states' 키 요구 — LIBERO 표준은 ee_pos/ee_ori 분리. ee 비활성화로 단순화.
            "seq_len": args.seq_len,
            "obs": {
                "modality": {
                    "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
                    "depth": [],
                    "low_dim": ["gripper_states", "joint_states"],
                },
            },
            "task_group_size": 1,
            "task_order_index": 0,
        },
        # shape_meta는 OrderedDict 포함 → omegaconf 호환 안됨. 별도 인자로 전달.
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

def _align_actions(dist, actions):
    """GMM dist (B, T_out) batch_shape에 맞게 actions trim (causal: 마지막 T_out개).

    V-JEPA은 어댑터 forward에서 T_in - 15 = T_out으로 줄어들기 때문에
    actions (B, T_in, ac_dim)도 마지막 T_out개로 정렬해야 log_prob shape match.
    """
    T_out = dist.batch_shape[1]
    T_act = actions.shape[1]
    if T_out == T_act:
        return actions
    if T_out > T_act:
        raise ValueError(f"dist T_out={T_out} > actions T_act={T_act} (causal trim 불가)")
    return actions[:, -T_out:]


def save_aug_check_png(policy, batch, image_keys, output_path, n_samples=2, n_steps=4):
    """첫 batch에서 augmentation 적용 전/후를 한 PNG에 저장.

    검증 포인트 (refactor_plan_2026-05-03 §3, V3 학습 시작 전 1회 시각 확인):
      - 한 row 내 인접 시점 (t-3..t)이 동일 augmentation 받는가 (TranslationAug crop offset)
      - 같은 sample의 다른 카메라가 함께 augmented되는가 (DataAugGroup dim=1 concat)

    Layout (sample s × camera c 별 2 row = raw / aug):
        s=0  cam=agent    [raw]   t-3  t-2  t-1  t
                          [aug]   t-3  t-2  t-1  t
        s=0  cam=wrist    [raw]   ...
                          [aug]   ...
        s=1  ...
    """
    import torch
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image

    if not policy.cfg.train.use_augmentation:
        print("[aug-check] use_augmentation=False, skipping viz")
        return

    raws = {k: batch["obs"][k][:n_samples, -n_steps:].detach().cpu() for k in image_keys}

    was_training = policy.img_aug.training
    policy.img_aug.train()
    with torch.no_grad():
        img_tuple = tuple(batch["obs"][k][:n_samples, -n_steps:] for k in image_keys)
        aug_out = policy.img_aug(img_tuple)
    policy.img_aug.train(was_training)

    augs = {k: aug_out[i].detach().cpu() for i, k in enumerate(image_keys)}

    rows = []
    for s in range(n_samples):
        for k in image_keys:
            rows.append(raws[k][s])
            rows.append(augs[k][s])
    all_imgs = torch.cat(rows, dim=0).clamp(0, 1)
    grid = make_grid(all_imgs, nrow=n_steps, padding=4, pad_value=1.0)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(grid).save(str(out))
    print(f"[aug-check] saved {out}")
    print(f"[aug-check] layout: {n_samples} samples × {len(image_keys)} cams × (raw, aug) × {n_steps} steps")


def _log_first_batch_stats(batch, image_keys):
    """첫 batch에서 image dtype/range/shape 출력 (encoder native 분포 일치 검증용)."""
    print("[debug] first batch obs/actions stats:")
    for k in image_keys:
        v = batch["obs"][k]
        print(f"  {k}: dtype={v.dtype} shape={tuple(v.shape)} "
              f"min={v.min().item():.4f} max={v.max().item():.4f} "
              f"mean={v.mean().item():.4f}")
    a = batch["actions"]
    print(f"  actions: dtype={a.dtype} shape={tuple(a.shape)} "
          f"min={a.min().item():.4f} max={a.max().item():.4f}")
    if "task_emb" in batch:
        t = batch["task_emb"]
        print(f"  task_emb: dtype={t.dtype} shape={tuple(t.shape)} "
              f"min={t.min().item():.4f} max={t.max().item():.4f}")


def train_one_epoch(policy, loader, optimizer, device, image_keys, img_size,
                     log_every=50, max_batches=None, debug_first_batch=False):
    policy.train()
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        # Move to device
        for k in batch["obs"]:
            batch["obs"][k] = batch["obs"][k].to(device, non_blocking=True)
        batch["actions"] = batch["actions"].to(device, non_blocking=True)
        if "task_emb" in batch:
            batch["task_emb"] = batch["task_emb"].to(device, non_blocking=True)

        # Resize obs to encoder native size
        resize_obs_inplace(batch, image_keys, img_size)

        if debug_first_batch and i == 0:
            _log_first_batch_stats(batch, image_keys)

        # Forward (returns GMM dist via policy_head)
        dist = policy(batch)
        # GMM negative log-likelihood (V-JEPA은 T_out < T_act → causal trim)
        actions_aligned = _align_actions(dist, batch["actions"])
        loss = -dist.log_prob(actions_aligned).mean()

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
        actions_aligned = _align_actions(dist, batch["actions"])
        loss = -dist.log_prob(actions_aligned).mean()
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
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="task ID 부분집합 (sanity test용). None이면 전체")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="epoch당 최대 batch 수 제한 (sanity용)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed-size", type=int, default=64,
                        help="BC-T embed_size (LIBERO default 64)")

    # V3 augmentation
    parser.add_argument("--img-size", type=int, default=224,
                        help="Encoder native input size for ColorJitter input_shape (V-JEPA은 384)")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="V2 호환: augmentation 끄기 (default: V3 aug 켜짐)")
    parser.add_argument("--aug-check-png", type=str, default=None,
                        help="첫 batch에서 augmentation 적용 전/후 PNG 저장 (시점 일관성 시각 검증). "
                             "None이면 저장 안 함. sanity 1잡에서 활성화 권장.")

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()
    args.use_augmentation = not args.no_augmentation

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
        "low_dim": ["gripper_states", "joint_states"],
    }

    task_indices = args.task_ids if args.task_ids is not None else list(range(n_tasks))
    print(f"Using task IDs: {task_indices}")

    for n, i in enumerate(task_indices):
        ds, sm = get_dataset(
            dataset_path=os.path.join(folder, benchmark.get_task_demonstration(i)),
            obs_modality=obs_modality,
            initialize_obs_utils=(n == 0),
            seq_len=args.seq_len,
        )
        if shape_meta is None:
            shape_meta = sm
        manip_datasets.append(ds)
        descriptions.append(benchmark.get_task(i).language)

    # ── 2. Task embeddings (CLIP text encoder, 512-d) ─────────────────────
    # LIBERO get_task_embs는 transformers 버전 호환 이슈 → 직접 호출
    # 이 conda env의 transformers는 get_text_features가 output object를 반환 →
    # text_model.pooler_output로 우회 (projection 미적용 — language_encoder MLP가 흡수)
    from transformers import CLIPTokenizer, CLIPModel
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    with torch.no_grad():
        toks = clip_tok(
            descriptions, padding="max_length", max_length=25,
            truncation=True, return_tensors="pt",
        ).to(device)
        text_out = clip_model.text_model(**toks)
        # pooler_output: (n_tasks, hidden_dim=512 for ViT-B/32)
        if hasattr(text_out, "pooler_output"):
            feats = text_out.pooler_output
        else:
            feats = text_out[1] if isinstance(text_out, tuple) else text_out
    task_embs = feats.detach().cpu()
    del clip_model, clip_tok

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

    # ── 6.5. (선택) augmentation 일관성 시각 검증 ─────────────────────────
    if args.aug_check_png:
        first_batch = next(iter(train_loader))
        for k in first_batch["obs"]:
            first_batch["obs"][k] = first_batch["obs"][k].to(device, non_blocking=True)
        resize_obs_inplace(first_batch, image_keys, img_size)
        save_aug_check_png(policy, first_batch, image_keys, args.aug_check_png)

    # ── 7. Training loop ─────────────────────────────────────────────────
    best_eval_loss = float("inf")
    history = {"train": [], "eval": []}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            policy, train_loader, optimizer, device, image_keys, img_size,
            max_batches=args.max_train_batches,
            debug_first_batch=(epoch == 1),
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
