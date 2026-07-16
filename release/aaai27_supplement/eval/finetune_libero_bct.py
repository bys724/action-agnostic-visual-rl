#!/usr/bin/env python
"""Training driver: LIBERO official BC-Transformer policy + our encoder adapter.

This script only produces the BC checkpoint; it does not use the simulator
(closed-loop rollout/evaluation lives in a separate driver, eval_libero.py).

Usage:
    python eval/finetune_libero_bct.py \\
        --encoder comp-mae \\
        --checkpoint <CHECKPOINT_DIR>/comp_mae_pretrain.pt \\
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
# robomimic 0.x uses the deprecated np.bool -> monkeypatch for numpy 1.20+.
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

# Bootstrap: make the supplement root (containing model/ and eval/) importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # supplement root (contains model/, eval/)

# libero is imported from the conda env site-packages (documented runtime dep).
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.utils import control_seed

from eval.policies.bc_transformer_adapted import AdaptedBCTransformerPolicy


# ============================================================================
# Config builder (compatible with the LIBERO BC-T config schema)
# ============================================================================

def build_cfg(args, shape_meta) -> OmegaConf:
    """Config compatible with LIBERO BCTransformerPolicy + our adapter."""
    embed_size = args.embed_size
    cfg_dict = {
        "encoder": {
            "type": args.encoder,
            "checkpoint": args.checkpoint,
            # pooling/use_m are comp-mae-specific -- avoid passing them as
            # redundant kwargs to other adapters.
            "adapter_kwargs": ({"pooling": args.pooling, "use_m": args.use_m}
                               if args.encoder == "comp-mae" else {}),
        },
        "train": {"use_augmentation": args.use_augmentation},
        "policy": {
            "policy_type": "AdaptedBCTransformerPolicy",
            "embed_size": embed_size,
            "extra_num_layers": 0,
            "extra_hidden_size": 128,
            # ColorJitter + Translation augmentation (LIBERO standard: DataAugGroup
            # concatenates viewpoints along dim=1 and applies a single random
            # transform, so viewpoint/camera consistency is preserved automatically).
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
                    # input_shape is auto-injected by BasePolicy.__init__ from
                    # shape_meta. translation=4 (LIBERO official default).
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
            # ExtraModalityTokens expects an 'ee_states' key, but the LIBERO
            # standard splits ee into ee_pos/ee_ori -> disable ee to keep it simple.
            "use_ee": False,
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
        # shape_meta contains an OrderedDict -> not omegaconf-compatible, so it
        # is passed as a separate argument.
        "task_embedding_format": "clip",
        "task_embedding_one_hot_offset": 1,
        "device": "cuda",
        "seed": args.seed,
    }
    return OmegaConf.create(cfg_dict)


# ============================================================================
# Image preprocessing (LIBERO obs 128 -> encoder native size)
# ============================================================================

def resize_obs_inplace(batch: dict, image_keys: list, target_size: int):
    """batch["obs"][k]: (B, T, C, H, W) -> (B, T, C, target, target)."""
    obs = batch["obs"]
    for k in image_keys:
        x = obs[k]
        B, T, C, H, W = x.shape
        if H == target_size and W == target_size:
            continue
        # robomimic handles uint8 -> float [0,1]; here we assume float input.
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
    """Trim actions to match the GMM dist's (B, T_out) batch_shape (causal:
    keep the last T_out steps).

    If an adapter shortens the time dim in its forward pass (T_in -> T_out),
    actions (B, T_in, ac_dim) must be aligned to the last T_out steps so the
    log_prob shape matches.
    """
    T_out = dist.batch_shape[1]
    T_act = actions.shape[1]
    if T_out == T_act:
        return actions
    if T_out > T_act:
        raise ValueError(f"dist T_out={T_out} > actions T_act={T_act} (cannot causal-trim)")
    return actions[:, -T_out:]


def apply_augmentation(policy, batch, image_keys):
    """Call the LIBERO `DataAugGroup` at training time (viewpoint/camera-
    consistent augmentation).

    AdaptedBCTransformerPolicy lacks the `image_encoders` attribute, so it does
    not go through the LIBERO `BasePolicy.preprocess_input` automatic flow ->
    it must be called explicitly here in train_one_epoch.

    Important: the TranslationAug crop_randomizer is auto-configured by
    BasePolicy.__init__ against the LIBERO native shape_meta (3, 128, 128).
    Therefore augmentation must be applied *before* `resize_obs_inplace`
    (i.e. on the raw 128x128 frames).
    """
    if not policy.cfg.train.use_augmentation:
        return
    img_tuple = tuple(batch["obs"][k] for k in image_keys)
    aug_out = policy.img_aug(img_tuple)
    for i, k in enumerate(image_keys):
        batch["obs"][k] = aug_out[i]


def save_aug_check_png(policy, batch, image_keys, output_path, n_samples=2, n_steps=4):
    """Save a before/after augmentation grid of the first batch to a single PNG.

    Verification points (visual sanity check before training):
      - Do adjacent timesteps (t-3..t) within a row receive the same
        augmentation (TranslationAug crop offset)?
      - Are different cameras of the same sample augmented together
        (DataAugGroup dim=1 concat)?

    Important: the caller must pass the batch *before* resize (LIBERO native
    128x128).

    Layout (2 rows per sample s x camera c = raw / aug):
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
    print(f"[aug-check] layout: {n_samples} samples x {len(image_keys)} cams x (raw, aug) x {n_steps} steps")


def _log_first_batch_stats(batch, image_keys):
    """Print image dtype/range/shape of the first batch (to verify the input
    distribution matches the encoder's native expectation)."""
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
                     log_every=50, max_batches=None, debug_first_batch=False, amp=False):
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

        # Augmentation runs on LIBERO native (128) -> apply before resize.
        apply_augmentation(policy, batch, image_keys)

        # Resize obs to encoder native size
        resize_obs_inplace(batch, image_keys, img_size)

        if debug_first_batch and i == 0:
            _log_first_batch_stats(batch, image_keys)

        # Forward (returns GMM dist via policy_head). AMP bf16 speeds up the
        # frozen ViT forward (autocast auto-promotes sensitive ops like logsumexp
        # to fp32, so the GMM log_prob stays safe; bf16 needs no GradScaler).
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            dist = policy(batch)
            # GMM negative log-likelihood (causal-trim actions if an adapter
            # shortened the time dim: T_out < T_act).
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
def evaluate(policy, loader, device, image_keys, img_size, amp=False):
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
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
                        choices=["comp-mae",
                                 "videomae-ours",
                                 "dinov2", "siglip", "vc1", "vjepa2-1"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Encoder checkpoint (may be None for encoders with "
                             "public pretrained weights)")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "attentive"],
                        help="comp-mae spatial pooling: mean | attentive "
                             "(per-stream learnable query; encoder frozen, only the query trains)")
    parser.add_argument("--use-m", action="store_true",
                        help="comp-mae: add the M encoder stream (|dL| curr-prev motion) "
                             "-> P_t (+) P_tk (+) M. M encoder stays frozen; only the pooler trains")
    parser.add_argument("--amp", action="store_true",
                        help="bf16 autocast (speeds up the frozen ViT forward ~1.5-2x; "
                             "negligible accuracy impact since the encoder is frozen)")
    parser.add_argument("--p-depth", type=int, default=12)
    parser.add_argument("--m-depth", type=int, default=6)

    # Data
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal",
                                 "libero_10", "libero_90"])
    parser.add_argument("--data-root", type=str, default=None,
                        help="LIBERO dataset root (HDF5 demos). "
                             "If None, uses libero's get_libero_path('datasets').")
    parser.add_argument("--bddl-folder", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length (25 for video encoders, 10 otherwise)")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Subset of task IDs (for sanity tests). None = all tasks")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Max batches per epoch (for sanity tests)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed-size", type=int, default=64,
                        help="BC-T embed_size (LIBERO default 64)")

    # Augmentation
    parser.add_argument("--img-size", type=int, default=224,
                        help="Encoder native input size for the ColorJitter input_shape "
                             "(video encoders may use 384)")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable augmentation (default: augmentation on)")
    parser.add_argument("--aug-check-png", type=str, default=None,
                        help="Save a before/after augmentation PNG from the first batch "
                             "(visual check of viewpoint consistency). None = don't save. "
                             "Recommended to enable for one sanity job.")

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()
    args.use_augmentation = not args.no_augmentation

    control_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Encoder: {args.encoder}")
    print(f"Task suite: {args.task_suite} | seq_len: {args.seq_len}")

    # -- 1. Benchmark + datasets (LIBERO official) ------------------------
    bm_cls = get_benchmark(args.task_suite)
    benchmark = bm_cls(0)  # task_order_index=0
    n_tasks = benchmark.n_tasks
    print(f"Loaded benchmark {args.task_suite} with {n_tasks} tasks")

    # Build cfg with placeholder shape_meta first; the real shape_meta is
    # obtained below. Load each task's HDF5 dataset.
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

    # -- 2. Task embeddings (CLIP text encoder, 512-d) --------------------
    # LIBERO get_task_embs has transformers-version compatibility issues, so we
    # call CLIP directly. In this env, get_text_features returns an output
    # object -> use text_model.pooler_output instead (no projection applied;
    # the language_encoder MLP absorbs the difference).
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

    # -- 3. Wrap each dataset with task_emb -------------------------------
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

    # -- 4. Build policy --------------------------------------------------
    cfg = build_cfg(args, shape_meta)
    policy = AdaptedBCTransformerPolicy(cfg, shape_meta).to(device)

    image_keys = list(cfg.data.obs.modality.rgb)
    img_size = policy.adapter.img_size
    print(f"Adapter native img_size: {img_size}")

    total_params = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy params: {total_params/1e6:.1f}M total | {trainable/1e6:.1f}M trainable")

    # -- 5. Optimizer + scheduler -----------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # -- 6. Output dir ----------------------------------------------------
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            f"outputs/libero_bct/"
            f"{args.encoder}_{args.task_suite}_seed{args.seed}_{ts}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")
    OmegaConf.save(cfg, os.path.join(args.output_dir, "config.yaml"))

    # -- 6.5. (optional) visual augmentation-consistency check ------------
    # NOTE: augmentation runs on LIBERO native (128) (TranslationAug
    # crop_randomizer is auto-set from shape_meta input_shape) -> call before resize.
    if args.aug_check_png:
        first_batch = next(iter(train_loader))
        for k in first_batch["obs"]:
            first_batch["obs"][k] = first_batch["obs"][k].to(device, non_blocking=True)
        save_aug_check_png(policy, first_batch, image_keys, args.aug_check_png)

    # -- 7. Training loop -------------------------------------------------
    best_eval_loss = float("inf")
    history = {"train": [], "eval": []}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            policy, train_loader, optimizer, device, image_keys, img_size,
            max_batches=args.max_train_batches,
            debug_first_batch=(epoch == 1),
            amp=args.amp,
        )
        eval_loss = evaluate(policy, eval_loader, device, image_keys, img_size, amp=args.amp)
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
