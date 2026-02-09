#!/usr/bin/env python
"""
Fine-tune Vision Encoder on LIBERO benchmark.

This script fine-tunes our pre-trained encoders (Two-Stream, Single-Stream, VideoMAE)
on LIBERO manipulation tasks with a simple action head.

Architecture:
    [img_t, img_tk] -> Encoder -> [B, num_patches, embed_dim]
                                          |
                                    Mean Pooling
                                          |
                                   [B, embed_dim]
                                          |
                                    Action Head
                                          |
                                   [B, 7] (action)

Usage:
    # Fine-tune Two-Stream encoder on LIBERO Spatial
    python scripts/finetune_libero.py --encoder two-stream \
        --checkpoint /workspace/data/checkpoints/two_stream/latest.pt \
        --task-suite libero_spatial --epochs 50

    # Fine-tune from scratch (no pre-training)
    python scripts/finetune_libero.py --encoder two-stream \
        --task-suite libero_spatial --epochs 100 --no-pretrain
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, '/workspace')

from src.models.openvla_encoder import (
    TwoStreamEncoderForOpenVLA,
    SingleStreamEncoderForOpenVLA,
    VideoMAEEncoderForOpenVLA,
)


# =============================================================================
# LIBERO Dataset
# =============================================================================

class LIBERODataset(Dataset):
    """
    LIBERO dataset loader for fine-tuning.

    Loads consecutive frame pairs (obs, next_obs) and actions from HDF5 files.
    """

    def __init__(
        self,
        data_root: str,
        task_suite: str = "libero_spatial",
        task_ids: Optional[List[int]] = None,
        img_size: int = 224,
        include_wrist: bool = False,
    ):
        """
        Args:
            data_root: Path to LIBERO datasets directory
            task_suite: Task suite name (libero_spatial, libero_object, etc.)
            task_ids: List of task IDs to include (None = all)
            img_size: Image resize target
            include_wrist: Include wrist camera as additional channel
        """
        self.data_root = Path(data_root)
        self.task_suite = task_suite
        self.img_size = img_size
        self.include_wrist = include_wrist

        # Find all HDF5 files for this task suite
        suite_dir = self.data_root / task_suite
        if not suite_dir.exists():
            raise FileNotFoundError(f"Task suite directory not found: {suite_dir}")

        hdf5_files = sorted(suite_dir.glob("*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {suite_dir}")

        # Filter by task IDs if specified
        if task_ids is not None:
            hdf5_files = [f for i, f in enumerate(hdf5_files) if i in task_ids]

        # Load all transitions
        self.samples = []  # List of (file_path, demo_idx, step_idx, task_description)

        for hdf5_path in tqdm(hdf5_files, desc="Loading LIBERO data"):
            with h5py.File(hdf5_path, "r") as f:
                # Get task description
                task_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
                task_description = task_info.get("language_instruction", "")

                # Iterate through demos
                for demo_key in sorted(f["data"].keys()):
                    if not demo_key.startswith("demo_"):
                        continue

                    demo_group = f[f"data/{demo_key}"]
                    num_samples = demo_group.attrs.get("num_samples", 0)

                    # Each step becomes a sample (except last, needs next_obs)
                    for step_idx in range(num_samples - 1):
                        self.samples.append((
                            str(hdf5_path),
                            demo_key,
                            step_idx,
                            task_description,
                        ))

        print(f"Loaded {len(self.samples)} transitions from {len(hdf5_files)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path, demo_key, step_idx, task_desc = self.samples[idx]

        with h5py.File(file_path, "r") as f:
            demo_group = f[f"data/{demo_key}"]

            # Get current observation image
            img_t = demo_group["obs/agentview_image"][step_idx]  # [H, W, C]
            img_tk = demo_group["obs/agentview_image"][step_idx + 1]  # Next frame

            # Get action
            action = demo_group["actions"][step_idx]  # [7]

        # Preprocess images
        img_t = self._preprocess_image(img_t)
        img_tk = self._preprocess_image(img_tk)

        # Stack as 6-channel input for encoder
        pixel_values = torch.cat([img_t, img_tk], dim=0)  # [6, H, W]

        return {
            "pixel_values": pixel_values,  # [6, H, W]
            "action": torch.from_numpy(action).float(),  # [7]
            "task_description": task_desc,
        }

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image: resize, normalize, convert to tensor."""
        # img: [H, W, C] uint8
        img = img.astype(np.float32) / 255.0  # [0, 1]

        # Resize if needed
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]

        return img


# =============================================================================
# Action Head
# =============================================================================

class ActionHead(nn.Module):
    """
    Simple action prediction head.

    Takes visual embeddings and predicts 7D action.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        action_dim: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, embed_dim] - visual embedding

        Returns:
            action: [B, action_dim]
        """
        return self.mlp(x)


# =============================================================================
# Fine-tuning Model
# =============================================================================

class EncoderWithActionHead(nn.Module):
    """
    Vision encoder with action prediction head for LIBERO fine-tuning.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 768,
        action_dim: int = 7,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        self.action_head = ActionHead(embed_dim=embed_dim, action_dim=action_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen (only action head will be trained)")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 6, H, W] - two stacked RGB images

        Returns:
            action: [B, 7] - predicted action
        """
        # Encode
        patch_embeddings = self.encoder(pixel_values)  # [B, num_patches, D]

        # Mean pooling over patches
        visual_embedding = patch_embeddings.mean(dim=1)  # [B, D]

        # Predict action
        action = self.action_head(visual_embedding)  # [B, 7]

        return action


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        target_action = batch["action"].to(device)

        optimizer.zero_grad()

        # Forward
        pred_action = model(pixel_values)

        # Loss (MSE)
        loss = F.mse_loss(pred_action, target_action)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_l1_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            target_action = batch["action"].to(device)

            pred_action = model(pixel_values)

            # Metrics
            loss = F.mse_loss(pred_action, target_action)
            l1_error = F.l1_loss(pred_action, target_action)

            total_loss += loss.item()
            total_l1_error += l1_error.item()
            num_batches += 1

    return {
        "mse_loss": total_loss / max(num_batches, 1),
        "l1_error": total_l1_error / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune encoder on LIBERO")

    # Encoder
    parser.add_argument("--encoder", type=str, default="two-stream",
                        choices=["two-stream", "single-stream", "videomae"],
                        help="Encoder type")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pre-trained encoder checkpoint")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Train from scratch (no pre-training)")
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                        help="Freeze encoder (train only action head)")
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="Unfreeze encoder (fine-tune all parameters)")

    # Data
    parser.add_argument("--data-root", type=str,
                        default="/workspace/data/libero/datasets",
                        help="LIBERO datasets root")
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal",
                                 "libero_10", "libero_90"],
                        help="LIBERO task suite")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Specific task IDs to train on")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction of data for evaluation")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Handle freeze/unfreeze
    if args.unfreeze_encoder:
        args.freeze_encoder = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create encoder
    print(f"\nCreating {args.encoder} encoder...")
    if args.encoder == "two-stream":
        encoder = TwoStreamEncoderForOpenVLA()
    elif args.encoder == "single-stream":
        encoder = SingleStreamEncoderForOpenVLA()
    elif args.encoder == "videomae":
        encoder = VideoMAEEncoderForOpenVLA()
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    # Load pre-trained weights
    if args.checkpoint and not args.no_pretrain:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Map weights (filter encoder-related only)
        encoder_state = {}
        for key, value in state_dict.items():
            # Handle different checkpoint formats
            if key.startswith("module."):
                key = key[7:]  # Remove DataParallel prefix

            if key.startswith("preprocessing.") or \
               key.startswith("encoder.") or \
               key.startswith("fusion."):
                encoder_state[key] = value
            elif key.startswith("patch_embed") or \
                 key.startswith("cls_token") or \
                 key.startswith("pos_embed") or \
                 key.startswith("blocks.") or \
                 key.startswith("norm"):
                encoder_state[key] = value

        if encoder_state:
            encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded {len(encoder_state)} weight tensors")
        else:
            print("Warning: No matching weights found in checkpoint")

    # Create model with action head
    model = EncoderWithActionHead(
        encoder=encoder,
        embed_dim=encoder.embed_dim,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Load data
    print(f"\nLoading LIBERO {args.task_suite} data...")
    try:
        full_dataset = LIBERODataset(
            data_root=args.data_root,
            task_suite=args.task_suite,
            task_ids=args.task_ids,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nLIBERO dataset not found. Please download it first:")
        print("  cd /workspace/third_party/openpi/third_party/libero")
        print(f"  python benchmark_scripts/download_libero_datasets.py --datasets {args.task_suite}")
        sys.exit(1)

    # Split train/eval
    eval_size = int(len(full_dataset) * args.eval_split)
    train_size = len(full_dataset) - eval_size

    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/workspace/data/checkpoints/libero_{args.encoder}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")

    # Save config
    config = vars(args)
    config["encoder_embed_dim"] = encoder.embed_dim
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("\nStarting training...")
    best_eval_loss = float("inf")
    history = {"train_loss": [], "eval_mse": [], "eval_l1": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        history["train_loss"].append(train_loss)

        # Evaluate
        eval_metrics = evaluate(model, eval_loader, device)
        history["eval_mse"].append(eval_metrics["mse_loss"])
        history["eval_l1"].append(eval_metrics["l1_error"])

        # Update scheduler
        scheduler.step()

        # Print progress
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Eval MSE: {eval_metrics['mse_loss']:.6f}, L1: {eval_metrics['l1_error']:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if eval_metrics["mse_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["mse_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "eval_mse": eval_metrics["mse_loss"],
                "config": config,
            }, os.path.join(args.output_dir, "best.pt"))
            print("  [Saved best model]")

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": config,
            }, os.path.join(args.output_dir, f"epoch_{epoch}.pt"))

    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": config,
    }, os.path.join(args.output_dir, "final.pt"))

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best eval MSE: {best_eval_loss:.6f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
