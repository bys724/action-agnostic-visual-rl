#!/usr/bin/env python
"""
Bridge V2 Action Probing Experiment

Frozen encoder에서 Bridge V2 robot actions를 예측할 수 있는지 테스트.

Cross-embodiment transfer 검증:
- EgoDex (human hands)로 학습한 표현이
- Bridge (robot arms)의 action도 예측 가능한가?

Usage:
    python scripts/probe_action_bridge.py \
        --encoder two-stream \
        --checkpoint data/checkpoints/two_stream/best_model.pt \
        --bridge-root /workspace/data/datasets/bridge_v2
"""

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace")

# Import encoder loading functions from EgoDex probing script
from probe_action import (
    load_encoder,
    encode_batch,
    LinearProbe,
    MLPProbe,
    compute_metrics,
)


# ============================================================================
# Bridge V2 Dataset for Action Probing
# ============================================================================

class BridgeProbingDataset(Dataset):
    """
    Bridge V2 dataset for action probing.

    Assumes data structure:
        bridge_v2/raw/bridge_data_v2/{collection}/{skill}/{date}/raw/traj_group{X}/traj{N}/
            - images0/im_{t}.jpg
            - action.pkl or obs_dict.pkl (contains actions)

    Action format (7-DoF):
        [x, y, z, roll, pitch, yaw, gripper]

    Returns:
        pixel_values: [6, 224, 224] - img_t + img_t+1 stacked
        action: [7] - end-effector action
    """

    def __init__(
        self,
        data_root: str,
        trajectory_paths: list,
        img_size: int = 224,
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.samples = []

        skipped = 0

        for traj_path in trajectory_paths:
            traj_path = Path(traj_path)
            images_dir = traj_path / "images0"

            if not images_dir.exists():
                skipped += 1
                continue

            # Load action data
            action_file = None
            for candidate in ["action.pkl", "obs_dict.pkl", "policy_out.pkl"]:
                candidate_path = traj_path / candidate
                if candidate_path.exists():
                    action_file = candidate_path
                    break

            if action_file is None:
                skipped += 1
                continue

            try:
                with open(action_file, "rb") as f:
                    data = pickle.load(f)

                # Extract actions (format varies)
                if isinstance(data, dict):
                    actions = data.get("actions", data.get("action", None))
                else:
                    actions = data

                if actions is None:
                    skipped += 1
                    continue

                actions = np.array(actions)

                # Count available images
                images = sorted(images_dir.glob("im_*.jpg"))
                num_frames = len(images)

                # Build (t, t+1) pairs
                for t in range(min(num_frames - 1, len(actions))):
                    self.samples.append({
                        "images_dir": str(images_dir),
                        "frame_idx": t,
                        "action": actions[t].astype(np.float32),
                    })

            except Exception as e:
                skipped += 1
                continue

        print(f"BridgeProbingDataset: {len(self.samples)} samples from {len(trajectory_paths)} trajectories")
        if skipped > 0:
            print(f"  Skipped: {skipped} trajectories")

    def _load_frame(self, images_dir: str, frame_idx: int) -> torch.Tensor:
        """Load a single frame."""
        img_path = Path(images_dir) / f"im_{frame_idx}.jpg"

        if not img_path.exists():
            raise ValueError(f"Image not found: {img_path}")

        frame = cv2.imread(str(img_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # [C, H, W]
        return frame

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images_dir = sample["images_dir"]
        t = sample["frame_idx"]

        img_t = self._load_frame(images_dir, t)
        img_t1 = self._load_frame(images_dir, t + 1)

        # Stack as 6-channel input
        pixel_values = torch.cat([img_t, img_t1], dim=0)  # [6, H, W]
        action = torch.from_numpy(sample["action"])  # [7] or variable dim

        return {"pixel_values": pixel_values, "action": action}


def build_datasets(bridge_root: str, max_trajectories: int = None, train_ratio: float = 0.8):
    """Build train/eval datasets with trajectory-level split."""
    data_root = Path(bridge_root)
    search_root = data_root / "raw" / "bridge_data_v2"

    if not search_root.exists():
        search_root = data_root  # fallback

    # Find all trajectory directories (containing images0/)
    trajectory_paths = []
    for images_dir in search_root.glob("**/images0"):
        traj_path = images_dir.parent
        trajectory_paths.append(traj_path)

    trajectory_paths = sorted(trajectory_paths)

    if max_trajectories:
        trajectory_paths = trajectory_paths[:max_trajectories]

    # Trajectory-level 80/20 split
    n_train = int(len(trajectory_paths) * train_ratio)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(trajectory_paths))
    train_trajs = [trajectory_paths[i] for i in indices[:n_train]]
    eval_trajs = [trajectory_paths[i] for i in indices[n_train:]]

    print(f"\nDataset split: {len(train_trajs)} train / {len(eval_trajs)} eval trajectories")

    train_ds = BridgeProbingDataset(bridge_root, train_trajs)
    eval_ds = BridgeProbingDataset(bridge_root, eval_trajs)

    return train_ds, eval_ds


# ============================================================================
# Training / Evaluation (reuse from probe_action.py)
# ============================================================================

def extract_embeddings(encoder, encoder_name, dataloader, device):
    """Extract frozen embeddings for the entire dataset."""
    all_embeddings = []
    all_actions = []

    encoder.eval()
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        actions = batch["action"]

        emb = encode_batch(encoder, encoder_name, pixel_values)  # [B, D]
        all_embeddings.append(emb.cpu())
        all_actions.append(actions)

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
    actions = torch.cat(all_actions, dim=0)  # [N, action_dim]
    return embeddings, actions


def train_probe(
    probe: nn.Module,
    train_emb: torch.Tensor,
    train_act: torch.Tensor,
    eval_emb: torch.Tensor,
    eval_act: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """Train the linear/MLP probe on pre-extracted embeddings."""
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(train_emb, train_act)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_r2 = -float("inf")
    best_metrics = None

    for epoch in range(1, epochs + 1):
        # Train
        probe.train()
        epoch_loss = 0
        n_batches = 0
        for emb_batch, act_batch in train_loader:
            emb_batch = emb_batch.to(device)
            act_batch = act_batch.to(device)

            pred = probe(emb_batch)
            loss = F.mse_loss(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Evaluate
        probe.eval()
        with torch.no_grad():
            eval_pred = probe(eval_emb.to(device)).cpu().numpy()
            eval_targets = eval_act.numpy()

        # Compute metrics (without per-joint breakdown for robot actions)
        metrics = compute_basic_metrics(eval_pred, eval_targets)
        metrics["train_mse"] = avg_train_loss

        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_metrics = metrics.copy()

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train MSE: {avg_train_loss:.6f} | "
            f"Eval R²: {metrics['r2']:.4f} | "
            f"MSE: {metrics['mse']:.6f} | "
            f"Cos: {metrics['cosine_sim']:.4f}"
        )

    return best_metrics


def compute_basic_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute R², MSE, Cosine Similarity (without joint breakdown)."""
    # MSE
    mse = np.mean((predictions - targets) ** 2)

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Cosine similarity
    pred_norm = np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8
    tgt_norm = np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8
    cos_sim = np.mean(np.sum(predictions * targets, axis=1) / (pred_norm.squeeze() * tgt_norm.squeeze()))

    return {
        "r2": float(r2),
        "mse": float(mse),
        "cosine_sim": float(cos_sim),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bridge V2 Action Probing")

    parser.add_argument("--encoder", type=str, required=True,
                        choices=["two-stream", "single-stream", "videomae", "clip", "dinov2"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Encoder checkpoint path (required for custom encoders)")
    parser.add_argument("--bridge-root", type=str, default="/workspace/data/datasets/bridge_v2",
                        help="Bridge V2 data root")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Probing epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Probe learning rate (default: 1e-3)")
    parser.add_argument("--probe", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Probe type (default: linear)")
    parser.add_argument("--max-trajectories", type=int, default=None,
                        help="Limit number of trajectories (for debugging)")
    parser.add_argument("--output-dir", type=str, default="data/probing_results",
                        help="Output directory")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Encoder: {args.encoder}")
    print(f"Probe: {args.probe}")
    print(f"Checkpoint: {args.checkpoint or '(pretrained)'}")

    # ---- 1. Load encoder ----
    print("\n" + "=" * 60)
    print("Loading encoder...")
    print("=" * 60)
    t0 = time.time()
    encoder, embed_dim = load_encoder(args.encoder, args.checkpoint, device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print(f"Encoder loaded in {time.time() - t0:.1f}s, embed_dim={embed_dim}")

    # ---- 2. Build datasets ----
    print("\n" + "=" * 60)
    print("Building datasets...")
    print("=" * 60)
    t0 = time.time()
    train_ds, eval_ds = build_datasets(args.bridge_root, args.max_trajectories)
    print(f"Datasets built in {time.time() - t0:.1f}s")

    if len(train_ds) == 0 or len(eval_ds) == 0:
        print("ERROR: No valid samples found. Check data path and action files.")
        sys.exit(1)

    # Infer action dimension from first sample
    action_dim = train_ds[0]["action"].shape[0]
    print(f"Action dimension: {action_dim}")

    # ---- 3. Extract embeddings ----
    print("\n" + "=" * 60)
    print("Extracting embeddings (frozen encoder)...")
    print("=" * 60)
    t0 = time.time()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_emb, train_act = extract_embeddings(encoder, args.encoder, train_loader, device)
    eval_emb, eval_act = extract_embeddings(encoder, args.encoder, eval_loader, device)

    print(f"Extracted in {time.time() - t0:.1f}s")
    print(f"  Train: {train_emb.shape} embeddings, {train_act.shape} actions")
    print(f"  Eval:  {eval_emb.shape} embeddings, {eval_act.shape} actions")

    # ---- 4. Train probe ----
    print("\n" + "=" * 60)
    print(f"Training {args.probe} probe...")
    print("=" * 60)

    if args.probe == "linear":
        probe = LinearProbe(embed_dim, action_dim=action_dim)
    else:
        probe = MLPProbe(embed_dim, action_dim=action_dim)

    print(f"Probe params: {sum(p.numel() for p in probe.parameters()):,}")

    best_metrics = train_probe(
        probe=probe,
        train_emb=train_emb,
        train_act=train_act,
        eval_emb=eval_emb,
        eval_act=eval_act,
        epochs=args.epochs,
        batch_size=min(256, len(train_ds)),
        lr=args.lr,
        device=device,
    )

    # ---- 5. Report ----
    print("\n" + "=" * 60)
    print("RESULTS (Bridge V2)")
    print("=" * 60)
    print(f"Encoder:    {args.encoder}")
    print(f"Probe:      {args.probe}")
    print(f"R²:         {best_metrics['r2']:.4f}  {'PASS' if best_metrics['r2'] > 0.7 else 'FAIL'} (threshold: 0.7)")
    print(f"MSE:        {best_metrics['mse']:.6f}")
    print(f"Cosine Sim: {best_metrics['cosine_sim']:.4f}")

    # ---- 6. Save results ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "dataset": "bridge_v2",
        "encoder": args.encoder,
        "probe": args.probe,
        "checkpoint": args.checkpoint,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_trajectories": args.max_trajectories,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "action_dim": action_dim,
        "embed_dim": embed_dim,
        "timestamp": timestamp,
        **best_metrics,
    }

    result_path = output_dir / f"probe_bridge_{args.encoder}_{args.probe}_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
