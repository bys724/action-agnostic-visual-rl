#!/usr/bin/env python
"""
DROID Action Probing — Cross-domain transfer 검증.

EgoDex로 학습한 frozen encoder가 DROID 로봇 action을 예측할 수 있는지 테스트.
프레임: /mnt/data/droid_frames/ext1/ep_XXXXXX/frame_XXXXXX.jpg
Action: /mnt/data/droid_frames/ext1/ep_XXXXXX/action.npy (T, 7)

Usage:
    python scripts/eval/probe_action_droid.py \
        --encoder two-stream \
        --checkpoint /mnt/data/checkpoints/two_stream/.../best_model.pt \
        --droid-root /mnt/data/droid_frames/ext1 \
        --gap 1 --max-episodes 500
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

_project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _project_root)
sys.path.insert(0, "/workspace")

# probe_action.py에서 공통 함수 재사용
from scripts.eval.probe_action import (
    load_encoder, encode_batch, LinearProbe, MLPProbe, compute_metrics,
)

ACTION_DIM = 7  # 6D cartesian + 1D gripper


class DROIDProbingDataset(Dataset):
    """DROID 프레임 + action.npy 기반 probing 데이터셋.

    Input: (img_t, img_t+gap) → 6ch concat
    Target: action delta = action[t+gap] - action[t] (7-dim)
    """

    def __init__(self, episode_dirs: list, gap: int = 1, img_size: int = 224):
        self.gap = gap
        self.img_size = img_size
        self.samples = []

        for ep_dir in episode_dirs:
            ep_dir = Path(ep_dir)
            action_path = ep_dir / "action.npy"
            if not action_path.exists():
                continue

            actions = np.load(action_path)  # [T, 7]
            frames = sorted(ep_dir.glob("frame_*.jpg"))
            n = min(len(frames), len(actions))

            for t in range(n - gap):
                delta = actions[t + gap] - actions[t]
                self.samples.append({
                    "frame_dir": str(ep_dir),
                    "frame_idx": t,
                    "action": delta.astype(np.float32),
                })

        print(f"DROIDProbingDataset: {len(self.samples)} samples from {len(episode_dirs)} episodes, gap={gap}")

    def _load_frame(self, frame_dir: str, idx: int) -> torch.Tensor:
        path = Path(frame_dir) / f"frame_{idx:06d}.jpg"
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        # 256x256 → center crop 224x224
        h, w = img.shape[:2]
        top = (h - self.img_size) // 2
        left = (w - self.img_size) // 2
        img = img[top:top + self.img_size, left:left + self.img_size]
        return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_t = self._load_frame(s["frame_dir"], s["frame_idx"])
        img_tk = self._load_frame(s["frame_dir"], s["frame_idx"] + self.gap)
        pixel_values = torch.cat([img_t, img_tk], dim=0)
        return {"pixel_values": pixel_values, "action": torch.from_numpy(s["action"])}


def build_datasets(droid_root: str, gap: int = 1, max_episodes: int = None, train_ratio: float = 0.8):
    """에피소드 단위 train/eval split."""
    root = Path(droid_root)
    # action.npy가 있는 에피소드만
    episode_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "action.npy").exists()])
    print(f"Found {len(episode_dirs)} episodes with actions")

    if max_episodes:
        episode_dirs = episode_dirs[:max_episodes]

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(episode_dirs))
    n_train = int(len(episode_dirs) * train_ratio)

    train_eps = [episode_dirs[i] for i in indices[:n_train]]
    eval_eps = [episode_dirs[i] for i in indices[n_train:]]

    print(f"Split: {len(train_eps)} train / {len(eval_eps)} eval episodes")
    train_ds = DROIDProbingDataset(train_eps, gap=gap)
    eval_ds = DROIDProbingDataset(eval_eps, gap=gap)
    return train_ds, eval_ds


def extract_embeddings(encoder, encoder_name, dataloader, device, cls_mode="average"):
    all_emb, all_act = [], []
    encoder.eval()
    for batch in dataloader:
        pv = batch["pixel_values"].to(device)
        emb = encode_batch(encoder, encoder_name, pv, cls_mode=cls_mode)
        all_emb.append(emb.cpu())
        all_act.append(batch["action"])
    return torch.cat(all_emb), torch.cat(all_act)


def train_probe(probe, train_emb, train_act, eval_emb, eval_act, epochs=20, batch_size=256, lr=1e-3, device="cuda"):
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    train_ds = torch.utils.data.TensorDataset(train_emb, train_act)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_r2 = -float("inf")
    best_metrics = None

    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss, n = 0, 0
        for emb, act in train_loader:
            emb, act = emb.to(device), act.to(device)
            pred = probe(emb)
            loss = F.mse_loss(pred, act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1

        probe.eval()
        with torch.no_grad():
            eval_pred = probe(eval_emb.to(device)).cpu().numpy()
        metrics = compute_metrics(eval_pred, eval_act.numpy())
        metrics["train_mse"] = total_loss / n

        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_metrics = metrics.copy()

        print(f"  Epoch {epoch:3d}/{epochs} | Train MSE: {total_loss/n:.6f} | "
              f"Eval R²: {metrics['r2']:.4f} | Cos: {metrics['cosine_sim']:.4f}")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="DROID Action Probing")
    parser.add_argument("--encoder", type=str, required=True,
                        choices=["two-stream", "videomae", "clip", "dinov2"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--cls-mode", type=str, default="patch_mean_concat",
                        choices=["average", "concat", "m_only", "p_only",
                                 "patch_mean", "patch_mean_concat", "patch_mean_m", "patch_mean_p"])
    parser.add_argument("--droid-root", type=str, default="/mnt/data/droid_frames/ext1")
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--output-dir", type=str, default="data/probing_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEncoder: {args.encoder}, CLS mode: {args.cls_mode}, Gap: {args.gap}")

    # 1. Encoder
    encoder, embed_dim = load_encoder(
        args.encoder, args.checkpoint, device,
        cls_mode=args.cls_mode, depth=args.depth, num_stages=args.num_stages,
    )
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print(f"Embed dim: {embed_dim}")

    # 2. Dataset
    train_ds, eval_ds = build_datasets(args.droid_root, gap=args.gap, max_episodes=args.max_episodes)
    if len(train_ds) == 0 or len(eval_ds) == 0:
        print("ERROR: No samples. Check action.npy extraction.")
        sys.exit(1)

    # 3. Extract embeddings
    print("\nExtracting embeddings...")
    t0 = time.time()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_emb, train_act = extract_embeddings(encoder, args.encoder, train_loader, device, args.cls_mode)
    eval_emb, eval_act = extract_embeddings(encoder, args.encoder, eval_loader, device, args.cls_mode)
    print(f"Extracted in {time.time()-t0:.1f}s — Train: {train_emb.shape}, Eval: {eval_emb.shape}")

    # 4. Train probe
    print(f"\nTraining {args.probe} probe (action_dim={ACTION_DIM})...")
    if args.probe == "linear":
        probe = LinearProbe(embed_dim, ACTION_DIM)
    else:
        probe = MLPProbe(embed_dim, action_dim=ACTION_DIM)

    best = train_probe(probe, train_emb, train_act, eval_emb, eval_act,
                       epochs=args.epochs, lr=args.lr, device=device)

    # 5. Results
    print(f"\n{'='*60}\nRESULTS (DROID)\n{'='*60}")
    print(f"Encoder:    {args.encoder}")
    print(f"CLS mode:   {args.cls_mode}")
    print(f"Gap:        {args.gap}")
    print(f"R²:         {best['r2']:.4f}")
    print(f"MSE:        {best['mse']:.6f}")
    print(f"Cosine Sim: {best['cosine_sim']:.4f}")

    # 6. Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "dataset": "droid",
        "encoder": args.encoder,
        "cls_mode": args.cls_mode,
        "gap": args.gap,
        "checkpoint": args.checkpoint,
        "max_episodes": args.max_episodes,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        **best,
    }
    path = output_dir / f"probe_droid_{args.encoder}_{args.cls_mode}_gap{args.gap}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
