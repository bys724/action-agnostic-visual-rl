#!/usr/bin/env python
"""
Action Probing Experiment

논문 핵심 주장 검증: "action-agnostic으로 학습해도 행동 정보가 인코딩된다"

EgoDex test 데이터의 (img_t, img_t+1)을 인코더에 넣고,
frozen embedding에서 linear probe로 hand pose delta(action)를 예측할 수 있는지 테스트.

GO/NO-GO 기준: R² > 0.7

인코더 5종:
    - two-stream: TwoStreamEncoderForOpenVLA (체크포인트 필요)
    - single-stream: SingleStreamEncoderForOpenVLA (체크포인트 필요)
    - videomae: VideoMAEEncoderForOpenVLA (체크포인트 필요)
    - clip: CLIPVisionModel (HuggingFace, pretrained)
    - dinov2: Dinov2Model (HuggingFace, pretrained)

Usage:
    # 커스텀 인코더
    python scripts/probe_action.py \\
        --encoder two-stream \\
        --checkpoint /path/to/best_model.pt \\
        --egodex-root /workspace/data/egodex

    # CLIP / DINOv2 baseline
    python scripts/probe_action.py --encoder clip
    python scripts/probe_action.py --encoder dinov2
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace")

# ============================================================================
# Target joints for action probing
# ============================================================================

TARGET_JOINTS = [
    "rightHand",
    "rightThumbTip",
    "rightIndexFingerTip",
    "rightMiddleFingerTip",
    "rightRingFingerTip",
    "rightLittleFingerTip",
]
ACTION_DIM = len(TARGET_JOINTS) * 3  # 6 joints × 3D = 18

CONFIDENCE_THRESHOLD = 0.3


# ============================================================================
# 1. EgoDexProbingDataset
# ============================================================================

class EgoDexProbingDataset(Dataset):
    """
    EgoDex dataset for action probing.

    MP4 비디오 + HDF5 손 포즈를 동시 로드.
    gap=1 고정 (연속 프레임 간 action 예측).

    Returns:
        pixel_values: [6, 224, 224] - img_t + img_t+1 채널 concat
        action: [18] - 6관절 × 3D position delta
    """

    def __init__(
        self,
        data_root: str,
        video_paths: list,
        img_size: int = 224,
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size

        # Build (video_path, frame_idx) pairs with valid actions
        self.samples = []
        skipped_conf = 0
        skipped_read = 0

        for vp in video_paths:
            video_path = Path(vp)
            hdf5_path = video_path.with_suffix(".hdf5")
            if not hdf5_path.exists():
                continue

            # Load HDF5 once to build valid frame indices
            try:
                with h5py.File(str(hdf5_path), "r") as f:
                    # Load transforms and confidences for target joints
                    transforms = {}
                    confidences = {}
                    for joint in TARGET_JOINTS:
                        t_key = f"transforms/{joint}"
                        c_key = f"confidences/{joint}"
                        if t_key not in f or c_key not in f:
                            break
                        transforms[joint] = f[t_key][()]  # (T, 4, 4)
                        confidences[joint] = f[c_key][()]  # (T,)
                    else:
                        # All joints found - build frame pairs
                        num_frames = transforms[TARGET_JOINTS[0]].shape[0]

                        # Get video frame count to avoid out-of-bounds
                        cap = cv2.VideoCapture(str(video_path))
                        vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        num_frames = min(num_frames, vid_frames)

                        for t in range(num_frames - 1):
                            # Check confidence for both frames
                            valid = True
                            for joint in TARGET_JOINTS:
                                if (confidences[joint][t] < CONFIDENCE_THRESHOLD or
                                        confidences[joint][t + 1] < CONFIDENCE_THRESHOLD):
                                    valid = False
                                    break

                            if not valid:
                                skipped_conf += 1
                                continue

                            # Compute action (position delta)
                            action = np.zeros(ACTION_DIM, dtype=np.float32)
                            for i, joint in enumerate(TARGET_JOINTS):
                                pos_t = transforms[joint][t, :3, 3]
                                pos_t1 = transforms[joint][t + 1, :3, 3]
                                action[i * 3 : (i + 1) * 3] = pos_t1 - pos_t

                            self.samples.append({
                                "video_path": str(video_path),
                                "frame_idx": t,
                                "action": action,
                            })
            except Exception as e:
                skipped_read += 1
                continue

        print(f"EgoDexProbingDataset: {len(self.samples)} samples from {len(video_paths)} videos")
        if skipped_conf > 0:
            print(f"  Skipped (low confidence): {skipped_conf}")
        if skipped_read > 0:
            print(f"  Skipped (read error): {skipped_read}")

    def _load_frame(self, video_path: str, frame_idx: int) -> torch.Tensor:
        """Load a single frame from video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = torch.from_numpy(frame).float() / 255.0  # [H, W, C]
        frame = frame.permute(2, 0, 1)  # [C, H, W]
        return frame

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        t = sample["frame_idx"]

        img_t = self._load_frame(video_path, t)
        img_t1 = self._load_frame(video_path, t + 1)

        # Stack as 6-channel input
        pixel_values = torch.cat([img_t, img_t1], dim=0)  # [6, H, W]
        action = torch.from_numpy(sample["action"])  # [18]

        return {"pixel_values": pixel_values, "action": action}


def build_datasets(egodex_root: str, max_videos: int = None, train_ratio: float = 0.8):
    """Build train/eval datasets with video-level split."""
    data_root = Path(egodex_root)
    test_dir = data_root / "test"

    video_paths = sorted(test_dir.glob("**/*.mp4"))
    if max_videos:
        video_paths = video_paths[:max_videos]

    # Video-level 80/20 split
    n_train = int(len(video_paths) * train_ratio)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(video_paths))
    train_videos = [video_paths[i] for i in indices[:n_train]]
    eval_videos = [video_paths[i] for i in indices[n_train:]]

    print(f"\nDataset split: {len(train_videos)} train / {len(eval_videos)} eval videos")

    train_ds = EgoDexProbingDataset(egodex_root, train_videos)
    eval_ds = EgoDexProbingDataset(egodex_root, eval_videos)

    return train_ds, eval_ds


# ============================================================================
# 2. Encoder loading
# ============================================================================

def load_encoder(name: str, checkpoint: str = None, device: str = "cuda"):
    """
    Load encoder by name.

    Returns:
        (encoder, embed_dim) - frozen encoder module and its output dimension
    """
    if name == "two-stream":
        from src.models.openvla_encoder import TwoStreamEncoderForOpenVLA
        assert checkpoint, "--checkpoint required for two-stream"
        encoder = TwoStreamEncoderForOpenVLA.from_checkpoint(checkpoint, device=device)
        embed_dim = encoder.embed_dim
        return encoder, embed_dim

    elif name == "single-stream":
        from src.models.openvla_encoder import SingleStreamEncoderForOpenVLA
        assert checkpoint, "--checkpoint required for single-stream"
        encoder = SingleStreamEncoderForOpenVLA.from_checkpoint(checkpoint, device=device)
        embed_dim = encoder.embed_dim
        return encoder, embed_dim

    elif name == "videomae":
        from src.models.openvla_encoder import VideoMAEEncoderForOpenVLA
        assert checkpoint, "--checkpoint required for videomae"
        encoder = VideoMAEEncoderForOpenVLA.from_checkpoint(checkpoint, device=device)
        embed_dim = encoder.embed_dim
        return encoder, embed_dim

    elif name == "clip":
        from transformers import CLIPVisionModel, CLIPImageProcessor
        model_id = "openai/clip-vit-base-patch16"
        encoder = CLIPVisionModel.from_pretrained(model_id)
        encoder.to(device)
        encoder.eval()
        # CLS from each frame → concat → 768*2 = 1536
        embed_dim = encoder.config.hidden_size * 2
        return encoder, embed_dim

    elif name == "dinov2":
        from transformers import AutoModel, AutoImageProcessor
        model_id = "facebook/dinov2-base"
        encoder = AutoModel.from_pretrained(model_id)
        encoder.to(device)
        encoder.eval()
        embed_dim = encoder.config.hidden_size * 2
        return encoder, embed_dim

    else:
        raise ValueError(f"Unknown encoder: {name}")


@torch.no_grad()
def encode_batch(encoder, name: str, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of 6-channel images into embeddings.

    Args:
        encoder: loaded encoder
        name: encoder name
        pixel_values: [B, 6, 224, 224]

    Returns:
        embeddings: [B, D]
    """
    if name in ("two-stream", "single-stream"):
        # CLS 토큰 사용 (global representation)
        return encoder.get_cls_embedding(pixel_values)  # [B, D]

    elif name == "videomae":
        # VideoMAE는 CLS 토큰 없음 → patch mean pooling
        patch_emb = encoder(pixel_values)  # [B, N, D]
        return patch_emb.mean(dim=1)  # [B, D]

    elif name == "clip":
        from transformers import CLIPImageProcessor
        # Split into two frames
        img_t = pixel_values[:, :3]   # [B, 3, H, W]
        img_t1 = pixel_values[:, 3:]  # [B, 3, H, W]

        # CLIP expects specific normalization - apply processor's normalization
        # CLIPVisionModel accepts pixel_values directly (already [B,3,H,W])
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=pixel_values.device).view(1, 3, 1, 1)
        img_t_norm = (img_t - mean) / std
        img_t1_norm = (img_t1 - mean) / std

        out_t = encoder(pixel_values=img_t_norm).last_hidden_state[:, 0]   # CLS [B, 768]
        out_t1 = encoder(pixel_values=img_t1_norm).last_hidden_state[:, 0]  # CLS [B, 768]
        return torch.cat([out_t, out_t1], dim=-1)  # [B, 1536]

    elif name == "dinov2":
        img_t = pixel_values[:, :3]
        img_t1 = pixel_values[:, 3:]

        # DINOv2 normalization (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(1, 3, 1, 1)
        img_t_norm = (img_t - mean) / std
        img_t1_norm = (img_t1 - mean) / std

        out_t = encoder(pixel_values=img_t_norm).last_hidden_state[:, 0]   # CLS [B, 768]
        out_t1 = encoder(pixel_values=img_t1_norm).last_hidden_state[:, 0]  # CLS [B, 768]
        return torch.cat([out_t, out_t1], dim=-1)  # [B, 1536]

    else:
        raise ValueError(f"Unknown encoder: {name}")


# ============================================================================
# 3. Probe definitions
# ============================================================================

class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int = ACTION_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, action_dim)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256, action_dim: int = ACTION_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# ============================================================================
# 4. Metrics
# ============================================================================

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute R², MSE, Cosine Similarity.

    Args:
        predictions: [N, 18]
        targets: [N, 18]
    """
    # MSE
    mse = np.mean((predictions - targets) ** 2)

    # R² (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Cosine similarity (per-sample, then average)
    pred_norm = np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8
    tgt_norm = np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8
    cos_sim = np.mean(np.sum(predictions * targets, axis=1) / (pred_norm.squeeze() * tgt_norm.squeeze()))

    # Per-joint R²
    per_joint_r2 = {}
    for i, joint in enumerate(TARGET_JOINTS):
        j_pred = predictions[:, i * 3 : (i + 1) * 3]
        j_tgt = targets[:, i * 3 : (i + 1) * 3]
        ss_res_j = np.sum((j_tgt - j_pred) ** 2)
        ss_tot_j = np.sum((j_tgt - j_tgt.mean(axis=0)) ** 2)
        per_joint_r2[joint] = float(1 - ss_res_j / (ss_tot_j + 1e-8))

    return {
        "r2": float(r2),
        "mse": float(mse),
        "cosine_sim": float(cos_sim),
        "per_joint_r2": per_joint_r2,
    }


# ============================================================================
# 5. Training / Evaluation loop
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
    actions = torch.cat(all_actions, dim=0)  # [N, 18]
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

    # Create simple tensor datasets
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

        metrics = compute_metrics(eval_pred, eval_targets)
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


# ============================================================================
# 6. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Action Probing Experiment")

    parser.add_argument("--encoder", type=str, required=True,
                        choices=["two-stream", "single-stream", "videomae", "clip", "dinov2"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Encoder checkpoint path (required for custom encoders)")
    parser.add_argument("--egodex-root", type=str, default="/workspace/data/egodex",
                        help="EgoDex data root")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Probing epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Probe learning rate (default: 1e-3)")
    parser.add_argument("--probe", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Probe type (default: linear)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit number of videos (for debugging)")
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
    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print(f"Encoder loaded in {time.time() - t0:.1f}s, embed_dim={embed_dim}")

    # ---- 2. Build datasets ----
    print("\n" + "=" * 60)
    print("Building datasets...")
    print("=" * 60)
    t0 = time.time()
    train_ds, eval_ds = build_datasets(args.egodex_root, args.max_videos)
    print(f"Datasets built in {time.time() - t0:.1f}s")

    if len(train_ds) == 0 or len(eval_ds) == 0:
        print("ERROR: No valid samples found. Check data path and HDF5 files.")
        sys.exit(1)

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

    # Action statistics
    act_mean = train_act.mean(dim=0)
    act_std = train_act.std(dim=0)
    print(f"  Action mean: {act_mean.abs().mean():.6f}")
    print(f"  Action std:  {act_std.mean():.6f}")

    # ---- 4. Train probe ----
    print("\n" + "=" * 60)
    print(f"Training {args.probe} probe...")
    print("=" * 60)

    if args.probe == "linear":
        probe = LinearProbe(embed_dim)
    else:
        probe = MLPProbe(embed_dim)

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
    print("RESULTS")
    print("=" * 60)
    print(f"Encoder:    {args.encoder}")
    print(f"Probe:      {args.probe}")
    print(f"R²:         {best_metrics['r2']:.4f}  {'PASS' if best_metrics['r2'] > 0.7 else 'FAIL'} (threshold: 0.7)")
    print(f"MSE:        {best_metrics['mse']:.6f}")
    print(f"Cosine Sim: {best_metrics['cosine_sim']:.4f}")
    print(f"\nPer-joint R²:")
    for joint, r2 in best_metrics["per_joint_r2"].items():
        print(f"  {joint:30s}: {r2:.4f}")

    # ---- 6. Save results ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "encoder": args.encoder,
        "probe": args.probe,
        "checkpoint": args.checkpoint,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_videos": args.max_videos,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "embed_dim": embed_dim,
        "timestamp": timestamp,
        **best_metrics,
    }

    result_path = output_dir / f"probe_{args.encoder}_{args.probe}_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
