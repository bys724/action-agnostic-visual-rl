#!/usr/bin/env python3
"""CoMP-MAE pretraining on EgoDex frame pairs.

Reproduces the reported CoMP-MAE-S / plain-control runs (see train/config_*.yaml
and the paper's Appendix A). This is a clean single-file training script:
argument parsing mirrors the config, and the training loop uses the exact recipe
of the reported runs (Fused AdamW, uniform weight decay 0.01, AdamW default betas
(0.9, 0.999), linear LR warmup over ~10% of epochs + cosine decay, BF16 autocast,
gradient clipping at 1.0).

The reported runs used 8 GPUs with DistributedDataParallel (global batch 1024,
per-GPU batch 128). This script runs single-GPU by default; wrap the model in
DDP and launch with torchrun to reproduce the multi-GPU setup (the optimization
math is identical, only the data is sharded across ranks).

Data: EgoDex frames are pre-extracted to `frames_root/<task>/<video>/frame_*.jpg`
(center-cropped to square, resized to 256x256). EgoDex is a public dataset and is
NOT redistributed here -- download it and extract frames separately. Each training
sample is a pair (frame_t, frame_{t+gap}) with gap ~ triangular over [1, max_gap]
(mode = sample_center).

Examples:
    # CoMP-MAE-S (ours)
    python train/pretrain.py --frames-root <EGODEX_FRAMES_ROOT> --comp-mae \
        --embed-dim 384 --num-heads 6 --m-depth 6 --epochs 50 \
        --batch-size 128 --lr 2.8e-4 --routing-mode v_from_p

    # Plain cross-modal MAE (matched control)
    python train/pretrain.py --frames-root <EGODEX_FRAMES_ROOT> --pixel-pred \
        --embed-dim 384 --num-heads 6 --m-depth 6 --epochs 50 \
        --batch-size 128 --lr 2.8e-4 --routing-mode v_from_m --masked-anchor \
        --independent-rotation-prob 0.0
"""
import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # supplement root (contains model/)

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model.comp_mae import CoMPMAE


# ---------------------------------------------------------------------------
# Dataset: EgoDex frame pairs (t, t+gap)
# ---------------------------------------------------------------------------

class EgoDexPairDataset(Dataset):
    """Sample (frame_t, frame_{t+gap}) RGB pairs from pre-extracted EgoDex frames.

    Directory layout: frames_root/<task>/<video>/frame_000000.jpg, ...
    Frames are read as [0, 1] RGB tensors (no ImageNet normalization -- the model
    consumes raw [0, 1] input; keep train/inference preprocessing identical).
    """

    def __init__(self, frames_root, img_size=224, max_gap=30, sample_center=15,
                 samples_per_video=100):
        self.img_size = img_size
        self.max_gap = max_gap
        self.sample_center = sample_center
        self.samples_per_video = samples_per_video
        root = Path(frames_root)
        self.videos = []
        for task in sorted(root.glob("*")):
            if not task.is_dir():
                continue
            for video in sorted(task.glob("*")):
                frames = sorted(video.glob("frame_*.jpg"))
                if len(frames) > max_gap + 1:
                    self.videos.append(frames)
        if not self.videos:
            raise FileNotFoundError(
                f"No EgoDex videos with > {max_gap+1} frames under {frames_root}")
        print(f"EgoDexPairDataset: {len(self.videos)} videos")

    def __len__(self):
        return len(self.videos) * self.samples_per_video

    def _load(self, path):
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        t = torch.frombuffer(img.tobytes(), dtype=torch.uint8).float() / 255.0
        return t.view(self.img_size, self.img_size, 3).permute(2, 0, 1)

    def __getitem__(self, idx):
        frames = self.videos[idx % len(self.videos)]
        # triangular gap in [1, max_gap], mode = sample_center
        gap = int(round(random.triangular(1, self.max_gap, self.sample_center)))
        gap = max(1, min(gap, self.max_gap))
        t = random.randint(0, len(frames) - gap - 1)
        return self._load(frames[t]), self._load(frames[t + gap])


# ---------------------------------------------------------------------------
# Optimizer / schedule helpers (reported recipe)
# ---------------------------------------------------------------------------

def build_optimizer(model, lr, weight_decay=0.01):
    """Fused AdamW with uniform weight decay and AdamW default betas (0.9, 0.999).

    This matches the reported CoMP-MAE / plain-control recipe (paper Appendix A):
    a single param group with uniform weight decay 0.01 and the default betas.
    There is no norm/bias/token no-decay split -- that split is used only for the
    VideoMAE baseline, which is not shipped in this supplement.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable, lr=lr, weight_decay=weight_decay,
        fused=torch.cuda.is_available(),
    )


def lr_at(step, total_steps, warmup_steps, base_lr):
    """Linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default=None, required=True,
                    help="pre-extracted EgoDex frames root (<task>/<video>/frame_*.jpg)")
    # objective (choose exactly one)
    ap.add_argument("--comp-mae", action="store_true", help="CoMP-MAE (ours)")
    ap.add_argument("--pixel-pred", action="store_true", help="plain cross-modal MAE control")
    # architecture
    ap.add_argument("--embed-dim", type=int, default=384)     # ViT-S
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--depth", type=int, default=12)          # P stream
    ap.add_argument("--m-depth", type=int, default=6)         # M stream
    ap.add_argument("--routing-mode", default="v_from_p", choices=["v_from_p", "v_from_m"])
    ap.add_argument("--masked-anchor", action="store_true")
    ap.add_argument("--mask-ratio-p", type=float, default=0.75)
    ap.add_argument("--m-recon-floor", type=float, default=0.1)
    # optimization
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.8e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--rotation-aug", action="store_true")
    ap.add_argument("--independent-rotation-prob", type=float, default=0.1)
    # sampling
    ap.add_argument("--max-gap", type=int, default=30)
    ap.add_argument("--sample-center", type=int, default=15)
    # io
    ap.add_argument("--output-dir", default="outputs/pretrain")
    ap.add_argument("--save-interval", type=int, default=4)
    args = ap.parse_args()

    assert args.comp_mae ^ args.pixel_pred, "choose exactly one of --comp-mae / --pixel-pred"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CoMPMAE(
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        p_depth=args.depth, m_depth=args.m_depth,
        routing_mode=args.routing_mode, use_sobel=False, pair_mode=True,
        mask_ratio_p=args.mask_ratio_p,
        rotation_aug=args.rotation_aug,
        independent_rotation_prob=args.independent_rotation_prob,
        masked_anchor=args.masked_anchor,
        comp_mae=args.comp_mae, pixel_pred=args.pixel_pred,
        m_recon_weight_floor=args.m_recon_floor,
    ).to(device)
    model.train()

    ds = EgoDexPairDataset(args.frames_root, img_size=224,
                           max_gap=args.max_gap, sample_center=args.sample_center)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optimizer = build_optimizer(model, args.lr)
    total_steps = args.epochs * len(loader)
    warmup_steps = max(1, total_steps // 10)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for img_t, img_tk in loader:
            img_t, img_tk = img_t.to(device), img_tk.to(device)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, args.lr)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                loss, _ = model.compute_loss(img_t, img_tk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_teacher(0.0)  # no-op for these objectives
            running += loss.item()
            step += 1
        print(f"epoch {epoch}/{args.epochs}  loss={running/len(loader):.4f}")

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       out / "latest.pt")


if __name__ == "__main__":
    main()
