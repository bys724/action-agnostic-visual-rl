#!/usr/bin/env python
"""
LIBERO BC fine-tuning wrapper for Two-Stream v11.

v11 모델(`TwoStreamV11Model`)은 기존 `TwoStreamEncoder`와 호환되지 않아
별도 wrapper가 필요. Probing(`probe_action_v11.py`)과 동일한 패턴:
representation 추출만 v11용으로 교체하고, 데이터/액션헤드/학습 루프는
기존 `finetune_libero.py`를 그대로 재사용.

기본 representation: `patch_mean_concat_enc_phase3` (A+D, EgoDex best).

Usage:
    python scripts/eval/finetune_libero_v11.py \\
        --encoder two-stream-v11 \\
        --checkpoint /proj/.../checkpoint_epoch0012.pt \\
        --data-root /proj/external_group/mrg/datasets/libero \\
        --task-suite libero_spatial \\
        --cls-mode patch_mean_concat_enc_phase3 \\
        --epochs 30 --batch-size 64
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _project_root)
sys.path.insert(0, "/workspace")

from scripts.eval.probe_action_v11 import (  # noqa: E402
    CLS_MODES_ALL,
    _embed_dim,
    extract_repr,
    load_v11_model,
)
from scripts.eval.finetune_libero import (  # noqa: E402
    ActionHead,
    LIBERODataset,
    evaluate,
    train_epoch,
)


# ============================================================================
# v11 + ActionHead wrapper
# ============================================================================

class V11WithActionHead(nn.Module):
    """Two-Stream v11 encoder + LIBERO action head.

    Encoder는 frozen 기본. `extract_repr`(probe_action_v11)을 그대로 사용해
    representation 추출 일관성 보장.
    """

    def __init__(
        self,
        v11_model: nn.Module,
        cls_mode: str,
        embed_dim: int,
        action_dim: int = 7,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = v11_model
        self.cls_mode = cls_mode
        self.action_head = ActionHead(embed_dim=embed_dim, action_dim=action_dim)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            print("Encoder frozen (only action head will be trained)")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 6, H, W] (img_prev ⊕ img_curr) → [B, 7] action."""
        # extract_repr 내부가 @torch.no_grad라 frozen일 때 그대로,
        # unfreeze에서는 grad enable 필요 → 분기.
        if any(p.requires_grad for p in self.encoder.parameters()):
            with torch.enable_grad():
                emb = extract_repr(self.encoder, pixel_values, self.cls_mode)
        else:
            emb = extract_repr(self.encoder, pixel_values, self.cls_mode)
        return self.action_head(emb)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LIBERO BC fine-tune — Two-Stream v11 wrapper"
    )

    # Encoder (kept for CLI parity with finetune_libero.py)
    parser.add_argument("--encoder", type=str, default="two-stream-v11",
                        choices=["two-stream-v11"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="v11 model checkpoint path")
    parser.add_argument("--p-depth", type=int, default=12)
    parser.add_argument("--m-depth", type=int, default=6)
    parser.add_argument("--cls-mode", type=str,
                        default="patch_mean_concat_enc_phase3",
                        choices=sorted(CLS_MODES_ALL),
                        help="v11 representation mode (default: A+D, EgoDex best)")
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="Unfreeze encoder (fine-tune all params)")

    # Data
    parser.add_argument("--data-root", type=str,
                        default="/proj/external_group/mrg/datasets/libero")
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal",
                                 "libero_10", "libero_90"])
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()

    if args.unfreeze_encoder:
        args.freeze_encoder = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder} (p_depth={args.p_depth}, m_depth={args.m_depth})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"CLS mode:   {args.cls_mode}")

    # ---- 1. Load v11 (mask_ratio=0 deterministic) ----
    print("\n" + "=" * 60)
    print("Loading v11 model (mask_ratio=0 for deterministic forward)...")
    print("=" * 60)
    t0 = time.time()
    v11 = load_v11_model(
        args.checkpoint,
        p_depth=args.p_depth,
        m_depth=args.m_depth,
        device=str(device),
    )
    embed_dim = _embed_dim(args.cls_mode)
    print(f"Loaded in {time.time() - t0:.1f}s, embed_dim={embed_dim}")

    # ---- 2. Wrap with action head ----
    model = V11WithActionHead(
        v11_model=v11,
        cls_mode=args.cls_mode,
        embed_dim=embed_dim,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # ---- 3. Data ----
    print(f"\nLoading LIBERO {args.task_suite} data...")
    full_dataset = LIBERODataset(
        data_root=args.data_root,
        task_suite=args.task_suite,
        task_ids=args.task_ids,
    )

    eval_size = int(len(full_dataset) * args.eval_split)
    train_size = len(full_dataset) - eval_size
    train_ds, eval_ds = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size]
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # ---- 4. Optimizer / scheduler ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # ---- 5. Output dir ----
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            f"/proj/external_group/mrg/checkpoints/libero/"
            f"{args.encoder}_{args.task_suite}_{ts}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput: {args.output_dir}")

    config = vars(args)
    config["encoder_embed_dim"] = embed_dim
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- 6. Training loop ----
    print("\nStarting training...")
    best_eval_loss = float("inf")
    history = {"train_loss": [], "eval_mse": [], "eval_l1": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, eval_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["eval_mse"].append(eval_metrics["mse_loss"])
        history["eval_l1"].append(eval_metrics["l1_error"])

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Eval MSE:   {eval_metrics['mse_loss']:.6f}, "
              f"L1: {eval_metrics['l1_error']:.6f}")
        print(f"  LR:         {scheduler.get_last_lr()[0]:.2e}")

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

        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "history": history,
                "config": config,
            }, os.path.join(args.output_dir, f"epoch_{epoch}.pt"))

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": config,
    }, os.path.join(args.output_dir, "final.pt"))

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best eval MSE: {best_eval_loss:.6f}")
    print(f"Checkpoints:   {args.output_dir}")


if __name__ == "__main__":
    main()
