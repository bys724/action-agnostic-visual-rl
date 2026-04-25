#!/usr/bin/env python
"""
DROID Action Probing for Two-Stream v11 (Motion-Guided Routing + Dual-Target).

`probe_action_v11.py` (EgoDex)와 `probe_action_droid.py` (DROID, v6/v10/VM)의
조합. v11 representation 추출 (12 mode 지원, mask=0 deterministic) +
DROID 데이터셋 (episode → 7-dim action delta).

Representation 옵션은 `probe_action_v11.CLS_MODES_ALL` 그대로.

Target: 7-dim DROID action delta (6D cartesian + 1D gripper)
        (EgoDex hand pose 18-dim과 다름)

Usage:
    python scripts/eval/probe_action_droid_v11.py \\
        --encoder two-stream-v11 \\
        --checkpoint /proj/.../two_stream_v11/<ts>/checkpoint_epoch0012.pt \\
        --droid-root /proj/external_group/mrg/datasets/droid_frames/ext1 \\
        --cls-mode patch_mean_p_features_tk \\
        --gap 1 --max-episodes 200
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _project_root)
sys.path.insert(0, "/workspace")

# v11 model loader + representation extractor
from scripts.eval.probe_action_v11 import (  # noqa: E402
    CLS_MODES_ALL,
    _embed_dim,
    extract_repr,
    load_v11_model,
)

# DROID dataset + episode-split builder
from scripts.eval.probe_action_droid import (  # noqa: E402
    DROIDProbingDataset,
    build_datasets,
    train_probe,
)

# Linear/MLP probe (action_dim 인자로 7 주입)
from scripts.eval.probe_action import (  # noqa: E402
    LinearProbe,
    MLPProbe,
)


ACTION_DIM = 7  # DROID: 6D cartesian + 1D gripper


# ============================================================================
# Embedding extraction loop (v11 representation)
# ============================================================================

def extract_embeddings_v11(model, dataloader, device, mode: str):
    """v11 모델로 batch마다 representation 추출. probe_action_v11.extract_repr 사용."""
    all_emb, all_act = [], []
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        emb = extract_repr(model, pixel_values, mode)
        all_emb.append(emb.cpu())
        all_act.append(batch["action"])
    return torch.cat(all_emb, dim=0), torch.cat(all_act, dim=0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DROID Action Probing — Two-Stream v11")
    parser.add_argument("--encoder", type=str, default="two-stream-v11",
                        choices=["two-stream-v11"],
                        help="Fixed for this script (kept for CLI compat)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="v11 model checkpoint path")
    parser.add_argument("--droid-root", type=str,
                        default="/proj/external_group/mrg/datasets/droid_frames/ext1")
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--cls-mode", type=str, default="patch_mean_p_features_tk",
                        choices=sorted(CLS_MODES_ALL))
    parser.add_argument("--p-depth", type=int, default=12)
    parser.add_argument("--m-depth", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="data/probing_results")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Encoder: {args.encoder} (p_depth={args.p_depth}, m_depth={args.m_depth})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"DROID root: {args.droid_root}, gap: {args.gap}, "
          f"max_episodes: {args.max_episodes}")
    print(f"CLS mode: {args.cls_mode}")

    # ---- 1. Load v11 model (mask_ratio=0 deterministic) ----
    print("\n" + "=" * 60)
    print("Loading v11 model (mask_ratio=0 for deterministic probing)...")
    print("=" * 60)
    t0 = time.time()
    model = load_v11_model(
        args.checkpoint,
        p_depth=args.p_depth, m_depth=args.m_depth, device=device,
    )
    embed_dim = _embed_dim(args.cls_mode)
    print(f"Model loaded in {time.time() - t0:.1f}s, embed_dim={embed_dim}")

    # ---- 2. Build DROID datasets (episode-level split) ----
    print("\n" + "=" * 60)
    print("Building DROID datasets...")
    print("=" * 60)
    t0 = time.time()
    train_ds, eval_ds = build_datasets(
        args.droid_root, gap=args.gap, max_episodes=args.max_episodes,
    )
    print(f"Datasets built in {time.time() - t0:.1f}s")
    if len(train_ds) == 0 or len(eval_ds) == 0:
        print("ERROR: No samples. Check action.npy extraction in DROID frames root.")
        sys.exit(1)

    # ---- 3. Extract embeddings ----
    print("\n" + "=" * 60)
    print("Extracting embeddings (frozen v11)...")
    print("=" * 60)
    t0 = time.time()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    train_emb, train_act = extract_embeddings_v11(model, train_loader, device,
                                                  args.cls_mode)
    eval_emb, eval_act = extract_embeddings_v11(model, eval_loader, device,
                                                args.cls_mode)

    print(f"Extracted in {time.time() - t0:.1f}s")
    print(f"  Train: {train_emb.shape} embeddings, {train_act.shape} actions")
    print(f"  Eval:  {eval_emb.shape} embeddings, {eval_act.shape} actions")
    print(f"  Action mean abs: {train_act.abs().mean():.6f}, "
          f"std: {train_act.std():.6f}")

    # ---- 4. Train probe ----
    print("\n" + "=" * 60)
    print(f"Training {args.probe} probe (action_dim={ACTION_DIM})...")
    print("=" * 60)
    if args.probe == "linear":
        probe = LinearProbe(embed_dim, action_dim=ACTION_DIM)
    else:
        probe = MLPProbe(embed_dim, action_dim=ACTION_DIM)
    print(f"Probe params: {sum(p.numel() for p in probe.parameters()):,}")

    best_metrics = train_probe(
        probe=probe,
        train_emb=train_emb, train_act=train_act,
        eval_emb=eval_emb, eval_act=eval_act,
        epochs=args.epochs,
        batch_size=min(256, len(train_ds)),
        lr=args.lr,
        device=device,
    )

    # ---- 5. Report ----
    print("\n" + "=" * 60)
    print("RESULTS (DROID, v11)")
    print("=" * 60)
    print(f"Encoder:    {args.encoder}")
    print(f"CLS mode:   {args.cls_mode}")
    print(f"Probe:      {args.probe}")
    print(f"Gap:        {args.gap}")
    print(f"R²:         {best_metrics['r2']:.4f}")
    print(f"MSE:        {best_metrics['mse']:.6f}")
    print(f"Cosine Sim: {best_metrics['cosine_sim']:.4f}")
    # NOTE: per_joint_r2는 EgoDex 6 joints × 3D 가정 → DROID 7-dim과 호환 안 됨.
    # compute_metrics 내부 슬라이스가 7-dim에서 OOB → 일부 항목은 부정확.
    # 전체 R²/MSE/cos는 정상이므로 보고 시 per_joint는 무시.

    # ---- 6. Save ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "dataset": "droid",
        "encoder": args.encoder,
        "probe": args.probe,
        "cls_mode": args.cls_mode,
        "gap": args.gap,
        "checkpoint": args.checkpoint,
        "max_episodes": args.max_episodes,
        "p_depth": args.p_depth,
        "m_depth": args.m_depth,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "embed_dim": embed_dim,
        "action_dim": ACTION_DIM,
        "timestamp": timestamp,
        **best_metrics,
    }
    ckpt_tag = Path(args.checkpoint).stem
    result_path = (output_dir /
                   f"probe_droid_v11_{ckpt_tag}_{args.cls_mode}_gap{args.gap}_"
                   f"{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
