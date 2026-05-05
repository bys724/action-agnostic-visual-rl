#!/usr/bin/env python3
"""Two-Stream v13 attention + reconstruction visualization.

v13Model은 v11Model을 상속하므로 attention hook (M/P encoder + motion-routing)
구조가 동일. visualize_attn_v11.py의 AttentionCapture / extract 로직 그대로 재사용.

차이점은 모델 초기화 + ckpt strict load만:
  - v13 init args: num_prototypes, dino_*, patch_pred_weight, cls_pred_weight 등
  - v13 ckpt에는 dino_head / dino_center / teacher_p.* 가중치가 포함됨 → strict=True 가능

Usage 동일:
    sbatch --export=ALL,CKPT=...,TAG=v13_ep8 scripts/cluster/viz_attn_v13.sbatch
"""

import argparse
import os
import sys

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.two_stream_v13 import TwoStreamV13Model
from scripts.eval.visualize_attn_compare import find_high_change_pairs, load_image
from scripts.eval.visualize_attn_v11 import (
    AttentionCapture,
    cls_attn_to_heatmap,
    anchor_attn_to_heatmap,
    motion_peak_anchor,
    motion_edge_bg,
    pred_to_image,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_v13_attention_and_predict(model, img_t_tensor, img_tk_tensor,
                                       anchor_idx, patch_grid=14):
    """v13 forward + attention hooks. v11 helper 그대로 사용 (구조 동일)."""
    x = img_t_tensor.unsqueeze(0).to(DEVICE)
    y = img_tk_tensor.unsqueeze(0).to(DEVICE)

    capture = AttentionCapture()
    capture.register(model)

    try:
        torch.manual_seed(0)
        with torch.no_grad():
            # v13.forward(image_current, image_future, image_future_global=None)
            out = model(x, y, None)
    finally:
        capture.remove()

    m_attn_hm = cls_attn_to_heatmap(capture.attn_weights["m_encoder_last"],
                                     patch_grid=patch_grid, mask=out["mask_m"])
    p_attn_hm = cls_attn_to_heatmap(capture.attn_weights["p_encoder_last"],
                                     patch_grid=patch_grid, mask=out["mask_p"])

    mr_hms = []
    for i in range(len(model.motion_routing)):
        key = f"motion_routing_{i}"
        if key in capture.attn_weights:
            mr_hms.append(anchor_attn_to_heatmap(
                capture.attn_weights[key], anchor_idx, patch_grid=patch_grid,
            ))
        else:
            mr_hms.append(np.zeros((224, 224)))

    return {
        "m_attn": m_attn_hm,
        "p_attn": p_attn_hm,
        "mr_attns": mr_hms,
        "pred_t": pred_to_image(out["pred_t"]),
        "pred_tk": pred_to_image(out["pred_tk"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egodex-dir", required=True)
    parser.add_argument("--droid-dir", required=True)
    parser.add_argument("--output", default="results/attn_v13.png")
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-depth", type=int, default=12)
    parser.add_argument("--m-depth", type=int, default=6)
    parser.add_argument("--num-motion-iters", type=int, default=2)
    parser.add_argument("--num-prototypes", type=int, default=1024,
                        help="v13 본 학습 cfg는 K=1024 (Option β)")
    parser.add_argument("--anchor-mode", default="motion-peak",
                        choices=["center", "motion-peak"])
    parser.add_argument("--mask-ratio-m", type=float, default=0.0)
    parser.add_argument("--mask-ratio-p", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading v13 checkpoint: {args.checkpoint}")
    model = TwoStreamV13Model(
        p_depth=args.p_depth, m_depth=args.m_depth,
        num_motion_iters=args.num_motion_iters,
        mask_ratio_m=args.mask_ratio_m, mask_ratio_p=args.mask_ratio_p,
        num_prototypes=args.num_prototypes,
    ).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                      for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  WARNING: {len(result.missing_keys)} missing keys "
              f"(first 3: {result.missing_keys[:3]})")
    if result.unexpected_keys:
        print(f"  WARNING: {len(result.unexpected_keys)} unexpected keys "
              f"(first 3: {result.unexpected_keys[:3]})")
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Epoch {epoch}, p_depth={args.p_depth}, m_depth={args.m_depth}, K={args.num_prototypes}")

    print(f"\nSelecting high-change pairs from EgoDex...")
    egodex_pairs = find_high_change_pairs(args.egodex_dir, 30, 2, args.gap, args.seed)
    print(f"  Selected: {[p[2] for p in egodex_pairs]}")

    print(f"Selecting high-change pairs from DROID...")
    droid_pairs = find_high_change_pairs(args.droid_dir, 30, 2, args.gap,
                                          args.seed + 1)
    print(f"  Selected: {[p[2] for p in droid_pairs]}")

    all_pairs = egodex_pairs + droid_pairs
    labels = ["EgoDex"] * 2 + ["DROID"] * 2

    patch_grid = 14
    fig, axes = plt.subplots(4, 8, figsize=(40, 20))
    col_titles = [
        "Frame t",
        f"Frame t+{args.gap}",
        "M encoder attn (CLS→patch)",
        "P encoder attn (CLS→patch)",
        "Motion-routing 0 (anchor)",
        "Motion-routing 1 (anchor)",
        "Recon t (Phase 1)",
        f"Recon t+{args.gap} (Phase 3)",
    ]

    for row, ((path_t, path_tk, ep_label, gap, mse), dataset) in enumerate(
            zip(all_pairs, labels)):
        img_t = load_image(path_t)
        img_tk = load_image(path_tk)

        crop = transforms.CenterCrop(224)
        img_t_c = crop(img_t.permute(2, 0, 1)).permute(1, 2, 0)
        img_tk_c = crop(img_tk.permute(2, 0, 1)).permute(1, 2, 0)

        img_t_tensor = img_t_c.permute(2, 0, 1)
        img_tk_tensor = img_tk_c.permute(2, 0, 1)

        img_t_np = img_t_c.numpy()
        img_tk_np = img_tk_c.numpy()

        if args.anchor_mode == "motion-peak":
            anchor_idx = motion_peak_anchor(img_t_np, img_tk_np, patch_grid)
        else:
            anchor_idx = (patch_grid // 2) * patch_grid + (patch_grid // 2)

        result = extract_v13_attention_and_predict(
            model, img_t_tensor, img_tk_tensor, anchor_idx, patch_grid=patch_grid,
        )

        edge_bg = motion_edge_bg(img_t_np, img_tk_np)

        axes[row][0].imshow(img_t_np)
        axes[row][1].imshow(img_tk_np)
        axes[row][2].imshow(edge_bg)
        axes[row][2].imshow(result["m_attn"], cmap="viridis", alpha=0.55)
        axes[row][3].imshow(edge_bg)
        axes[row][3].imshow(result["p_attn"], cmap="viridis", alpha=0.55)
        for mi, mr_hm in enumerate(result["mr_attns"][:2]):
            ax = axes[row][4 + mi]
            ax.imshow(edge_bg)
            ax.imshow(mr_hm, cmap="viridis", alpha=0.55)
            ar, ac = anchor_idx // patch_grid, anchor_idx % patch_grid
            ax.scatter([ac * 16 + 8], [ar * 16 + 8], s=40, c="red",
                       marker="x", linewidths=2)
        axes[row][6].imshow(result["pred_t"])
        axes[row][7].imshow(result["pred_tk"])

        mse_pt = ((img_t_np - result["pred_t"]) ** 2).mean()
        mse_ptk = ((img_tk_np - result["pred_tk"]) ** 2).mean()

        axes[row][0].set_ylabel(
            f"{dataset}\n{ep_label[:25]}\nΔ={mse:.3f}",
            fontsize=9, rotation=0, labelpad=80, va="center",
        )
        axes[row][6].set_xlabel(f"MSE={mse_pt:.4f}", fontsize=8)
        axes[row][7].set_xlabel(f"MSE={mse_ptk:.4f}", fontsize=8)

        for col in range(8):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for col in range(8):
        axes[0][col].set_title(col_titles[col], fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Two-Stream v13 Epoch {epoch} — Attention + Reconstruction "
        f"(gap={args.gap}, motion_iters={args.num_motion_iters}, K={args.num_prototypes})",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
