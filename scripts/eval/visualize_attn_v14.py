#!/usr/bin/env python3
"""[DEPRECATED] v14 폐기 lineup viz. 코드만 reference 보존.

Two-Stream v14 attention + reconstruction visualization (motion-routing focus).

v14는 v11 base + V-JEPA + DINO. motion-routing block은 v11과 동일 구조 (self.motion_routing).
visualize_attn_v11과 같은 grid + V-JEPA motion-routed prediction 컬럼 추가.

Grid (4 sample × 9 column):
  col 1: Frame t
  col 2: Frame t+k
  col 3: M encoder last-layer attention (CLS → patches, unmasked)
  col 4: P encoder last-layer attention (CLS → patches, no-mask inference)
  col 5: Motion-routing iter 0 attention (Q/K from M, anchor patch)
  col 6: Motion-routing iter 1 attention (same anchor)
  col 7: Recon t (P-MAE no-mask)
  col 8: Recon t+k (P-MAE no-mask)
  col 9: V-JEPA motion-routed → recon_head (encoder-level latent 디코드)
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.two_stream_v14 import TwoStreamV14Model
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


@torch.no_grad()
def extract_v14_attention_and_predict(
    model: TwoStreamV14Model,
    img_t_tensor: torch.Tensor,
    img_tk_tensor: torch.Tensor,
    anchor_idx: int,
    patch_grid: int = 14,
):
    """v14 forward + attention hooks. Returns dict of arrays for plotting."""
    x = img_t_tensor.unsqueeze(0).to(DEVICE)
    y = img_tk_tensor.unsqueeze(0).to(DEVICE)

    capture = AttentionCapture()
    # AttentionCapture.register는 v11 model API 가정 (blocks_m, blocks_p, motion_routing).
    # v14는 v11 상속 + 동일 attribute → 그대로 사용 가능.
    capture.register(model)

    try:
        torch.manual_seed(0)
        # global views None → V-JEPA/DINO loss 0, predicted_tk_repr는 계산됨
        out = model(x, y, None, None)
    finally:
        capture.remove()

    # v14는 M 항상 unmasked → mask=None (full grid)
    m_attn_hm = cls_attn_to_heatmap(
        capture.attn_weights["m_encoder_last"], patch_grid=patch_grid, mask=None,
    )
    # P encoder: viz 시 mask_ratio_p=0 으로 호출 → mask는 zeros placeholder, full grid
    p_attn_hm = cls_attn_to_heatmap(
        capture.attn_weights["p_encoder_last"], patch_grid=patch_grid, mask=None,
    )

    mr_hms = []
    for i in range(len(model.motion_routing)):
        key = f"motion_routing_{i}"
        if key in capture.attn_weights:
            mr_hms.append(
                anchor_attn_to_heatmap(
                    capture.attn_weights[key], anchor_idx, patch_grid=patch_grid,
                )
            )
        else:
            mr_hms.append(np.zeros((224, 224)))

    pred_t_img = pred_to_image(out["pred_t"])
    pred_tk_img = pred_to_image(out["pred_tk"])

    # V-JEPA motion-routed → recon_head (mild OOD 진단)
    predicted_full = out["predicted_tk_repr"]  # [1, 1+N, D]
    predicted_decoded = model._run_interpreter(
        predicted_full + model.dec_pos_embed_p,
        model.interpreter_1,
        model.interpreter_1_norm,
    )
    patch_pred_motion = model.recon_head(predicted_decoded[:, 1:])
    pred_tk_motion_img = pred_to_image(model._unpatchify(patch_pred_motion))

    return {
        "m_attn": m_attn_hm,
        "p_attn": p_attn_hm,
        "mr_attns": mr_hms,
        "pred_t": pred_t_img,
        "pred_tk": pred_tk_img,
        "pred_tk_motion": pred_tk_motion_img,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egodex-dir", required=True)
    parser.add_argument("--droid-dir", required=True)
    parser.add_argument("--output", default="paper_artifacts/v14_main_train_samples/attn_v14.png")
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-depth", type=int, default=12)
    parser.add_argument("--m-depth", type=int, default=6)
    parser.add_argument("--num-prototypes", type=int, default=1024)
    parser.add_argument("--lambda-pred", type=float, default=1.0)
    parser.add_argument("--lambda-dino", type=float, default=0.01)
    parser.add_argument("--dino-n-crop", type=int, default=2)
    parser.add_argument("--dino-teacher-temp", type=float, default=0.04)
    parser.add_argument("--dino-student-temp", type=float, default=0.1)
    parser.add_argument("--dino-center-momentum", type=float, default=0.9)
    parser.add_argument("--routing-mode", default="v_from_p")
    parser.add_argument(
        "--anchor-mode",
        default="motion-peak",
        choices=["center", "motion-peak"],
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading v14 checkpoint: {args.checkpoint}")
    model = TwoStreamV14Model(
        p_depth=args.p_depth,
        m_depth=args.m_depth,
        mask_ratio_p=0.75,  # init only — viz용 mask=0
        rotation_aug=True,
        routing_mode=args.routing_mode,
        lambda_pred=args.lambda_pred,
        lambda_dino=args.lambda_dino,
        dino_n_crop=args.dino_n_crop,
        num_prototypes=args.num_prototypes,
        dino_teacher_temp=args.dino_teacher_temp,
        dino_student_temp=args.dino_student_temp,
        dino_center_momentum=args.dino_center_momentum,
    ).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
    result = model.load_state_dict(state_dict, strict=True)
    print(f"  state_dict loaded: {result}")
    model.eval()
    # No-mask inference
    model.mask_ratio_p = 0.0
    epoch = ckpt.get("epoch", "?")
    print(f"  Epoch {epoch}, p_depth={args.p_depth}, m_depth={args.m_depth}")

    print(f"\nSelecting high-change pairs from EgoDex...")
    egodex_pairs = find_high_change_pairs(args.egodex_dir, 30, 2, args.gap, args.seed)
    print(f"  Selected: {[p[2] for p in egodex_pairs]} "
          f"(MSE: {[f'{p[4]:.4f}' for p in egodex_pairs]})")

    print(f"Selecting high-change pairs from DROID...")
    droid_pairs = find_high_change_pairs(args.droid_dir, 30, 2, args.gap, args.seed + 1)
    print(f"  Selected: {[p[2] for p in droid_pairs]} "
          f"(MSE: {[f'{p[4]:.4f}' for p in droid_pairs]})")

    all_pairs = egodex_pairs + droid_pairs
    labels = ["EgoDex"] * 2 + ["DROID"] * 2
    patch_grid = 14

    fig, axes = plt.subplots(4, 9, figsize=(45, 20))
    col_titles = [
        "Frame t",
        f"Frame t+{args.gap}",
        "M encoder attn (CLS→patch)",
        "P encoder attn (CLS→patch)",
        "Motion-routing 0 (anchor)",
        "Motion-routing 1 (anchor)",
        "Recon t (P-MAE)",
        f"Recon t+{args.gap} (P-MAE)",
        "V-JEPA pred → recon_head",
    ]

    for row, ((path_t, path_tk, ep_label, gap, mse), dataset) in enumerate(
        zip(all_pairs, labels)
    ):
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

        result = extract_v14_attention_and_predict(
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
        axes[row][8].imshow(result["pred_tk_motion"])

        mse_pt = ((img_t_np - result["pred_t"]) ** 2).mean()
        mse_ptk = ((img_tk_np - result["pred_tk"]) ** 2).mean()
        mse_pmot = ((img_tk_np - result["pred_tk_motion"]) ** 2).mean()

        axes[row][0].set_ylabel(
            f"{dataset}\n{ep_label[:25]}\nΔ={mse:.3f}",
            fontsize=9, rotation=0, labelpad=80, va="center",
        )
        axes[row][6].set_xlabel(f"MSE={mse_pt:.4f}", fontsize=8)
        axes[row][7].set_xlabel(f"MSE={mse_ptk:.4f}", fontsize=8)
        axes[row][8].set_xlabel(f"MSE={mse_pmot:.4f}", fontsize=8)

        for col in range(9):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for col in range(9):
        axes[0][col].set_title(col_titles[col], fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Two-Stream v14 Epoch {epoch} — Attention + Reconstruction "
        f"(gap={args.gap}, motion routing 2 iter, anchor=red ×)",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
