#!/usr/bin/env python3
"""Two-Stream rotation diagnostic.

하나의 sample 에 대해 입력 이미지를 0/90/180/270도 회전시키며
M/P attention 이 content-driven (equivariant) 인지, 아니면
position-prior 고정 (invariant) 인지 진단.

진짜 motion content 를 읽는다면 attention 은 입력과 함께 회전해야 함.
2D RoPE 는 절대좌표 기반이라 equivariance 를 구조적으로 보장하지 않음
→ 그럼에도 attention 이 회전을 따라가면 "content 의존", 아니면 "RoPE + 학습된 prior 붕괴".
"""

import argparse
import os
import sys

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TwoStreamModel

# visualize_attn_compare 의 헬퍼 재사용
from scripts.eval.visualize_attn_compare import (
    load_image, extract_attention_and_predict, attn_to_heatmap,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROTATIONS = [0, 90, 180, 270]


def rotate_frame(img_chw, k):
    """torch rot90: [C, H, W] -> [C, H, W] counter-clockwise by k*90°."""
    return torch.rot90(img_chw, k=k // 90, dims=(-2, -1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-t", required=True, help="Frame t 이미지 경로")
    parser.add_argument("--frame-tk", required=True, help="Frame t+k 이미지 경로")
    parser.add_argument("--output", default="results/viz_rotation/attn_rotation.png")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-ratio-p", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load model
    print(f"Loading: {args.checkpoint}")
    model = TwoStreamModel(
        depth=args.depth, num_stages=args.num_stages,
        mask_ratio=args.mask_ratio, mask_ratio_p=args.mask_ratio_p,
    ).to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    epoch = ck.get('epoch', '?')
    print(f"  Epoch {epoch}")

    # Load frames → center crop 224
    img_t = load_image(args.frame_t)
    img_tk = load_image(args.frame_tk)
    crop = transforms.CenterCrop(224)
    img_t_c = crop(img_t.permute(2, 0, 1))   # [3, 224, 224]
    img_tk_c = crop(img_tk.permute(2, 0, 1))

    # Build figure: 4 rows (rotations) × 6 cols
    fig, axes = plt.subplots(4, 6, figsize=(30, 20))
    col_titles = ['Frame t (rot)', 'Frame t+k (rot)',
                  'M Attn on ΔL', 'P Attn on ΔL', 'Pred M', 'Pred P']

    for row, deg in enumerate(ROTATIONS):
        # Rotate input pair *identically*
        img_t_rot = rotate_frame(img_t_c, deg)
        img_tk_rot = rotate_frame(img_tk_c, deg)

        attn_maps, pred_m, pred_p, delta_l_rgb = extract_attention_and_predict(
            model, img_t_rot, img_tk_rot)

        img_t_vis = img_t_rot.permute(1, 2, 0).numpy()
        img_tk_vis = img_tk_rot.permute(1, 2, 0).numpy()
        m_hm = attn_to_heatmap(attn_maps['m'])
        p_hm = attn_to_heatmap(attn_maps['p'])

        axes[row][0].imshow(img_t_vis)
        axes[row][1].imshow(img_tk_vis)
        axes[row][2].imshow(delta_l_rgb)
        axes[row][2].imshow(m_hm, cmap='hot', alpha=0.5)
        axes[row][3].imshow(delta_l_rgb)
        axes[row][3].imshow(p_hm, cmap='hot', alpha=0.5)
        axes[row][4].imshow(pred_m)
        axes[row][5].imshow(pred_p)

        mse_m = ((img_tk_vis - pred_m) ** 2).mean()
        mse_p = ((img_tk_vis - pred_p) ** 2).mean()

        axes[row][0].set_ylabel(
            f'{deg}°',
            fontsize=14, rotation=0, labelpad=40, va='center', fontweight='bold',
        )
        axes[row][4].set_xlabel(f'MSE={mse_m:.4f}', fontsize=9)
        axes[row][5].set_xlabel(f'MSE={mse_p:.4f}', fontsize=9)

        for col in range(6):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for col in range(6):
        axes[0][col].set_title(col_titles[col], fontsize=12, fontweight='bold')

    sample_id = os.path.basename(os.path.dirname(args.frame_t))
    fig.suptitle(
        f'Two-Stream Epoch {epoch} — Rotation Diagnostic (sample: {sample_id})\n'
        f'If model is content-driven, attention should rotate WITH input.',
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
