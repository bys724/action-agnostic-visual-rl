#!/usr/bin/env python3
"""v8 rotation diagnostic — attn_compare_v8와 동일한 샘플을 0/90/180/270 회전.

각 샘플에 대해 4 rotation × 5 column grid를 한 파일로 저장:
  cols = [Frame t(rot), Frame t+k(rot), Student M attn, Student P attn, Teacher P attn]
  rows = [0°, 90°, 180°, 270°]

attention이 입력과 함께 회전하면 content-driven. 같은 절대좌표 유지하면 position prior 고정.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TwoStreamModel
from scripts.eval.visualize_attn_compare_v8 import (
    load_image,
    find_high_change_pairs,
    forward_student_and_extract,
    forward_teacher_and_extract,
    cls_attn_to_heatmap,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROTATIONS = [0, 90, 180, 270]


def run_rotation_grid(model, img_t_tensor, img_tk_tensor, title, output_path, gap):
    n_rows, n_cols = 4, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    col_titles = [
        'Frame t (rot)',
        f'Frame t+{gap} (rot)',
        'Student M Attn\n(on ΔL)',
        'Student P Attn\n(on Frame t)',
        'Teacher P Attn\n(on Frame t+k)',
    ]

    for row, deg in enumerate(ROTATIONS):
        k = deg // 90
        img_t_rot = torch.rot90(img_t_tensor, k=k, dims=(-2, -1))
        img_tk_rot = torch.rot90(img_tk_tensor, k=k, dims=(-2, -1))

        attn_maps, _pred_m, _delta_l_rgb, cls_p_stu = forward_student_and_extract(
            model, img_t_rot, img_tk_rot)
        attn_p_teacher, cls_p_tea = forward_teacher_and_extract(model, img_tk_rot)

        cos_st = F.cosine_similarity(cls_p_stu, cls_p_tea, dim=-1).item()

        img_t_np = img_t_rot.permute(1, 2, 0).numpy()
        img_tk_np = img_tk_rot.permute(1, 2, 0).numpy()
        m_hm = cls_attn_to_heatmap(attn_maps['m'])
        p_hm_s = cls_attn_to_heatmap(attn_maps['p'])
        p_hm_t = cls_attn_to_heatmap(attn_p_teacher)

        axes[row][0].imshow(img_t_np)
        axes[row][1].imshow(img_tk_np)
        axes[row][2].imshow(img_t_np)
        axes[row][2].imshow(m_hm, cmap='hot', alpha=0.55)
        axes[row][3].imshow(img_t_np)
        axes[row][3].imshow(p_hm_s, cmap='hot', alpha=0.55)
        axes[row][4].imshow(img_tk_np)
        axes[row][4].imshow(p_hm_t, cmap='hot', alpha=0.55)

        axes[row][0].set_ylabel(
            f'{deg}°\ncos(s,t)={cos_st:.3f}',
            fontsize=11, rotation=0, labelpad=45, va='center', fontweight='bold',
        )
        for col in range(n_cols):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for col in range(n_cols):
        axes[0][col].set_title(col_titles[col], fontsize=11, fontweight='bold')

    fig.suptitle(title, fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="v8 rotation diagnostic (same samples as attn_compare_v8)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egodex-dir", required=True)
    parser.add_argument("--droid-dir", default=None)
    parser.add_argument("--output-dir", default="docs/architecture/rotation_v8")
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--epoch-tag", default="ep12")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {args.checkpoint}")
    model = TwoStreamModel(
        depth=args.depth, num_stages=args.num_stages,
        mask_ratio=0.3, mask_ratio_p=0.5,
        use_ape=True, rotation_aug=True,
        v8_mode=True, pred_head_ratio=2.0,
    ).to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    epoch = ck.get('epoch', args.epoch_tag)
    print(f"  Epoch {epoch}")

    # 같은 샘플 선택 (seed=42 EgoDex, seed=43 DROID — attn_compare_v8와 동일)
    print("\nSelecting EgoDex pairs...")
    egodex_pairs = find_high_change_pairs(
        args.egodex_dir, 30, args.num_samples, args.gap, args.seed)
    print(f"  Selected: {[p[2] for p in egodex_pairs]}")

    pairs = [(p, 'EgoDex') for p in egodex_pairs]
    if args.droid_dir and os.path.exists(args.droid_dir):
        print("Selecting DROID pairs...")
        droid_pairs = find_high_change_pairs(
            args.droid_dir, 30, args.num_samples, args.gap, args.seed + 1)
        print(f"  Selected: {[p[2] for p in droid_pairs]}")
        pairs += [(p, 'DROID') for p in droid_pairs]

    crop = transforms.CenterCrop(224)
    for (path_t, path_tk, ep_label, gap, mse), dataset in pairs:
        img_t = load_image(path_t)
        img_tk = load_image(path_tk)
        img_t_c = crop(img_t.permute(2, 0, 1))
        img_tk_c = crop(img_tk.permute(2, 0, 1))

        safe_label = ep_label.replace('/', '_').replace(' ', '_')[:40]
        out_path = os.path.join(
            args.output_dir,
            f"rotation_v8_{args.epoch_tag}_{dataset}_{safe_label}.png",
        )
        title = (
            f'v8 Epoch {epoch} — Rotation Diagnostic\n'
            f'{dataset}: {ep_label}  (ΔMSE={mse:.3f}, gap={gap})'
        )
        run_rotation_grid(model, img_t_c, img_tk_c, title, out_path, gap)


if __name__ == "__main__":
    main()
