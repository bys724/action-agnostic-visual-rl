"""v13 ckpt → mask 없이 reconstruction 시각화.

학습 중 자동 저장되는 sample은 학습 forward를 그대로 호출 → MAE-style random mask 적용된 상태.
시각화엔 mask=0 inference가 더 직관적 (model이 "본 그대로" 어떻게 복원하는지).

Usage:
    python scripts/eval/visualize_v13_no_mask.py \
        --ckpt /proj/external_group/mrg/checkpoints/two_stream_v13/<ts>/latest.pt
"""
import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets import EgoDexDataset
from src.models import TwoStreamV13Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument(
        "--out-dir",
        default="paper_artifacts/v13_main_train_samples",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--data-root", default="/proj/external_group/mrg/datasets/egodex/frames"
    )
    # v13 hyperparams (sbatch defaults — ckpt와 일치해야 strict load 성공)
    ap.add_argument("--num-prototypes", type=int, default=1024)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    epoch = ckpt.get("epoch", 0)
    print(f"  epoch={epoch}, train_loss={ckpt.get('train_loss', 'N/A')}")

    model = TwoStreamV13Model(
        p_depth=12,
        m_depth=6,
        mask_ratio_m=0.3,
        mask_ratio_p=0.75,
        rotation_aug=False,
        routing_mode="v_from_p",
        patch_pred_weight=1.5,
        cls_pred_weight=0.01,
        dino_center_momentum=0.95,
        num_prototypes=args.num_prototypes,
    )
    msg = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"  state_dict loaded: {msg}")
    model.eval().cuda()

    # Mask off for inference visualization
    model.mask_ratio_p = 0.0
    model.mask_ratio_m = 0.0

    ds = EgoDexDataset(
        data_root=args.data_root,
        split="test",
        max_gap=60,
        sample_dist="triangular",
        sample_center=30,
        train=False,
    )
    print(f"Dataset: {len(ds)} samples")

    indices = random.sample(range(len(ds)), args.num_samples)

    rows = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for idx in indices:
            try:
                img_t, img_tk, gap = ds[idx]
            except Exception as e:
                print(f"  skip idx={idx}: {e}")
                continue
            x = img_t.unsqueeze(0).cuda()
            y = img_tk.unsqueeze(0).cuda()
            out = model(x, y, None)
            # pred_t: frame_t single-image reconstruction (interpreter_1 + recon_head)
            # pred_tk_single: frame_{t+k} single-image reconstruction (interpreter_1 + recon_head)
            # pred_tk_motion: motion-routed encoder-level latent → interpreter_1 → recon_head
            #                 (encoder-level routing 디자인에서는 decoder path가 학습 분포 안)
            pred_t = (
                out["pred_t"].squeeze(0).float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            )
            pred_tk_single = (
                out["pred_tk"].squeeze(0).float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            )
            # Motion-routed reconstruction (encoder-level routing 디자인 정합):
            #   predicted_p_tk_full = [CLS, patches] (encoder-level, teacher target과 align)
            #     → interpreter_1 → recon_head → pixel
            # interpreter_1는 frame_t/tk encoder repr에서 reconstruction 학습 → motion-routed
            # encoder-level repr도 같은 manifold라 in-distribution decoder.
            predicted_full = torch.cat(
                [out["predicted_cls_tk"].unsqueeze(1), out["predicted_patches_tk"]],
                dim=1,
            )  # [B, 1+N, D]
            predicted_decoded = model._run_interpreter(
                predicted_full, model.interpreter_1, model.interpreter_1_norm,
            )
            patch_pred_motion = model.recon_head(predicted_decoded[:, 1:])
            pred_tk_motion = (
                model._unpatchify(patch_pred_motion)
                .squeeze(0)
                .float()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .clip(0, 1)
            )
            rows.append(
                {
                    "img_t": img_t.permute(1, 2, 0).numpy().clip(0, 1),
                    "img_tk": img_tk.permute(1, 2, 0).numpy().clip(0, 1),
                    "pred_t": pred_t,
                    "pred_tk_single": pred_tk_single,
                    "pred_tk_motion": pred_tk_motion,
                    "gap": int(gap),
                }
            )

    if not rows:
        print("ERROR: no samples produced")
        return

    N = len(rows)
    fig, axes = plt.subplots(N, 5, figsize=(20, 4 * N))
    if N == 1:
        axes = [axes]
    col_titles = [
        "Frame t",
        "Frame t+k",
        "Pred t (Ph1, single-image)",
        "Pred t+k (Ph1, single-image)",
        "Pred t+k (motion-routed → recon_head)",
    ]
    for r, row in enumerate(rows):
        for c, key in enumerate(
            ["img_t", "img_tk", "pred_t", "pred_tk_single", "pred_tk_motion"]
        ):
            axes[r][c].imshow(row[key])
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=10)
        axes[r][0].text(
            -0.06,
            0.5,
            f"gap={row['gap']}",
            transform=axes[r][0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
    fig.suptitle(
        f"v13 epoch {epoch} — no-mask reconstruction "
        f"(col 5 = motion-routed encoder-level latent → interpreter_1 → recon_head)",
        fontsize=12,
        y=0.995,
        fontweight="bold",
    )
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"epoch_{epoch:03d}_nomask.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
