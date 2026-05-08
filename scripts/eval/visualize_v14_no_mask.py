"""v14 ckpt → mask 없이 reconstruction 시각화.

학습 중 자동 저장 sample은 학습 forward (mask 0.75 random) 적용 상태라 직관적이지 않음.
no-mask inference로 model이 "본 그대로" 어떻게 복원하는지 확인.

5 columns: [frame_t, frame_t+k, pred_t (MAE), pred_tk (MAE), pred_tk (V-JEPA motion-routed)]

- pred_t / pred_tk: P encoder visible(=all) → recon decoder → recon_head
- pred_tk (V-JEPA): P_enc_unmasked(t) → motion-routing(K/V from M_completed) → interpreter_2
                    → predicted_tk_repr [B,1+N,D] → recon decoder (interpreter_1 + recon_head)
                    학습된 teacher_p.unmasked output 분포에 정합되도록 학습되지만,
                    interpreter_1 입장에서는 약간 off-distribution. 진단용 시각화.

Usage:
    python scripts/eval/visualize_v14_no_mask.py \
        --ckpt /proj/external_group/mrg/checkpoints/two_stream_v14/<ts>/checkpoint_epoch0004.pt
"""
import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.datasets import EgoDexDataset
from src.models import TwoStreamV14Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument(
        "--out-dir",
        default="paper_artifacts/v14_main_train_samples",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--data-root", default="/proj/external_group/mrg/datasets/egodex/frames"
    )
    # v14 main run hparams (config 매칭 — strict load 성공해야 함)
    ap.add_argument("--p-depth", type=int, default=12)
    ap.add_argument("--m-depth", type=int, default=6)
    ap.add_argument("--num-prototypes", type=int, default=1024)
    ap.add_argument("--dino-teacher-temp", type=float, default=0.04)
    ap.add_argument("--dino-student-temp", type=float, default=0.1)
    ap.add_argument("--dino-center-momentum", type=float, default=0.9)
    ap.add_argument("--lambda-pred", type=float, default=1.0)
    ap.add_argument("--lambda-dino", type=float, default=0.01)
    ap.add_argument("--dino-n-crop", type=int, default=2)
    ap.add_argument("--routing-mode", default="v_from_p")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    epoch = ckpt.get("epoch", 0)
    print(f"  epoch={epoch}, train_loss={ckpt.get('train_loss', 'N/A')}")

    model = TwoStreamV14Model(
        p_depth=args.p_depth,
        m_depth=args.m_depth,
        mask_ratio_p=0.75,
        rotation_aug=True,
        routing_mode=args.routing_mode,
        lambda_pred=args.lambda_pred,
        lambda_dino=args.lambda_dino,
        dino_n_crop=args.dino_n_crop,
        num_prototypes=args.num_prototypes,
        dino_teacher_temp=args.dino_teacher_temp,
        dino_student_temp=args.dino_student_temp,
        dino_center_momentum=args.dino_center_momentum,
    )
    msg = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"  state_dict loaded: {msg}")
    model.eval().cuda()

    # No-mask inference
    model.mask_ratio_p = 0.0

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
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for idx in indices:
            try:
                img_t, img_tk, gap = ds[idx]
            except Exception as e:
                print(f"  skip idx={idx}: {e}")
                continue
            x = img_t.unsqueeze(0).cuda()
            y = img_tk.unsqueeze(0).cuda()

            # Standard forward (image_current_global=None ⇒ V-JEPA/DINO loss 0,
            # but predicted_tk_repr is computed regardless).
            out = model(x, y, None, None)
            pred_t = (
                out["pred_t"].squeeze(0).float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            )
            pred_tk_single = (
                out["pred_tk"].squeeze(0).float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            )

            # V-JEPA motion-routed prediction → recon_head 통과해 픽셀로 시각화.
            # predicted_tk_repr는 encoder-level (teacher_p.forward_unmasked target).
            # interpreter_1은 학습 시 visible-only encoder repr 분포에서 학습됨 → mild OOD.
            predicted_full = out["predicted_tk_repr"]                          # [B,1+N,D]
            predicted_decoded = model._run_interpreter(
                predicted_full + model.dec_pos_embed_p,
                model.interpreter_1,
                model.interpreter_1_norm,
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
        "Pred t (P-MAE, no-mask)",
        "Pred t+k (P-MAE, no-mask)",
        "Pred t+k (V-JEPA motion-routed → recon_head)",
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
        f"v14 epoch {epoch} — no-mask reconstruction "
        f"(col 5 = V-JEPA motion-routed encoder-level latent → interpreter_1 → recon_head, mild OOD)",
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
