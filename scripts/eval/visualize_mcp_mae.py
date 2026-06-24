"""MCP-MAE(§9 pixel-pred) ckpt → 복구/예측 가시화 (seen vs unseen).

쌍(frame_t, frame_t+k)당 **3개 출력**(loss 3항과 1:1):
  recover_t    = predict_pixels(P_t,  M_null=M(t,t))    → 정지 routing → frame_t   복구   (L_t)
  recover_t+k  = predict_pixels(P_tk, M_null=M(tk,tk))  → 정지 routing → frame_t+k 복구   (L_tk)
  predict_t+k  = predict_pixels(P_t,  M_real=M(t,t+k))  → 모션 routing → frame_t+k 예측   (L_pred)
같은 메커니즘인데 M routing(null vs real)만 다름. predict_t+k가 recover_t보다 frame_t+k에
가까우면 모션 라우팅이 "미래 예측"을 만든다는 증거.

학습 분포대로 **masked(0.75) 복원**: visible(입력 원본) + masked(예측) 합성 = 표준 MAE 뷰.
**seen**(EgoDex, 학습 도메인) vs **unseen**(DROID, cross-domain)으로 일반화 비교.

사용:
  python scripts/eval/visualize_mcp_mae.py --ckpt <pixel_pred.pt> --no-sobel \
      --embed-dim 384 --num-heads 6 --m-depth 6
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets import EgoDexDataset, DROIDDataset
from src.models import TwoStreamV15Model


def to_np(t):
    return t.detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--no-sobel", action="store_true")
    ap.add_argument("--embed-dim", type=int, default=384)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--p-depth", type=int, default=12)
    ap.add_argument("--m-depth", type=int, default=6)
    ap.add_argument("--routing-mode", default="v_from_p")
    ap.add_argument("--routing-source", default="m", choices=["m", "p"])
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--num-per-source", type=int, default=2)   # 쌍 개수(seen/unseen 각)
    ap.add_argument("--egodex-root", default="/proj/external_group/mrg/datasets/egodex/frames")
    ap.add_argument("--seen-split", default="part1")     # 학습 도메인
    ap.add_argument("--unseen-split", default="part4")   # held-out (모델 미학습) — 같은 도메인 generalization
    ap.add_argument("--droid-root", default="/proj/external_group/mrg/datasets/droid_frames")
    ap.add_argument("--droid-cameras", default="ext1")
    ap.add_argument("--unseen-droid", action="store_true",  # cross-domain DROID (느린 스캔 — GPU 잡 권장)
                    help="unseen을 DROID(cross-domain)로. 미설정 시 EgoDex held-out split(빠름).")
    ap.add_argument("--max-videos", type=int, default=100)
    ap.add_argument("--out", default="paper_artifacts/mcp_mae_sanity_recon/recon.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"  epoch={ckpt.get('epoch', '?')}, train_loss={ckpt.get('train_loss', '?')}")
    model = TwoStreamV15Model(
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        p_depth=args.p_depth, m_depth=args.m_depth,
        mask_ratio_p=args.mask_ratio, routing_mode=args.routing_mode,
        routing_source=args.routing_source, use_sobel=not args.no_sobel,
        pair_mode=True, use_compose=False, pixel_pred=True,
    )
    sd = ckpt["model_state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    print("  load:", model.load_state_dict(sd, strict=True))
    model.eval().to(device)
    pp = model.preprocessing

    ds_seen = EgoDexDataset(
        data_root=args.egodex_root, split=args.seen_split, max_gap=30,
        sample_dist="triangular", sample_center=15, train=False,
        return_triple=False, max_videos=args.max_videos,
    )
    if args.unseen_droid:
        ds_unseen = DROIDDataset(
            data_root=args.droid_root, cameras=args.droid_cameras.split(","), max_gap=15,
            sample_dist="triangular", sample_center=8, train=False,
            return_triple=False, max_videos=args.max_videos,
        )
        unseen_label = "UNSEEN (DROID x-domain)"
    else:
        ds_unseen = EgoDexDataset(
            data_root=args.egodex_root, split=args.unseen_split, max_gap=30,
            sample_dist="triangular", sample_center=15, train=False,
            return_triple=False, max_videos=args.max_videos,
        )
        unseen_label = f"UNSEEN (EgoDex {args.unseen_split})"
    sources = [(f"SEEN (EgoDex {args.seen_split})", ds_seen), (unseen_label, ds_unseen)]
    n_patch = (224 // 16) ** 2

    def recon(x_vis_src, x_loss_tgt, m_routing, masked):
        """masked=True: visible(원본)+masked(예측) 합성(학습 분포). False: recon_head 전체 출력(nomask, OOD)."""
        if masked:
            mask = model._random_mask(1, device, args.mask_ratio)
        else:
            mask = torch.zeros(1, n_patch, dtype=torch.bool, device=device)
        p_vis = model._student_p_encode_visible(pp.compute_p_channel(x_vis_src), mask)
        pred = model._predict_pixels(p_vis, mask, m_routing, x_loss_tgt)[1]
        if masked:
            vis = model._patchify(x_vis_src)
            m = mask.unsqueeze(-1).float()
            img = model._unpatchify(vis * (1 - m) + pred * m)
            std = pred[0][mask[0]].std().item()
        else:
            img = model._unpatchify(pred)             # 전체 patch 예측 (마스크 없음)
            std = pred[0].std().item()
        return to_np(img[0]), std

    rows = []
    with torch.no_grad():
        for label, ds in sources:
            for k in random.sample(range(len(ds)), args.num_per_source):
                img_t, img_tk, gap = ds[k]
                x_t, x_tk = img_t.unsqueeze(0).to(device), img_tk.unsqueeze(0).to(device)
                m_null = model._encode_m_unmasked(pp.compute_m_channel(x_t, x_t))
                m_real = model._encode_m_unmasked(pp.compute_m_channel(x_t, x_tk))
                outs, stds = [], []
                # 3 출력 × {mask, nomask}: recover_t(M_null) / recover_t+k(M_null) / predict_t+k(M_real)
                for (xv, xt, mr) in [(x_t, x_t, m_null), (x_tk, x_tk, m_null), (x_t, x_tk, m_real)]:
                    for masked in (True, False):
                        im, s = recon(xv, xt, mr, masked)
                        outs.append(im); stds.append(s)
                print(f"  [{label}] gap={int(gap)} | masked std rec_t={stds[0]:.3f} "
                      f"rec_tk={stds[2]:.3f} pred_tk={stds[4]:.3f} | tgt={model._patchify(x_tk).std().item():.3f}")
                rows.append((label, int(gap), to_np(img_t), to_np(img_tk), *outs))

    titles = ["frame_t", "frame_t+k",
              "rec_t mask", "rec_t nomask", "rec_t+k mask", "rec_t+k nomask",
              "pred_t+k mask", "pred_t+k nomask"]
    ncol = len(titles)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.5 * ncol, 4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for r, (label, gap, *imgs) in enumerate(rows):
        for col, im in enumerate(imgs):
            axes[r][col].imshow(im)
            axes[r][col].axis("off")
            if r == 0:
                axes[r][col].set_title(titles[col], fontsize=11)
        axes[r][0].set_ylabel(f"{label}\ngap={gap}", fontsize=9, rotation=0, labelpad=55, va="center")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
