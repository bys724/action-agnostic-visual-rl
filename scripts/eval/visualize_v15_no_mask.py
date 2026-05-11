"""v15 ckpt → mask 없이 reconstruction + 3-segment motion routing + composition 가시화.

v15 forward의 모든 representation 경로를 픽셀로 시각화:

  Col 1-3  GT frames: t / t+n / t+m
  Col 4-6  P MAE nomask reconstruction × 3 frame
  Col 7-9  V-JEPA P motion routing × 3 segment:
             short : Teacher_P(t)   + m_chan(t→t+n)  → pred ≈ P(t+n)
             step  : Teacher_P(t+n) + m_chan(t+n→t+m)→ pred ≈ P(t+m)
             long  : Teacher_P(t)   + m_chan(t→t+m)  → pred ≈ P(t+m)
  Col 10   Composition (L_compose path):
             m_predicted = composition_head(m_local_short, m_local_long)
                         = m_long − m_short          (linear_residual mode)
             routing source = m_predicted (GT m_step의 surrogate)
             anchor         = Teacher_P(t+n)
             → pred ≈ P(t+m)

Col 7-10은 encoder-level latent → interpreter_1 → recon_head 통과로 픽셀화.
interpreter_1은 학습 시 (student p_visible + mask_token) 분포에서 학습됨 → mild OOD.
ep4 단계는 recon decoder 미숙해 patch grid speckle 위주일 수 있음 (v14 ep4 viz와 동일).

Usage:
    python scripts/eval/visualize_v15_no_mask.py \
        --ckpt /proj/external_group/mrg/checkpoints/two_stream_v15/<ts>/checkpoint_epoch0004.pt
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
from src.datasets.droid import DROIDDataset
from src.models import TwoStreamV15Model


def to_np_image(t: torch.Tensor) -> np.ndarray:
    """[3, H, W] → [H, W, 3] in [0, 1]."""
    return t.squeeze(0).float().cpu().permute(1, 2, 0).numpy().clip(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-egodex", type=int, default=2)
    ap.add_argument("--num-droid", type=int, default=2,
                    help="DROID cross-domain sample 수 (v15 학습은 EgoDex만 → DROID는 unseen)")
    ap.add_argument("--out-dir", default="paper_artifacts/v15_main_train_samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--egodex-data-root", default="/proj/external_group/mrg/datasets/egodex/frames",
    )
    ap.add_argument(
        "--droid-data-root", default="/proj/external_group/mrg/datasets/droid_frames",
    )
    ap.add_argument("--droid-cameras", default="ext1",
                    help="DROID camera (comma-separated). default: ext1")
    ap.add_argument("--droid-max-videos", type=int, default=10,
                    help="DROID는 클러스터에 sanity 분량(10 ep)만 frame 추출됨. 빈 ep retry 회피용")
    # EgoDex 학습 분포 (30Hz, max_gap=30 ≈ 1s, center=15 ≈ 0.5s)
    ap.add_argument("--egodex-max-gap", type=int, default=30)
    ap.add_argument("--egodex-sample-center", type=int, default=15)
    # DROID 15Hz → 동일 시간 의미 위해 절반
    ap.add_argument("--droid-max-gap", type=int, default=15)
    ap.add_argument("--droid-sample-center", type=int, default=8)
    # v15 main run hparams (cluster_sessions.md:125 일치)
    ap.add_argument("--p-depth", type=int, default=12)
    ap.add_argument("--m-depth", type=int, default=6)
    ap.add_argument("--routing-mode", default="v_from_p")
    ap.add_argument("--composition-mode", default="linear_residual")
    ap.add_argument("--mask-ratio-m-jepa", type=float, default=0.5)
    ap.add_argument("--lambda-pred", type=float, default=1.0)
    ap.add_argument("--lambda-m-jepa", type=float, default=1.0)
    ap.add_argument("--lambda-compose", type=float, default=1.0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    epoch = ckpt.get("epoch", 0)
    print(f"  epoch={epoch}, train_loss={ckpt.get('train_loss', 'N/A')}")

    model = TwoStreamV15Model(
        p_depth=args.p_depth,
        m_depth=args.m_depth,
        mask_ratio_p=0.75,
        rotation_aug=True,
        routing_mode=args.routing_mode,
        lambda_pred=args.lambda_pred,
        lambda_m_jepa=args.lambda_m_jepa,
        lambda_compose=args.lambda_compose,
        mask_ratio_m_jepa=args.mask_ratio_m_jepa,
        composition_mode=args.composition_mode,
    )
    msg = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"  state_dict loaded: {msg}")
    model.eval().cuda()
    model.mask_ratio_p = 0.0  # no-mask inference

    ds_egodex = EgoDexDataset(
        data_root=args.egodex_data_root,
        split="test",
        max_gap=args.egodex_max_gap,
        sample_dist="triangular",
        sample_center=args.egodex_sample_center,
        train=False,
        return_triple=True,
    )
    print(f"EgoDex (test, return_triple): {len(ds_egodex)} samples")

    # NOTE: 클러스터 ext1은 sanity 추출분(처음 10 ep)만 frame이 채워져 있음.
    # 빈 ep에 retry 5회 모두 걸리면 fallback (검정 텐서)이 나오므로 max_videos로 추출된 ep만 제한.
    ds_droid = DROIDDataset(
        data_root=args.droid_data_root,
        cameras=args.droid_cameras.split(","),
        max_gap=args.droid_max_gap,
        sample_dist="triangular",
        sample_center=args.droid_sample_center,
        train=False,
        return_triple=True,
        max_videos=args.droid_max_videos,
    )
    print(f"DROID  (cross-domain, return_triple): {len(ds_droid)} samples")

    sources = []
    if args.num_egodex > 0:
        sources.append(("EgoDex", ds_egodex, random.sample(range(len(ds_egodex)), args.num_egodex)))
    if args.num_droid > 0:
        sources.append(("DROID",  ds_droid,  random.sample(range(len(ds_droid)),  args.num_droid)))

    def to_pixel_from_repr(repr_full: torch.Tensor) -> np.ndarray:
        """encoder-level [B,1+N,D] → interpreter_1 → recon_head → unpatchify → np image."""
        decoded = model._run_interpreter(
            repr_full + model.dec_pos_embed_p,
            model.interpreter_1, model.interpreter_1_norm,
        )
        patch_pred = model.recon_head(decoded[:, 1:])
        img = model._unpatchify(patch_pred)
        return to_np_image(img)

    rows = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for dataset_name, ds, idx_list in sources:
          for idx in idx_list:
            try:
                img_t, img_tn, img_tm, gap_n, gap_m_offset = ds[idx]
            except Exception as e:
                print(f"  skip [{dataset_name}] idx={idx}: {e}")
                continue

            x_t = img_t.unsqueeze(0).cuda()
            x_tn = img_tn.unsqueeze(0).cuda()
            x_tm = img_tm.unsqueeze(0).cuda()

            # Channels
            p_t = model.preprocessing.compute_p_channel(x_t)
            p_tn = model.preprocessing.compute_p_channel(x_tn)
            p_tm = model.preprocessing.compute_p_channel(x_tm)

            m_short = model.preprocessing.compute_m_channel(x_t, x_tn)
            m_step = model.preprocessing.compute_m_channel(x_tn, x_tm)
            m_long = model.preprocessing.compute_m_channel(x_t, x_tm)

            # M encoder unmasked × 3 (routing source + composition operand)
            m_local_short = model._encode_m_unmasked(m_short)
            m_local_step = model._encode_m_unmasked(m_step)
            m_local_long = model._encode_m_unmasked(m_long)

            # P MAE nomask × 3 frame
            _, patch_pred_t, _, _ = model._mae_one_frame(x_t, p_t)
            _, patch_pred_tn, _, _ = model._mae_one_frame(x_tn, p_tn)
            _, patch_pred_tm, _, _ = model._mae_one_frame(x_tm, p_tm)
            recon_t = to_np_image(model._unpatchify(patch_pred_t))
            recon_tn = to_np_image(model._unpatchify(patch_pred_tn))
            recon_tm = to_np_image(model._unpatchify(patch_pred_tm))

            # V-JEPA P motion routing × 3 segment
            _, pred_short, _ = model._vjepa_p_one_segment(
                m_short, p_t, p_tn, m_local_routing=m_local_short,
            )
            _, pred_step, _ = model._vjepa_p_one_segment(
                m_step, p_tn, p_tm, m_local_routing=m_local_step,
            )
            _, pred_long, _ = model._vjepa_p_one_segment(
                m_long, p_t, p_tm, m_local_routing=m_local_long,
            )
            motion_short = to_pixel_from_repr(pred_short)
            motion_step = to_pixel_from_repr(pred_step)
            motion_long = to_pixel_from_repr(pred_long)

            # Composition path: m_predicted (= m_long − m_short for linear_residual)
            # routing source = m_predicted, anchor = Teacher_P(t+n) → predicted ≈ P(t+m)
            m_predicted = model.composition_head(m_local_short, m_local_long)
            anchor_T = model.teacher_p.forward_unmasked(p_tn)
            p_state = anchor_T
            for step in model.p_motion_decoder:
                p_state = step(p_state, m_predicted)
            pred_compose = model.p_motion_decoder_norm(p_state)
            motion_compose = to_pixel_from_repr(pred_compose)

            rows.append({
                "dataset": dataset_name,
                "img_t": to_np_image(x_t),
                "img_tn": to_np_image(x_tn),
                "img_tm": to_np_image(x_tm),
                "recon_t": recon_t,
                "recon_tn": recon_tn,
                "recon_tm": recon_tm,
                "motion_short": motion_short,
                "motion_step": motion_step,
                "motion_long": motion_long,
                "motion_compose": motion_compose,
                "gap_n": int(gap_n),
                "gap_m_offset": int(gap_m_offset),
            })

    if not rows:
        print("ERROR: no samples produced")
        return

    N = len(rows)
    fig, axes = plt.subplots(N, 10, figsize=(30, 3.2 * N))
    if N == 1:
        axes = [axes]
    col_titles = [
        "GT t", "GT t+n", "GT t+m",
        "Recon t\n(P MAE nomask)",
        "Recon t+n\n(P MAE nomask)",
        "Recon t+m\n(P MAE nomask)",
        "Motion t→t+n\n(short, GT m)",
        "Motion t+n→t+m\n(step, GT m)",
        "Motion t→t+m\n(long, GT m)",
        "Composition t+n→t+m\n(m_predicted = m_long−m_short)",
    ]
    keys = [
        "img_t", "img_tn", "img_tm",
        "recon_t", "recon_tn", "recon_tm",
        "motion_short", "motion_step", "motion_long",
        "motion_compose",
    ]
    for r, row in enumerate(rows):
        for c, key in enumerate(keys):
            axes[r][c].imshow(row[key])
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=9)
        axes[r][0].text(
            -0.08, 0.5,
            f"{row['dataset']}\nn={row['gap_n']}\nΔ={row['gap_m_offset']}",
            transform=axes[r][0].transAxes,
            ha="right", va="center", fontsize=9,
            fontweight="bold",
        )
    fig.suptitle(
        f"v15 epoch {epoch} — no-mask reconstruction + motion routing × 3 segment + L_compose path\n"
        f"(col 7-10 = encoder-level latent → interpreter_1 → recon_head; mild OOD)",
        fontsize=11, y=0.995, fontweight="bold",
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
