"""Parvo(pair) ckpt → 2-frame nomask reconstruction + motion routing 가시화.

pair_mode 모델(2-frame: t, t+k)의 표현 경로를 픽셀로 시각화:

  Col 1-2  GT frames: t / t+k
  Col 3-4  P MAE nomask reconstruction: recon(t) / recon(t+k)
  Col 5    V-JEPA P motion routing: Teacher_P(t) + m_chan(t→t+k) → pred ≈ P(t+k)

행 = 데이터 출처:
  · EgoDex-train(part1)  = SEEN     (학습에 사용한 분포)
  · DROID                = NEW      (cross-domain, 미학습)

Col 3-5는 encoder-level latent → interpreter_1 → recon_head 통과로 픽셀화.
interpreter_1은 학습 시 (student p_visible + mask_token, 75% mask) 분포에서 학습됨
→ nomask inference는 mild OOD (학습 초기일수록 흐릿할 수 있음).

Usage:
    python scripts/eval/visualize_v15_no_mask.py \
        --ckpt /proj/.../parvo_pair_ssim/<ts>/latest.pt --no-sobel
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
    ap.add_argument("--no-sobel", action="store_true",
                    help="Parvo/Paper2 no-Sobel 모델 (P=RGB 3ch, M=ΔL 1ch)")
    ap.add_argument("--masked-anchor", action="store_true",
                    help="Run B masked anchor 모델 (decode_first 라우팅 순서 일치 + masked 라우팅 컬럼)")
    ap.add_argument("--num-egodex-train", type=int, default=3,
                    help="EgoDex train split(part1, SEEN) sample 수 — 학습데이터 복구")
    ap.add_argument("--egodex-train-split", default="part1")
    ap.add_argument("--num-droid", type=int, default=3,
                    help="DROID cross-domain(NEW) sample 수 — 학습은 EgoDex만")
    ap.add_argument("--out-dir", default="paper_artifacts/parvo_pair_recon_samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--egodex-data-root", default="/proj/external_group/mrg/datasets/egodex/frames")
    ap.add_argument("--droid-data-root", default="/proj/external_group/mrg/datasets/droid_frames")
    ap.add_argument("--droid-cameras", default="ext1")
    ap.add_argument("--droid-max-videos", type=int, default=10,
                    help="DROID는 클러스터에 sanity 분량(10 ep)만 frame 추출됨")
    # EgoDex 학습 분포 (30Hz, max_gap=30 ≈ 1s, center=15 ≈ 0.5s)
    ap.add_argument("--egodex-max-gap", type=int, default=30)
    ap.add_argument("--egodex-sample-center", type=int, default=15)
    # DROID 15Hz → 동일 시간 의미 위해 절반
    ap.add_argument("--droid-max-gap", type=int, default=15)
    ap.add_argument("--droid-sample-center", type=int, default=8)
    # 모델 아키텍처 (본 학습 hparams 일치)
    ap.add_argument("--p-depth", type=int, default=12)
    ap.add_argument("--m-depth", type=int, default=6)
    ap.add_argument("--routing-mode", default="v_from_p")
    ap.add_argument("--mask-ratio-m-jepa", type=float, default=0.5)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    epoch = ckpt.get("epoch", 0)
    print(f"  epoch={epoch}, train_loss={ckpt.get('train_loss', 'N/A')}")

    # pair_mode=True로 생성 → composition_head 미생성 → pair ckpt와 정확히 일치 (strict)
    model = TwoStreamV15Model(
        p_depth=args.p_depth,
        m_depth=args.m_depth,
        mask_ratio_p=0.75,
        rotation_aug=True,
        routing_mode=args.routing_mode,
        use_sobel=not args.no_sobel,
        mask_ratio_m_jepa=args.mask_ratio_m_jepa,
        pair_mode=True,
        masked_anchor=args.masked_anchor,
    )
    sd = ckpt["model_state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=True)
    print(f"  state_dict loaded (no_sobel={args.no_sobel}, pair_mode=True): {msg}")
    model.eval().cuda()
    model.mask_ratio_p = 0.0  # no-mask inference

    ds_egodex = EgoDexDataset(
        data_root=args.egodex_data_root,
        split=args.egodex_train_split,
        max_gap=args.egodex_max_gap,
        sample_dist="triangular",
        sample_center=args.egodex_sample_center,
        train=False,
        return_triple=False,
    )
    print(f"EgoDex (train={args.egodex_train_split}, SEEN): {len(ds_egodex)} samples")

    ds_droid = DROIDDataset(
        data_root=args.droid_data_root,
        cameras=args.droid_cameras.split(","),
        max_gap=args.droid_max_gap,
        sample_dist="triangular",
        sample_center=args.droid_sample_center,
        train=False,
        return_triple=False,
        max_videos=args.droid_max_videos,
    )
    print(f"DROID  (cross-domain, NEW): {len(ds_droid)} samples")

    sources = []
    if args.num_egodex_train > 0:
        sources.append(("EgoDex-train(SEEN)", ds_egodex,
                        random.sample(range(len(ds_egodex)), args.num_egodex_train)))
    if args.num_droid > 0:
        sources.append(("DROID(NEW/x-domain)", ds_droid,
                        random.sample(range(len(ds_droid)), args.num_droid)))

    def to_pixel_from_repr(repr_full: torch.Tensor) -> np.ndarray:
        """encoder-level [B,1+N,D] → interpreter_1 → recon_head → unpatchify → np image."""
        decoded = model._run_interpreter(
            repr_full + model.dec_pos_embed_p,
            model.interpreter_1, model.interpreter_1_norm,
        )
        patch_pred = model.recon_head(decoded[:, 1:])
        return to_np_image(model._unpatchify(patch_pred))

    rows = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for dataset_name, ds, idx_list in sources:
          for idx in idx_list:
            try:
                img_t, img_tk, gap = ds[idx]
            except Exception as e:
                print(f"  skip [{dataset_name}] idx={idx}: {e}")
                continue

            x_t = img_t.unsqueeze(0).cuda()
            x_tk = img_tk.unsqueeze(0).cuda()

            p_t = model.preprocessing.compute_p_channel(x_t)
            p_tk = model.preprocessing.compute_p_channel(x_tk)
            m_chan = model.preprocessing.compute_m_channel(x_t, x_tk)  # t → t+k

            # P MAE masked recon (학습 분포 = 75% mask): visible=GT, masked=예측 합성.
            # 학습과 동일 조건이라 모델의 실제 복구 능력을 봄 (nomask는 OOD).
            def combine_masked(patch_pred, mask, img):
                tgt = model._patchify(img)
                me = mask.unsqueeze(-1).to(patch_pred.dtype)
                return to_np_image(model._unpatchify(patch_pred * me + tgt * (1 - me)))

            model.mask_ratio_p = 0.75
            _, pp_t_m, mask_t, _ = model._mae_one_frame(x_t, p_t)
            _, pp_tk_m, mask_tk, _ = model._mae_one_frame(x_tk, p_tk)
            recon_t_masked = combine_masked(pp_t_m, mask_t, x_t)
            recon_tk_masked = combine_masked(pp_tk_m, mask_tk, x_tk)

            # P MAE nomask recon (참고; recon_head가 masked 분포 학습 → mild OOD)
            model.mask_ratio_p = 0.0
            _, patch_pred_t, _, _ = model._mae_one_frame(x_t, p_t)
            _, patch_pred_tk, _, _ = model._mae_one_frame(x_tk, p_tk)
            recon_t = to_np_image(model._unpatchify(patch_pred_t))
            recon_tk = to_np_image(model._unpatchify(patch_pred_tk))

            # V-JEPA P motion routing (t → t+k): student anchor + M routing → pred ≈ P(t+k)
            m_local = model._encode_m_unmasked(m_chan)
            _, pred_motion, _ = model._vjepa_p_one_segment(
                m_chan, p_t, p_tk, m_local_routing=m_local,
            )
            motion = to_pixel_from_repr(pred_motion)

            # 동일 프레임 self-pair 라우팅: M(x,x)≈0 (zero-motion) anchor=target=p_t → 라우팅.
            # 단일프레임 inference 모드(full pipeline 사용). ⚠️ M엔 OOD(학습은 실제 motion만 봄)
            # → "정지 관찰을 라우팅이 어떻게 변환하나" 진단. M=0이라 pred≈P_t 예상.
            m_chan_self = model.preprocessing.compute_m_channel(x_t, x_t)
            m_local_self = model._encode_m_unmasked(m_chan_self)
            _, pred_motion_self, _ = model._vjepa_p_one_segment(
                m_chan_self, p_t, p_t, m_local_routing=m_local_self,
            )
            motion_self = to_pixel_from_repr(pred_motion_self)

            # masked frame_t → 모션 라우팅 → frame_t+k 예측 → pixels (Run B 학습 경로 = masked anchor).
            # masked 복구(col3-4) 다음에 배치 = "본 것으로 안 본 미래 예측" 흐름 일치.
            motion_masked = None
            if args.masked_anchor:
                model.mask_ratio_p = 0.75
                _, _, mask_mt, p_t_vis = model._mae_one_frame(x_t, p_t)
                _, pred_motion_masked, _ = model._vjepa_p_masked(p_t_vis, mask_mt, p_tk, m_local)
                motion_masked = to_pixel_from_repr(pred_motion_masked)

            rows.append({
                "dataset": dataset_name,
                "img_t": to_np_image(x_t),
                "img_tk": to_np_image(x_tk),
                "recon_t_masked": recon_t_masked,
                "recon_tk_masked": recon_tk_masked,
                "motion_masked": motion_masked,
                "recon_t": recon_t,
                "recon_tk": recon_tk,
                "motion": motion,
                "motion_self": motion_self,
                "gap": int(gap),
            })

    if not rows:
        print("ERROR: no samples produced")
        return

    N = len(rows)
    col_titles = ["GT t", "GT t+k",
                  "Recon t\n(75% masked)", "Recon t+k\n(75% masked)"]
    keys = ["img_t", "img_tk", "recon_t_masked", "recon_tk_masked"]
    if args.masked_anchor:
        col_titles.append("Masked t→routing\n(→t+k 예측)")
        keys.append("motion_masked")
    col_titles += ["Recon t\n(nomask, OOD)", "Recon t+k\n(nomask, OOD)",
                   "Motion t→t+k\n(V-JEPA P routing)",
                   "Motion x→x\n(self-pair, M≈0)"]
    keys += ["recon_t", "recon_tk", "motion", "motion_self"]
    ncols = len(keys)
    fig, axes = plt.subplots(N, ncols, figsize=(3 * ncols, 3.2 * N))
    if N == 1:
        axes = [axes]
    for r, row in enumerate(rows):
        for c, key in enumerate(keys):
            axes[r][c].imshow(row[key])
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=9)
        axes[r][0].text(
            -0.08, 0.5,
            f"{row['dataset']}\ngap={row['gap']}",
            transform=axes[r][0].transAxes,
            ha="right", va="center", fontsize=9, fontweight="bold",
        )
    _tag = "Parvo no-Sobel" if args.no_sobel else "v15b"
    fig.suptitle(
        f"{_tag} pair epoch {epoch} — P MAE 복구(3-4) + V-JEPA P motion routing(5)\n"
        f"행: EgoDex-train(SEEN) / DROID(NEW, x-domain). col 3-5 = latent → interpreter_1 → recon_head",
        fontsize=11, y=0.995, fontweight="bold",
    )
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"epoch_{epoch:03d}_pair.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
