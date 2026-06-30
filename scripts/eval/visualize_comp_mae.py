"""CoMP-MAE(v16, 대칭 cross-recon) ckpt → P-recon + M-recon 가시화 (seen vs unseen).

쌍(frame_t, frame_t+k)당 두 stream을 시각화:
  P-recon (RGB, deliverable encoder):
    rec_t      = predict_pixels(P_t,  M_null=learned null token)  → 정지 routing → frame_t   복구 (L_t)
    rec_t+k    = predict_pixels(P_tk, M_null)                     → 정지 routing → frame_t+k 복구 (L_tk)
    pred_t+k   = predict_pixels(P_t,  M_real=M(t,t+k))            → 모션 routing → frame_t+k 예측 (L_pred)
  M-recon (ΔL, motion grounding — v16 신규):
    ΔL_target    = M(t,t+k) = lum(t+k) − lum(t)                   → 실제 motion field
    ΔL_recon(B)  = masked ΔL(t,tk) → P-grouping(helper=frame_t) 복구 (Case B, L_mB)
    static(t,t)  = masked ΔL(t,t)=0 → 복구 (Case A, helper=frame_t)   → **출력 0 기대**(calibration)
    static(tk,tk)= masked ΔL(tk,tk)=0 → 복구 (Case A, helper=frame_tk) → 0 기대 (tk appearance 일반화)
    "ΔL_recon(B)가 target motion을 복원 + static이 0이면 M이 motion을 표상 & 정지 calibration 성공."

학습 분포대로 **masked 복원**(P=0.75, M=0.5): visible(원본)+masked(예측) 합성 = 표준 MAE 뷰.
nomask는 학습 분포 밖(OOD 외삽)이라 제외. seen(part1) vs unseen(part4 held-out, +--add-droid면 DROID
cross-domain)으로 일반화 비교. comp_mae_plan §6 검증용.

사용:
  python scripts/eval/visualize_comp_mae.py --ckpt <comp_mae.pt> \
      --embed-dim 384 --num-heads 6 --m-depth 6
"""
import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets import EgoDexDataset, DROIDDataset
from src.models import TwoStreamV15Model


class _FixedEpDataset(DROIDDataset):
    """DROIDDataset의 _scan_frame_dirs만 고정 ep 리스트로 대체 (viz 전용).

    DROID 프레임은 부분 추출 상태(다수 ep가 action.npy만 보유) → 전수 glob+is_dir stat이
    매우 느림(GPFS metadata). 프레임 보유 ep 경로를 직접 주면 스캔을 건너뛰고, 샘플링·crop·
    정규화(/255 [0,1])는 DROIDDataset/VideoFrameDataset에서 그대로 상속 = train parity 유지.
    """

    def __init__(self, fixed_dirs, **kw):
        self._fixed_dirs = list(fixed_dirs)
        super().__init__(**kw)

    def _scan_frame_dirs(self):
        return self._fixed_dirs


def to_np_rgb(t):
    return t.detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1)


def to_np_dl(t):
    """[1,1,H,W] ΔL → [H,W] numpy (diverging colormap용, [-1,1])."""
    return t.detach().float().cpu()[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--embed-dim", type=int, default=384)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--p-depth", type=int, default=12)
    ap.add_argument("--m-depth", type=int, default=6)
    ap.add_argument("--mask-ratio", type=float, default=0.75)      # P stream
    ap.add_argument("--mask-ratio-m", type=float, default=0.5)     # M stream (guard 8: P보다 낮게)
    ap.add_argument("--dl-vlim", type=float, default=0.3, help="ΔL colormap 범위 [-vlim, vlim]")
    ap.add_argument("--num-per-source", type=int, default=2)
    ap.add_argument("--egodex-root", default="/proj/external_group/mrg/datasets/egodex/frames")
    ap.add_argument("--seen-split", default="part1")
    ap.add_argument("--unseen-split", default="part4")
    ap.add_argument("--droid-root", default="/proj/external_group/mrg/datasets/droid_frames")
    ap.add_argument("--droid-cameras", default="ext1")
    ap.add_argument("--add-droid", action="store_true",
                    help="DROID(cross-domain) 행 추가 (part4 held-out과 함께). 고정 ep만 사용 → 전수 스캔 없음.")
    ap.add_argument("--droid-eps", nargs="+",
                    default=["ext1/ep_000002", "ext1/ep_000004", "ext1/ep_000006"],
                    help="DROID ep 디렉터리(droid-root 기준 상대경로) 고정. 프레임 보유 ep만 — 95k 전수 스캔 회피.")
    ap.add_argument("--max-videos", type=int, default=100)
    ap.add_argument("--out", default="scratch/viz/comp_mae_s/recon.png",
                    help="기본=gitignored scratch. 모델별 폴더(comp_mae_{s|b}) + 파일명 recon_ep{NNNN}_{domain}.png. 논문 승격 시 paper_artifacts/figN/ 명시")
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
        mask_ratio_p=args.mask_ratio, use_sobel=False,
        pair_mode=True, use_compose=False, comp_mae=True,
        mask_ratio_m_recon=args.mask_ratio_m,
    )
    sd = ckpt["model_state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    print("  load:", model.load_state_dict(sd, strict=True))
    model.eval().to(device)
    pp = model.preprocessing
    n_patch = (224 // 16) ** 2
    P, ps = 224 // 16, 16

    def unpatchify_1ch(patches):
        """[1, N, ps*ps] → [1,1,H,W]."""
        x = patches.reshape(1, P, P, ps, ps, 1).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(1, 1, P * ps, P * ps)

    def recon_p(x_vis_src, x_loss_tgt, m_routing):
        """P-recon masked 복원: visible(원본)+masked(예측) 합성."""
        mask = model._random_mask(1, device, args.mask_ratio)
        p_vis = model._student_p_encode_visible(pp.compute_p_channel(x_vis_src), mask)
        pred = model._predict_pixels(p_vis, mask, m_routing, x_loss_tgt)[1]
        vis = model._patchify(x_vis_src)
        m = mask.unsqueeze(-1).float()
        img = model._unpatchify(vis * (1 - m) + pred * m)
        return to_np_rgb(img[0]), pred[0][mask[0]].std().item()

    def recon_m(x_dl_a, x_dl_b, x_helper):
        """M-recon: masked ΔL(a,b) → P-grouping routing(helper) → 1ch 복구 (masked = 학습 분포).

        Case B(동적): (t, tk, t) → 실제 ΔL 복구.
        Case A(정지): (t, t, t) / (tk, tk, tk) → ΔL≡0 입력 → 출력 0 기대(calibration 검증).
        """
        m_chan = pp.compute_m_channel(x_dl_a, x_dl_b)          # [1,1,H,W] ΔL (정지면 0)
        mask = model._random_mask(1, device, args.mask_ratio_m)
        m_vis = model._encode_m_masked(m_chan, mask)
        m_state = model._inject_mask_tokens(m_vis, mask, model.mask_token_m) + model.dec_pos_embed_m
        p_helper = model._encode_p_unmasked(pp.compute_p_channel(x_helper))
        for step in model.m_recon_decoder:
            m_state = step(m_state, p_helper)
        m_state = model.m_recon_decoder_norm(m_state)
        pred = model.m_recon_head(m_state[:, 1:])              # [1, N, ps*ps]
        tgt = model._patchify(m_chan)                          # [1, N, ps*ps]
        m = mask.unsqueeze(-1).float()
        dl_img = unpatchify_1ch(tgt * (1 - m) + pred * m)      # visible(원본) + masked(예측) 합성
        # std는 masked 위치 예측값만 (Case A면 ~0이어야 calibration 성공)
        return to_np_dl(dl_img), pred[0][mask[0]].std().item()

    ds_seen = EgoDexDataset(
        data_root=args.egodex_root, split=args.seen_split, max_gap=30,
        sample_dist="triangular", sample_center=15, train=False,
        return_triple=False, max_videos=args.max_videos,
    )
    ds_unseen = EgoDexDataset(
        data_root=args.egodex_root, split=args.unseen_split, max_gap=30,
        sample_dist="triangular", sample_center=15, train=False,
        return_triple=False, max_videos=args.max_videos,
    )
    sources = [(f"SEEN (EgoDex {args.seen_split})", ds_seen),
               (f"UNSEEN (EgoDex {args.unseen_split})", ds_unseen)]
    if args.add_droid:
        # cross-domain DROID 행. 고정 ep만 로드(전수 스캔 회피, _FixedEpDataset 참고).
        fixed = [Path(args.droid_root) / e for e in args.droid_eps]
        ds_droid = _FixedEpDataset(
            fixed, data_root=args.droid_root, cameras=args.droid_cameras.split(","), max_gap=15,
            sample_dist="triangular", sample_center=8, train=False,
            return_triple=False, max_videos=args.max_videos,
        )
        sources.append(("UNSEEN (DROID x-domain)", ds_droid))

    # 각 cell = (array, kind): kind 'rgb' → imshow / 'dl' → imshow(cmap=seismic, vlim)
    rows = []
    with torch.no_grad():
        for label, ds in sources:
            for k in random.sample(range(len(ds)), args.num_per_source):
                img_t, img_tk, gap = ds[k]
                x_t, x_tk = img_t.unsqueeze(0).to(device), img_tk.unsqueeze(0).to(device)
                m_null = model._null_routing(1, device)
                m_real = model._encode_m_unmasked(pp.compute_m_channel(x_t, x_tk))
                dl_target = pp.compute_m_channel(x_t, x_tk)

                rec_t, s_t = recon_p(x_t, x_t, m_null)
                rec_tk, s_tk = recon_p(x_tk, x_tk, m_null)
                pred_tk, s_pred = recon_p(x_t, x_tk, m_real)
                dl_recB, s_mB = recon_m(x_t, x_tk, x_t)          # Case B (동적): 실제 ΔL
                dl_statA_t, s_mA_t = recon_m(x_t, x_t, x_t)      # Case A (정지, frame_t): →0?
                dl_statA_tk, s_mA_tk = recon_m(x_tk, x_tk, x_tk)  # Case A (정지, frame_tk): →0?

                cells = [
                    (to_np_rgb(img_t), "rgb"), (to_np_rgb(img_tk), "rgb"),
                    (rec_t, "rgb"), (rec_tk, "rgb"), (pred_tk, "rgb"),
                    (to_np_dl(dl_target), "dl"), (dl_recB, "dl"),
                    (dl_statA_t, "dl"), (dl_statA_tk, "dl"),
                ]
                print(f"  [{label}] gap={int(gap)} | P rec_t={s_t:.3f} pred_tk={s_pred:.3f} | "
                      f"M-recon(B)={s_mB:.3f} target={dl_target.std().item():.3f} | "
                      f"Case-A(→0 기대) static_t={s_mA_t:.4f} static_tk={s_mA_tk:.4f}")
                rows.append((label, int(gap), cells))

    titles = ["frame_t", "frame_t+k", "P:rec_t", "P:rec_t+k", "P:pred_t+k",
              "M:ΔL_target", "M:ΔL_recon(B)", "M:static(t,t)→0?", "M:static(tk,tk)→0?"]
    ncol = len(titles)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.3 * ncol, 3.6 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for r, (label, gap, cells) in enumerate(rows):
        for col, (im, kind) in enumerate(cells):
            ax = axes[r][col]
            if kind == "dl":
                ax.imshow(im, cmap="seismic", vmin=-args.dl_vlim, vmax=args.dl_vlim)
            else:
                ax.imshow(im)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[col], fontsize=10)
        axes[r][0].set_ylabel(f"{label}\ngap={gap}", fontsize=9, rotation=0, labelpad=55, va="center")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
