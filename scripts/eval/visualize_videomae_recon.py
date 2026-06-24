"""VideoMAE-ours 2-frame reconstruction 가시화 — SEEN(EgoDex) vs UNSEEN(DROID).

목적: VideoMAE가 학습한 도메인(EgoDex)과 안 본 도메인(DROID)에서 복원 품질이 얼마나
다른지 = 도메인 의존성 직접 확인. 입력 2장(tubelet)을 학습과 동일하게 받음.

컬럼 (epoch_030_pair.png 구도): GT t / GT t+k / Masked-in / Masked recon / Full recon
- **Masked recon** (학습 mask 0.5): visible=GT, masked=모델 예측 합성 → 학습 task 그대로.
  안 본 도메인에서 흐려지면 도메인 의존성.
- **Full recon** (mask 0, 전체 토큰 디코드): full-context 복원 = 참조(상한).

주의: VideoMAE-ours는 normalize_target=True → 디코더는 per-patch 정규화 패치를 예측.
픽셀 가시화는 GT 패치의 per-patch mean/std로 de-normalize (구조 예측 + GT 밝기).

사용:
  sbatch --export=ALL,CKPT=/proj/external_group/mrg/checkpoints/videomae/20260415_012017/best_model.pt,\
VIDEOMAE_RECON=1 scripts/cluster/viz_v15.sbatch   # (또는 아래 python 직접)
  python scripts/eval/visualize_videomae_recon.py --ckpt <best_model.pt>
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.videomae import VideoMAEModel  # noqa: E402
from src.datasets.egodex import EgoDexDataset  # noqa: E402
from src.datasets.droid import DROIDDataset  # noqa: E402


def to_np(t: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] or [3,H,W], [0,1] → np HWC clipped."""
    if t.dim() == 4:
        t = t[0]
    return t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mask-ratio", type=float, default=0.5, help="학습값 = 0.5 (vm_mask)")
    ap.add_argument("--num-egodex-train", type=int, default=3)
    ap.add_argument("--egodex-train-split", default="part1")
    ap.add_argument("--num-droid", type=int, default=3)
    ap.add_argument("--out-dir", default="scratch/viz/videomae_recon",
                    help="기본=gitignored scratch. 논문 승격 시 명시적으로 paper_artifacts/fig4_recon_quality/ 지정")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--egodex-data-root", default="/proj/external_group/mrg/datasets/egodex/frames")
    ap.add_argument("--droid-data-root", default="/proj/external_group/mrg/datasets/droid_frames")
    ap.add_argument("--droid-cameras", default="ext1")
    ap.add_argument("--droid-max-videos", type=int, default=10)
    ap.add_argument("--egodex-max-gap", type=int, default=30)
    ap.add_argument("--egodex-sample-center", type=int, default=15)
    ap.add_argument("--droid-max-gap", type=int, default=15)
    ap.add_argument("--droid-sample-center", type=int, default=8)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoMAEModel(mask_ratio=args.mask_ratio, normalize_target=True).to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if k.startswith(("encoder.", "decoder.", "encoder_to_decoder", "mask_token"))]
    assert not bad, f"VideoMAE 가중치 미로드: {bad[:5]}"
    print(f"loaded {args.ckpt} | missing {len(missing)} unexpected {len(unexpected)}")

    C, P, tub = model.in_channels, model.patch_size, model.tubelet_size
    N = model.num_patches
    T = 2
    gh = int(round((N / (T // tub)) ** 0.5))  # 14
    assert gh * gh * (T // tub) == N, f"patch grid mismatch N={N} gh={gh}"
    pp = N  # patches per frame-grid (single temporal token)

    def patch_stats(patches):  # [1,N,D] → mean,std [1,N,1,C]
        B, Np, D = patches.shape
        x = patches.reshape(B, Np, D // C, C)
        mean = x.mean(dim=-2, keepdim=True)
        std = x.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
        return mean, std

    def denorm(pred_norm, mean, std):  # [1,n,D] norm → pixel [1,n,D]
        B, n, D = pred_norm.shape
        x = pred_norm.reshape(B, n, D // C, C) * std + mean
        return x.reshape(B, n, D)

    def unpatchify(patches):  # [1,N,tub*P*P*C] → [1,C,T,H,W] (역 _patchify)
        x = patches.reshape(1, T // tub, gh, gh, tub, P, P, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()  # B C (T//p0) p0 (H//p1) p1 (W//p1) p2
        return x.reshape(1, C, T, gh * P, gh * P)

    def frames(video5d):  # [1,C,2,H,W] → (np_t, np_tk)
        return to_np(video5d[:, :, 0]), to_np(video5d[:, :, 1])

    ds_egodex = EgoDexDataset(
        data_root=args.egodex_data_root, split=args.egodex_train_split,
        max_gap=args.egodex_max_gap, sample_dist="triangular",
        sample_center=args.egodex_sample_center, train=False, return_triple=False,
    )
    print(f"EgoDex (train={args.egodex_train_split}, SEEN): {len(ds_egodex)} samples")
    ds_droid = DROIDDataset(
        data_root=args.droid_data_root, cameras=args.droid_cameras.split(","),
        max_gap=args.droid_max_gap, sample_dist="triangular",
        sample_center=args.droid_sample_center, train=False, return_triple=False,
        max_videos=args.droid_max_videos,
    )
    print(f"DROID (cross-domain, UNSEEN): {len(ds_droid)} samples")

    sources = []
    if args.num_egodex_train > 0:
        sources.append(("EgoDex-train(SEEN)", ds_egodex,
                        random.sample(range(len(ds_egodex)), args.num_egodex_train)))
    if args.num_droid > 0:
        sources.append(("DROID(UNSEEN)", ds_droid,
                        random.sample(range(len(ds_droid)), args.num_droid)))

    rows = []
    with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
        for ds_name, ds, idxs in sources:
            for idx in idxs:
                try:
                    img_t, img_tk, gap = ds[idx]
                except Exception as e:
                    print(f"  skip [{ds_name}] {idx}: {e}")
                    continue
                x_t = img_t.unsqueeze(0).to(device).float()
                x_tk = img_tk.unsqueeze(0).to(device).float()
                if x_t.shape[-1] != gh * P:  # 224 보정
                    x_t = F.interpolate(x_t, size=(gh * P, gh * P), mode="bilinear", align_corners=False)
                    x_tk = F.interpolate(x_tk, size=(gh * P, gh * P), mode="bilinear", align_corners=False)
                video = torch.stack([x_t, x_tk], dim=2)  # [1,3,2,H,W]

                gt_patches = model._patchify(video)        # [1,N,D]
                gt_mean, gt_std = patch_stats(gt_patches)

                # --- Masked recon (학습 mask 0.5) ---
                model.mask_ratio = args.mask_ratio
                mask = model.random_masking(1, N, video.device)  # [1,N] True=masked
                pred_norm, _ = model._forward_masked(video, mask)  # [1,nm,D] normalized
                nm = int(mask.sum().item())
                D = gt_patches.shape[-1]
                mean_m = gt_mean[mask].reshape(1, nm, 1, C)   # GT 패치 통계로 de-normalize
                std_m = gt_std[mask].reshape(1, nm, 1, C)
                pred_pix = (pred_norm.float().reshape(1, nm, D // C, C) * std_m + mean_m).reshape(1, nm, D)
                full_masked = gt_patches.clone().float()
                full_masked[mask] = pred_pix.reshape(nm, D)
                rec_m_t, rec_m_tk = frames(unpatchify(full_masked))

                # masked input viz (visible=GT, masked=회색)
                masked_in = gt_patches.clone().float()
                masked_in[mask] = 0.5
                min_t, min_tk = frames(unpatchify(masked_in))

                # --- Full recon (mask 0, 전체 토큰 디코드) ---
                mask_none = torch.zeros(1, N, dtype=torch.bool, device=video.device)
                vis = model.encoder(video, mask_none)           # [1,N,enc_dim] 전체 visible
                vis = model.encoder_to_decoder(vis)
                pos = model.decoder_pos_embed.expand(1, -1, -1).type_as(video).to(video.device)
                pred_full_norm = model.decoder(vis + pos, N)     # [1,N,D] 전체 예측
                pred_full_pix = denorm(pred_full_norm.float(), gt_mean, gt_std)
                rec_f_t, rec_f_tk = frames(unpatchify(pred_full_pix))

                rows.append({
                    "dataset": ds_name, "gap": int(gap),
                    "gt_t": to_np(x_t), "gt_tk": to_np(x_tk),
                    "min_t": min_t, "min_tk": min_tk,
                    "rec_m_t": rec_m_t, "rec_m_tk": rec_m_tk,
                    "rec_f_t": rec_f_t, "rec_f_tk": rec_f_tk,
                })

    if not rows:
        print("ERROR: no samples")
        return

    keys = ["gt_t", "gt_tk", "min_t", "min_tk", "rec_m_t", "rec_m_tk", "rec_f_t", "rec_f_tk"]
    titles = ["GT t", "GT t+k",
              f"Masked-in t\n({int(args.mask_ratio*100)}%)", "Masked-in t+k",
              "Recon t\n(masked)", "Recon t+k\n(masked)",
              "Recon t\n(full)", "Recon t+k\n(full)"]
    N_rows, ncols = len(rows), len(keys)
    fig, axes = plt.subplots(N_rows, ncols, figsize=(2.7 * ncols, 3.0 * N_rows))
    if N_rows == 1:
        axes = [axes]
    for r, row in enumerate(rows):
        for c, k in enumerate(keys):
            axes[r][c].imshow(row[k])
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(titles[c], fontsize=9)
        axes[r][0].text(-0.10, 0.5, f"{row['dataset']}\ngap={row['gap']}",
                        transform=axes[r][0].transAxes, ha="right", va="center",
                        fontsize=9, fontweight="bold")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = ck.get("epoch", "NA")
    out_path = out_dir / f"videomae_recon_seen_vs_unseen_ep{ep}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
