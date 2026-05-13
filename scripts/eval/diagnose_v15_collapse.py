"""v15 P encoder collapse 진단.

3가지 측정:
1. CLS-level diversity (TB scalar 일치 확인용)
2. Patch token diversity (사용자 가설 검증 — CLS만 collapse vs patch도 collapse)
3. Reconstruction image diversity (pairwise MSE — recon이 mean pattern으로 수렴했는지)

EgoDex (in-domain) + DROID (cross-domain) 비교.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.datasets import EgoDexDataset
from src.datasets.droid import DROIDDataset
from src.models import TwoStreamV15Model


def diagnose(model, dataset, name, n_samples, seed):
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    img_t_list, img_tn_list, img_tm_list = [], [], []
    for idx in indices:
        try:
            img_t, img_tn, img_tm, _, _ = dataset[idx]
            img_t_list.append(img_t)
            img_tn_list.append(img_tn)
            img_tm_list.append(img_tm)
        except Exception as e:
            print(f"  skip idx={idx}: {e}")
            continue
    if len(img_t_list) < 2:
        print(f"  ERROR: insufficient samples for {name}")
        return
    img_t = torch.stack(img_t_list).cuda()
    img_tn = torch.stack(img_tn_list).cuda()
    img_tm = torch.stack(img_tm_list).cuda()
    B = img_t.shape[0]
    print(f"\n=== {name} (B={B}) ===")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(img_t, img_tn, img_tm)

    cls_p = out['cls_p'].float()                  # [B, D]
    cls_m = out['cls_m'].float()                  # [B, D]
    p_features = out['p_features_t'].float()      # [B, N_vis, D]
    pred_t = out['pred_t'].float()                # [B, 3, H, W] — full image (unpatchify)

    eye = ~torch.eye(B, device=cls_p.device, dtype=torch.bool)

    # 1. CLS-level
    cls_p_std = cls_p.std(dim=0).mean().item()
    cls_m_std = cls_m.std(dim=0).mean().item()
    cls_p_n = F.normalize(cls_p, dim=-1)
    cls_m_n = F.normalize(cls_m, dim=-1)
    cls_p_cos = (cls_p_n @ cls_p_n.T)[eye].mean().item()
    cls_m_cos = (cls_m_n @ cls_m_n.T)[eye].mean().item()

    # 2. Patch token diversity (P encoder)
    Bf, Nf, Df = p_features.shape
    # per-position batch-axis std → average
    patch_std = p_features.std(dim=0).mean().item()
    # 같은 patch position에서 다른 sample 끼리 cosine, 모든 position 평균
    p_n = F.normalize(p_features, dim=-1)               # [B, N, D]
    cos_per_pos = torch.einsum('bnd,cnd->nbc', p_n, p_n)  # [N, B, B]
    patch_cos = cos_per_pos[:, eye].mean().item()

    # 3. Reconstruction image diversity
    def pairwise_mse(x):
        x = x.reshape(x.shape[0], -1)
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        mse = (diff ** 2).mean(dim=-1)
        off = mse[eye]
        return off.mean().item(), off.std().item()

    pred_mse_mean, pred_mse_std = pairwise_mse(pred_t)
    gt_mse_mean, gt_mse_std = pairwise_mse(img_t)
    ratio = pred_mse_mean / gt_mse_mean if gt_mse_mean > 1e-8 else float('nan')

    # per-sample fidelity
    fidelity_mse = ((pred_t - img_t) ** 2).mean(dim=(1, 2, 3))

    print(f"  p_features shape: [B={Bf}, N_vis={Nf}, D={Df}]")
    print(f"  CLS-level (TB scalar 일치 확인):")
    print(f"    cls_p_std       = {cls_p_std:.4f}   cls_p_cos_intra = {cls_p_cos:.4f}")
    print(f"    cls_m_std       = {cls_m_std:.4f}   cls_m_cos_intra = {cls_m_cos:.4f}")
    print(f"  Patch token diversity (P encoder, batch axis):")
    print(f"    patch_std       = {patch_std:.4f}   patch_cos_intra = {patch_cos:.4f}")
    print(f"  Reconstruction image pairwise MSE (B={B} → {B*(B-1)} 쌍):")
    print(f"    pred pairwise MSE = {pred_mse_mean:.5f} ± {pred_mse_std:.5f}")
    print(f"    GT   pairwise MSE = {gt_mse_mean:.5f} ± {gt_mse_std:.5f}")
    print(f"    ratio pred/GT     = {ratio:.4f}  ← 1.0이면 GT 수준 다양성, <<1.0이면 mean pattern 수렴")
    print(f"  Per-sample fidelity (pred_t vs GT, MSE):")
    print(f"    {fidelity_mse.mean().item():.5f} ± {fidelity_mse.std().item():.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--egodex-data-root", default="/proj/external_group/mrg/datasets/egodex/frames")
    ap.add_argument("--droid-data-root", default="/proj/external_group/mrg/datasets/droid_frames")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    epoch = ckpt.get("epoch", 0)
    print(f"  epoch={epoch}")

    model = TwoStreamV15Model(
        p_depth=12, m_depth=6,
        mask_ratio_p=0.75, rotation_aug=True,
        routing_mode="v_from_p",
        lambda_pred=1.0, lambda_m_jepa=1.0, lambda_compose=1.0,
        mask_ratio_m_jepa=0.5,
        composition_mode="linear_residual",
    )
    msg = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"  state_dict: {msg}")
    model.eval().cuda()
    # viz와 동일 조건 (사용자가 "복구 퀄리티 비슷"이라 본 그 setting)
    model.mask_ratio_p = 0.0

    ds_egodex = EgoDexDataset(
        data_root=args.egodex_data_root,
        split="test", max_gap=30, sample_dist="triangular", sample_center=15,
        train=False, return_triple=True,
    )
    print(f"EgoDex test: {len(ds_egodex)} samples")
    diagnose(model, ds_egodex, "EgoDex (in-domain)", n_samples=args.n_samples, seed=args.seed)

    ds_droid = DROIDDataset(
        data_root=args.droid_data_root, cameras=["ext1"],
        max_gap=15, sample_dist="triangular", sample_center=8,
        train=False, return_triple=True, max_videos=10,
    )
    print(f"DROID test (max_videos=10): {len(ds_droid)} samples")
    diagnose(model, ds_droid, "DROID (cross-domain)", n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
