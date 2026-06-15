"""L_t = MSE + 0.1·SSIM 의 term별 스케일 분해 진단.

가설: SSIM term이 raw MSE를 압도하면 복구가 SSIM에만 끌려 픽셀 디테일 소실.
lambda_ssim 토글(0 vs 0.1)로 같은 mask에서 MSE term과 0.1·SSIM term을 분리.
ep3(L_t 최저, 복구됨) vs ep11(퇴화) 비교.
"""
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from src.datasets import EgoDexDataset
from src.models import TwoStreamV15Model

CKPTS = {
    "ep3 (L_t 최저)": "/proj/external_group/mrg/checkpoints/parvo_pair_ssim/20260613_131606/checkpoint_epoch0003.pt",
    "ep11 (퇴화)": "/proj/external_group/mrg/checkpoints/parvo_pair_ssim/20260613_131606/latest.pt",
}

ds = EgoDexDataset(
    data_root="/proj/external_group/mrg/datasets/egodex/frames",
    split="part1", max_gap=30, sample_dist="triangular", sample_center=15,
    train=False, return_triple=False,
)
random.seed(0)
idxs = random.sample(range(len(ds)), 8)

for name, cp in CKPTS.items():
    ck = torch.load(cp, map_location="cuda", weights_only=False)
    m = TwoStreamV15Model(
        p_depth=12, m_depth=6, mask_ratio_p=0.75, rotation_aug=True,
        routing_mode="v_from_p", use_sobel=False, mask_ratio_m_jepa=0.5, pair_mode=True,
    )
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}
    m.load_state_dict(sd, strict=True)
    m.eval().cuda()

    mses, ssim_terms = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for idx in idxs:
            x_t, _, _ = ds[idx]
            x = x_t.unsqueeze(0).cuda()
            p = m.preprocessing.compute_p_channel(x)
            # 같은 mask 보장: 매 호출 전 seed 고정 (_random_mask가 torch.rand 사용)
            torch.manual_seed(1000 + idx)
            m.lambda_ssim = 0.0
            l_mse, _, _, _ = m._mae_one_frame(x, p)
            torch.manual_seed(1000 + idx)
            m.lambda_ssim = 0.1
            l_full, _, _, _ = m._mae_one_frame(x, p)
            mses.append(l_mse.item())
            ssim_terms.append((l_full - l_mse).item())

    mse = float(np.mean(mses))
    ss = float(np.mean(ssim_terms))
    print(f"{name:20s}  MSE={mse:.5f}  0.1*SSIM={ss:.5f}  ratio(SSIM/MSE)={ss/max(mse,1e-9):.1f}  L_t={mse+ss:.5f}")
