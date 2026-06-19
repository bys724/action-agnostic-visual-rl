#!/usr/bin/env python
"""V-JEPA P trivial 진단 (2026-05-28).

질문: student-anchor 수정 후 cos(pred,tgt)=1.0, L_pred≈0 → P encoder가 motion 압력 받나?
M routing은 v_from_p (Q,K=M, V=P)라 ΔL 직접 누출 아님. trivial 원인 구분:

측정 (EgoDex part1, 40 frame pair, anchor=t, target=t+n):
  1. baseline cos(teacher_P(t), teacher_P(t+n))  — frame 자체 유사도
  2. predictor cos(pred, target)                 — 정상 routing
  3. M=0 ablation cos(pred_M0, target)           — M routing 제거

해석:
  baseline 높음(0.99) → frame 유사 = task 쉬움 (A)
  M=0 ≈ normal        → M 무의미, predictor가 anchor만으로 trivial
  M=0 << normal       → M이 실제 motion 정보 제공 (성공)
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.two_stream_v15 import TwoStreamV15Model
from src.datasets import EgoDexDataset

SANITY_CKPT = "/proj/external_group/mrg/checkpoints/two_stream_v15_sanity_student_anchor/20260528_165202/latest.pt"
EP50_CKPT = "/proj/external_group/mrg/checkpoints/two_stream_v15/20260511_045319/latest.pt"
FRAMES_ROOT = "/proj/external_group/mrg/datasets/egodex/frames"
DEVICE = "cuda"
N_SAMPLES = 100  # gap bucket 통계 위해 늘림


def cos_tok(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=SANITY_CKPT)
    ap.add_argument("--tag", default="sanity_3ep")
    ap.add_argument("--no-sobel", action="store_true",
                    help="Parvo/Paper2 no-Sobel 모델 (P=RGB 3ch, M=ΔL 1ch)")
    ap.add_argument("--masked-anchor", action="store_true",
                    help="Run B masked anchor (decode_first 라우팅 순서 일치)")
    args = ap.parse_args()
    CKPT = args.ckpt
    print(f"[ckpt] {args.tag}: {CKPT}  (no_sobel={args.no_sobel})")

    model = TwoStreamV15Model(routing_mode="v_from_p", use_sobel=not args.no_sobel,
                              pair_mode=True, masked_anchor=args.masked_anchor)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(DEVICE).eval()

    ds = EgoDexDataset(data_root=FRAMES_ROOT, split="part1", max_gap=30,
                       max_videos=10, sample_center=15, return_triple=True)
    print(f"[data] EgoDex part1: {len(ds)} samples")

    cb, cn, ca, cp = [], [], [], []  # cp = cos(pred, 현재 target=teacher_t)
    cls_list, patch_list = [], []  # 교차샘플 collapse 진단 (CLS vs patch)
    by_gap = defaultdict(list)  # gap bucket → baseline cos
    with torch.no_grad():
        for i in range(min(N_SAMPLES, len(ds))):
            img_t, img_tn, img_tm, gn, gm = ds[i]
            gap_n = int(gn)
            img_t = img_t.unsqueeze(0).to(DEVICE)
            img_tn = img_tn.unsqueeze(0).to(DEVICE)

            p_ch_t = model.preprocessing.compute_p_channel(img_t)
            p_ch_tn = model.preprocessing.compute_p_channel(img_tn)
            m_ch = model.preprocessing.compute_m_channel(img_t, img_tn)

            tgt_t = model.teacher_p.forward_unmasked(p_ch_t)
            tgt_tn = model.teacher_p.forward_unmasked(p_ch_tn)
            cls_list.append(tgt_t[0, 0].float().cpu())          # CLS token
            patch_list.append(tgt_t[0, 1:].mean(0).float().cpu())  # patch-mean
            bcos = cos_tok(tgt_t, tgt_tn)
            cb.append(bcos)
            # gap bucket (5단위)
            by_gap[(gap_n // 5) * 5].append(bcos)

            anchor_s = model._encode_p_unmasked(p_ch_t)
            m_local = model._encode_m_unmasked(m_ch)

            p_state = anchor_s
            for step in model.p_motion_decoder:
                p_state = step(p_state, m_local)
            pred = model.p_motion_decoder_norm(p_state)
            cn.append(cos_tok(pred, tgt_tn))
            cp.append(cos_tok(pred, tgt_t))   # 예측이 *현재*(t)에 가까운가 = identity 의심

            p_state = anchor_s
            mz = torch.zeros_like(m_local)
            for step in model.p_motion_decoder:
                p_state = step(p_state, mz)
            pred_a = model.p_motion_decoder_norm(p_state)
            ca.append(cos_tok(pred_a, tgt_tn))

    print(f"\n=== V-JEPA P trivial 진단 [{args.tag}] (EgoDex part1, anchor=t, target=t+n, n={len(cb)}) ===")
    print(f"1. baseline  cos(teacher_t, teacher_tn)  = {np.mean(cb):+.4f}  (std {np.std(cb):.3f})")
    print(f"2. predictor cos(pred, target)           = {np.mean(cn):+.4f}  (std {np.std(cn):.3f})")
    print(f"3. M=0 ablt  cos(pred_M0, target)        = {np.mean(ca):+.4f}  (std {np.std(ca):.3f})")
    print(f"Δ(predictor − baseline) = {np.mean(cn) - np.mean(cb):+.4f}")
    print(f"Δ(predictor − M=0)      = {np.mean(cn) - np.mean(ca):+.4f}  (M routing 기여)")
    print(f"\n=== trivial vs 진짜 예측 판정 (핵심) ===")
    print(f"  baseline cos(현재 t, 미래 t+n)       = {np.mean(cb):+.4f}  ← 1에 가까우면 frame이 안 변함(애초에 trivial 쉬움)")
    print(f"  cos(pred, 미래 t+n)                  = {np.mean(cn):+.4f}")
    print(f"  cos(pred, 현재 t)                    = {np.mean(cp):+.4f}")
    print(f"  Δ = cos(pred,미래) − cos(pred,현재)  = {np.mean(cn) - np.mean(cp):+.4f}")
    print(f"  → Δ>0 & baseline 낮으면: 예측이 *미래*에 더 가까움 = 진짜 예측(non-trivial)")
    print(f"  → cos(pred,현재)≈cos(pred,미래) & baseline 높으면: identity = trivial")

    # CLS vs patch collapse 진단 (교차샘플): collapse면 cross-sample std→0, cos→1.
    # recon은 patch로 됨 → patch가 healthy면 'CLS만 collapse'라 recon 모순 없음.
    cls_stack = torch.stack(cls_list)      # [N, D]
    patch_stack = torch.stack(patch_list)  # [N, D]
    cls_xstd = cls_stack.std(0).mean().item()
    patch_xstd = patch_stack.std(0).mean().item()
    def _xcos(x):
        xn = F.normalize(x, dim=-1)
        g = xn @ xn.T
        n = g.shape[0]
        return (g.sum() - n) / (n * (n - 1))  # off-diagonal mean
    print(f"\n--- 교차샘플 collapse 진단 (n={len(cls_list)}) ---")
    print(f"  CLS   : cross-sample std={cls_xstd:.4f}  cos={_xcos(cls_stack):+.4f}  (→0/→1이면 CLS collapse)")
    print(f"  patch : cross-sample std={patch_xstd:.4f}  cos={_xcos(patch_stack):+.4f}  (healthy면 std>0, cos<1 → recon 가능)")

    print(f"\n--- gap별 baseline cos (gap 클수록 frame 차이 커야 = cos 낮아야) ---")
    for g in sorted(by_gap):
        v = by_gap[g]
        print(f"  gap [{g:2d}-{g+4:2d}]: cos={np.mean(v):+.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
