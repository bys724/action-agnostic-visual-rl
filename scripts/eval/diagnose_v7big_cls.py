#!/usr/bin/env python3
"""Two-Stream v7-big CLS 특화 진단.

ep4 attention viz에서 CLS_P_bg / CLS_P_motion의 spatial attention이 유사하게 나온
원인을 검증하기 위한 두 가지 테스트:

1) **Cosine similarity**: encoder 출력 CLS embedding 자체가 얼마나 다른지
   - cos(CLS_P_bg_i, CLS_P_motion_i)  — 같은 샘플 내 두 CLS의 차이
   - cos(CLS_P_bg_i, CLS_P_bg_j)      — 샘플 간 CLS_P_bg가 샘플 정보를 담는지 (sanity)
   - cos(CLS_P_motion_i, CLS_P_motion_j)
   - cos(CLS_M_i, CLS_P_bg_i), cos(CLS_M_i, CLS_P_motion_i)

2) **Decoder swap ablation**: decoder 입력 CLS를 서로 바꿨을 때 예측이 변하는지
   - normal:  BG_dec(m_patches, CLS_P_bg), Motion_dec(m_patches, CLS_P_motion)
   - swapped: BG_dec(m_patches, CLS_P_motion), Motion_dec(m_patches, CLS_P_bg)
   - MSE(pred_normal vs pred_swapped), MSE(pred_swapped vs target) 비교
   - swap MSE 변화가 미미 → CLS 차이 무관, decoder weight가 일을 함

Usage:
    python scripts/eval/diagnose_v7big_cls.py \\
        --checkpoint /path/to/v7big_isolated/checkpoint_epoch0004.pt \\
        --egodex-dir /proj/external_group/mrg/datasets/egodex/frames/part4 \\
        --num-samples 64
"""

import argparse
import os
import sys
import random

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TwoStreamModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 2  # v7-big: CLS_P 개수


def load_image(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float() / 255.0


def sample_pairs(frames_dir, num_samples, gap, seed):
    rng = random.Random(seed)
    episode_dirs = []
    for root, _, files in os.walk(frames_dir):
        if any(f.endswith('.jpg') for f in files):
            episode_dirs.append(root)
    episode_dirs.sort()
    rng.shuffle(episode_dirs)

    pairs = []
    for ep_dir in episode_dirs:
        frames = sorted([f for f in os.listdir(ep_dir) if f.endswith('.jpg')])
        if len(frames) < gap + 5:
            continue
        idx = rng.randint(0, len(frames) - gap - 1)
        path_t = os.path.join(ep_dir, frames[idx])
        path_tk = os.path.join(ep_dir, frames[idx + gap])
        pairs.append((path_t, path_tk))
        if len(pairs) >= num_samples:
            break
    return pairs


def forward_encoder(model, x, y):
    """Encoder forward (no masking) → m_patches, p_cls_bg, p_cls_motion, m_cls."""
    from src.models.two_stream import apply_rope

    with torch.no_grad():
        m_channel, p_channel = model.preprocessing(x, y)
    encoder = model.encoder
    use_ape = encoder.use_ape
    freqs_cis = None if use_ape else encoder.freqs_cis

    with torch.no_grad():
        B = m_channel.shape[0]
        m_patches = encoder.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        p_patches = encoder.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        m_cls = encoder.cls_token_m.expand(B, -1, -1)
        p_cls = encoder.cls_token_p.expand(B, -1, -1)
        m_tokens = torch.cat([m_cls, m_patches], dim=1)
        p_tokens = torch.cat([p_cls, p_patches], dim=1)
        if use_ape:
            m_tokens = m_tokens + encoder.pos_embed_m
            p_tokens = p_tokens + encoder.pos_embed_p

        for si in range(encoder.num_stages):
            for bm in encoder.blocks_m[si]:
                m_tokens = bm(m_tokens, freqs_cis=freqs_cis)
            for bp in encoder.blocks_p[si]:
                p_tokens = bp(p_tokens, freqs_cis=freqs_cis)
            m_cls_tok = m_tokens[:, 0:1]
            p_cls_tok = p_tokens[:, 0:K]
            cls_ex = encoder.cls_exchange[si](torch.cat([m_cls_tok, p_cls_tok], dim=1))
            m_tokens = torch.cat([cls_ex[:, 0:1], m_tokens[:, 1:]], dim=1)
            p_tokens = torch.cat([cls_ex[:, 1:1+K], p_tokens[:, K:]], dim=1)

        m_tokens = encoder.norm_m(m_tokens)
        p_tokens = encoder.norm_p(p_tokens)
        m_cls_out = m_tokens[:, 0:1]                  # [B,1,D]
        p_cls_bg = p_tokens[:, 0:1]                    # [B,1,D]
        p_cls_motion = p_tokens[:, 1:2]                # [B,1,D]
        m_patches_out = m_tokens[:, 1:]                # [B,N,D]
        return m_patches_out, p_cls_bg, p_cls_motion, m_cls_out, m_channel, p_channel


def batched_cos(a, b):
    """Cosine similarity per-sample: a,b shape [B,D]."""
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return F.cosine_similarity(a, b, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egodex-dir", required=True)
    parser.add_argument("--droid-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.03)
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    model = TwoStreamModel(
        depth=args.depth, num_stages=args.num_stages,
        mask_ratio=0.1, mask_ratio_p=0.5,
        use_ape=True, rotation_aug=False,
        v7_big_mode=True, sigma=args.sigma,
    ).to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    print(f"  Epoch {ck.get('epoch', '?')}")

    # Collect pairs
    all_pairs = []
    src_labels = []
    if args.egodex_dir:
        egodex_pairs = sample_pairs(args.egodex_dir, args.num_samples, args.gap, args.seed)
        all_pairs += egodex_pairs
        src_labels += ['EgoDex'] * len(egodex_pairs)
    if args.droid_dir and os.path.exists(args.droid_dir):
        droid_pairs = sample_pairs(args.droid_dir, args.num_samples // 2, args.gap, args.seed + 1)
        all_pairs += droid_pairs
        src_labels += ['DROID'] * len(droid_pairs)
    print(f"\nSamples: {len(all_pairs)} pairs (EgoDex={src_labels.count('EgoDex')}, "
          f"DROID={src_labels.count('DROID')})")

    crop = transforms.CenterCrop(224)

    # Batched forward
    cls_bg_list, cls_motion_list, cls_m_list = [], [], []
    m_patches_list = []
    frame_tk_list = []
    batch_size = 8

    for i in range(0, len(all_pairs), batch_size):
        batch = all_pairs[i:i+batch_size]
        xs, ys = [], []
        for (pt, ptk) in batch:
            img_t = load_image(pt)
            img_tk = load_image(ptk)
            xt = crop(img_t.permute(2, 0, 1))
            xtk = crop(img_tk.permute(2, 0, 1))
            xs.append(xt)
            ys.append(xtk)
            frame_tk_list.append(xtk.permute(1, 2, 0).numpy())
        x = torch.stack(xs).to(DEVICE)
        y = torch.stack(ys).to(DEVICE)
        m_patches, p_cls_bg, p_cls_motion, m_cls_out, m_channel, p_channel = forward_encoder(model, x, y)
        cls_bg_list.append(p_cls_bg.squeeze(1).cpu())
        cls_motion_list.append(p_cls_motion.squeeze(1).cpu())
        cls_m_list.append(m_cls_out.squeeze(1).cpu())
        m_patches_list.append(m_patches.cpu())

    cls_bg = torch.cat(cls_bg_list, dim=0)        # [N,D]
    cls_motion = torch.cat(cls_motion_list, dim=0)
    cls_m = torch.cat(cls_m_list, dim=0)
    m_patches_all = torch.cat(m_patches_list, dim=0)  # [N,P,D]
    N = cls_bg.shape[0]

    # ============== 1) Cosine similarity ==============
    print("\n" + "=" * 60)
    print("1) CLS cosine similarity")
    print("=" * 60)

    # Per-sample pairs
    cos_bg_motion = batched_cos(cls_bg, cls_motion)           # i-th pair
    cos_m_bg = batched_cos(cls_m, cls_bg)
    cos_m_motion = batched_cos(cls_m, cls_motion)

    # Cross-sample: shuffle to get random pairs
    perm = torch.randperm(N)
    cos_bg_bg_cross = batched_cos(cls_bg, cls_bg[perm])       # random sample pairs
    cos_motion_motion_cross = batched_cos(cls_motion, cls_motion[perm])

    def fmt(name, t):
        return f"  {name:40s}  mean={t.mean():+.4f}  std={t.std():.4f}  " \
               f"[{t.min():+.4f}, {t.max():+.4f}]"

    print(fmt("cos(CLS_P_bg_i, CLS_P_motion_i)  [same-sample]", cos_bg_motion))
    print(fmt("cos(CLS_P_bg_i, CLS_P_bg_j)      [cross-sample]", cos_bg_bg_cross))
    print(fmt("cos(CLS_P_mot_i, CLS_P_mot_j)    [cross-sample]", cos_motion_motion_cross))
    print(fmt("cos(CLS_M_i, CLS_P_bg_i)", cos_m_bg))
    print(fmt("cos(CLS_M_i, CLS_P_motion_i)", cos_m_motion))

    # 진단 해석
    mean_same = cos_bg_motion.mean().item()
    mean_cross_bg = cos_bg_bg_cross.mean().item()
    print("\n  해석:")
    print(f"   - 동일 샘플 내 bg↔motion cos ≈ {mean_same:+.3f}")
    if mean_same > 0.95:
        print(f"     → 두 CLS embedding 거의 동일. 특화 실패 (CLS 레벨).")
    elif mean_same > 0.7:
        print(f"     → 유사하지만 분리 흔적 있음.")
    else:
        print(f"     → CLS는 분리됨. attention pooling만 비슷했을 수 있음.")
    print(f"   - 샘플간 bg↔bg cos ≈ {mean_cross_bg:+.3f}")
    if mean_cross_bg > 0.95:
        print(f"     → CLS_P_bg가 샘플에 무관하게 거의 일정 벡터. (입력 정보 인코딩 약함)")
    elif mean_cross_bg > mean_same - 0.02:
        print(f"     → 샘플간 변동 ≈ 샘플 내 bg/motion 변동. 특화 신호 없음.")

    # ============== 2) Swap ablation ==============
    print("\n" + "=" * 60)
    print("2) Decoder swap ablation")
    print("=" * 60)

    # Batched decode
    pred_bg_norm_list, pred_bg_swap_list = [], []
    pred_motion_norm_list, pred_motion_swap_list = [], []
    tgt_list = []

    for i in range(0, N, batch_size):
        sl = slice(i, i + batch_size)
        m_p = m_patches_all[sl].to(DEVICE)
        bg = cls_bg[sl].unsqueeze(1).to(DEVICE)
        mo = cls_motion[sl].unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            p_bg_n = model.decoder_m_bg(m_p, bg)
            p_mo_n = model.decoder_m_motion(m_p, mo)
            p_bg_s = model.decoder_m_bg(m_p, mo)     # swap: motion CLS into BG decoder
            p_mo_s = model.decoder_m_motion(m_p, bg) # swap: bg CLS into Motion decoder
        pred_bg_norm_list.append(p_bg_n.cpu())
        pred_bg_swap_list.append(p_bg_s.cpu())
        pred_motion_norm_list.append(p_mo_n.cpu())
        pred_motion_swap_list.append(p_mo_s.cpu())

    pred_bg_norm = torch.cat(pred_bg_norm_list)       # [N,3,H,W]
    pred_bg_swap = torch.cat(pred_bg_swap_list)
    pred_motion_norm = torch.cat(pred_motion_norm_list)
    pred_motion_swap = torch.cat(pred_motion_swap_list)
    tgt = torch.from_numpy(np.stack(frame_tk_list)).permute(0, 3, 1, 2).float()  # [N,3,H,W]

    def per_sample_mse(a, b):
        return ((a - b) ** 2).mean(dim=(1, 2, 3))

    mse_bg_target_norm = per_sample_mse(pred_bg_norm, tgt)
    mse_bg_target_swap = per_sample_mse(pred_bg_swap, tgt)
    mse_motion_target_norm = per_sample_mse(pred_motion_norm, tgt)
    mse_motion_target_swap = per_sample_mse(pred_motion_swap, tgt)

    mse_bg_norm_vs_swap = per_sample_mse(pred_bg_norm, pred_bg_swap)
    mse_motion_norm_vs_swap = per_sample_mse(pred_motion_norm, pred_motion_swap)

    def fmt2(name, t):
        return f"  {name:45s}  mean={t.mean().item():.6f}  std={t.std().item():.6f}"

    print("\n  vs target (ground truth frame t+k)")
    print(fmt2("MSE(Pred_BG_normal,     target)", mse_bg_target_norm))
    print(fmt2("MSE(Pred_BG_swapped,    target)", mse_bg_target_swap))
    print(fmt2("MSE(Pred_Motion_normal, target)", mse_motion_target_norm))
    print(fmt2("MSE(Pred_Motion_swapped,target)", mse_motion_target_swap))

    print("\n  prediction divergence (normal vs swapped)")
    print(fmt2("MSE(Pred_BG_normal,     Pred_BG_swapped)", mse_bg_norm_vs_swap))
    print(fmt2("MSE(Pred_Motion_normal, Pred_Motion_swapped)", mse_motion_norm_vs_swap))

    # 해석
    bg_dmse = mse_bg_target_swap.mean() - mse_bg_target_norm.mean()
    motion_dmse = mse_motion_target_swap.mean() - mse_motion_target_norm.mean()
    print("\n  해석:")
    print(f"   - BG decoder: ΔMSE(swap−normal) = {bg_dmse.item():+.6f}")
    print(f"   - Motion decoder: ΔMSE(swap−normal) = {motion_dmse.item():+.6f}")
    print(f"   - BG pred shift (normal↔swapped MSE): {mse_bg_norm_vs_swap.mean().item():.6f}")
    print(f"   - Motion pred shift (normal↔swapped MSE): {mse_motion_norm_vs_swap.mean().item():.6f}")
    if mse_bg_norm_vs_swap.mean() < 1e-4 and mse_motion_norm_vs_swap.mean() < 1e-4:
        print(f"     → 예측이 swap에 무반응. Decoder weight가 전부 담당, CLS 분화 무의미.")
    elif bg_dmse.abs() < 1e-4 and motion_dmse.abs() < 1e-4:
        print(f"     → 예측은 살짝 변하지만 target 기준 품질 동일. CLS 차이가 decoder 출력을 바꾸지만 "
              f"특화된 정보는 아님 (noise에 가까움).")
    else:
        print(f"     → Swap 시 품질 저하. CLS가 실제로 다른 정보를 담고 있음.")

    # 간단 요약 파일
    summary_path = os.path.join(
        os.path.dirname(args.checkpoint), f"cls_diagnose_ep{ck.get('epoch', '?')}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"v7-big CLS diagnosis — epoch {ck.get('epoch', '?')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Samples: {N}\n\n")
        f.write("[Cosine similarity]\n")
        f.write(f"  cos(bg_i, motion_i) same-sample : mean={cos_bg_motion.mean():+.4f} "
                f"std={cos_bg_motion.std():.4f}\n")
        f.write(f"  cos(bg_i, bg_j) cross-sample    : mean={cos_bg_bg_cross.mean():+.4f} "
                f"std={cos_bg_bg_cross.std():.4f}\n")
        f.write(f"  cos(motion_i, motion_j) cross   : mean={cos_motion_motion_cross.mean():+.4f} "
                f"std={cos_motion_motion_cross.std():.4f}\n")
        f.write(f"  cos(M_i, bg_i)                  : mean={cos_m_bg.mean():+.4f}\n")
        f.write(f"  cos(M_i, motion_i)              : mean={cos_m_motion.mean():+.4f}\n")
        f.write("\n[Swap ablation, MSE vs target]\n")
        f.write(f"  BG normal      : {mse_bg_target_norm.mean():.6f}\n")
        f.write(f"  BG swapped     : {mse_bg_target_swap.mean():.6f}  "
                f"(Δ={bg_dmse.item():+.6f})\n")
        f.write(f"  Motion normal  : {mse_motion_target_norm.mean():.6f}\n")
        f.write(f"  Motion swapped : {mse_motion_target_swap.mean():.6f}  "
                f"(Δ={motion_dmse.item():+.6f})\n")
        f.write("\n[Prediction divergence]\n")
        f.write(f"  BG normal vs swapped     : {mse_bg_norm_vs_swap.mean():.6f}\n")
        f.write(f"  Motion normal vs swapped : {mse_motion_norm_vs_swap.mean():.6f}\n")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
