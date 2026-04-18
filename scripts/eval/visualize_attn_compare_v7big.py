#!/usr/bin/env python3
"""Two-Stream v7-big attention + prediction 비교 시각화.

v7-big는 CLS_P가 2개(bg, motion), M decoder도 2개:
- CLS_M → M patches attention (M stream)
- CLS_P_bg → P patches attention (specialization: 정적 영역 기대)
- CLS_P_motion → P patches attention (specialization: motion 영역 기대)
- Pred BG, Pred Motion (M decoder 2개 출력)

CLS_P_bg와 CLS_P_motion의 attention 맵이 공간적으로 구별되면 specialization 성공 신호.

위치: 각 stream의 last stage × last block self-attention
     (마지막 CLS exchange 직전, 즉 CLS가 자기 stream patches를 보는 "최종 시선")

Usage:
    python scripts/eval/visualize_attn_compare_v7big.py \\
        --checkpoint /path/to/v7big/checkpoint_epoch0004.pt \\
        --egodex-dir /proj/external_group/mrg/datasets/egodex/frames/part4 \\
        --droid-dir /proj/external_group/mrg/datasets/droid_frames/ext1 \\
        --output results/attn_compare_v7big.png
"""

import argparse
import os
import sys
import random

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TwoStreamModel
from src.models.two_stream import apply_rope

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 2  # v7-big: CLS_P 개수


def load_image(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float() / 255.0


def find_high_change_pairs(frames_dir, num_candidates=30, num_select=2, gap=30, seed=42):
    rng = random.Random(seed)
    episode_dirs = []
    for root, _, files in os.walk(frames_dir):
        if any(f.endswith('.jpg') for f in files):
            episode_dirs.append(root)
    episode_dirs.sort()
    rng.shuffle(episode_dirs)

    candidates = []
    for ep_dir in episode_dirs[:200]:
        frames = sorted([f for f in os.listdir(ep_dir) if f.endswith('.jpg')])
        if len(frames) < gap + 5:
            continue
        idx = rng.randint(0, len(frames) - gap - 1)
        path_t = os.path.join(ep_dir, frames[idx])
        path_tk = os.path.join(ep_dir, frames[idx + gap])
        img_t = load_image(path_t)
        img_tk = load_image(path_tk)
        mse = ((img_t - img_tk) ** 2).mean().item()
        ep_label = os.path.relpath(ep_dir, frames_dir)
        candidates.append((path_t, path_tk, ep_label, gap, mse))
        if len(candidates) >= num_candidates:
            break

    candidates.sort(key=lambda x: x[4], reverse=True)
    return candidates[:num_select]


def manual_attention(block, tokens, freqs_cis):
    """TransformerBlock forward를 수동으로 실행하며 attention weights 반환."""
    h = block.norm1(tokens)
    B, N, D = h.shape
    qkv = block.qkv(h).reshape(B, N, 3, block.num_heads, block.head_dim)
    q, k, v = qkv.unbind(dim=2)
    if freqs_cis is not None:
        q = apply_rope(q, freqs_cis, has_cls=True)
        k = apply_rope(k, freqs_cis, has_cls=True)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scale = block.head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    attn_avg = attn.mean(dim=1)  # [B, N, N] heads 평균
    out = (attn @ v).transpose(1, 2).reshape(B, N, D)
    tokens = tokens + block.proj(out)
    tokens = tokens + block.mlp(block.norm2(tokens))
    return tokens, attn_avg.detach().cpu()


def extract_v7big_attention_and_predict(model, img_t_tensor, img_tk_tensor):
    """v7-big: 3개 CLS attention + 2개 decoder 출력 추출."""
    x = img_t_tensor.unsqueeze(0).to(DEVICE)
    y = img_tk_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        m_channel, p_channel = model.preprocessing(x, y)

    encoder = model.encoder
    last_stage = encoder.num_stages - 1
    attn_maps = {}

    use_ape = encoder.use_ape
    freqs_cis = None if use_ape else encoder.freqs_cis

    with torch.no_grad():
        B = m_channel.shape[0]
        m_patches = encoder.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        p_patches = encoder.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        m_cls = encoder.cls_token_m.expand(B, -1, -1)
        p_cls = encoder.cls_token_p.expand(B, -1, -1)  # [B, K=2, D]
        m_tokens = torch.cat([m_cls, m_patches], dim=1)
        p_tokens = torch.cat([p_cls, p_patches], dim=1)  # [B, K+N, D]

        if use_ape:
            m_tokens = m_tokens + encoder.pos_embed_m
            p_tokens = p_tokens + encoder.pos_embed_p

        for si in range(encoder.num_stages):
            for bi, bm in enumerate(encoder.blocks_m[si]):
                if si == last_stage and bi == len(encoder.blocks_m[si]) - 1:
                    m_tokens, attn_maps['m'] = manual_attention(bm, m_tokens, freqs_cis)
                else:
                    m_tokens = bm(m_tokens, freqs_cis=freqs_cis)
            for bi, bp in enumerate(encoder.blocks_p[si]):
                if si == last_stage and bi == len(encoder.blocks_p[si]) - 1:
                    p_tokens, attn_maps['p'] = manual_attention(bp, p_tokens, freqs_cis)
                else:
                    p_tokens = bp(p_tokens, freqs_cis=freqs_cis)

            # CLS exchange: M(1) + P(K=2) = 3 토큰
            m_cls_tok = m_tokens[:, 0:1]
            p_cls_tok = p_tokens[:, 0:K]
            cls_ex = encoder.cls_exchange[si](torch.cat([m_cls_tok, p_cls_tok], dim=1))
            m_tokens = torch.cat([cls_ex[:, 0:1], m_tokens[:, 1:]], dim=1)
            p_tokens = torch.cat([cls_ex[:, 1:1+K], p_tokens[:, K:]], dim=1)

        m_tokens = encoder.norm_m(m_tokens)
        p_tokens = encoder.norm_p(p_tokens)

        # v7-big decoders (CLS_M 제외, CLS_P_bg/motion만)
        p_cls_bg = p_tokens[:, 0:1]
        p_cls_motion = p_tokens[:, 1:2]
        m_patches_final = m_tokens[:, 1:]
        pred_bg = model.decoder_m_bg(m_patches_final, p_cls_bg)
        pred_motion = model.decoder_m_motion(m_patches_final, p_cls_motion)

    pred_bg_np = pred_bg.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
    pred_motion_np = pred_motion.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

    # ΔL grayscale for overlay
    delta_l = m_channel[0, 0].cpu().numpy()
    delta_l_norm = (delta_l - delta_l.min()) / (delta_l.max() - delta_l.min() + 1e-8)
    delta_l_rgb = np.stack([delta_l_norm] * 3, axis=-1)

    return attn_maps, pred_bg_np, pred_motion_np, delta_l_rgb


def cls_attn_to_heatmap(attn_weights, cls_idx, patches_offset, size=224, patch_grid=14):
    """특정 CLS의 patch attention weight → heatmap.

    Args:
        attn_weights: [B, N, N] softmax attention
        cls_idx: query CLS 인덱스 (M: 0, P_bg: 0, P_motion: 1)
        patches_offset: 각 stream에서 patches 시작 인덱스
                        (M stream: 1, P stream: K=2)
    """
    cls_attn = attn_weights[0, cls_idx, patches_offset:].numpy()
    cls_attn = cls_attn.reshape(patch_grid, patch_grid)
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    hm = np.array(Image.fromarray((cls_attn * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR)) / 255.0
    return hm


def main():
    parser = argparse.ArgumentParser(description="v7-big attention diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egodex-dir", required=True,
                        help="EgoDex frames dir (e.g. .../egodex/frames/part4)")
    parser.add_argument("--droid-dir", default=None,
                        help="(optional) DROID frames dir (e.g. .../droid_frames/ext1)")
    parser.add_argument("--output", default="results/attn_compare_v7big.png")
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.03)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load v7-big model
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
    epoch = ck.get('epoch', '?')
    print(f"  Epoch {epoch}")

    # Select high-change pairs
    print(f"\nSelecting high-change pairs from EgoDex...")
    egodex_pairs = find_high_change_pairs(args.egodex_dir, 30, 2, args.gap, args.seed)
    print(f"  Selected: {[p[2] for p in egodex_pairs]} "
          f"(MSE: {[f'{p[4]:.4f}' for p in egodex_pairs]})")

    pairs = list(egodex_pairs)
    labels = ['EgoDex'] * len(egodex_pairs)

    if args.droid_dir and os.path.exists(args.droid_dir):
        print(f"Selecting high-change pairs from DROID...")
        droid_pairs = find_high_change_pairs(args.droid_dir, 30, 2, args.gap, args.seed + 1)
        print(f"  Selected: {[p[2] for p in droid_pairs]} "
              f"(MSE: {[f'{p[4]:.4f}' for p in droid_pairs]})")
        pairs += droid_pairs
        labels += ['DROID'] * len(droid_pairs)

    n_rows = len(pairs)
    n_cols = 7  # Frame t, Frame t+k, M attn, P_bg attn, P_motion attn, Pred BG, Pred Motion

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = ['Frame t', f'Frame t+{args.gap}',
                  'M Attn on ΔL',
                  'P_bg Attn on ΔL', 'P_motion Attn on ΔL',
                  'Pred BG', 'Pred Motion']

    for row, ((path_t, path_tk, ep_label, gap, mse), dataset) in enumerate(zip(pairs, labels)):
        img_t = load_image(path_t)
        img_tk = load_image(path_tk)

        crop = transforms.CenterCrop(224)
        img_t_c = crop(img_t.permute(2, 0, 1)).permute(1, 2, 0)
        img_tk_c = crop(img_tk.permute(2, 0, 1)).permute(1, 2, 0)
        img_t_tensor = img_t_c.permute(2, 0, 1)
        img_tk_tensor = img_tk_c.permute(2, 0, 1)

        attn_maps, pred_bg, pred_motion, delta_l_rgb = extract_v7big_attention_and_predict(
            model, img_t_tensor, img_tk_tensor)

        img_t_np = img_t_c.numpy()
        img_tk_np = img_tk_c.numpy()

        # M stream: 1 CLS → patches 시작 idx=1, query idx=0
        m_hm = cls_attn_to_heatmap(attn_maps['m'], cls_idx=0, patches_offset=1)
        # P stream: 2 CLS → patches 시작 idx=2, query: 0(bg), 1(motion)
        p_bg_hm = cls_attn_to_heatmap(attn_maps['p'], cls_idx=0, patches_offset=K)
        p_motion_hm = cls_attn_to_heatmap(attn_maps['p'], cls_idx=1, patches_offset=K)

        axes[row][0].imshow(img_t_np)
        axes[row][1].imshow(img_tk_np)
        axes[row][2].imshow(delta_l_rgb)
        axes[row][2].imshow(m_hm, cmap='hot', alpha=0.5)
        axes[row][3].imshow(delta_l_rgb)
        axes[row][3].imshow(p_bg_hm, cmap='hot', alpha=0.5)
        axes[row][4].imshow(delta_l_rgb)
        axes[row][4].imshow(p_motion_hm, cmap='hot', alpha=0.5)
        axes[row][5].imshow(pred_bg)
        axes[row][6].imshow(pred_motion)

        mse_bg = ((img_tk_np - pred_bg) ** 2).mean()
        mse_motion = ((img_tk_np - pred_motion) ** 2).mean()

        axes[row][0].set_ylabel(
            f'{dataset}\n{ep_label[:25]}\nΔ={mse:.3f}',
            fontsize=9, rotation=0, labelpad=80, va='center',
        )
        axes[row][5].set_xlabel(f'MSE={mse_bg:.4f}', fontsize=8)
        axes[row][6].set_xlabel(f'MSE={mse_motion:.4f}', fontsize=8)

        for col in range(n_cols):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for col in range(n_cols):
        axes[0][col].set_title(col_titles[col], fontsize=11, fontweight='bold')

    fig.suptitle(
        f'v7-big Epoch {epoch} — 3-CLS Attention + Dual Decoder '
        f'(high-change samples, gap={args.gap}, σ={args.sigma})',
        fontsize=14, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.output}")
    print("\nSpecialization check: P_bg Attn와 P_motion Attn이 공간적으로 구별되는지 관찰.")


if __name__ == "__main__":
    main()
