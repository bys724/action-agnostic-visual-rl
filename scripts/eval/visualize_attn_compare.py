#!/usr/bin/env python3
"""Two-Stream attention + prediction 비교 시각화 (cross-domain).

EgoDex / DROID 에서 변화가 큰 샘플을 자동 선별 후:
- Frame t, Frame t+k
- M stream attention overlay (on frame t)
- P stream attention overlay (on frame t+k)
- Pred M, Pred P
를 4행 6열 그림 한 장에 출력.
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


def load_image(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, 3]


def find_high_change_pairs(frames_dir, num_candidates=30, num_select=2, gap=30, seed=42):
    """변화량(pixel MSE)이 큰 프레임 쌍 선별."""
    rng = random.Random(seed)
    episode_dirs = []
    for root, dirs, files in os.walk(frames_dir):
        if any(f.endswith('.jpg') for f in files):
            episode_dirs.append(root)
    # os.walk 반환 순서는 FS 의존적이라 서버/환경마다 다름 → sorted로 고정해서
    # seed와 결합 시 완전 재현 가능하게 만듦.
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


def extract_attention_and_predict(model, img_t_tensor, img_tk_tensor):
    """Run model forward, extract last-layer attention maps + predictions.

    APE/RoPE 자동 감지: encoder에 pos_embed_m이 있으면 APE, 아니면 RoPE.
    """
    x = img_t_tensor.unsqueeze(0).to(DEVICE)
    y = img_tk_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        m_channel, p_channel = model.preprocessing(x, y)

    encoder = model.encoder
    last_stage = encoder.num_stages - 1
    attn_maps = {}

    # APE vs RoPE 감지
    use_ape = hasattr(encoder, 'pos_embed_m')
    freqs_cis = None if use_ape else encoder.freqs_cis

    def manual_attention(block, tokens, freqs_cis):
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
        attn_avg = attn.mean(dim=1)  # [B, N, N]
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        tokens = tokens + block.proj(out)
        tokens = tokens + block.mlp(block.norm2(tokens))
        return tokens, attn_avg.detach().cpu()

    with torch.no_grad():
        B = m_channel.shape[0]
        m_patches = encoder.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        p_patches = encoder.patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        m_tokens = torch.cat([encoder.cls_token_m.expand(B, -1, -1), m_patches], dim=1)
        p_tokens = torch.cat([encoder.cls_token_p.expand(B, -1, -1), p_patches], dim=1)

        # APE: position embedding 더하기
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
            m_cls, p_cls = m_tokens[:, 0:1], p_tokens[:, 0:1]
            cls_ex = encoder.cls_exchange[si](torch.cat([m_cls, p_cls], dim=1))
            m_tokens = torch.cat([cls_ex[:, 0:1], m_tokens[:, 1:]], dim=1)
            p_tokens = torch.cat([cls_ex[:, 1:2], p_tokens[:, 1:]], dim=1)

        m_tokens = encoder.norm_m(m_tokens)
        p_tokens = encoder.norm_p(p_tokens)

        cls_emb = (m_tokens[:, 0] + p_tokens[:, 0]) / 2
        pred_m = model.decoder_m(m_tokens[:, 1:], cls_emb)
        pred_p = model.decoder_p(p_tokens[:, 1:], cls_emb)

    pred_m_np = pred_m.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
    pred_p_np = pred_p.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

    # ΔL → grayscale RGB for overlay
    delta_l = m_channel[0, 0].cpu().numpy()
    delta_l_norm = (delta_l - delta_l.min()) / (delta_l.max() - delta_l.min() + 1e-8)
    delta_l_rgb = np.stack([delta_l_norm] * 3, axis=-1)

    return attn_maps, pred_m_np, pred_p_np, delta_l_rgb


def attn_to_heatmap(attn_weights, size=224, patch_grid=14):
    cls_attn = attn_weights[0, 0, 1:].numpy().reshape(patch_grid, patch_grid)
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    hm = np.array(Image.fromarray((cls_attn * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR)) / 255.0
    return hm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egodex-dir", required=True)
    parser.add_argument("--droid-dir", required=True)
    parser.add_argument("--output", default="results/attn_compare.png")
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-ratio-p", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load model
    print(f"Loading: {args.checkpoint}")
    model = TwoStreamModel(
        depth=args.depth, num_stages=args.num_stages,
        mask_ratio=args.mask_ratio, mask_ratio_p=args.mask_ratio_p,
    ).to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    epoch = ck.get('epoch', '?')
    print(f"  Epoch {epoch}")

    # Select high-change pairs
    print(f"\nSelecting high-change pairs from EgoDex...")
    egodex_pairs = find_high_change_pairs(args.egodex_dir, 30, 2, args.gap, args.seed)
    print(f"  Selected: {[p[2] for p in egodex_pairs]} (MSE: {[f'{p[4]:.4f}' for p in egodex_pairs]})")

    print(f"Selecting high-change pairs from DROID...")
    droid_pairs = find_high_change_pairs(args.droid_dir, 30, 2, args.gap, args.seed + 1)
    print(f"  Selected: {[p[2] for p in droid_pairs]} (MSE: {[f'{p[4]:.4f}' for p in droid_pairs]})")

    all_pairs = egodex_pairs + droid_pairs
    labels = ['EgoDex'] * 2 + ['DROID'] * 2

    # Build figure: 4 rows × 6 cols
    fig, axes = plt.subplots(4, 6, figsize=(30, 20))
    col_titles = ['Frame t', f'Frame t+{args.gap}',
                  'M Attn on ΔL', 'P Attn on ΔL', 'Pred M', 'Pred P']

    for row, ((path_t, path_tk, ep_label, gap, mse), dataset) in enumerate(zip(all_pairs, labels)):
        img_t = load_image(path_t)
        img_tk = load_image(path_tk)

        crop = transforms.CenterCrop(224)
        img_t_c = crop(img_t.permute(2, 0, 1)).permute(1, 2, 0)
        img_tk_c = crop(img_tk.permute(2, 0, 1)).permute(1, 2, 0)

        img_t_tensor = img_t_c.permute(2, 0, 1)  # [3, 224, 224]
        img_tk_tensor = img_tk_c.permute(2, 0, 1)

        attn_maps, pred_m, pred_p, delta_l_rgb = extract_attention_and_predict(
            model, img_t_tensor, img_tk_tensor)

        img_t_np = img_t_c.numpy()
        img_tk_np = img_tk_c.numpy()
        m_hm = attn_to_heatmap(attn_maps['m'])
        p_hm = attn_to_heatmap(attn_maps['p'])

        # Col 0: Frame t
        axes[row][0].imshow(img_t_np)
        # Col 1: Frame t+k
        axes[row][1].imshow(img_tk_np)
        # Col 2: M attention overlay on ΔL
        axes[row][2].imshow(delta_l_rgb)
        axes[row][2].imshow(m_hm, cmap='hot', alpha=0.5)
        # Col 3: P attention overlay on ΔL
        axes[row][3].imshow(delta_l_rgb)
        axes[row][3].imshow(p_hm, cmap='hot', alpha=0.5)
        # Col 4: Pred M
        axes[row][4].imshow(pred_m)
        # Col 5: Pred P
        axes[row][5].imshow(pred_p)

        mse_m = ((img_tk_np - pred_m) ** 2).mean()
        mse_p = ((img_tk_np - pred_p) ** 2).mean()

        axes[row][0].set_ylabel(
            f'{dataset}\n{ep_label[:25]}\nΔ={mse:.3f}',
            fontsize=9, rotation=0, labelpad=80, va='center',
        )
        axes[row][4].set_xlabel(f'MSE={mse_m:.4f}', fontsize=8)
        axes[row][5].set_xlabel(f'MSE={mse_p:.4f}', fontsize=8)

        for col in range(6):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for col in range(6):
        axes[0][col].set_title(col_titles[col], fontsize=12, fontweight='bold')

    fig.suptitle(
        f'Two-Stream Epoch {epoch} — Attention + Prediction (high-change samples, gap={args.gap})',
        fontsize=15, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
