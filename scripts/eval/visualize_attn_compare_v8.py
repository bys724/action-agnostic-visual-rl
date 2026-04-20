#!/usr/bin/env python3
"""Two-Stream v8 attention 비교 시각화.

v8 구조 (v4-style single CLS + EMA P teacher):
- Student encoder: M stream (ΔL+Sobel) + P stream (frame_t)
- Teacher encoder: shared M (zero input) + EMA P (frame_{t+k})
- 학습 loss: L_M (pixel) + λ·L_P (student p_patches[masked] → teacher p_patches[matching])

시각화 컬럼 (가로):
1. Frame t
2. Frame t+k
3. Student M CLS attn (on ΔL)
4. Student P CLS attn (on Frame t) — P는 image_current(=frame_t) 사용
5. Teacher P CLS attn (on Frame t+k) — teacher 입력 (t+k, t+k)
6. Pred M (student decoder_m 출력, L_M 복원 결과)
7. cos(CLS_P_student, CLS_P_teacher) — 텍스트로 기록

Student/teacher P attention이 **서로 다른 영역**에 attend하면 teacher→student prediction 학습이 유효한 signal임을 의미.
두 attention이 너무 비슷하면 L_P가 trivial해진다는 경고 신호.

Usage:
    python scripts/eval/visualize_attn_compare_v8.py \\
        --checkpoint /path/to/v8/checkpoint_epoch0004.pt \\
        --egodex-dir /proj/external_group/mrg/datasets/egodex/frames/part4 \\
        --droid-dir /proj/external_group/mrg/datasets/droid_frames/ext1 \\
        --output docs/architecture/attn_v8_ep4.png
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
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TwoStreamModel
from src.models.two_stream import apply_rope

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    """Block forward 수동 실행하며 self-attention weights 반환."""
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
    attn_avg = attn.mean(dim=1)  # heads 평균 [B, N, N]
    out = (attn @ v).transpose(1, 2).reshape(B, N, D)
    tokens = tokens + block.proj(out)
    tokens = tokens + block.mlp(block.norm2(tokens))
    return tokens, attn_avg.detach().cpu()


def forward_student_and_extract(model, img_t_tensor, img_tk_tensor):
    """Student encoder forward + last-stage M/P attention + pred_m + CLS_P embedding."""
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
        m_tokens = torch.cat([encoder.cls_token_m.expand(B, -1, -1), m_patches], dim=1)
        p_tokens = torch.cat([encoder.cls_token_p.expand(B, -1, -1), p_patches], dim=1)

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
        cls_p_student = p_tokens[:, 0]  # [B=1, D]

    pred_m_np = pred_m.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

    delta_l = m_channel[0, 0].cpu().numpy()
    delta_l_norm = (delta_l - delta_l.min()) / (delta_l.max() - delta_l.min() + 1e-8)
    delta_l_rgb = np.stack([delta_l_norm] * 3, axis=-1)

    return attn_maps, pred_m_np, delta_l_rgb, cls_p_student


def forward_teacher_and_extract(model, img_tk_tensor):
    """Teacher forward: (img_tk, img_tk) → zero M + EMA P. Last stage P attention 추출.

    Teacher는 M은 shared, P만 EMA copy. 입력 ΔL=0 (zero M).
    """
    y = img_tk_tensor.unsqueeze(0).to(DEVICE)
    B = y.shape[0]

    with torch.no_grad():
        p_channel = model.preprocessing.compute_p_channel(y)

    encoder = model.encoder
    last_stage = encoder.num_stages - 1
    use_ape = encoder.use_ape
    freqs_cis = None if use_ape else encoder.freqs_cis

    attn_p = None

    with torch.no_grad():
        # Zero M input (teacher 설계: (t+k, t+k) → ΔL=0)
        H, W = y.shape[2], y.shape[3]
        m_channel = torch.zeros(B, 3, H, W, device=y.device, dtype=y.dtype)

        m_patches = encoder.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        p_patches = model.ema_patch_embed_p(p_channel).flatten(2).transpose(1, 2)
        m_tokens = torch.cat([encoder.cls_token_m.expand(B, -1, -1), m_patches], dim=1)
        p_tokens = torch.cat([model.ema_cls_token_p.expand(B, -1, -1), p_patches], dim=1)

        if use_ape:
            m_tokens = m_tokens + encoder.pos_embed_m
            p_tokens = p_tokens + model.ema_pos_embed_p

        for si in range(encoder.num_stages):
            for bm in encoder.blocks_m[si]:
                m_tokens = bm(m_tokens, freqs_cis=freqs_cis)
            for bi, bp in enumerate(model.ema_blocks_p[si]):
                if si == last_stage and bi == len(model.ema_blocks_p[si]) - 1:
                    p_tokens, attn_p = manual_attention(bp, p_tokens, freqs_cis)
                else:
                    p_tokens = bp(p_tokens, freqs_cis=freqs_cis)

            m_cls, p_cls = m_tokens[:, 0:1], p_tokens[:, 0:1]
            cls_ex = model.ema_cls_exchange[si](torch.cat([m_cls, p_cls], dim=1))
            m_tokens = torch.cat([cls_ex[:, 0:1], m_tokens[:, 1:]], dim=1)
            p_tokens = torch.cat([cls_ex[:, 1:2], p_tokens[:, 1:]], dim=1)

        p_tokens = model.ema_norm_p(p_tokens)
        cls_p_teacher = p_tokens[:, 0]

    return attn_p, cls_p_teacher


def cls_attn_to_heatmap(attn_weights, size=224, patch_grid=14):
    cls_attn = attn_weights[0, 0, 1:].numpy().reshape(patch_grid, patch_grid)
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    hm = np.array(Image.fromarray((cls_attn * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR)) / 255.0
    return hm


def main():
    parser = argparse.ArgumentParser(description="v8 attention diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egodex-dir", required=True)
    parser.add_argument("--droid-dir", default=None)
    parser.add_argument("--output", default="results/attn_compare_v8.png")
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=2,
                        help="각 데이터셋에서 선택할 high-change 샘플 수")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load v8 model: training args와 동일 설정
    # (pretrain.sbatch: --use-ape --rotation-aug --mask-ratio 0.3 --mask-ratio-p 0.5 --v8-mode)
    print(f"Loading: {args.checkpoint}")
    model = TwoStreamModel(
        depth=args.depth, num_stages=args.num_stages,
        mask_ratio=0.3, mask_ratio_p=0.5,
        use_ape=True, rotation_aug=True,
        v8_mode=True, pred_head_ratio=2.0,
    ).to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    epoch = ck.get('epoch', '?')
    print(f"  Epoch {epoch}")
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    # Select high-change pairs
    print(f"\nSelecting high-change pairs from EgoDex...")
    egodex_pairs = find_high_change_pairs(
        args.egodex_dir, 30, args.num_samples, args.gap, args.seed)
    print(f"  Selected: {[p[2] for p in egodex_pairs]} "
          f"(MSE: {[f'{p[4]:.4f}' for p in egodex_pairs]})")

    pairs = list(egodex_pairs)
    labels = ['EgoDex'] * len(egodex_pairs)

    if args.droid_dir and os.path.exists(args.droid_dir):
        print(f"Selecting high-change pairs from DROID...")
        droid_pairs = find_high_change_pairs(
            args.droid_dir, 30, args.num_samples, args.gap, args.seed + 1)
        print(f"  Selected: {[p[2] for p in droid_pairs]} "
              f"(MSE: {[f'{p[4]:.4f}' for p in droid_pairs]})")
        pairs += droid_pairs
        labels += ['DROID'] * len(droid_pairs)

    n_rows = len(pairs)
    n_cols = 6  # Frame t, Frame t+k, M attn, P student attn, P teacher attn, Pred M

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = [
        'Frame t',
        f'Frame t+{args.gap}',
        'Student M Attn\n(on ΔL)',
        'Student P Attn\n(on Frame t)',
        'Teacher P Attn\n(on Frame t+k)',
        'Pred M',
    ]

    for row, ((path_t, path_tk, ep_label, gap, mse), dataset) in enumerate(zip(pairs, labels)):
        img_t = load_image(path_t)
        img_tk = load_image(path_tk)

        crop = transforms.CenterCrop(224)
        img_t_c = crop(img_t.permute(2, 0, 1)).permute(1, 2, 0)
        img_tk_c = crop(img_tk.permute(2, 0, 1)).permute(1, 2, 0)
        img_t_tensor = img_t_c.permute(2, 0, 1)
        img_tk_tensor = img_tk_c.permute(2, 0, 1)

        attn_maps, pred_m, delta_l_rgb, cls_p_stu = forward_student_and_extract(
            model, img_t_tensor, img_tk_tensor)
        attn_p_teacher, cls_p_tea = forward_teacher_and_extract(model, img_tk_tensor)

        img_t_np = img_t_c.numpy()
        img_tk_np = img_tk_c.numpy()
        m_hm = cls_attn_to_heatmap(attn_maps['m'])
        p_hm_student = cls_attn_to_heatmap(attn_maps['p'])
        p_hm_teacher = cls_attn_to_heatmap(attn_p_teacher)

        cos_st = F.cosine_similarity(cls_p_stu, cls_p_tea, dim=-1).item()

        axes[row][0].imshow(img_t_np)
        axes[row][1].imshow(img_tk_np)
        axes[row][2].imshow(delta_l_rgb)
        axes[row][2].imshow(m_hm, cmap='hot', alpha=0.5)
        axes[row][3].imshow(img_t_np)
        axes[row][3].imshow(p_hm_student, cmap='hot', alpha=0.5)
        axes[row][4].imshow(img_tk_np)
        axes[row][4].imshow(p_hm_teacher, cmap='hot', alpha=0.5)
        axes[row][5].imshow(pred_m)

        mse_m = ((img_tk_np - pred_m) ** 2).mean()

        axes[row][0].set_ylabel(
            f'{dataset}\n{ep_label[:25]}\nΔ={mse:.3f}\ncos(s,t)={cos_st:.3f}',
            fontsize=9, rotation=0, labelpad=85, va='center',
        )
        axes[row][5].set_xlabel(f'MSE(M)={mse_m:.4f}', fontsize=8)

        for col in range(n_cols):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for col in range(n_cols):
        axes[0][col].set_title(col_titles[col], fontsize=11, fontweight='bold')

    fig.suptitle(
        f'v8 Epoch {epoch} — Student/Teacher Attention + L_M Reconstruction '
        f'(high-change samples, gap={args.gap})',
        fontsize=14, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.output}")
    print("\n진단 포인트:")
    print(" - Student P attn (on frame_t) vs Teacher P attn (on frame_t+k): 공간 패턴 비교")
    print("   · 완전 일치: teacher→student prediction trivial, L_P 학습 기여 낮을 우려")
    print("   · 적절한 차이: motion/temporal 정보가 prediction 신호로 작용")
    print(" - cos(CLS_P_student, CLS_P_teacher): ep 초반엔 ~1.0 (EMA 초기화)에서 점차 분화 기대")


if __name__ == "__main__":
    main()
