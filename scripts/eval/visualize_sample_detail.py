"""
Two-Stream 단일 샘플 상세 시각화.

하나의 프레임 쌍에 대해:
1. 원본 입력 (frame_t, frame_tk)
2. M channel 입력 (ΔL, Sobel_x(ΔL), Sobel_y(ΔL))
3. P channel 입력 (Sobel_x, Sobel_y, R, G, B)
4. M/P stream의 마지막 layer attention map (CLS → patches)
5. 모델 예측 (Pred M, Pred P)

Usage:
    python scripts/eval/visualize_sample_detail.py \
        --checkpoint /mnt/data/checkpoints/two_stream/.../checkpoint_epoch0005.pt \
        --frames-dir /mnt/data/egodex_frames/part1 \
        --output-dir docs/architecture/sample_detail \
        --gap 30 --episode deal_gather_cards
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_crop(path, crop_params, size=256):
    img = Image.open(path).convert("RGB").resize((size, size))
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return transforms.functional.crop(t, *crop_params)


def find_episode(frames_dir, episode_filter=None, min_frames=60):
    """프레임 수가 충분한 에피소드 탐색."""
    candidates = []
    for root, dirs, files in os.walk(frames_dir):
        jpgs = sorted([f for f in files if f.endswith('.jpg')])
        if len(jpgs) >= min_frames:
            if episode_filter is None or episode_filter in root:
                candidates.append((root, jpgs))
        if len(candidates) > 100:  # 너무 많이 탐색하지 않도록
            break
    if not candidates:
        raise ValueError(f"No episodes with {min_frames}+ frames found")
    return random.choice(candidates)


def extract_attention_maps(model, m_channel, p_channel):
    """마지막 transformer layer에서 CLS→patches attention 추출.

    커스텀 TransformerBlock + RoPE 지원. 마지막 stage 마지막 block에서
    QKV를 직접 계산하여 attention weight를 추출.
    """
    from src.models.two_stream import apply_rope

    attention_maps = {'m': None, 'p': None}
    encoder = model.encoder
    last_stage = encoder.num_stages - 1

    def manual_attention(block, x, freqs_cis):
        """TransformerBlock의 self-attention을 수동으로 실행하여 attn weight 추출."""
        h = block.norm1(x)
        B, N, D = h.shape
        qkv = block.qkv(h).reshape(B, N, 3, block.num_heads, block.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, N, H, D_head]

        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis, has_cls=True)
            k = apply_rope(k, freqs_cis, has_cls=True)

        # [B, H, N, D_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 수동 attention (softmax)
        scale = block.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)

        # head 평균
        attn_avg = attn.mean(dim=1)  # [B, N, N]

        # 정상 attention output (forward 계속하기 위해)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + block.proj(out)
        x = x + block.mlp(block.norm2(x))
        return x, attn_avg.detach().cpu()

    with torch.no_grad():
        batch_size = m_channel.shape[0]

        # Patch embedding
        m_patches = encoder.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        p_patches = encoder.patch_embed_p(p_channel).flatten(2).transpose(1, 2)

        m_cls = encoder.cls_token_m.expand(batch_size, -1, -1)
        p_cls = encoder.cls_token_p.expand(batch_size, -1, -1)
        m_tokens = torch.cat([m_cls, m_patches], dim=1)
        p_tokens = torch.cat([p_cls, p_patches], dim=1)

        for stage_idx in range(encoder.num_stages):
            for block_idx, block_m in enumerate(encoder.blocks_m[stage_idx]):
                if stage_idx == last_stage and block_idx == len(encoder.blocks_m[stage_idx]) - 1:
                    m_tokens, attn_w = manual_attention(block_m, m_tokens, encoder.freqs_cis)
                    attention_maps['m'] = attn_w
                else:
                    m_tokens = block_m(m_tokens, freqs_cis=encoder.freqs_cis)

            for block_idx, block_p in enumerate(encoder.blocks_p[stage_idx]):
                if stage_idx == last_stage and block_idx == len(encoder.blocks_p[stage_idx]) - 1:
                    p_tokens, attn_w = manual_attention(block_p, p_tokens, encoder.freqs_cis)
                    attention_maps['p'] = attn_w
                else:
                    p_tokens = block_p(p_tokens, freqs_cis=encoder.freqs_cis)

            # CLS exchange
            m_cls_ex = m_tokens[:, 0:1]
            p_cls_ex = p_tokens[:, 0:1]
            cls_combined = torch.cat([m_cls_ex, p_cls_ex], dim=1)
            cls_exchanged = encoder.cls_exchange[stage_idx](cls_combined)
            m_tokens = torch.cat([cls_exchanged[:, 0:1], m_tokens[:, 1:]], dim=1)
            p_tokens = torch.cat([cls_exchanged[:, 1:2], p_tokens[:, 1:]], dim=1)

        m_tokens = encoder.norm_m(m_tokens)
        p_tokens = encoder.norm_p(p_tokens)

    return attention_maps, m_tokens, p_tokens


def save_channel_viz(channel, titles, save_path, suptitle):
    """멀티채널 텐서를 채널별로 시각화."""
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (title, ax) in enumerate(zip(titles, axes)):
        data = channel[0, i].cpu().numpy()
        im = ax.imshow(data, cmap='RdBu_r' if data.min() < 0 else 'viridis')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def save_attention_map(attn_weights, save_path, title, patch_grid=14):
    """CLS → patches attention을 heatmap으로 시각화."""
    # attn_weights: [1, N+1, N+1] — CLS row (index 0) → patches
    cls_attn = attn_weights[0, 0, 1:].numpy()  # [N] — CLS가 각 patch에 주는 attention
    cls_attn = cls_attn.reshape(patch_grid, patch_grid)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(cls_attn, cmap='hot', interpolation='bilinear')
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def _resize_attn(cls_attn_2d, size=224):
    """14x14 attention map → 224x224 리사이즈."""
    from PIL import Image as PILImage
    attn_norm = (cls_attn_2d * 255 / (cls_attn_2d.max() + 1e-8)).astype(np.uint8)
    return np.array(PILImage.fromarray(attn_norm).resize((size, size), PILImage.BILINEAR)) / 255.0


def save_attention_overlay(imgs_and_labels, attn_weights, save_path, title, patch_grid=14):
    """여러 이미지에 attention heatmap 오버레이.

    Args:
        imgs_and_labels: [(img_np, label), ...] — 오버레이할 이미지 목록
        attn_weights: [1, N+1, N+1] attention matrix
        save_path: 저장 경로
        title: 그림 제목
    """
    cls_attn = attn_weights[0, 0, 1:].numpy().reshape(patch_grid, patch_grid)
    attn_resized = _resize_attn(cls_attn)

    n_imgs = len(imgs_and_labels)
    # columns: attention map + (image, overlay) per img
    n_cols = 1 + n_imgs * 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Attention heatmap
    axes[0].imshow(cls_attn, cmap='hot', interpolation='bilinear')
    axes[0].set_title('Attention (CLS→patches)', fontsize=10)
    axes[0].axis('off')

    # 이미지별 원본 + 오버레이
    for i, (img_np, label) in enumerate(imgs_and_labels):
        col_img = 1 + i * 2
        col_overlay = 2 + i * 2
        axes[col_img].imshow(img_np)
        axes[col_img].set_title(label, fontsize=10)
        axes[col_img].axis('off')

        axes[col_overlay].imshow(img_np)
        axes[col_overlay].imshow(attn_resized, cmap='hot', alpha=0.5)
        axes[col_overlay].set_title(f'{label} + Attn', fontsize=10)
        axes[col_overlay].axis('off')

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", default="docs/architecture/sample_detail")
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--episode", type=str, default=None)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.0,
                        help="MAE mask ratio (체크포인트와 맞춰야 함)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 로드
    print(f"Loading: {args.checkpoint}")
    model = TwoStreamModel(depth=args.depth, num_stages=args.num_stages, mask_ratio=args.mask_ratio).to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    # strict=False: compositor 등 학습 전용 모듈이 체크포인트에 있을 수 있음
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    epoch = ck.get('epoch', '?')
    print(f"  Epoch {epoch}, Loss {ck.get('train_loss', '?'):.4f}")

    # 에피소드 선택
    ep_dir, frames = find_episode(args.frames_dir, args.episode, min_frames=args.gap + 5)
    ep_name = os.path.relpath(ep_dir, args.frames_dir)
    print(f"Episode: {ep_name} ({len(frames)} frames), gap={args.gap}")

    # 프레임 로드
    start_idx = len(frames) // 4
    future_idx = start_idx + args.gap
    path_t = os.path.join(ep_dir, frames[start_idx])
    path_tk = os.path.join(ep_dir, frames[future_idx])

    dummy = Image.open(path_t).convert("RGB").resize((256, 256))
    dummy_t = torch.from_numpy(np.array(dummy)).float().permute(2, 0, 1) / 255.0
    crop_t = transforms.RandomCrop.get_params(dummy_t, (224, 224))
    crop_tk = transforms.RandomCrop.get_params(dummy_t, (224, 224))

    img_t = load_and_crop(path_t, crop_t)
    img_tk = load_and_crop(path_tk, crop_tk)

    img_t_np = img_t.permute(1, 2, 0).numpy()
    img_tk_np = img_tk.permute(1, 2, 0).numpy()

    # === 1. 원본 입력 저장 ===
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_t_np); axes[0].set_title(f'Frame t (#{start_idx})'); axes[0].axis('off')
    axes[1].imshow(img_tk_np); axes[1].set_title(f'Frame t+{args.gap} (#{future_idx})'); axes[1].axis('off')
    fig.suptitle(f'{ep_name}, gap={args.gap}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, '01_input_frames.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_input_frames.png")

    # === 2. 전처리: M/P 채널 ===
    x = img_t.unsqueeze(0).to(DEVICE)
    y = img_tk.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        m_channel, p_channel = model.preprocessing(x, y)

    # M channel: [1, 3, H, W] = [ΔL, Sobel_x(ΔL), Sobel_y(ΔL)]
    save_channel_viz(m_channel, ['ΔL (brightness change)', 'Sobel_x(ΔL)', 'Sobel_y(ΔL)'],
                     os.path.join(args.output_dir, '02_m_channel_input.png'),
                     'M Channel Input (3ch: change detection)')
    print("  Saved: 02_m_channel_input.png")

    # P channel: [1, 5, H, W] = [Sobel_x, Sobel_y, R, G, B]
    save_channel_viz(p_channel, ['Sobel_x (edge)', 'Sobel_y (edge)', 'R', 'G', 'B'],
                     os.path.join(args.output_dir, '03_p_channel_input.png'),
                     'P Channel Input (5ch: appearance + structure)')
    print("  Saved: 03_p_channel_input.png")

    # === 3. Attention Maps ===
    attn_maps, m_tokens, p_tokens = extract_attention_maps(model, m_channel, p_channel)

    # ΔL 이미지를 grayscale RGB로 변환 (오버레이용)
    delta_l = m_channel[0, 0].cpu().numpy()  # ΔL channel
    delta_l_norm = (delta_l - delta_l.min()) / (delta_l.max() - delta_l.min() + 1e-8)
    delta_l_rgb = np.stack([delta_l_norm] * 3, axis=-1)  # [H,W,3]

    # M stream attention: ΔL 이미지에 오버레이
    save_attention_overlay(
        [(delta_l_rgb, 'ΔL (brightness diff)'), (img_t_np, 'Frame t')],
        attn_maps['m'],
        os.path.join(args.output_dir, '04_m_attention.png'),
        'M Stream: Last Layer Attention (CLS → patches)')
    print("  Saved: 04_m_attention.png")

    # P stream attention: frame_t + frame_tk 둘 다에 오버레이
    save_attention_overlay(
        [(img_t_np, 'Frame t'), (img_tk_np, f'Frame t+{args.gap}')],
        attn_maps['p'],
        os.path.join(args.output_dir, '05_p_attention.png'),
        'P Stream: Last Layer Attention (CLS → patches)')
    print("  Saved: 05_p_attention.png")

    # === 4. 모델 예측 ===
    with torch.no_grad():
        cls_embedding = (m_tokens[:, 0] + p_tokens[:, 0]) / 2
        m_patches = m_tokens[:, 1:]
        p_patches = p_tokens[:, 1:]
        pred_m = model.decoder_m(m_patches, cls_embedding)
        pred_p = model.decoder_p(p_patches, cls_embedding)

    pred_m_np = pred_m.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
    pred_p_np = pred_p.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(axes,
        [img_t_np, img_tk_np, pred_m_np, pred_p_np],
        ['Frame t', f'Frame t+{args.gap} (target)', 'Pred M (ΔL+Sobel)', 'Pred P (Sobel+RGB)']):
        ax.imshow(img); ax.set_title(title, fontsize=10); ax.axis('off')
    fig.suptitle(f'Predictions — Epoch {epoch}, gap={args.gap}', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, '06_predictions.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_predictions.png")

    # === 5. 전체 요약 ===
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    # Row 1: Input & channels
    axes[0][0].imshow(img_t_np); axes[0][0].set_title('Frame t'); axes[0][0].axis('off')
    axes[0][1].imshow(img_tk_np); axes[0][1].set_title(f'Frame t+{args.gap}'); axes[0][1].axis('off')
    axes[0][2].imshow(m_channel[0, 0].cpu().numpy(), cmap='RdBu_r')
    axes[0][2].set_title('M: ΔL'); axes[0][2].axis('off')
    axes[0][3].imshow(p_channel[0, 2].cpu().numpy(), cmap='gray')
    axes[0][3].set_title('P: R channel'); axes[0][3].axis('off')
    axes[0][4].imshow(delta_l_rgb)
    axes[0][4].set_title('ΔL (normalized)'); axes[0][4].axis('off')

    # Row 2: Attention & predictions
    m_attn = attn_maps['m'][0, 0, 1:].numpy().reshape(14, 14)
    p_attn = attn_maps['p'][0, 0, 1:].numpy().reshape(14, 14)
    m_attn_r = _resize_attn(m_attn)
    p_attn_r = _resize_attn(p_attn)

    # M attention on ΔL
    axes[1][0].imshow(delta_l_rgb)
    axes[1][0].imshow(m_attn_r, cmap='hot', alpha=0.5)
    axes[1][0].set_title('M Attn on ΔL'); axes[1][0].axis('off')

    # P attention on frame_t
    axes[1][1].imshow(img_t_np)
    axes[1][1].imshow(p_attn_r, cmap='hot', alpha=0.5)
    axes[1][1].set_title('P Attn on Frame t'); axes[1][1].axis('off')

    # P attention on frame_tk
    axes[1][2].imshow(img_tk_np)
    axes[1][2].imshow(p_attn_r, cmap='hot', alpha=0.5)
    axes[1][2].set_title(f'P Attn on Frame t+{args.gap}'); axes[1][2].axis('off')

    axes[1][3].imshow(pred_m_np); axes[1][3].set_title('Pred M'); axes[1][3].axis('off')
    axes[1][4].imshow(pred_p_np); axes[1][4].set_title('Pred P'); axes[1][4].axis('off')

    fig.suptitle(f'Two-Stream — {ep_name}, gap={args.gap}, epoch {epoch}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, '00_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 00_summary.png")

    print(f"\nAll saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
