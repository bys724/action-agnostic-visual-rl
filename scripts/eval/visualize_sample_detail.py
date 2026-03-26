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

    nn.TransformerEncoderLayer의 내부 MultiheadAttention에 hook을 걸어서
    attention weight를 캡처합니다.
    """
    attention_maps = {'m': [], 'p': []}

    hooks = []

    # 마지막 stage의 마지막 block에서 attention 추출
    last_stage = model.encoder.num_stages - 1
    last_block_m = model.encoder.blocks_m[last_stage][-1]
    last_block_p = model.encoder.blocks_p[last_stage][-1]

    def make_hook(stream_name):
        def hook_fn(module, args, kwargs, output):
            # TransformerEncoderLayer 내부에서 self_attn을 직접 호출
            pass
        return hook_fn

    # MultiheadAttention에 직접 hook
    def attn_hook(name):
        def hook_fn(module, args, output):
            # output = (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) == 2:
                attn_weights = output[1]  # [B, N, N]
                if attn_weights is not None:
                    attention_maps[name].append(attn_weights.detach().cpu())
        return hook_fn

    # TransformerEncoderLayer.self_attn에 hook 등록
    h1 = last_block_m.self_attn.register_forward_hook(attn_hook('m'))
    h2 = last_block_p.self_attn.register_forward_hook(attn_hook('p'))
    hooks.extend([h1, h2])

    # need_weights=True로 설정해야 attention weight가 반환됨
    # TransformerEncoderLayer는 기본적으로 need_weights=False
    # 임시로 변경
    orig_m = last_block_m.self_attn.batch_first
    orig_p = last_block_p.self_attn.batch_first

    # Forward with attention
    with torch.no_grad():
        # 수동으로 encoder forward를 재현하되 마지막 block에서만 attention 캡처
        # 간단하게: 전체 forward를 하되, hook에서 attention을 잡음

        # 문제: TransformerEncoderLayer는 내부적으로 self_attn(need_weights=False)를 호출
        # 이를 우회하기 위해 _sa_block을 패치

        # 원본 _sa_block 저장
        orig_sa_m = last_block_m._sa_block
        orig_sa_p = last_block_p._sa_block

        def patched_sa_block(self_ref, attn_module, name):
            def _sa_block(x, attn_mask=None, key_padding_mask=None):
                attn_out, attn_w = attn_module(x, x, x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=True)
                attention_maps[name].append(attn_w.detach().cpu())
                return attn_module.out_proj(attn_out) if hasattr(attn_module, '_qkv_same_embed_dim') else attn_out
            return _sa_block

        # 더 간단한 접근: forward를 직접 호출하면서 중간 결과를 저장
        # encoder.forward를 그대로 호출하되, 마지막 stage 마지막 block만
        # need_weights=True로 패치

        # 가장 간단한 방법: 전체 forward 후 마지막 layer attention을 근사
        # → encoder forward를 수동 재현

        batch_size = m_channel.shape[0]
        encoder = model.encoder

        # Patch embedding
        m_patches = encoder.patch_embed_m(m_channel).flatten(2).transpose(1, 2)
        p_patches = encoder.patch_embed_p(p_channel).flatten(2).transpose(1, 2)

        m_cls = encoder.cls_token_m.expand(batch_size, -1, -1)
        p_cls = encoder.cls_token_p.expand(batch_size, -1, -1)

        m_tokens = torch.cat([m_cls, m_patches], dim=1) + encoder.pos_embed_m
        p_tokens = torch.cat([p_cls, p_patches], dim=1) + encoder.pos_embed_p

        for stage_idx in range(encoder.num_stages):
            for block_idx, block_m in enumerate(encoder.blocks_m[stage_idx]):
                if stage_idx == last_stage and block_idx == len(encoder.blocks_m[stage_idx]) - 1:
                    # 마지막 block: attention 추출
                    normed = block_m.norm1(m_tokens)
                    attn_out, attn_w = block_m.self_attn(
                        normed, normed, normed, need_weights=True, average_attn_weights=True)
                    attention_maps['m'] = attn_w.detach().cpu()
                    m_tokens = m_tokens + attn_out
                    m_tokens = m_tokens + block_m._ff_block(block_m.norm2(m_tokens))
                else:
                    m_tokens = block_m(m_tokens)

            for block_idx, block_p in enumerate(encoder.blocks_p[stage_idx]):
                if stage_idx == last_stage and block_idx == len(encoder.blocks_p[stage_idx]) - 1:
                    normed = block_p.norm1(p_tokens)
                    attn_out, attn_w = block_p.self_attn(
                        normed, normed, normed, need_weights=True, average_attn_weights=True)
                    attention_maps['p'] = attn_w.detach().cpu()
                    p_tokens = p_tokens + attn_out
                    p_tokens = p_tokens + block_p._ff_block(block_p.norm2(p_tokens))
                else:
                    p_tokens = block_p(p_tokens)

            # CLS exchange
            m_cls_ex = m_tokens[:, 0:1]
            p_cls_ex = p_tokens[:, 0:1]
            cls_combined = torch.cat([m_cls_ex, p_cls_ex], dim=1)
            cls_exchanged = encoder.cls_exchange[stage_idx](cls_combined)
            m_tokens = torch.cat([cls_exchanged[:, 0:1], m_tokens[:, 1:]], dim=1)
            p_tokens = torch.cat([cls_exchanged[:, 1:2], p_tokens[:, 1:]], dim=1)

        m_tokens = encoder.norm_m(m_tokens)
        p_tokens = encoder.norm_p(p_tokens)

    # cleanup hooks
    for h in hooks:
        h.remove()

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


def save_attention_overlay(img_np, attn_weights, save_path, title, patch_grid=14):
    """원본 이미지 위에 attention heatmap 오버레이."""
    cls_attn = attn_weights[0, 0, 1:].numpy().reshape(patch_grid, patch_grid)

    # Resize attention to image size
    from PIL import Image as PILImage
    attn_resized = np.array(PILImage.fromarray(
        (cls_attn * 255 / cls_attn.max()).astype(np.uint8)
    ).resize((224, 224), PILImage.BILINEAR)) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title('Input', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(cls_attn, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Attention (CLS→patches)', fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(img_np)
    axes[2].imshow(attn_resized, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay', fontsize=10)
    axes[2].axis('off')

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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 로드
    print(f"Loading: {args.checkpoint}")
    model = TwoStreamModel().to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
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

    # M stream attention overlay on frame_t
    save_attention_overlay(img_t_np, attn_maps['m'],
                          os.path.join(args.output_dir, '04_m_attention.png'),
                          'M Stream: Last Layer Attention (CLS → patches)')
    print("  Saved: 04_m_attention.png")

    # P stream attention overlay on frame_t
    save_attention_overlay(img_t_np, attn_maps['p'],
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
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Input & channels
    axes[0][0].imshow(img_t_np); axes[0][0].set_title('Frame t'); axes[0][0].axis('off')
    axes[0][1].imshow(img_tk_np); axes[0][1].set_title(f'Frame t+{args.gap}'); axes[0][1].axis('off')
    axes[0][2].imshow(m_channel[0, 0].cpu().numpy(), cmap='RdBu_r')
    axes[0][2].set_title('M: ΔL'); axes[0][2].axis('off')
    axes[0][3].imshow(p_channel[0, 2].cpu().numpy(), cmap='gray')
    axes[0][3].set_title('P: R channel'); axes[0][3].axis('off')

    # Row 2: Attention & predictions
    m_attn = attn_maps['m'][0, 0, 1:].numpy().reshape(14, 14)
    p_attn = attn_maps['p'][0, 0, 1:].numpy().reshape(14, 14)
    axes[1][0].imshow(img_t_np)
    from PIL import Image as PILImage
    m_attn_r = np.array(PILImage.fromarray((m_attn * 255 / m_attn.max()).astype(np.uint8)).resize((224, 224), PILImage.BILINEAR)) / 255.0
    axes[1][0].imshow(m_attn_r, cmap='hot', alpha=0.5)
    axes[1][0].set_title('M Attention'); axes[1][0].axis('off')

    p_attn_r = np.array(PILImage.fromarray((p_attn * 255 / p_attn.max()).astype(np.uint8)).resize((224, 224), PILImage.BILINEAR)) / 255.0
    axes[1][1].imshow(img_t_np)
    axes[1][1].imshow(p_attn_r, cmap='hot', alpha=0.5)
    axes[1][1].set_title('P Attention'); axes[1][1].axis('off')

    axes[1][2].imshow(pred_m_np); axes[1][2].set_title('Pred M'); axes[1][2].axis('off')
    axes[1][3].imshow(pred_p_np); axes[1][3].set_title('Pred P'); axes[1][3].axis('off')

    fig.suptitle(f'Two-Stream v3 — {ep_name}, gap={args.gap}, epoch {epoch}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, '00_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 00_summary.png")

    print(f"\nAll saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
