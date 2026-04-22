#!/usr/bin/env python3
"""Two-Stream rotation diagnostic (multi-model 지원).

하나의 sample에 대해 입력 이미지를 0/90/180/270도 회전시키며
M/P attention이 content-driven (equivariant) 인지, 아니면
position-prior 고정 (invariant) 인지 진단.

진짜 motion content를 읽는다면 attention은 입력과 함께 회전해야 함.

Legacy (APE + nn.TransformerEncoderLayer) 및
현재 (RoPE + custom TransformerBlock) 체크포인트 모두 지원.
"""

import argparse
import os
import sys

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TwoStreamModel

from scripts.eval.visualize_attn_compare import (
    load_image, extract_attention_and_predict, attn_to_heatmap,
    reconstruct_from_residual_pred,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROTATIONS = [0, 90, 180, 270]


# ============================================================================
# Legacy checkpoint 호환
# ============================================================================

def _detect_model_config(state_dict):
    """체크포인트에서 num_stages, use_ape 자동 감지."""
    stages = set()
    for k in state_dict:
        if k.startswith('encoder.blocks_m.'):
            stages.add(int(k.split('.')[2]))
    num_stages = len(stages) if stages else 2

    use_ape = 'encoder.pos_embed_m' in state_dict

    # Legacy key 감지: nn.TransformerEncoderLayer 패턴
    is_legacy = any('self_attn.in_proj_weight' in k
                     for k in state_dict if k.startswith('encoder.blocks_'))

    return num_stages, use_ape, is_legacy


def _remap_legacy_encoder_keys(state_dict):
    """nn.TransformerEncoderLayer → custom TransformerBlock key 변환.

    self_attn.in_proj_weight → qkv.weight
    self_attn.in_proj_bias   → qkv.bias
    self_attn.out_proj.*     → proj.*
    linear1.*                → mlp.0.*
    linear2.*                → mlp.2.*
    """
    remap = {
        'self_attn.in_proj_weight': 'qkv.weight',
        'self_attn.in_proj_bias': 'qkv.bias',
        'self_attn.out_proj.weight': 'proj.weight',
        'self_attn.out_proj.bias': 'proj.bias',
        'linear1.weight': 'mlp.0.weight',
        'linear1.bias': 'mlp.0.bias',
        'linear2.weight': 'mlp.2.weight',
        'linear2.bias': 'mlp.2.bias',
    }
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith('encoder.blocks_'):
            for old_suffix, new_suffix in remap.items():
                if k.endswith(old_suffix):
                    new_k = k[:k.rfind(old_suffix)] + new_suffix
                    break
        new_sd[new_k] = v
    return new_sd


def load_model(checkpoint_path, device):
    """체크포인트 로드 — num_stages, APE/RoPE, legacy keys 자동 처리."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ck['model_state_dict']

    num_stages, use_ape, is_legacy = _detect_model_config(sd)
    epoch = ck.get('epoch', '?')
    pos_type = 'APE' if use_ape else 'RoPE'

    print(f"  config: stages={num_stages}, pos={pos_type}, "
          f"legacy={is_legacy}, epoch={epoch}")

    if is_legacy:
        sd = _remap_legacy_encoder_keys(sd)

    model = TwoStreamModel(depth=12, num_stages=num_stages)

    # APE: 현재 모델에 pos_embed 버퍼 등록 후 로드
    if use_ape:
        model.encoder.register_buffer('pos_embed_m', sd['encoder.pos_embed_m'])
        model.encoder.register_buffer('pos_embed_p', sd['encoder.pos_embed_p'])

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # freqs_cis는 buffer라 missing에 안 뜸, pos_embed은 APE면 이미 등록됨
    # 의미 있는 missing만 출력
    real_missing = [k for k in missing if 'freqs_cis' not in k]
    if real_missing:
        print(f"  ⚠ missing keys: {real_missing[:5]}")

    model.to(device).eval()
    return model, epoch, num_stages, use_ape


def rotate_frame(img_chw, k):
    """torch rot90: [C, H, W] → [C, H, W] counter-clockwise by k*90°."""
    return torch.rot90(img_chw, k=k // 90, dims=(-2, -1))


def _cls_attn_grid(attn_weights, patch_grid=14):
    """[1, N, N] CLS→patch attention → [patch_grid, patch_grid] numpy."""
    return attn_weights[0, 0, 1:].reshape(patch_grid, patch_grid).numpy()


def _pearson_corr(a, b):
    """Zero-mean normalized correlation. Uniform input/softmax 영향 제거."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_rotation_diagnostic(model, img_t_c, img_tk_c, model_label, epoch,
                            output_path, p_target='future'):
    """4×6 rotation diagnostic + 원본 좌표계 정렬 후 attn correlation.

    회전 후 attention map을 역회전하여 0° 좌표계로 정렬 → 0° attn과 Pearson corr 계산.
    - high corr (>0.8): content-driven (원본 content 위치 따라감)
    - low corr (<0.3): position-driven (회전된 이미지의 absolute position에 고정)
    """
    fig, axes = plt.subplots(4, 6, figsize=(30, 20))
    _p_tgt_label = {
        'future': 'frame_t+k',
        'current': 'frame_t',
        'residual': 'Recon frame_t+k',
    }[p_target]
    col_titles = ['Frame t (rot)', 'Frame t+k (rot)',
                  'M Attn on ΔL', 'P Attn on ΔL',
                  'Pred M', f'Pred P (→{_p_tgt_label})']

    # rotation별 attention grid 저장 (정렬 후 비교용)
    m_grids, p_grids = {}, {}

    for row, deg in enumerate(ROTATIONS):
        img_t_rot = rotate_frame(img_t_c, deg)
        img_tk_rot = rotate_frame(img_tk_c, deg)

        attn_maps, pred_m, pred_p, delta_l_rgb, pred_p_tensor = \
            extract_attention_and_predict(model, img_t_rot, img_tk_rot)

        img_t_vis = img_t_rot.permute(1, 2, 0).numpy()
        img_tk_vis = img_tk_rot.permute(1, 2, 0).numpy()
        m_hm = attn_to_heatmap(attn_maps['m'])
        p_hm = attn_to_heatmap(attn_maps['p'])

        m_grids[deg] = _cls_attn_grid(attn_maps['m'])
        p_grids[deg] = _cls_attn_grid(attn_maps['p'])

        # residual: decoder output(normalized) → reconstructed frame_{t+k}
        if p_target == 'residual':
            with torch.no_grad():
                it_b = img_t_rot.unsqueeze(0).to(DEVICE)
                itk_b = img_tk_rot.unsqueeze(0).to(DEVICE)
                recon = reconstruct_from_residual_pred(pred_p_tensor, it_b, itk_b)
            pred_p = recon.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

        axes[row][0].imshow(img_t_vis)
        axes[row][1].imshow(img_tk_vis)
        axes[row][2].imshow(delta_l_rgb)
        axes[row][2].imshow(m_hm, cmap='hot', alpha=0.5)
        axes[row][3].imshow(delta_l_rgb)
        axes[row][3].imshow(p_hm, cmap='hot', alpha=0.5)
        axes[row][4].imshow(pred_m)
        axes[row][5].imshow(pred_p)

        mse_m = ((img_tk_vis - pred_m) ** 2).mean()
        mse_p = ((img_tk_vis - pred_p) ** 2).mean()

        axes[row][0].set_ylabel(
            f'{deg}°',
            fontsize=14, rotation=0, labelpad=40, va='center', fontweight='bold',
        )
        axes[row][4].set_xlabel(f'MSE={mse_m:.4f}', fontsize=9)
        axes[row][5].set_xlabel(f'MSE={mse_p:.4f}', fontsize=9)

        for col in range(6):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    # 원본 좌표계 정렬 (rot90 역회전) 후 0°와 correlation
    m_ref, p_ref = m_grids[0], p_grids[0]
    corr_lines = []
    for deg in ROTATIONS:
        k_inv = -(deg // 90)
        m_aligned = np.rot90(m_grids[deg], k=k_inv)
        p_aligned = np.rot90(p_grids[deg], k=k_inv)
        cm = _pearson_corr(m_ref, m_aligned)
        cp = _pearson_corr(p_ref, p_aligned)
        corr_lines.append(f'{deg}°: corr(M)={cm:+.3f}, corr(P)={cp:+.3f}')
        # 각 row 하단에도 수치 표시
        axes[deg // 90][2].set_xlabel(f'corr(M vs 0°)={cm:+.3f}', fontsize=9)
        axes[deg // 90][3].set_xlabel(f'corr(P vs 0°)={cp:+.3f}', fontsize=9)

    for col in range(6):
        axes[0][col].set_title(col_titles[col], fontsize=12, fontweight='bold')

    print(f"  [corr vs 0° after inverse-rotating attention to original frame]")
    for line in corr_lines:
        print(f"    {line}")

    fig.suptitle(
        f'{model_label} (epoch {epoch}) — Rotation Diagnostic\n'
        f'content-driven ⇒ high corr after inverse-rotating attn to 0° frame. '
        + ' | '.join(corr_lines),
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model rotation diagnostic")
    parser.add_argument("--checkpoints", nargs='+', required=True,
                        help="체크포인트 경로 (여러 개 가능)")
    parser.add_argument("--labels", nargs='+', default=None,
                        help="모델 라벨 (--checkpoints와 같은 수)")
    parser.add_argument("--frame-t", required=True)
    parser.add_argument("--frame-tk", required=True)
    parser.add_argument("--output-dir", default="results/viz_rotation")
    parser.add_argument("--p-target", choices=['future', 'current', 'residual'],
                        default='future',
                        help='v9 P decoder target (Pred P 시각화 방식 결정)')
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [f"model_{i}" for i in range(len(args.checkpoints))]
    assert len(args.labels) == len(args.checkpoints)

    os.makedirs(args.output_dir, exist_ok=True)

    # 프레임 로드 + center crop 224
    img_t = load_image(args.frame_t)
    img_tk = load_image(args.frame_tk)
    crop = transforms.CenterCrop(224)
    img_t_c = crop(img_t.permute(2, 0, 1))
    img_tk_c = crop(img_tk.permute(2, 0, 1))

    sample_id = os.path.basename(os.path.dirname(args.frame_t))

    for ckpt_path, label in zip(args.checkpoints, args.labels):
        print(f"\n=== {label} ===")
        print(f"Loading: {ckpt_path}")
        model, epoch, num_stages, use_ape = load_model(ckpt_path, DEVICE)

        safe_label = label.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(
            args.output_dir, f"rotation_{safe_label}_{sample_id}.png")
        run_rotation_diagnostic(
            model, img_t_c, img_tk_c, label, epoch, output_path,
            p_target=args.p_target)


if __name__ == "__main__":
    main()
