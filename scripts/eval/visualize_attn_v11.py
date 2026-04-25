#!/usr/bin/env python3
"""Two-Stream v11 attention + reconstruction visualization.

v11은 v6/v10과 구조가 완전히 다름 (stream-independent M(6-layer)/P(12-layer) encoder +
3-phase P decoder + motion-routing). 기존 visualize_attn_compare.py는 v11에 사용 불가.

Grid (4 sample × 8 column):
  col 1: Frame t
  col 2: Frame t+k
  col 3: M encoder last-layer attention (CLS → patches)
  col 4: P encoder last-layer attention (CLS → patches)
  col 5: Motion-routing iter 0 attention (Q/K from M, averaged over heads,
           anchor patch = frame 중심)
  col 6: Motion-routing iter 1 attention (same anchor)
  col 7: Reconstruction t  (Phase 1 output, pred_t)
  col 8: Reconstruction t+k (Phase 3 output, pred_tk)

Attention 추출: PyTorch hook으로 manual QK softmax 재계산. 모델 forward 동작 변경 없음.
Motion-routing은 SDPA를 사용하지만 Q/K projection 결과를 hook에서 capture 후 직접 계산.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.two_stream_v11 import TwoStreamV11Model
from scripts.eval.visualize_attn_compare import find_high_change_pairs, load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Attention capture utilities
# ============================================================================

class AttentionCapture:
    """Forward pre-hook + custom forward로 attention weight capture.

    TransformerBlock은 F.scaled_dot_product_attention을 사용해 attention weight를
    바깥에서 직접 볼 수 없음. forward hook으로 block 입력을 가로챈 뒤, block의
    qkv/proj/norm 모듈을 재사용해 attention softmax를 수동 재계산.
    """

    def __init__(self):
        self.attn_weights = {}  # key → [B, N, N] (heads averaged)
        self.handles = []

    def _make_transformer_hook(self, key: str):
        """TransformerBlock용 hook. block 입력을 받아 attention 재계산."""
        def hook(block, inputs):
            x = inputs[0]  # [B, N, D]
            B, N, D = x.shape
            h = block.norm1(x)
            qkv = block.qkv(h).reshape(B, N, 3, block.num_heads, block.head_dim)
            q, k, _ = qkv.unbind(dim=2)
            # 2D RoPE는 freqs_cis=None이면 적용 안됨 (v11은 APE만 사용)
            q = q.transpose(1, 2)  # [B, H, N, D_head]
            k = k.transpose(1, 2)
            scale = block.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self.attn_weights[key] = attn.mean(dim=1).detach().cpu()  # [B, N, N]
        return hook

    def _make_motion_routing_hook(self, key: str):
        """MotionRoutingBlock용 hook.

        MotionRoutingBlock.forward(p_state, m_completed) → Q/K는 m_completed에서.
        block의 norm_m/qk_m을 재사용해 attention 재계산.
        """
        def hook(block, inputs):
            p_state, m_completed = inputs
            B, N, D = m_completed.shape
            m_normed = block.norm_m(m_completed)
            qk = block.qk_m(m_normed).reshape(B, N, 2, block.num_heads, block.head_dim)
            q, k = qk.unbind(dim=2)
            q = q.transpose(1, 2)  # [B, H, N, D_head]
            k = k.transpose(1, 2)
            scale = block.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self.attn_weights[key] = attn.mean(dim=1).detach().cpu()  # [B, N, N]
        return hook

    def register(self, model: TwoStreamV11Model):
        """M/P encoder 마지막 block + motion-routing 블록 전부에 hook 등록."""
        # M encoder last block
        last_m = model.blocks_m[-1]
        self.handles.append(last_m.register_forward_pre_hook(
            self._make_transformer_hook("m_encoder_last")
        ))
        # P encoder last block
        last_p = model.blocks_p[-1]
        self.handles.append(last_p.register_forward_pre_hook(
            self._make_transformer_hook("p_encoder_last")
        ))
        # Motion-routing blocks
        for i, mr_block in enumerate(model.motion_routing):
            self.handles.append(mr_block.register_forward_pre_hook(
                self._make_motion_routing_hook(f"motion_routing_{i}")
            ))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================================
# Visualization helpers
# ============================================================================

def cls_attn_to_heatmap(attn_weights: torch.Tensor, size: int = 224,
                        patch_grid: int = 14,
                        mask: torch.Tensor = None) -> np.ndarray:
    """[1, 1+N_vis, 1+N_vis] → [size, size] heatmap (CLS query → visible patches).

    v11 encoder는 visible-only (N_vis = (1-mask_ratio) × N). mask가 주어지면
    visible patch attention을 원래 spatial grid에 scatter, masked 위치는 0.
    mask=None이면 full sequence 가정 (N+1 tokens).
    """
    cls_row = attn_weights[0, 0, 1:].numpy()  # [N_vis] or [N]
    N = patch_grid * patch_grid

    if mask is not None:
        # mask: [1, N] bool, True=masked. visible = ~mask
        visible_mask = (~mask[0]).cpu().numpy()  # [N]
        full = np.zeros(N, dtype=np.float32)
        full[visible_mask] = cls_row
        grid = full.reshape(patch_grid, patch_grid)
    else:
        grid = cls_row.reshape(patch_grid, patch_grid)

    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    hm = np.array(
        Image.fromarray((grid * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR)
    ) / 255.0
    return hm


def anchor_attn_to_heatmap(attn_weights: torch.Tensor, anchor_idx: int,
                           size: int = 224, patch_grid: int = 14) -> np.ndarray:
    """[1, N+1, N+1] → [size, size] heatmap (anchor patch → patches).

    anchor_idx: 0-indexed patch position (0 ~ N-1). attn row index = anchor_idx+1
    (CLS가 row 0). column도 patches만 slice.
    """
    row = anchor_idx + 1  # CLS offset
    anchor_attn = attn_weights[0, row, 1:].numpy().reshape(patch_grid, patch_grid)
    anchor_attn = (anchor_attn - anchor_attn.min()) / (
        anchor_attn.max() - anchor_attn.min() + 1e-8
    )
    hm = np.array(
        Image.fromarray((anchor_attn * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR)
    ) / 255.0
    return hm


def pred_to_image(pred_tensor: torch.Tensor) -> np.ndarray:
    """[1, 3, H, W] → [H, W, 3] numpy in [0,1]."""
    return pred_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)


# ============================================================================
# Main
# ============================================================================

def extract_v11_attention_and_predict(model: TwoStreamV11Model,
                                       img_t_tensor: torch.Tensor,
                                       img_tk_tensor: torch.Tensor,
                                       anchor_idx: int,
                                       patch_grid: int = 14):
    """v11 forward + attention hooks. Returns dict of arrays for plotting."""
    x = img_t_tensor.unsqueeze(0).to(DEVICE)
    y = img_tk_tensor.unsqueeze(0).to(DEVICE)

    capture = AttentionCapture()
    capture.register(model)

    try:
        # v11 model.training=False 이므로 mask 적용되지만 random — seed 고정으로
        # 시드별 재현. 여기서는 viz 시점 이미지가 의미있으면 OK (spec 제약).
        torch.manual_seed(0)
        with torch.no_grad():
            out = model(x, y)
    finally:
        capture.remove()

    # Encoder attention은 visible-only → mask로 spatial grid 복원
    m_attn_hm = cls_attn_to_heatmap(capture.attn_weights["m_encoder_last"],
                                     patch_grid=patch_grid, mask=out["mask_m"])
    p_attn_hm = cls_attn_to_heatmap(capture.attn_weights["p_encoder_last"],
                                     patch_grid=patch_grid, mask=out["mask_p"])

    mr_hms = []
    for i in range(len(model.motion_routing)):
        key = f"motion_routing_{i}"
        if key in capture.attn_weights:
            mr_hms.append(anchor_attn_to_heatmap(
                capture.attn_weights[key], anchor_idx, patch_grid=patch_grid,
            ))
        else:
            mr_hms.append(np.zeros((224, 224)))

    pred_t_img = pred_to_image(out["pred_t"])
    pred_tk_img = pred_to_image(out["pred_tk"])

    return {
        "m_attn": m_attn_hm,
        "p_attn": p_attn_hm,
        "mr_attns": mr_hms,
        "pred_t": pred_t_img,
        "pred_tk": pred_tk_img,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egodex-dir", required=True)
    parser.add_argument("--droid-dir", required=True)
    parser.add_argument("--output", default="results/attn_v11.png")
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-depth", type=int, default=12)
    parser.add_argument("--m-depth", type=int, default=6)
    parser.add_argument("--num-motion-iters", type=int, default=2)
    # Viz는 mask=0으로 (downstream inference와 일치, deterministic)
    parser.add_argument("--mask-ratio-m", type=float, default=0.0)
    parser.add_argument("--mask-ratio-p", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load v11 model
    print(f"Loading v11 checkpoint: {args.checkpoint}")
    model = TwoStreamV11Model(
        p_depth=args.p_depth, m_depth=args.m_depth,
        num_motion_iters=args.num_motion_iters,
        mask_ratio_m=args.mask_ratio_m, mask_ratio_p=args.mask_ratio_p,
    ).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # DDP "module." prefix 제거
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                      for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  WARNING: {len(result.missing_keys)} missing keys "
              f"(first 3: {result.missing_keys[:3]})")
    if result.unexpected_keys:
        print(f"  WARNING: {len(result.unexpected_keys)} unexpected keys "
              f"(first 3: {result.unexpected_keys[:3]})")
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Epoch {epoch}, p_depth={args.p_depth}, m_depth={args.m_depth}")

    # Sample selection (deterministic — same pattern as compare script)
    print(f"\nSelecting high-change pairs from EgoDex...")
    egodex_pairs = find_high_change_pairs(args.egodex_dir, 30, 2, args.gap, args.seed)
    print(f"  Selected: {[p[2] for p in egodex_pairs]} "
          f"(MSE: {[f'{p[4]:.4f}' for p in egodex_pairs]})")

    print(f"Selecting high-change pairs from DROID...")
    droid_pairs = find_high_change_pairs(args.droid_dir, 30, 2, args.gap,
                                          args.seed + 1)
    print(f"  Selected: {[p[2] for p in droid_pairs]} "
          f"(MSE: {[f'{p[4]:.4f}' for p in droid_pairs]})")

    all_pairs = egodex_pairs + droid_pairs
    labels = ["EgoDex"] * 2 + ["DROID"] * 2

    # Motion-routing anchor patch: 이미지 중심 (14×14 grid의 (7, 7) → idx=7*14+7=105)
    patch_grid = 14
    anchor_idx = (patch_grid // 2) * patch_grid + (patch_grid // 2)

    # Figure: 4 rows × 8 cols
    fig, axes = plt.subplots(4, 8, figsize=(40, 20))
    col_titles = [
        "Frame t",
        f"Frame t+{args.gap}",
        "M encoder attn (CLS→patch)",
        "P encoder attn (CLS→patch)",
        "Motion-routing 0 (anchor)",
        "Motion-routing 1 (anchor)",
        "Recon t (Phase 1)",
        f"Recon t+{args.gap} (Phase 3)",
    ]

    for row, ((path_t, path_tk, ep_label, gap, mse), dataset) in enumerate(
            zip(all_pairs, labels)):
        img_t = load_image(path_t)
        img_tk = load_image(path_tk)

        crop = transforms.CenterCrop(224)
        img_t_c = crop(img_t.permute(2, 0, 1)).permute(1, 2, 0)
        img_tk_c = crop(img_tk.permute(2, 0, 1)).permute(1, 2, 0)

        img_t_tensor = img_t_c.permute(2, 0, 1)
        img_tk_tensor = img_tk_c.permute(2, 0, 1)

        result = extract_v11_attention_and_predict(
            model, img_t_tensor, img_tk_tensor, anchor_idx, patch_grid=patch_grid,
        )

        img_t_np = img_t_c.numpy()
        img_tk_np = img_tk_c.numpy()

        # Col 0: Frame t
        axes[row][0].imshow(img_t_np)
        # Col 1: Frame t+k
        axes[row][1].imshow(img_tk_np)
        # Col 2: M encoder attn overlay on frame t
        axes[row][2].imshow(img_t_np)
        axes[row][2].imshow(result["m_attn"], cmap="viridis", alpha=0.55)
        # Col 3: P encoder attn overlay on frame t+k
        axes[row][3].imshow(img_tk_np)
        axes[row][3].imshow(result["p_attn"], cmap="viridis", alpha=0.55)
        # Col 4-5: Motion-routing attention overlays (on frame t+k, anchor marker)
        for mi, mr_hm in enumerate(result["mr_attns"][:2]):
            ax = axes[row][4 + mi]
            ax.imshow(img_tk_np)
            ax.imshow(mr_hm, cmap="viridis", alpha=0.55)
            # Mark anchor patch center
            ar, ac = anchor_idx // patch_grid, anchor_idx % patch_grid
            ax.scatter([ac * 16 + 8], [ar * 16 + 8], s=40, c="red",
                       marker="x", linewidths=2)
        # Col 6: Pred t
        axes[row][6].imshow(result["pred_t"])
        # Col 7: Pred t+k
        axes[row][7].imshow(result["pred_tk"])

        mse_pt = ((img_t_np - result["pred_t"]) ** 2).mean()
        mse_ptk = ((img_tk_np - result["pred_tk"]) ** 2).mean()

        axes[row][0].set_ylabel(
            f"{dataset}\n{ep_label[:25]}\nΔ={mse:.3f}",
            fontsize=9, rotation=0, labelpad=80, va="center",
        )
        axes[row][6].set_xlabel(f"MSE={mse_pt:.4f}", fontsize=8)
        axes[row][7].set_xlabel(f"MSE={mse_ptk:.4f}", fontsize=8)

        for col in range(8):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for col in range(8):
        axes[0][col].set_title(col_titles[col], fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Two-Stream v11 Epoch {epoch} — Attention + Reconstruction "
        f"(gap={args.gap}, motion_iters={args.num_motion_iters})",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
