"""DINOv2-style PCA overlay 가시화 (project page용).

Frozen encoder의 patch features를 PCA로 3D 축소 → RGB로 매핑 → input frame에
alpha-blend overlay. 비슷한 의미·motion의 patch가 비슷한 색으로 보임.

가시화 대상 3 encoder 비교:
- v15 (P encoder, paper main)
- DINOv2 base (internet-scale SSL 대비군)
- VideoMAE-ours (same corpus, no scaffolding 대비군)

입력 source:
- LIBERO demo HDF5 (prototype, agentview_rgb)
- 또는 numpy frame stack (.npy, rollout video 적용 시)

산출:
- per-encoder GIF (overlay 시간 progression)
- side-by-side GIF (raw + 3 encoder)
- static comparison PNG (sample frame 3개 × 4 panel)

Plan §3 가드:
- PCA는 전체 video patches에서 한 번만 fit → frame 간 색 일관성
- encoder별 별도 PCA (feature space 다름)
- CLS token 제외, patch only
- percentile clip (1~99%) 정규화 → outlier 강건
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

IMG_SIZE = 224
DEFAULT_V15_CKPT = "/proj/external_group/mrg/checkpoints/two_stream_v15/20260511_045319/latest.pt"
DEFAULT_VIDEOMAE_CKPT = "/proj/external_group/mrg/checkpoints/videomae/20260415_012017/best_model.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Frame source
# ─────────────────────────────────────────────────────────────────────────────

def load_frames_libero(hdf5_path: str, demo_key: str, view: str = "agentview_rgb") -> np.ndarray:
    """(T, H, W, 3) uint8."""
    with h5py.File(hdf5_path, "r") as f:
        return np.asarray(f[f"data/{demo_key}/obs/{view}"])


def load_frames_npy(npy_path: str) -> np.ndarray:
    """(T, H, W, 3) uint8."""
    arr = np.load(npy_path)
    if arr.dtype != np.uint8:
        raise ValueError(f"expected uint8 frame stack, got {arr.dtype}")
    return arr


def preprocess_frames(frames_uint8: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (T, 3, img_size, img_size) [0,1] float."""
    x = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Encoder별 patch feature 추출
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def patches_v15(ckpt_path: str, frames: torch.Tensor, device: torch.device, batch: int = 16):
    """v15 P encoder (single-frame mode: preprocessing(f, f))로 patch features 추출.

    Returns (T, N_patch, D) — CLS 제외, 14×14=196 patches.
    """
    from scripts.eval.probe_action_v11 import load_v11_model, _p_encoder_forward
    model = load_v11_model(ckpt_path, p_depth=12, m_depth=6, device=str(device))
    out = []
    for s in range(0, frames.shape[0], batch):
        x = frames[s:s + batch].to(device)
        _, p_channel = model.preprocessing(x, x)
        feat = _p_encoder_forward(model, p_channel)  # (B, 1+N, D)
        out.append(feat[:, 1:].cpu())                # drop CLS
    return torch.cat(out, dim=0), (14, 14)


@torch.no_grad()
def patches_dinov2(frames: torch.Tensor, device: torch.device, batch: int = 16):
    """DINOv2-base via SingleFrameAdapter (HF cache 통과, 자체 normalize buffer).

    patch=14, 224 input → 16×16=256 patches.
    """
    from src.encoders.adapters.single_frame import SingleFrameAdapter
    adapter = SingleFrameAdapter("dinov2").to(device).eval()
    out = []
    for s in range(0, frames.shape[0], batch):
        x = frames[s:s + batch].to(device)
        x_n = adapter._normalize(x)
        feat = adapter.model(pixel_values=x_n).last_hidden_state  # (B, 1+N, D)
        out.append(feat[:, 1:].cpu())                              # drop CLS
    return torch.cat(out, dim=0), (16, 16)


@torch.no_grad()
def patches_videomae(ckpt_path: str, frames: torch.Tensor, device: torch.device, batch: int = 16):
    """VideoMAE-ours patch features (CLS 없는 모델).

    VideoMAEEncoderForVLA.forward(pixel_values: (B,6,H,W)) → (B, 196, 768).
    single-frame mode는 same-frame 6-ch replica로 입력 (probe_action에서 동일 패턴).
    """
    from src.encoders.adapters.videomae import VideoMAEOursAdapter
    adapter = VideoMAEOursAdapter(checkpoint_path=ckpt_path, mode="paired").to(device).eval()
    out = []
    for s in range(0, frames.shape[0], batch):
        x = frames[s:s + batch].to(device)
        pv = torch.cat([x, x], dim=1)  # (B, 6, 224, 224)
        patches = adapter.model(pv)    # (B, 196, 768) — CLS 없음
        out.append(patches.cpu())
    return torch.cat(out, dim=0), (14, 14)


# ─────────────────────────────────────────────────────────────────────────────
# PCA + overlay
# ─────────────────────────────────────────────────────────────────────────────

def fit_pca_to_rgb(patches: torch.Tensor, grid_shape: tuple) -> np.ndarray:
    """patches: (T, N_patch, D) → rgb_per_frame: (T, Hp, Wp, 3) [0,1].

    - 전체 frame patches 모아 PCA fit (frame 간 색 일관성)
    - PCA = mean center + SVD top-3 components (numpy 기반, sklearn 의존성 X)
    - 1~99 percentile clip + min-max → outlier 강건
    """
    T, N, D = patches.shape
    Hp, Wp = grid_shape
    assert N == Hp * Wp, f"patch count {N} != grid {Hp}×{Wp}"

    flat = patches.reshape(T * N, D).numpy().astype(np.float32)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    # SVD: centered = U S Vt; eigenvalues of covariance ∝ S² / (n-1)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3]                   # (3, D)
    proj = centered @ components.T        # (T*N, 3)

    # explained variance ratio (top-3 / total)
    eigvals = (sv ** 2) / max(centered.shape[0] - 1, 1)
    evr = eigvals / eigvals.sum()
    print(f"[viz]   explained var top3 = {evr[0]:.3f}, {evr[1]:.3f}, {evr[2]:.3f}  "
          f"(cum {evr[:3].sum():.3f})")

    # 각 component별 1~99 percentile clip → min-max
    for c in range(3):
        lo, hi = np.percentile(proj[:, c], [1, 99])
        proj[:, c] = np.clip(proj[:, c], lo, hi)
        rng = hi - lo if hi > lo else 1.0
        proj[:, c] = (proj[:, c] - lo) / rng

    rgb = proj.reshape(T, Hp, Wp, 3).astype(np.float32)
    return rgb, evr[:3]


def overlay_rgb_on_frame(frame_uint8: np.ndarray, rgb_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """frame_uint8: (H, W, 3) uint8, rgb_map: (Hp, Wp, 3) [0,1] float → (H, W, 3) uint8."""
    h, w = frame_uint8.shape[:2]
    rgb_resized = np.asarray(
        Image.fromarray((rgb_map * 255).astype(np.uint8)).resize((w, h), Image.NEAREST),
        dtype=np.float32,
    ) / 255.0
    blended = (1 - alpha) * (frame_uint8.astype(np.float32) / 255.0) + alpha * rgb_resized
    return (blended * 255).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# GIF / PNG 출력
# ─────────────────────────────────────────────────────────────────────────────

def write_gif(frames_uint8_list: list, out_path: Path, fps: int = 10):
    """frames_uint8_list: list of (H, W, 3) uint8 → GIF (PIL save_all)."""
    pil_frames = [Image.fromarray(f) for f in frames_uint8_list]
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )


def make_side_by_side(frames_dict: dict, evr_dict: dict = None, fps: int = 10) -> list:
    """frames_dict: {'raw': [...], 'v15': [...], ...} → side-by-side concat per frame.

    Order: raw | v15 | dinov2 | videomae-ours (encoder별 PC1-3 explained variance %).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["raw", "v15", "dinov2", "videomae-ours"]
    T = len(frames_dict["raw"])
    composites = []
    for t in range(T):
        fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.5))
        for ax, k in zip(axes, keys):
            ax.imshow(frames_dict[k][t])
            if k == "raw" or evr_dict is None:
                title = k
            else:
                e = evr_dict[k]
                title = f"{k}\nPC1-3 var: {e[0]*100:.0f}/{e[1]*100:.0f}/{e[2]*100:.0f}%"
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(f"frame {t}", y=0.02, fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        composites.append(buf[..., :3].copy())
        plt.close(fig)
    return composites


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["libero", "npy"])
    ap.add_argument("--hdf5", default=None, help="LIBERO HDF5 path (source=libero)")
    ap.add_argument("--demo-key", default="demo_0")
    ap.add_argument("--view", default="agentview_rgb")
    ap.add_argument("--npy-path", default=None, help="frame stack .npy (source=npy)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--v15-ckpt", default=DEFAULT_V15_CKPT)
    ap.add_argument("--videomae-ckpt", default=DEFAULT_VIDEOMAE_CKPT)
    ap.add_argument("--encoders", nargs="+", default=["v15", "dinov2", "videomae-ours"])
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None, help="prototype용 frame 제한")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] device={device}, out_dir={out_dir}")

    # 1) Load frames
    if args.source == "libero":
        if args.hdf5 is None:
            raise SystemExit("--hdf5 required for source=libero")
        frames_uint8 = load_frames_libero(args.hdf5, args.demo_key, args.view)
    else:
        if args.npy_path is None:
            raise SystemExit("--npy-path required for source=npy")
        frames_uint8 = load_frames_npy(args.npy_path)
    if args.max_frames is not None:
        frames_uint8 = frames_uint8[: args.max_frames]
    T, H, W, _ = frames_uint8.shape
    print(f"[viz] frames: T={T}, HxW={H}x{W}")
    frames_tensor = preprocess_frames(frames_uint8)

    # 2) per-encoder patch feature → PCA → RGB
    per_enc_rgb = {}
    per_enc_evr = {}
    for enc in args.encoders:
        print(f"[viz] === {enc}: extract patches ===")
        if enc == "v15":
            patches, grid = patches_v15(args.v15_ckpt, frames_tensor, device)
        elif enc == "dinov2":
            patches, grid = patches_dinov2(frames_tensor, device)
        elif enc == "videomae-ours":
            patches, grid = patches_videomae(args.videomae_ckpt, frames_tensor, device)
        else:
            raise ValueError(f"unknown encoder: {enc}")
        print(f"[viz]   patches shape={tuple(patches.shape)}, grid={grid}")
        rgb_per_frame, evr = fit_pca_to_rgb(patches, grid)
        per_enc_rgb[enc] = rgb_per_frame
        per_enc_evr[enc] = evr

    # 3) per-encoder overlay GIF
    frames_overlaid = {"raw": [f for f in frames_uint8]}
    for enc in args.encoders:
        overlays = [overlay_rgb_on_frame(frames_uint8[t], per_enc_rgb[enc][t], args.alpha) for t in range(T)]
        frames_overlaid[enc] = overlays
        gp = out_dir / f"overlay_{enc}.gif"
        write_gif(overlays, gp, fps=args.fps)
        print(f"[viz] saved {gp}")

    # 4) side-by-side GIF
    print("[viz] composing side-by-side ...")
    composites = make_side_by_side(frames_overlaid, evr_dict=per_enc_evr, fps=args.fps)
    sbs = out_dir / "side_by_side.gif"
    write_gif(composites, sbs, fps=args.fps)
    print(f"[viz] saved {sbs}")

    # 5) static grid (3 sample frame × 4 panel)
    sample_idx = [T // 6, T // 2, T * 5 // 6]
    static_frames = [composites[i] for i in sample_idx]
    static = out_dir / "static_grid.png"
    Image.fromarray(np.concatenate(static_frames, axis=0)).save(static)
    print(f"[viz] saved {static}")

    print("[viz] done.")


if __name__ == "__main__":
    main()
