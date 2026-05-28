"""DINOv2-style PCA overlay 가시화 (project page용).

Frozen encoder의 patch features를 PCA로 3D 축소 → RGB로 매핑 → input frame에
alpha-blend overlay. 비슷한 의미·motion의 patch가 비슷한 색으로 보임.

가시화 대상 5 encoder 비교 (paper main lineup):
- ours      (= Two-Stream v15, P encoder)
- DINOv2 base
- SigLIP base
- VC-1 base
- VideoMAE-ours

입력 source (BC 수행 안 한 데이터셋 = demo / BC 수행 데이터셋 = rollout video frame .npy):
- libero   : LIBERO HDF5 demo (agentview_rgb)
- egodex   : EgoDex pretraining frames (jpg sequence in task dir)
- droid    : DROID frames (jpg sequence in episode dir)
- calvin   : CALVIN episode_*.npz (rgb_static)
- npy      : .npy frame stack (T, H, W, 3) uint8 — 로컬 rollout video 적용용

산출:
- per-encoder GIF
- side-by-side GIF (raw + 5 encoder, 1 row × 6 col)
- static comparison PNG (3 sample frame)

변경 이력 (단일 파일 정책):
- v1 (2026-05-27): 3 encoder, LIBERO HDF5만
- v2 (2026-05-27): alpha=0.4 + explained variance 출력
- v3 (2026-05-27):
  · 5 encoder (+ SigLIP, VC-1)
  · 다중 dataset (egodex/droid/calvin/libero/npy)
  · 라벨 "v15" → "ours"
  · 6 panel side-by-side
  · VC-1 patches는 ViT forward 재현 (forward_features는 CLS만 반환)
- v4 (2026-05-27, 본 버전): encoder overlay panel만 grayscale base 적용
  (이유: 원본 frame RGB와 PCA RGB가 겹쳐 색 혼동 — 사용자 요청).
  raw panel은 RGB 그대로 보존.
"""
from __future__ import annotations

import argparse
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
# Frame source loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_frames_libero(hdf5_path: str, demo_key: str, view: str = "agentview_rgb") -> np.ndarray:
    """(T, H, W, 3) uint8."""
    with h5py.File(hdf5_path, "r") as f:
        return np.asarray(f[f"data/{demo_key}/obs/{view}"])


def load_frames_npy(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.dtype != np.uint8:
        raise ValueError(f"expected uint8 frame stack, got {arr.dtype}")
    return arr


def load_frames_egodex(task_dir: str, max_frames: int = 60, stride: int = 5) -> np.ndarray:
    """EgoDex task directory에서 frame_*.jpg 시퀀스 → (T, H, W, 3) uint8.

    task_dir 예: /proj/external_group/mrg/datasets/egodex/frames/part1/<task>/<uuid>/
    또는 1단계 위 디렉토리에서 첫 uuid 자동 선택.
    """
    p = Path(task_dir)
    # 자동 탐색: jpg 직접 또는 하위 uuid 디렉토리
    jpgs = sorted(p.glob("*.jpg"))
    if not jpgs:
        # 하위 uuid 디렉토리 시도
        sub = next((d for d in sorted(p.iterdir()) if d.is_dir()), None)
        if sub is not None:
            jpgs = sorted(sub.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"no jpg frames under {task_dir}")
    jpgs = jpgs[::stride][:max_frames]
    frames = [np.asarray(Image.open(j).convert("RGB")) for j in jpgs]
    return np.stack(frames, axis=0)


def load_frames_droid(episode_dir: str, max_frames: int = 60, stride: int = 5) -> np.ndarray:
    """DROID episode directory (frame_NNNNNN.jpg) → (T, H, W, 3) uint8."""
    return load_frames_egodex(episode_dir, max_frames=max_frames, stride=stride)  # 동일 jpg 시퀀스


def load_frames_calvin(split_dir: str, start_id: int, end_id: int,
                       view: str = "rgb_static", stride: int = 2,
                       max_frames: int = 60) -> np.ndarray:
    """CALVIN episode_NNNNNNN.npz 시퀀스 → (T, H, W, 3) uint8.

    split_dir 예: /proj/external_group/mrg/datasets/calvin/task_ABCD_D/validation/
    (start_id, end_id): CALVIN segment 또는 episode 범위.
    """
    p = Path(split_dir)
    ids = list(range(start_id, end_id + 1, stride))[:max_frames]
    frames = []
    for fid in ids:
        with np.load(p / f"episode_{fid:07d}.npz") as f:
            frames.append(np.asarray(f[view]))
    return np.stack(frames, axis=0)


def preprocess_frames(frames_uint8: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    x = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Encoder별 patch feature 추출 (5 encoder)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def patches_ours(ckpt_path: str, frames: torch.Tensor, device: torch.device, batch: int = 16):
    """Two-Stream v15 P encoder (= "ours"), single-frame mode. CLS 제외, 14×14 patches."""
    from scripts.eval.probe_action_v11 import load_v11_model, _p_encoder_forward
    model = load_v11_model(ckpt_path, p_depth=12, m_depth=6, device=str(device))
    out = []
    for s in range(0, frames.shape[0], batch):
        x = frames[s:s + batch].to(device)
        _, p_channel = model.preprocessing(x, x)
        feat = _p_encoder_forward(model, p_channel)
        out.append(feat[:, 1:].cpu())
    return torch.cat(out, dim=0), (14, 14)


@torch.no_grad()
def patches_dinov2(frames: torch.Tensor, device: torch.device, batch: int = 16):
    """DINOv2-base. patch=14, 224 input → 16×16=256 patches."""
    from src.encoders.adapters.single_frame import SingleFrameAdapter
    adapter = SingleFrameAdapter("dinov2").to(device).eval()
    out = []
    for s in range(0, frames.shape[0], batch):
        x = frames[s:s + batch].to(device)
        x_n = adapter._normalize(x)
        feat = adapter.model(pixel_values=x_n).last_hidden_state
        out.append(feat[:, 1:].cpu())  # drop CLS
    return torch.cat(out, dim=0), (16, 16)


@torch.no_grad()
def patches_siglip(frames: torch.Tensor, device: torch.device, batch: int = 16):
    """SigLIP base patch16-224. last_hidden_state는 CLS 없음 (196 patches × 768)."""
    from src.encoders.adapters.single_frame import SingleFrameAdapter
    adapter = SingleFrameAdapter("siglip").to(device).eval()
    out = []
    for s in range(0, frames.shape[0], batch):
        x = frames[s:s + batch].to(device)
        x_n = adapter._normalize(x)
        feat = adapter.model(pixel_values=x_n).last_hidden_state  # (B, 196, 768) — no CLS
        out.append(feat.cpu())
    return torch.cat(out, dim=0), (14, 14)


@torch.no_grad()
def patches_vc1(frames: torch.Tensor, device: torch.device, batch: int = 16):
    """VC-1 (eai-vc). VC-1.forward_features는 CLS pool 후 (B, D)만 반환 →
    표준 timm ViT 단계 재현 (patch_embed → +cls+pos → blocks → norm) → CLS drop.
    """
    from vc_models.models.vit import model_utils
    model, _, _, _ = model_utils.load_model(model_utils.VC1_BASE_NAME)
    model = model.to(device).eval()
    # ImageNet normalize (single_frame.py vc1 컨벤션과 동일)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    out = []
    for s in range(0, frames.shape[0], batch):
        x = frames[s:s + batch].to(device)
        x_n = (x - mean) / std
        # timm ViT forward 재현
        tok = model.patch_embed(x_n)                                 # (B, N, D)
        B = tok.shape[0]
        cls = model.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1) + model.pos_embed         # (B, 1+N, D)
        for blk in model.blocks:
            tok = blk(tok)
        tok = model.norm(tok)
        out.append(tok[:, 1:].cpu())                                 # drop CLS
    return torch.cat(out, dim=0), (14, 14)


@torch.no_grad()
def patches_videomae(ckpt_path: str, frames: torch.Tensor, device: torch.device, batch: int = 16):
    """VideoMAE-ours: VideoMAEEncoderForVLA.forward(pixel_values=(B,6,H,W)) → (B, 196, 768).
    single-frame mode = same-frame 6-channel replica.
    """
    from src.encoders.adapters.videomae import VideoMAEOursAdapter
    adapter = VideoMAEOursAdapter(checkpoint_path=ckpt_path, mode="paired").to(device).eval()
    out = []
    for s in range(0, frames.shape[0], batch):
        x = frames[s:s + batch].to(device)
        pv = torch.cat([x, x], dim=1)
        patches = adapter.model(pv)                                  # (B, 196, 768), no CLS
        out.append(patches.cpu())
    return torch.cat(out, dim=0), (14, 14)


# ─────────────────────────────────────────────────────────────────────────────
# PCA + overlay
# ─────────────────────────────────────────────────────────────────────────────

def fit_pca_to_rgb(patches: torch.Tensor, grid_shape: tuple):
    """(T, N, D) → (T, Hp, Wp, 3) [0,1] + (3,) explained variance ratio.

    - 전체 video patches에서 한 번만 PCA fit (frame 간 색 일관성)
    - SVD top-3 directions, mean-center
    - 1~99 percentile clip + min-max → outlier 강건
    """
    T, N, D = patches.shape
    Hp, Wp = grid_shape
    assert N == Hp * Wp, f"patch count {N} != grid {Hp}×{Wp}"

    flat = patches.reshape(T * N, D).numpy().astype(np.float32)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3]                       # (3, D)
    proj = centered @ components.T            # (T*N, 3)

    eigvals = (sv ** 2) / max(centered.shape[0] - 1, 1)
    evr = eigvals / eigvals.sum()
    print(f"[viz]   explained var top3 = {evr[0]:.3f}, {evr[1]:.3f}, {evr[2]:.3f}  "
          f"(cum {evr[:3].sum():.3f})")

    for c in range(3):
        lo, hi = np.percentile(proj[:, c], [1, 99])
        proj[:, c] = np.clip(proj[:, c], lo, hi)
        rng = hi - lo if hi > lo else 1.0
        proj[:, c] = (proj[:, c] - lo) / rng

    return proj.reshape(T, Hp, Wp, 3).astype(np.float32), evr[:3]


def overlay_rgb_on_frame(frame_uint8: np.ndarray, rgb_map: np.ndarray,
                         alpha: float = 0.4, grayscale_base: bool = True) -> np.ndarray:
    """grayscale_base=True: 원본 frame을 luminance grayscale로 변환 후 PCA RGB overlay
    → 원본 색과 PCA 색 혼동 방지. raw panel에는 grayscale_base=False로 호출."""
    h, w = frame_uint8.shape[:2]
    rgb_resized = np.asarray(
        Image.fromarray((rgb_map * 255).astype(np.uint8)).resize((w, h), Image.NEAREST),
        dtype=np.float32,
    ) / 255.0
    if grayscale_base:
        gray = (0.299 * frame_uint8[..., 0] + 0.587 * frame_uint8[..., 1]
                + 0.114 * frame_uint8[..., 2])
        base = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32) / 255.0
    else:
        base = frame_uint8.astype(np.float32) / 255.0
    blended = (1 - alpha) * base + alpha * rgb_resized
    return (blended * 255).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# GIF / PNG
# ─────────────────────────────────────────────────────────────────────────────

def write_gif(frames_uint8_list: list, out_path: Path, fps: int = 10):
    pil_frames = [Image.fromarray(f) for f in frames_uint8_list]
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )


def make_side_by_side(frames_dict: dict, evr_dict: dict, encoder_order: list) -> list:
    """frames_dict: {'raw': [...], 'ours': [...], 'dinov2': [...], ...} → 1×N panel per frame.

    encoder_order: keys (raw 제외)의 표시 순서.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["raw"] + list(encoder_order)
    T = len(frames_dict["raw"])
    composites = []
    for t in range(T):
        fig, axes = plt.subplots(1, len(keys), figsize=(2.5 * len(keys), 3.0))
        if len(keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, keys):
            ax.imshow(frames_dict[k][t])
            if k == "raw" or evr_dict.get(k) is None:
                title = k
            else:
                e = evr_dict[k]
                title = f"{k}\nPC1-3: {e[0]*100:.0f}/{e[1]*100:.0f}/{e[2]*100:.0f}%"
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"frame {t}", y=0.03, fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        composites.append(buf[..., :3].copy())
        plt.close(fig)
    return composites


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ENCODER_BUILDERS = {
    "ours":          lambda fr, dev, ckpt_v15, ckpt_vmae: patches_ours(ckpt_v15, fr, dev),
    "dinov2":        lambda fr, dev, ckpt_v15, ckpt_vmae: patches_dinov2(fr, dev),
    "siglip":        lambda fr, dev, ckpt_v15, ckpt_vmae: patches_siglip(fr, dev),
    "vc1":           lambda fr, dev, ckpt_v15, ckpt_vmae: patches_vc1(fr, dev),
    "videomae-ours": lambda fr, dev, ckpt_v15, ckpt_vmae: patches_videomae(ckpt_vmae, fr, dev),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["libero", "npy", "egodex", "droid", "calvin"])
    # 공통
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoders", nargs="+",
                    default=["ours", "dinov2", "siglip", "vc1", "videomae-ours"])
    ap.add_argument("--v15-ckpt", default=DEFAULT_V15_CKPT)
    ap.add_argument("--videomae-ckpt", default=DEFAULT_VIDEOMAE_CKPT)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.40)
    ap.add_argument("--device", default="cuda")
    # source별
    ap.add_argument("--hdf5", default=None, help="LIBERO HDF5 (source=libero)")
    ap.add_argument("--demo-key", default="demo_0", help="LIBERO demo key")
    ap.add_argument("--view", default="agentview_rgb")
    ap.add_argument("--npy-path", default=None, help="frame stack .npy (source=npy)")
    ap.add_argument("--task-dir", default=None, help="EgoDex task dir (source=egodex)")
    ap.add_argument("--episode-dir", default=None, help="DROID episode dir (source=droid)")
    ap.add_argument("--calvin-split", default=None, help="CALVIN training/ or validation/ (source=calvin)")
    ap.add_argument("--calvin-start", type=int, default=0, help="CALVIN segment start id")
    ap.add_argument("--calvin-end", type=int, default=120, help="CALVIN segment end id")
    ap.add_argument("--frame-stride", type=int, default=5,
                    help="EgoDex/DROID/CALVIN frame downsample stride")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] device={device}, out_dir={out_dir}, source={args.source}")

    # 1) Load frames
    if args.source == "libero":
        if args.hdf5 is None: raise SystemExit("--hdf5 required for source=libero")
        frames_uint8 = load_frames_libero(args.hdf5, args.demo_key, args.view)
    elif args.source == "npy":
        if args.npy_path is None: raise SystemExit("--npy-path required for source=npy")
        frames_uint8 = load_frames_npy(args.npy_path)
    elif args.source == "egodex":
        if args.task_dir is None: raise SystemExit("--task-dir required for source=egodex")
        frames_uint8 = load_frames_egodex(args.task_dir,
                                          max_frames=args.max_frames or 60,
                                          stride=args.frame_stride)
    elif args.source == "droid":
        if args.episode_dir is None: raise SystemExit("--episode-dir required for source=droid")
        frames_uint8 = load_frames_droid(args.episode_dir,
                                         max_frames=args.max_frames or 60,
                                         stride=args.frame_stride)
    elif args.source == "calvin":
        if args.calvin_split is None: raise SystemExit("--calvin-split required for source=calvin")
        frames_uint8 = load_frames_calvin(args.calvin_split, args.calvin_start, args.calvin_end,
                                          stride=args.frame_stride,
                                          max_frames=args.max_frames or 60)
    if args.max_frames is not None and args.source in ("libero", "npy"):
        frames_uint8 = frames_uint8[: args.max_frames]
    T, H, W, _ = frames_uint8.shape
    print(f"[viz] frames: T={T}, HxW={H}x{W}")
    frames_tensor = preprocess_frames(frames_uint8)

    # 2) per-encoder patch feature → PCA → RGB
    per_enc_rgb = {}
    per_enc_evr = {}
    for enc in args.encoders:
        if enc not in ENCODER_BUILDERS:
            raise ValueError(f"unknown encoder: {enc}")
        print(f"[viz] === {enc}: extract patches ===")
        patches, grid = ENCODER_BUILDERS[enc](
            frames_tensor, device, args.v15_ckpt, args.videomae_ckpt,
        )
        print(f"[viz]   patches shape={tuple(patches.shape)}, grid={grid}")
        rgb_per_frame, evr = fit_pca_to_rgb(patches, grid)
        per_enc_rgb[enc] = rgb_per_frame
        per_enc_evr[enc] = evr

    # 3) per-encoder overlay GIF
    frames_overlaid = {"raw": [f for f in frames_uint8]}
    for enc in args.encoders:
        overlays = [overlay_rgb_on_frame(frames_uint8[t], per_enc_rgb[enc][t], args.alpha)
                    for t in range(T)]
        frames_overlaid[enc] = overlays
        gp = out_dir / f"overlay_{enc}.gif"
        write_gif(overlays, gp, fps=args.fps)
        print(f"[viz] saved {gp}")

    # 4) side-by-side GIF
    print("[viz] composing side-by-side ...")
    composites = make_side_by_side(frames_overlaid, per_enc_evr, encoder_order=args.encoders)
    sbs = out_dir / "side_by_side.gif"
    write_gif(composites, sbs, fps=args.fps)
    print(f"[viz] saved {sbs}")

    # 5) static comparison PNG (3 sample frames stacked vertically)
    sample_idx = [T // 6, T // 2, T * 5 // 6]
    static_frames = [composites[i] for i in sample_idx]
    static = out_dir / "static_grid.png"
    Image.fromarray(np.concatenate(static_frames, axis=0)).save(static)
    print(f"[viz] saved {static}")

    print("[viz] done.")


if __name__ == "__main__":
    main()
