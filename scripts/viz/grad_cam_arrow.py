"""Grad-CAM for ViT — Motion Arrow Overlay (plan #1).

Concat probe (P_t patches mean ⊕ P_tk patches mean → 7-DoF action) 위에 patch별
contribution = (∂Δ / ∂p_i) · p_i 계산하여 Δx, Δy 벡터 → quiver plot.

Plan §4 표 'concat' input mode: "Grad-CAM 각 frame별 분리 적용 — 두 frame motion 기여
비교 가능 (가장 motion-aware)". 본 prototype은 frame_t (P_t patches) 기여만 그림 —
P_tk 기여는 별 figure로 확장 가능.

워크플로우:
1. LIBERO demo HDF5 → (frame_t, frame_tk, target_ee_delta) triple 추출 (gap=20)
2. encoder 통과 → P_t/P_tk patches mean
3. nn.Linear(2*D, 7) probe 학습 (Adam 20 ep) — 같은 demo 활용 (prototype, cherry-pick 인정)
4. 각 frame pair에 대해:
   - patches_t = encoder(frame_t); leaf로 detach + requires_grad
   - pooled = concat(patches_t.mean(0), patches_tk.mean(0))
   - pred = probe(pooled); for dim ∈ {0=Δx, 1=Δy}: grad = autograd.grad(pred[dim], patches_t)
   - contribution_dim = (grad * patches_t).sum(-1) → (N,) per patch
5. (cx, cy) map → quiver overlay on frame_t
6. static PNG (3 sample frame pair) + GIF (전체 frame pair sequence)

Plan §4 가드:
- Δx, Δy만 시각화 (화면 평면)
- frame 내 max-norm 정규화 (frame 간 magnitude scale 차이 흡수)
- agentview only (eye-in-hand 제외)
- probe 학습 split과 visualization split 분리 권장 — 본 prototype은 단일 demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

IMG_SIZE = 224
ACTION_DIM = 7
DEFAULT_V15_CKPT = "/proj/external_group/mrg/checkpoints/two_stream_v15/20260511_045319/latest.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Frame + action target
# ─────────────────────────────────────────────────────────────────────────────

def load_demo(hdf5_path: str, demo_key: str, view: str = "agentview_rgb"):
    with h5py.File(hdf5_path, "r") as f:
        d = f[f"data/{demo_key}/obs"]
        frames = np.asarray(d[view])                       # (T, H, W, 3) uint8
        ee_pos = np.asarray(d["ee_pos"])                   # (T, 3)
        ee_ori = np.asarray(d["ee_ori"])                   # (T, 3) euler
        actions = np.asarray(f[f"data/{demo_key}/actions"])  # (T, 7)
    return frames, ee_pos, ee_ori, actions


def libero_action_target(ee_pos, ee_ori, actions, t, k):
    """LIBERO 7-DoF: pos Δ + rotvec Δ + gripper binary."""
    from scipy.spatial.transform import Rotation as R
    target_pos = ee_pos[t + k] - ee_pos[t]
    r1 = R.from_euler("xyz", ee_ori[t])
    r2 = R.from_euler("xyz", ee_ori[t + k])
    target_rotvec = (r1.inv() * r2).as_rotvec()
    target_gripper = actions[t + k - 1, 6:7]
    return np.concatenate([target_pos, target_rotvec, target_gripper], dtype=np.float32)


def preprocess_frames(frames_uint8: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    x = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Encoder forward (single-frame patch features)
# ─────────────────────────────────────────────────────────────────────────────

def build_v15(ckpt_path: str, device: torch.device):
    """v15 P encoder (single-frame, drop CLS). Returns (forward, grid=(14,14), embed_dim=768)."""
    from scripts.eval.probe_action_v11 import load_v11_model, _p_encoder_forward
    model = load_v11_model(ckpt_path, p_depth=12, m_depth=6, device=str(device))
    for p in model.parameters():
        p.requires_grad = False

    def forward(frame: torch.Tensor) -> torch.Tensor:
        _, p_channel = model.preprocessing(frame, frame)
        feat = _p_encoder_forward(model, p_channel)
        return feat[:, 1:]  # drop CLS

    return forward, (14, 14), 768


def build_dinov2(device: torch.device):
    """DINOv2 base patch features (single-frame, drop CLS). grid=(16,16), embed_dim=768."""
    from src.encoders.adapters.single_frame import SingleFrameAdapter
    adapter = SingleFrameAdapter("dinov2").to(device).eval()
    for p in adapter.parameters():
        p.requires_grad = False

    def forward(frame: torch.Tensor) -> torch.Tensor:
        x_n = adapter._normalize(frame)
        feat = adapter.model(pixel_values=x_n).last_hidden_state
        return feat[:, 1:]

    return forward, (16, 16), 768


def build_videomae(ckpt_path: str, device: torch.device):
    """VideoMAE-ours patch features (same-frame replica, no CLS). grid=(14,14), embed_dim=768."""
    from src.encoders.adapters.videomae import VideoMAEOursAdapter
    adapter = VideoMAEOursAdapter(checkpoint_path=ckpt_path, mode="paired").to(device).eval()
    for p in adapter.parameters():
        p.requires_grad = False

    def forward(frame: torch.Tensor) -> torch.Tensor:
        pv = torch.cat([frame, frame], dim=1)  # (B, 6, 224, 224)
        return adapter.model(pv)               # (B, 196, 768) — no CLS

    return forward, (14, 14), 768


def build_encoder(encoder_type: str, v15_ckpt: str, videomae_ckpt: str, device: torch.device):
    if encoder_type == "v15":
        return build_v15(v15_ckpt, device)
    if encoder_type == "dinov2":
        return build_dinov2(device)
    if encoder_type == "videomae-ours":
        return build_videomae(videomae_ckpt, device)
    raise ValueError(f"Unknown encoder: {encoder_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Probe 학습 (concat input)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_demo_means(forward_fn, frames_tensor, device, batch=32):
    """frames_tensor: (T, 3, H, W) → (T, D) patch mean per frame."""
    with torch.no_grad():
        feats = []
        for s in range(0, frames_tensor.shape[0], batch):
            x = frames_tensor[s:s + batch].to(device)
            feats.append(forward_fn(x).cpu())            # (b, N, D)
        feats = torch.cat(feats, dim=0)                  # (T, N, D)
    return feats.mean(dim=1)                             # (T, D)


def build_probe_dataset(forward_fn, frames_tensor, ee_pos, ee_ori, actions, gap, device):
    """LIBERO demo 1개 → (X concat, Y action target)."""
    T = frames_tensor.shape[0]; n = T - gap
    means = _encode_demo_means(forward_fn, frames_tensor, device)
    X = torch.cat([means[:n], means[gap:]], dim=-1)
    Y = torch.stack([
        torch.from_numpy(libero_action_target(ee_pos, ee_ori, actions, t, gap))
        for t in range(n)
    ])
    return X, Y


def build_multi_demo_dataset(forward_fn, hdf5_dir: Path, gap: int, device,
                             view: str = "agentview_rgb",
                             holdout_demos_per_task: int = 5):
    """LIBERO suite 전체 task × demo로 probe 학습 데이터 구축.

    holdout_demos_per_task: 각 task의 마지막 N demo를 visualization용 holdout으로 남김.

    Returns:
        X, Y: 학습 데이터
        holdout_index: list of (hdf5_path, demo_key) for viz (cherry-pick 회피)
    """
    Xs, Ys = [], []
    holdouts = []
    hdf5_paths = sorted(hdf5_dir.glob("*.hdf5"))
    print(f"[viz] multi-demo: {len(hdf5_paths)} tasks in {hdf5_dir}")
    for hp in hdf5_paths:
        with h5py.File(hp, "r") as f:
            demos = sorted(k for k in f["data"].keys() if k.startswith("demo_"))
        train_demos = demos[:-holdout_demos_per_task] if holdout_demos_per_task > 0 else demos
        holdout_demos = demos[-holdout_demos_per_task:] if holdout_demos_per_task > 0 else []
        for d in train_demos:
            frames, ee_pos, ee_ori, actions = load_demo(str(hp), d, view)
            if frames.shape[0] <= gap + 1: continue
            ft = preprocess_frames(frames)
            X, Y = build_probe_dataset(lambda x: forward_fn(x), ft, ee_pos, ee_ori, actions, gap, device)
            Xs.append(X); Ys.append(Y)
        for d in holdout_demos:
            holdouts.append((str(hp), d))
        print(f"[viz]   {hp.name}: {len(train_demos)} train demos, {len(holdout_demos)} holdout")
    return torch.cat(Xs, dim=0), torch.cat(Ys, dim=0), holdouts


def train_linear_probe(X, Y, epochs=80, lr=1e-3, batch=64, device="cuda"):
    D = X.shape[1]
    probe = nn.Linear(D, ACTION_DIM).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    X_dev = X.to(device); Y_dev = Y.to(device)
    n = X_dev.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            pred = probe(X_dev[idx])
            loss = F.mse_loss(pred, Y_dev[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    # Train R² (sanity)
    with torch.no_grad():
        pred_all = probe(X_dev).cpu()
    Y_np = Y.numpy(); P_np = pred_all.numpy()
    ss_res = np.sum((Y_np - P_np) ** 2)
    ss_tot = np.sum((Y_np - Y_np.mean(0)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    print(f"[viz] probe trained: R² (train, all gap pairs) = {r2:+.3f}")
    return probe


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM contribution (per patch, per action dim)
# ─────────────────────────────────────────────────────────────────────────────

def compute_contribution(
    forward_fn, probe, frame_t: torch.Tensor, frame_tk: torch.Tensor,
    dims: tuple = (0, 1), source: str = "t", device: str = "cuda",
):
    """concat probe (P_t | P_tk) 위 Grad-CAM. source='t' 또는 'tk' 중 한 쪽 patches에만 grad.

    Args:
        source: 't' → P_t patches contribution (frame_t 위에 그림 권장)
                'tk' → P_tk patches contribution (frame_tk 위에 그림 권장)

    Returns:
        dict {dim_idx: (N,) numpy} for each requested action dim.
    """
    with torch.no_grad():
        patches_t = forward_fn(frame_t.unsqueeze(0)).squeeze(0)    # (N, D)
        patches_tk = forward_fn(frame_tk.unsqueeze(0)).squeeze(0)  # (N, D)

    if source == "t":
        leaf = patches_t.detach().clone().requires_grad_(True)
        other = patches_tk
        pooled = torch.cat([leaf.mean(dim=0), other.mean(dim=0)], dim=-1).unsqueeze(0)
    elif source == "tk":
        leaf = patches_tk.detach().clone().requires_grad_(True)
        other = patches_t
        pooled = torch.cat([other.mean(dim=0), leaf.mean(dim=0)], dim=-1).unsqueeze(0)
    else:
        raise ValueError(f"source must be 't' or 'tk', got {source}")

    pred = probe(pooled).squeeze(0)                                # (7,)
    out = {}
    for d in dims:
        grad = torch.autograd.grad(pred[d], leaf, retain_graph=True)[0]
        out[d] = (grad * leaf).sum(dim=-1).detach().cpu().numpy()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Quiver overlay
# ─────────────────────────────────────────────────────────────────────────────

def _draw_quiver_on_ax(ax, frame_uint8, cx, cy, grid_shape, title,
                       shared_norm: float, sum_pixel_scale: float,
                       sum_color: str, top_percentile: float = 60.0,
                       gt_dxdy_pixel: tuple = None,
                       diag_label: str = ""):
    """변경 이력:
    - v6 (2026-05-27): patch arrow scale 5→12 (작게), sum vector pixel-scale, alpha=mag·keep
    - v8 (2026-05-27): V 부호 flip (사용자 시각: 그리퍼 down=image up). U=cy, V=-cx
    - v9 (2026-05-27): axis-alignment 정량 분석 추가
    - v10 (2026-05-27): yellow (kept patches mean) vs red sum direction cosine 진단 출력 추가
      (이유: 사용자가 "노란색 평균과 빨간색 sum이 반대 방향" 우려 — 정량 검증 필요)
    """
    """ax 위에 quiver overlay + sum vector (별도 pixel scale).

    LIBERO agentview camera (pos=[0.589,0,1.49], quat=[0.638,0.305,0.305,0.638]) →
    world → image mapping (matplotlib imshow upper-origin, y-axis inverted):
      U (image right)  = +Δy_world  (camera +x axis = world +y)
      V (image down)   = +Δx_world  (Δz 무시 근사. 정확: +0.628·Δx − 0.778·Δz)

    pivot="tail": patch center가 arrow 꼬리. 끝점이 motion direction.

    sum_pixel_scale: sum vector를 image pixel 좌표계에서 직접 길이로 그림.
                    paired_overlay에서 P_t/P_tk shared scale 전달 → 상대 magnitude 유지.
    """
    Hp, Wp = grid_shape
    H, W = frame_uint8.shape[:2]
    cx_2d = cx.reshape(Hp, Wp); cy_2d = cy.reshape(Hp, Wp)

    cx_n = cx_2d / shared_norm; cy_n = cy_2d / shared_norm
    mag_n = np.sqrt(cx_n ** 2 + cy_n ** 2)
    mag_n_clipped = np.clip(mag_n, 0, 1)

    thresh = np.percentile(mag_n_clipped, top_percentile)
    keep = mag_n_clipped >= thresh

    step_y, step_x = H / Hp, W / Wp
    ys = (np.arange(Hp) + 0.5) * step_y
    xs = (np.arange(Wp) + 0.5) * step_x
    Xg, Yg = np.meshgrid(xs, ys)

    rgba = np.zeros((Hp * Wp, 4))
    rgba[:, 0] = 1.0; rgba[:, 1] = 1.0; rgba[:, 2] = 0.0  # yellow
    rgba[:, 3] = np.where(keep.flatten(), 0.4 + 0.6 * mag_n_clipped.flatten(), 0.0)

    ax.imshow(frame_uint8)
    # per-patch arrows (yellow). V 부호 flip — 사용자 시각 검증 기준 (그리퍼 down = image up).
    ax.quiver(Xg.flatten(), Yg.flatten(),
              cy_n.flatten(), -cx_n.flatten(),
              color=rgba, scale=12.0, width=0.006,
              headwidth=4, headlength=5, pivot="tail")

    # Sum vector: data 좌표 직접 (pixel scale). frame 중앙에서 시작.
    sum_cx = cx_n.sum(); sum_cy = cy_n.sum()
    sum_mag = np.sqrt(sum_cx ** 2 + sum_cy ** 2)

    # 진단: yellow (kept patches mean) vs red sum direction cosine
    # 사용자 우려: yellow 평균 방향이 red sum과 반대로 보임 → 정량 align 검증.
    if keep.sum() > 0:
        ky_cx = cx_n[keep].mean(); ky_cy = cy_n[keep].mean()
        ky_mag = np.sqrt(ky_cx ** 2 + ky_cy ** 2)
        cos_yk_sum = float((ky_cx * sum_cx + ky_cy * sum_cy) / (ky_mag * sum_mag + 1e-8))
    else:
        cos_yk_sum = float("nan")
    # all patches mean (unfiltered) vs sum — 이건 trivially 정합 (mean=sum/N, sign 동일).
    am_cx = cx_n.mean(); am_cy = cy_n.mean()
    am_mag = np.sqrt(am_cx ** 2 + am_cy ** 2)
    cos_all_sum = float((am_cx * sum_cx + am_cy * sum_cy) / (am_mag * sum_mag + 1e-8))
    print(f"  [{diag_label}] yellow_kept_mean vs red_sum cos={cos_yk_sum:+.3f}  "
          f"all_mean vs sum cos={cos_all_sum:+.3f}  "
          f"(kept {int(keep.sum())}/{keep.size}, kept_mag={ky_mag:.3f}, sum_mag={sum_mag:.3f})")

    cx_frame, cy_frame = W / 2, H / 2
    dx_pix = sum_cy * sum_pixel_scale       # image right (U)
    dy_pix = -sum_cx * sum_pixel_scale      # V flip: world Δx 양수 = image up (사용자 시각 검증)
    if sum_mag > 1e-6:
        ax.annotate(
            "", xy=(cx_frame + dx_pix, cy_frame + dy_pix), xytext=(cx_frame, cy_frame),
            arrowprops=dict(arrowstyle="->", color=sum_color, lw=2.5,
                            mutation_scale=18),
        )
    # label: 외곽 (frame 좌상단)
    ax.text(6, 16, f"|Σ|={sum_mag:.2f}", color=sum_color,
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=sum_color, alpha=0.85))

    # Ground-truth motion vector (green) — frame 중앙에서 시작, ee_pos delta → image plane
    if gt_dxdy_pixel is not None:
        gt_dx, gt_dy = gt_dxdy_pixel
        ax.annotate(
            "", xy=(cx_frame + gt_dx, cy_frame + gt_dy), xytext=(cx_frame, cy_frame),
            arrowprops=dict(arrowstyle="->", color="lime", lw=2.5, mutation_scale=18,
                            linestyle="--"),
        )
        ax.text(6, 36, "GT motion (ee_pos Δ)", color="lime",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                          edgecolor="lime", alpha=0.7))

    ax.set_title(title, fontsize=10)
    ax.axis("off")


def paired_overlay(frame_t_uint8, frame_tk_uint8, contribs_t, contribs_tk,
                   grid_shape, t: int, tk: int, save_path: Path,
                   top_percentile: float = 60.0,
                   sum_target_ratio: float = 0.30,
                   gt_world_delta: np.ndarray = None):
    """frame pair (t, tk) 나란히. P_t와 P_tk 색 구분 (red / cyan).

    sum_target_ratio: 더 큰 sum vector를 frame width의 이 비율 pixel 길이로 fit.
                     P_t/P_tk 사이 상대 비율 유지.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cx_t, cy_t = contribs_t[0], contribs_t[1]
    cx_tk, cy_tk = contribs_tk[0], contribs_tk[1]
    mag_t = np.sqrt(cx_t ** 2 + cy_t ** 2)
    mag_tk = np.sqrt(cx_tk ** 2 + cy_tk ** 2)
    shared_norm = max(mag_t.max(), mag_tk.max(), 1e-6)

    # P_t/P_tk sum magnitude 사전 계산 (정규화된 cx_n/cy_n 기준)
    sum_mag_t  = np.sqrt((cx_t / shared_norm).sum() ** 2 + (cy_t / shared_norm).sum() ** 2)
    sum_mag_tk = np.sqrt((cx_tk / shared_norm).sum() ** 2 + (cy_tk / shared_norm).sum() ** 2)
    max_sum = max(sum_mag_t, sum_mag_tk, 1e-6)
    W = frame_t_uint8.shape[1]
    # max_sum 데이터 단위 ↔ sum_target_ratio·W 픽셀
    sum_pixel_scale = (sum_target_ratio * W) / max_sum

    # GT motion (world Δx, Δy, Δz) → image plane projection
    #   u_gt =  Δy_world                       (image right)
    #   v_gt = -(+0.628·Δx − 0.778·Δz)         (V flip, 사용자 시각 검증: 그리퍼 down=image up)
    gt_dxdy_pix = None
    if gt_world_delta is not None:
        u_gt = gt_world_delta[1]
        v_gt = -(0.628 * gt_world_delta[0] - 0.778 * gt_world_delta[2])
        gt_mag_world = np.sqrt(u_gt ** 2 + v_gt ** 2)
        if gt_mag_world > 1e-6:
            gt_target_pixel = sum_target_ratio * W
            gt_scale = gt_target_pixel / gt_mag_world
            gt_dxdy_pix = (u_gt * gt_scale, v_gt * gt_scale)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.6))
    _draw_quiver_on_ax(axL, frame_t_uint8, cx_t, cy_t, grid_shape,
                       title=f"frame t={t}  (P_t patch contrib)",
                       shared_norm=shared_norm,
                       sum_pixel_scale=sum_pixel_scale,
                       sum_color="red",
                       top_percentile=top_percentile,
                       gt_dxdy_pixel=gt_dxdy_pix,
                       diag_label=f"P_t  t={t:03d}")
    _draw_quiver_on_ax(axR, frame_tk_uint8, cx_tk, cy_tk, grid_shape,
                       title=f"frame t+gap={tk}  (P_(t+k) patch contrib)",
                       shared_norm=shared_norm,
                       sum_pixel_scale=sum_pixel_scale,
                       sum_color="cyan",
                       top_percentile=top_percentile,
                       gt_dxdy_pixel=gt_dxdy_pix,
                       diag_label=f"P_tk tk={tk:03d}")
    fig.suptitle(
        "v15 Grad-CAM (Δx, Δy)  —  yellow=per-patch, red=Σ_P_t, cyan=Σ_P_tk, "
        "green dashed=GT motion (ee_pos Δ projected)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="visualization demo HDF5 (--multi-demo면 동일 task suite 디렉토리 사용)")
    ap.add_argument("--demo-key", default="demo_0", help="visualization 대상 demo key")
    ap.add_argument("--view", default="agentview_rgb")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder", default="v15", choices=["v15", "dinov2", "videomae-ours"])
    ap.add_argument("--v15-ckpt", default=DEFAULT_V15_CKPT)
    ap.add_argument("--videomae-ckpt", default="/proj/external_group/mrg/checkpoints/videomae/20260415_012017/best_model.pt")
    ap.add_argument("--gap", type=int, default=20)
    ap.add_argument("--probe-epochs", type=int, default=80)
    ap.add_argument("--multi-demo", action="store_true",
                    help="True: --hdf5의 부모 디렉토리 모든 task × demo로 probe 학습 (holdout 5 demo/task)")
    ap.add_argument("--holdout-demos", type=int, default=5)
    ap.add_argument("--sample-indices", type=int, nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] device={device}, out_dir={out_dir}")

    # 1) Encoder
    forward_fn, grid, embed_dim = build_encoder(args.encoder, args.v15_ckpt, args.videomae_ckpt, device)
    print(f"[viz] encoder={args.encoder}, grid={grid}, embed_dim={embed_dim}")

    # 2) Probe 학습
    if args.multi_demo:
        hdf5_dir = Path(args.hdf5).parent
        X, Y, holdouts = build_multi_demo_dataset(
            forward_fn, hdf5_dir, args.gap, device, args.view,
            holdout_demos_per_task=args.holdout_demos,
        )
        print(f"[viz] multi-demo probe dataset: X={tuple(X.shape)}, Y={tuple(Y.shape)}, "
              f"holdout={len(holdouts)} demos")
    else:
        frames_u8_train, ee_pos_t, ee_ori_t, actions_t = load_demo(args.hdf5, args.demo_key, args.view)
        ft_train = preprocess_frames(frames_u8_train)
        X, Y = build_probe_dataset(forward_fn, ft_train, ee_pos_t, ee_ori_t, actions_t, args.gap, device)
        print(f"[viz] single-demo probe dataset: X={tuple(X.shape)}, Y={tuple(Y.shape)}")
    probe = train_linear_probe(X, Y, epochs=args.probe_epochs, device=device)
    probe.requires_grad_(True)

    # 3) Visualization demo 로드 (holdout 권장)
    frames_uint8, ee_pos, ee_ori, actions = load_demo(args.hdf5, args.demo_key, args.view)
    T = frames_uint8.shape[0]
    print(f"[viz] viz demo: T={T}, gap={args.gap}, n_pairs={T - args.gap}")
    frames_tensor = preprocess_frames(frames_uint8)

    # 4) Sample frame pairs → contribution + quiver
    if args.sample_indices is None:
        n_max = T - args.gap
        idxs = [n_max // 6, n_max // 3, n_max // 2, 2 * n_max // 3, 5 * n_max // 6]
    else:
        idxs = args.sample_indices

    print(f"[viz] computing Grad-CAM (paired t/t+gap) for {len(idxs)} frame pairs: {idxs}")
    sample_paths = []
    for i, t in enumerate(idxs):
        tk = t + args.gap
        ft = frames_tensor[t].to(device)
        ftk = frames_tensor[tk].to(device)
        contribs_t = compute_contribution(forward_fn, probe, ft, ftk, dims=(0, 1), source="t",  device=str(device))
        contribs_tk = compute_contribution(forward_fn, probe, ft, ftk, dims=(0, 1), source="tk", device=str(device))
        # Ground-truth motion in world frame (probe target과 동일 계산)
        gt_delta = ee_pos[tk] - ee_pos[t]  # (3,) world Δx, Δy, Δz
        p = out_dir / f"arrow_pair_t{t:03d}_tk{tk:03d}.png"
        paired_overlay(frames_uint8[t], frames_uint8[tk], contribs_t, contribs_tk,
                       grid, t=t, tk=tk, save_path=p, gt_world_delta=gt_delta)
        sample_paths.append(p)
        m_t = np.sqrt(contribs_t[0] ** 2 + contribs_t[1] ** 2).max()
        m_tk = np.sqrt(contribs_tk[0] ** 2 + contribs_tk[1] ** 2).max()
        gt_mag = float(np.linalg.norm(gt_delta))
        print(f"[viz]   saved {p}  |  mag max  P_t={m_t:.4f}  P_tk={m_tk:.4f}  "
              f"|  GT world Δ=[{gt_delta[0]:+.3f}, {gt_delta[1]:+.3f}, {gt_delta[2]:+.3f}] |Δ|={gt_mag:.3f}m")

    # 6) 전체 frame pair (n_pairs = T-gap) 정량 분석: P_t/P_tk sum vs GT axis-alignment
    print(f"\n[viz] axis-alignment analysis over all {T - args.gap} frame pairs ...")
    import csv as _csv
    rows = []
    for t in range(T - args.gap):
        tk = t + args.gap
        ft = frames_tensor[t].to(device)
        ftk = frames_tensor[tk].to(device)
        cT = compute_contribution(forward_fn, probe, ft, ftk, dims=(0, 1), source="t",  device=str(device))
        cTk = compute_contribution(forward_fn, probe, ft, ftk, dims=(0, 1), source="tk", device=str(device))
        gt = ee_pos[tk] - ee_pos[t]
        # image plane mapping (V flip 후): u = Δy_world, v = -(0.628·Δx − 0.778·Δz)
        gt_u = float(gt[1])
        gt_v = -(0.628 * float(gt[0]) - 0.778 * float(gt[2]))
        # sum (V flip 후): u = sum(cy), v = -sum(cx)
        st_u  = float(cT[1].sum());  st_v  = -float(cT[0].sum())
        stk_u = float(cTk[1].sum()); stk_v = -float(cTk[0].sum())
        gt_mag = float(np.sqrt(gt_u ** 2 + gt_v ** 2))
        rows.append({
            "t": t, "tk": tk,
            "gt_u": gt_u, "gt_v": gt_v, "gt_mag": gt_mag,
            "P_t_u":  st_u,  "P_t_v":  st_v,
            "P_tk_u": stk_u, "P_tk_v": stk_v,
            "P_t_u_sign_match":  int(np.sign(st_u)  == np.sign(gt_u)),
            "P_t_v_sign_match":  int(np.sign(st_v)  == np.sign(gt_v)),
            "P_tk_u_sign_match": int(np.sign(stk_u) == np.sign(gt_u)),
            "P_tk_v_sign_match": int(np.sign(stk_v) == np.sign(gt_v)),
        })

    csv_path = out_dir / "axis_alignment.csv"
    with csv_path.open("w") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[viz] saved {csv_path}")

    # Summary statistics
    def split(threshold):
        big  = [r for r in rows if r["gt_mag"] >= threshold]
        small = [r for r in rows if r["gt_mag"] <  threshold]
        return big, small

    big, small = split(0.10)  # 10cm 기준
    print(f"\n[viz] axis-alignment summary (sign match rate, V-flip 후 image axis 기준):")
    print(f"{'group':12s} {'n':>4s}  {'P_t u':>7s} {'P_t v':>7s} {'P_tk u':>7s} {'P_tk v':>7s}")
    for label, group in [("all", rows), ("|GT|≥10cm", big), ("|GT|<10cm", small)]:
        if not group:
            continue
        Pu  = np.mean([r["P_t_u_sign_match"]  for r in group])
        Pv  = np.mean([r["P_t_v_sign_match"]  for r in group])
        Tu  = np.mean([r["P_tk_u_sign_match"] for r in group])
        Tv  = np.mean([r["P_tk_v_sign_match"] for r in group])
        print(f"{label:12s} {len(group):>4d}  {Pu:>7.2f} {Pv:>7.2f} {Tu:>7.2f} {Tv:>7.2f}")

    # 7) Probe weight 직접 분석: W[:D] vs -W[D:] 정합 → implicit subtraction 직접 증명
    print(f"\n[viz] probe weight analysis (implicit P_tk − P_t 가설 검증):")
    W = probe.weight.data.cpu().numpy()  # (7, 2D)
    D = embed_dim
    W_t  = W[:, :D]   # P_t portion
    W_tk = W[:, D:]   # P_tk portion
    dim_names = ["Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "grip"]
    print(f"{'dim':6s} {'name':>6s}  {'cos(W_t, -W_tk)':>16s} {'sign_match':>11s} {'|W_t|':>8s} {'|W_tk|':>8s}")
    weight_rows = []
    for d in range(7):
        cos = float(np.dot(W_t[d], -W_tk[d]) /
                    (np.linalg.norm(W_t[d]) * np.linalg.norm(W_tk[d]) + 1e-8))
        sm  = float(np.mean(np.sign(W_t[d]) == np.sign(-W_tk[d])))
        nt  = float(np.linalg.norm(W_t[d])); ntk = float(np.linalg.norm(W_tk[d]))
        print(f"{d:>6d} {dim_names[d]:>6s}  {cos:>+16.4f} {sm:>11.3f} {nt:>8.3f} {ntk:>8.3f}")
        weight_rows.append({"dim": d, "name": dim_names[d],
                            "cos_Wt_negWtk": cos, "sign_match_Wt_negWtk": sm,
                            "norm_Wt": nt, "norm_Wtk": ntk})

    wcsv = out_dir / "probe_weight_summary.csv"
    with wcsv.open("w") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(weight_rows[0].keys()))
        w.writeheader(); w.writerows(weight_rows)
    print(f"[viz] saved {wcsv}")

    print(f"\n[viz] interpretation:")
    print(f"  cos(W_t, -W_tk) > 0  → probe가 implicit (P_tk − P_t) subtraction 학습")
    print(f"  cos(W_t, -W_tk) ≈ 1  → P_tk MINUS P_t 완벽 difference (motion = frame embedding 차이)")
    print(f"  cos(W_t, -W_tk) ≈ 0  → 두 frame이 독립적 정보 source")
    print(f"  cos(W_t, -W_tk) < 0  → P_t와 P_tk가 같은 방향으로 contribute (sum, not difference)")

    # 5) Composite static PNG: 5 frame pairs × 2 panel = vertical stack
    from PIL import Image as PILImage
    imgs = [PILImage.open(p) for p in sample_paths]
    w, h = imgs[0].size
    canvas = PILImage.new("RGB", (w, h * len(imgs)), (255, 255, 255))
    for i, im in enumerate(imgs):
        canvas.paste(im, (0, i * h))
    static_path = out_dir / "static_arrow_grid.png"
    canvas.save(static_path)
    print(f"[viz] saved composite {static_path}")
    print("[viz] done.")


if __name__ == "__main__":
    main()
