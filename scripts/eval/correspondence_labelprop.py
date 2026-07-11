#!/usr/bin/env python
"""Correspondence label-propagation eval — JHMDB keypoint PCK (readout-free).

계획: docs/correspondence_eval_plan.md (옵션 A, 경로 2).
- 학습 head 0 (concat-probe artifact 면역): frozen patch feature 유사도만으로
  첫 프레임 keypoint 라벨을 후속 프레임에 전파 → PCK@{0.1,0.2}.
- 전파 로직 = DINO eval_video_segmentation.py 포팅 (context queue + spatial
  neighborhood + top-k + τ). PCK 규약 = videowalk(TimeCycle 계열) eval_pck.py:
  per-frame visible-joint bbox 대각선 × 0.6 정규화, per-joint PCK 평균, 첫 프레임 제외.

🔴 parity (eval_protocols §0): 전 encoder 동일 14×14 patch grid @ 실효 224
  (dinov2 = patch14라 입력 196×196 + HF pos-embed 자동 보간으로 grid 매칭),
  동일 τ·topk·radius·n_last_frames. 위반 시 assert.
🔴 preprocessing parity: parvo·videomae-vla = [0,1] raw (EgoDex 학습 규약,
  ImageNet norm 금지) / dinov2·siglip = 각자 native norm (adapters 규약과 동일).

M-stream 규약 (plan §5): "frame t의 motion feature" = ΔL(t-1, t). t=0은
ΔL(0,0)=0 (static) — CoMP 학습의 Case-A(static recon, prob 0.25)로 in-distribution.
P/M은 별도 track으로 각각 측정 (--parvo-stream).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

GRID = 14          # 🔴 parity 앵커: 전 encoder 14×14
N_JOINTS = 15
PCK_THRESHOLDS = (0.1, 0.2)

# ─────────────────────────────────────────────────────────────────────────
# JHMDB 로더
# ─────────────────────────────────────────────────────────────────────────

def list_jhmdb_test_clips(root: Path, split: int = 1) -> list[tuple[str, str]]:
    """splits/<action>_test_split<k>.txt에서 label==2(test) 클립 → (action, clip)."""
    clips = []
    for f in sorted((root / "splits").glob(f"*_test_split{split}.txt")):
        action = f.name.replace(f"_test_split{split}.txt", "")
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == "2":
                clips.append((action, parts[0].replace(".avi", "")))
    return clips


def load_jhmdb_clip(root: Path, action: str, clip: str):
    """frames (T,H,W,3) uint8 RGB + keypoints (2,15,T) 0-indexed 원본 좌표."""
    frame_files = sorted(glob.glob(str(root / "Rename_Images" / action / clip / "*.png")))
    assert frame_files, f"no frames: {action}/{clip}"
    frames = np.stack([cv2.imread(f)[:, :, ::-1] for f in frame_files])  # BGR→RGB
    mat = loadmat(str(root / "joint_positions" / action / clip / "joint_positions.mat"))
    kps = mat["pos_img"].astype(np.float64) - 1.0  # (2,15,T), 1-indexed → 0-indexed
    t = min(len(frames), kps.shape[2])  # 일부 클립은 frame/GT 길이 1~2 어긋남 → 공통 길이
    return frames[:t], kps[:, :, :t]


# ─────────────────────────────────────────────────────────────────────────
# Frozen dense-feature 추출기 (pooling 전 patch grid — AttentivePoolProbe 미사용)
# ─────────────────────────────────────────────────────────────────────────

_NORM = {  # single_frame.py _ENCODER_SPECS와 동일 (native norm)
    "dinov2": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "siglip": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


class DenseExtractor:
    """video frames (T,H,W,3) uint8 → (T, GRID*GRID, D) fp32 (L2-norm은 전파 단계)."""

    def __init__(self, encoder: str, checkpoint: str | None, parvo_stream: str,
                 device: torch.device, batch: int = 64):
        self.encoder, self.stream, self.device, self.batch = encoder, parvo_stream, device, batch
        if encoder == "parvo":
            from scripts.eval.probe_action_libero import build_parvo_encoder
            self.model = build_parvo_encoder(checkpoint, device)
            self.img_size = 224
        elif encoder == "videomae-vla":
            from scripts.eval.probe_action_libero import build_videomae_token_encoder
            self.model = build_videomae_token_encoder(checkpoint, device)
            self.img_size = 224
        elif encoder in ("dinov2", "siglip"):
            if encoder == "dinov2":
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained("facebook/dinov2-base")
                self.img_size = 196  # patch14 → 196/14 = 14×14 grid (pos-embed 자동 보간)
            else:
                from transformers import SiglipVisionModel
                self.model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
                self.img_size = 224
            self.model.to(device).eval()
            for p in self.model.parameters():
                p.requires_grad = False
            mean, std = _NORM[encoder]
            self._mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
            self._std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        else:
            raise ValueError(f"encoder: {encoder}")

    @torch.no_grad()
    def extract(self, frames_uint8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(frames_uint8.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        outs = []
        for s in range(0, x.shape[0], self.batch):
            xb = x[s:s + self.batch].to(self.device)
            if self.encoder == "parvo":
                if self.stream == "p":
                    p_ch = self.model.preprocessing.compute_p_channel(xb)
                    tok = self.model._encode_p_unmasked(p_ch)[:, 1:]
                else:  # m: ΔL(t-1, t), 배치 경계는 호출측에서 prev 프레임 전달로 처리
                    raise RuntimeError("m-stream은 extract_m() 사용")
            elif self.encoder == "videomae-vla":
                tok = self.model(torch.cat([xb, xb], dim=1))  # self-pair replica (§C7)
            else:
                xb = (xb - self._mean) / self._std
                hid = self.model(pixel_values=xb).last_hidden_state
                tok = hid[:, 1:] if self.encoder == "dinov2" else hid  # siglip은 CLS 없음
            assert tok.shape[1] == GRID * GRID, \
                f"parity 위반: {self.encoder} grid {tok.shape[1]} ≠ {GRID*GRID}"
            outs.append(tok.float().cpu())
        return torch.cat(outs)

    @torch.no_grad()
    def extract_m(self, frames_uint8: np.ndarray) -> torch.Tensor:
        """M-stream: feature[t] = M(ΔL(t-1, t)); t=0 → ΔL(0,0)=0 (Case-A static)."""
        x = torch.from_numpy(frames_uint8.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        prev = torch.cat([x[:1], x[:-1]], dim=0)  # t=0은 self-pair
        outs = []
        for s in range(0, x.shape[0], self.batch):
            pb, cb = prev[s:s + self.batch].to(self.device), x[s:s + self.batch].to(self.device)
            m_ch = self.model.preprocessing.compute_m_channel(pb, cb)
            tok = self.model._encode_m_unmasked(m_ch)[:, 1:]
            assert tok.shape[1] == GRID * GRID
            outs.append(tok.float().cpu())
        return torch.cat(outs)

    def features(self, frames_uint8: np.ndarray) -> torch.Tensor:
        if self.encoder == "parvo" and self.stream == "m":
            return self.extract_m(frames_uint8)
        return self.extract(frames_uint8)


# ─────────────────────────────────────────────────────────────────────────
# Label propagation (DINO eval_video_segmentation.py 포팅)
# ─────────────────────────────────────────────────────────────────────────

def restrict_neighborhood(h: int, w: int, radius: int, device) -> torch.Tensor:
    """(h*w, h*w) 0/1 mask — query (i,j)가 참조 가능한 source = 공간 반경 radius."""
    ii, jj = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    q = torch.stack([ii.flatten(), jj.flatten()], dim=1).float()  # (hw, 2)
    d = (q[:, None, :] - q[None, :, :]).abs().max(dim=-1).values  # (hw, hw) Chebyshev
    return (d <= radius).float().to(device)


@torch.no_grad()
def propagate_video(feats: torch.Tensor, first_label: torch.Tensor, args,
                    device) -> torch.Tensor:
    """feats (T, N, D) → 전파된 soft label (T, C, N). first_label (C, N)."""
    T, N, _ = feats.shape
    feats = F.normalize(feats, dim=-1, p=2).to(device)
    mask = restrict_neighborhood(GRID, GRID, args.radius, device)  # (N, N)

    labels = [first_label.to(device)]  # frame0 = GT
    queue: list[tuple[torch.Tensor, torch.Tensor]] = []  # (feat (N,D), label (C,N))
    for t in range(1, T):
        ctx_feats = [feats[0]] + [f for f, _ in queue]
        ctx_labels = [labels[0]] + [l for _, l in queue]
        f_src = torch.stack(ctx_feats)                      # (C_ctx, N, D)
        f_tar = feats[t].unsqueeze(0).expand_as(f_src)      # (C_ctx, N, D)
        aff = torch.exp(torch.bmm(f_tar, f_src.transpose(1, 2)) / args.temperature)
        aff = aff * mask.unsqueeze(0)                       # (C_ctx, N_tar, N_src)
        aff = aff.transpose(2, 1).reshape(-1, N)            # (C_ctx*N_src, N_tar)
        tk, _ = torch.topk(aff, dim=0, k=args.topk)
        aff[aff < tk.min(dim=0).values] = 0
        aff = aff / aff.sum(dim=0, keepdim=True).clamp_min(1e-12)
        lbl_src = torch.cat(ctx_labels, dim=1)              # (C, C_ctx*N_src)
        lbl_t = lbl_src @ aff                               # (C, N_tar)
        labels.append(lbl_t)
        queue.append((feats[t], lbl_t))
        if len(queue) > args.n_last_frames:
            queue.pop(0)
    return torch.stack(labels)  # (T, C, N)


# ─────────────────────────────────────────────────────────────────────────
# Keypoint ↔ heatmap (grid)
# ─────────────────────────────────────────────────────────────────────────

def kps_to_heatmap(kps_xy: np.ndarray, orig_w: int, orig_h: int) -> torch.Tensor:
    """(2, J) 원본 좌표 → (J, GRID*GRID) nearest-cell one-hot.

    hard one-hot = DINO read_seg(nearest resize) 규약. bilinear splat은 반반 가중치
    joint에서 전파 노이즈에 argmax가 플립되는 취약성 관측 → 배제.
    """
    hm = torch.zeros(N_JOINTS, GRID, GRID)
    gx = np.clip(np.round(kps_xy[0] / orig_w * GRID - 0.5), 0, GRID - 1).astype(int)
    gy = np.clip(np.round(kps_xy[1] / orig_h * GRID - 0.5), 0, GRID - 1).astype(int)
    for j in range(N_JOINTS):
        hm[j, gy[j], gx[j]] = 1.0
    return hm.reshape(N_JOINTS, -1)


def heatmap_to_kps(hm: torch.Tensor, orig_w: int, orig_h: int) -> np.ndarray:
    """(J, GRID*GRID) → argmax patch 중심의 원본 좌표 (2, J). (videowalk argmax 규약)"""
    idx = hm.argmax(dim=1).cpu().numpy()
    gy, gx = idx // GRID, idx % GRID
    return np.stack([(gx + 0.5) / GRID * orig_w, (gy + 0.5) / GRID * orig_h])


# ─────────────────────────────────────────────────────────────────────────
# PCK (videowalk eval_pck.py 규약)
# ─────────────────────────────────────────────────────────────────────────

def pck_distances(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """preds/gts (2, J, T) 원본 좌표 → normalized dist (J, T-1) (frame0 제외).

    정규화 = per-frame GT joint bbox 대각선 × 0.6 (videowalk).
    """
    T = gts.shape[2]
    dists = np.full((N_JOINTS, T - 1), np.inf)
    for k in range(1, T):
        mn, mx = gts[:, :, k].min(axis=1), gts[:, :, k].max(axis=1)
        box = 0.6 * np.linalg.norm(mx - mn)
        if box <= 0:
            continue
        d = np.linalg.norm(preds[:, :, k] - gts[:, :, k], axis=0)
        dists[:, k - 1] = d / box
    return dists


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("JHMDB correspondence label-propagation")
    ap.add_argument("--encoder", required=True,
                    choices=["parvo", "dinov2", "siglip", "videomae-vla"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--parvo-stream", default="p", choices=["p", "m"],
                    help="CoMP stream — 별도 track으로 각각 측정 (plan §5)")
    ap.add_argument("--jhmdb-root", default="/proj/external_group/mrg/datasets/jhmdb")
    ap.add_argument("--split", type=int, default=1)
    # 전파 하이퍼 (🔴 parity: 전 encoder 동일값 고정 — DINO 기본값의 14×14 스케일 조정)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--radius", type=int, default=3,
                    help="DINO 12 @ ~60폭 grid → 14폭 스케일 ≈ 3")
    ap.add_argument("--n-last-frames", type=int, default=7)
    ap.add_argument("--max-videos", type=int, default=0, help="0=전체 (sanity용 제한)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.jhmdb_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = list_jhmdb_test_clips(root, args.split)
    if args.max_videos:
        clips = clips[: args.max_videos]
    tag = args.encoder + (f"-{args.parvo_stream}" if args.encoder == "parvo" else "")
    print(f"[corr] encoder={tag} clips={len(clips)} device={device.type} "
          f"τ={args.temperature} topk={args.topk} r={args.radius} "
          f"n_last={args.n_last_frames} grid={GRID}")

    extractor = DenseExtractor(args.encoder, args.checkpoint, args.parvo_stream,
                               device, args.batch)

    all_dists, per_video = [], []
    t0 = time.time()
    for i, (action, clip) in enumerate(clips):
        frames, kps = load_jhmdb_clip(root, action, clip)
        T, H, W = frames.shape[:3]
        feats = extractor.features(frames)                       # (T, N, D)
        first = kps_to_heatmap(kps[:, :, 0], W, H)               # (J, N)
        soft = propagate_video(feats, first, args, device)       # (T, J, N)
        preds = np.stack([heatmap_to_kps(soft[t], W, H) for t in range(T)], axis=2)
        d = pck_distances(preds, kps)                            # (J, T-1)
        all_dists.append(d)
        pck_v = {f"pck@{th}": float(np.nanmean((d <= th).astype(float)))
                 for th in PCK_THRESHOLDS}
        per_video.append({"action": action, "clip": clip, "frames": T, **pck_v})
        if (i + 1) % 25 == 0 or i + 1 == len(clips):
            print(f"  [{i+1}/{len(clips)}] {action}/{clip} T={T} "
                  f"pck@0.1={pck_v['pck@0.1']:.3f} ({time.time()-t0:.0f}s)")

    # videowalk 규약: joint별 정답률 → 15-joint 평균
    cat = np.concatenate(all_dists, axis=1)  # (J, Σ(T-1))
    summary = {
        "encoder": tag, "checkpoint": args.checkpoint,
        "n_clips": len(clips), "n_frames_eval": int(cat.shape[1]),
        "grid": GRID, "temperature": args.temperature, "topk": args.topk,
        "radius": args.radius, "n_last_frames": args.n_last_frames,
        "split": args.split,
    }
    for th in PCK_THRESHOLDS:
        per_joint = (cat <= th).astype(float)
        valid = np.isfinite(cat)
        pj = [float(per_joint[j][valid[j]].mean()) for j in range(N_JOINTS)]
        summary[f"pck@{th}"] = float(np.mean(pj))
        summary[f"pck@{th}_per_joint"] = pj
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_video.json").write_text(json.dumps(per_video, indent=2))
    print(f"[corr] DONE pck@0.1={summary['pck@0.1']:.4f} "
          f"pck@0.2={summary['pck@0.2']:.4f} → {out_dir}/summary.json")


if __name__ == "__main__":
    main()
