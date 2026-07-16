#!/usr/bin/env python
"""LIBERO action probing.

Complements the near-zero absolute R^2 obtained when probing action from a static
real-robot dataset, using the cleaner LIBERO simulation environment. One job =
(encoder x suite); the inner loop sweeps 4 temporal gaps to produce 4 cells.

Target (pose-derived, NOT a cumulative sum of actions):
    - pos:     ee_pos[t+k] - ee_pos[t]                                                    (3,)
    - rotvec:  (R.from_rotvec(ee_ori[t]).inv() * R.from_rotvec(ee_ori[t+k])).as_rotvec()  (3,)
    - gripper: actions[t+k-1, 6]                                                          (1,)
    -> 7-DoF target

Gap-to-time mapping (LIBERO runs at 20 Hz):
    - gap=1   ~0.05 s
    - gap=13  ~0.65 s
    - gap=20  ~1.00 s   (matches the ~1 s temporal scale of the pretraining data)
    - gap=40  ~2.00 s
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the supplement root importable so the `model` and `eval` packages resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # supplement root (contains model/, eval/)

import argparse
import csv
import json
import os
import time
from typing import List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, TensorDataset

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal")
SUPPORTED_ENCODERS = (
    "comp-mae",           # CoMP-MAE (shipped, fully functional path)
    "videomae-ours",   # baseline -- loader omitted from this release (stub)
    "siammae",         # baseline -- loader omitted from this release (stub)
    "dinov2",          # baseline -- loader omitted from this release (stub)
    "siglip",          # baseline -- loader omitted from this release (stub)
    "vc1",             # baseline -- loader omitted from this release (stub)
)
DEFAULT_GAPS = [1, 13, 20, 40]
ACTION_DIM = 7  # 3 pos + 3 rotvec + 1 gripper


# ─────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ─────────────────────────────────────────────────────────────────────────

def list_demos(hdf5_path: str) -> List[str]:
    with h5py.File(hdf5_path, "r") as f:
        return sorted([k for k in f["data"].keys() if k.startswith("demo_")])


def load_demo(hdf5_path: str, demo_key: str, view: str = "agentview_rgb"):
    """Load one demo: frames(uint8), ee_pos, ee_ori (axis-angle), actions."""
    with h5py.File(hdf5_path, "r") as f:
        d = f[f"data/{demo_key}"]
        frames = np.asarray(d[f"obs/{view}"])      # (T, H, W, 3) uint8
        ee_pos = np.asarray(d["obs/ee_pos"])        # (T, 3)
        ee_ori = np.asarray(d["obs/ee_ori"])        # (T, 3) axis-angle
        actions = np.asarray(d["actions"])          # (T, 7)
    return frames, ee_pos, ee_ori, actions


def libero_action_target(eef_pos, ee_ori, actions, t, k):
    """Pose-derived target (NOT a cumulative sum of actions)."""
    target_pos = eef_pos[t + k] - eef_pos[t]                         # (3,)
    r1 = R.from_rotvec(ee_ori[t])
    r2 = R.from_rotvec(ee_ori[t + k])
    target_rotvec = (r1.inv() * r2).as_rotvec()                      # (3,)
    target_gripper = actions[t + k - 1, 6:7]                         # (1,)
    return np.concatenate([target_pos, target_rotvec, target_gripper], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Encoder build
# ─────────────────────────────────────────────────────────────────────────

def build_standard_encoder(encoder_type: str, checkpoint: str | None, device: torch.device,
                           **adapter_kwargs):
    # Baseline encoders (videomae-ours adapter path, siammae, dinov2, siglip, vc1)
    # are all routed through here.
    raise NotImplementedError(
        "Baseline encoder loaders are omitted from this anonymized release. "
        "The headline CoMP-MAE / plain-control rows use the shipped checkpoints; see README."
    )


# ─────────────────────────────────────────────────────────────────────────
# Pair encode (img_t, img_{t+gap})
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_pairs_via_adapter(
    adapter, frames_prev: torch.Tensor, frames_curr: torch.Tensor,
    device: torch.device, batch: int = 64,
) -> torch.Tensor:
    """frames_prev/curr: (N, 3, H, W) [0,1] float -> (N, embed_dim).

    Adapter T>1 path: obs_seq=[prev, curr]. The first step is a self-pair; the second
    step is the (prev, curr) pair. We keep the embedding of the second step.
    """
    out = []
    for s in range(0, frames_prev.shape[0], batch):
        p = frames_prev[s:s+batch].to(device, non_blocking=True)  # (n, 3, H, W)
        c = frames_curr[s:s+batch].to(device, non_blocking=True)
        seq = torch.stack([p, c], dim=1)  # (n, 2, 3, H, W)
        adapter.reset()
        e = adapter(seq)[:, 1]  # take t=1 embedding (pair-formed)
        out.append(e.cpu())
    return torch.cat(out, dim=0)


# ─────────────────────────────────────────────────────────────────────────
# CoMP-MAE (CoMPMAE, no-Sobel) -- headline / plain-control path.
# Same convention as the adapter: infer architecture from the checkpoint and
# detect the M-recon (CoMP-MAE) branch.
# no-Sobel: P = RGB [0,1] 3ch, M = |dL| 1ch. Input is the [0,1] raw produced by
# preprocess_frames (do NOT apply ImageNet normalization -- train/inference parity).
# ─────────────────────────────────────────────────────────────────────────

def build_comp_mae_encoder(checkpoint: str, device: torch.device):
    from model.comp_mae import CoMPMAE
    assert checkpoint and checkpoint != "random", "comp-mae encoder requires --checkpoint"
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    _ed = next(v.shape[-1] for k, v in sd.items() if k == "pos_embed_p")
    _md = len({k.split(".")[1] for k in sd if k.startswith("blocks_m.")})
    _comp = any("m_recon" in k for k in sd)  # CoMP-MAE = has the M-recon branch
    model = CoMPMAE(
        embed_dim=_ed, num_heads=_ed // 64, m_depth=_md, comp_mae=_comp,
        pair_mode=True, use_sobel=False, masked_anchor=True,
    )
    missing, _ = model.load_state_dict(sd, strict=False)
    _enc_missing = [k for k in missing if k.startswith(("blocks_p", "blocks_m", "patch_embed_p", "patch_embed_m"))]
    assert not _enc_missing, f"comp-mae: P/M encoder weights not loaded {_enc_missing[:5]}"
    for p in model.parameters():
        p.requires_grad = False
    model.to(device).eval()
    return model


@torch.no_grad()
def encode_pairs_comp_mae(
    model, frames_prev: torch.Tensor, frames_curr: torch.Tensor,
    device: torch.device, mode: str = "p_t_p_tk", readout: str = "mean", batch: int = 64,
) -> torch.Tensor:
    """CoMP-MAE 2-stream readout. Stream order = [P(t), P(tk)] or [P(t), M(t,tk)].

    mode:    p_t_p_tk = deployed P readout (appearance) / p_t_m = P(t) + M(t,tk) (with motion)
    readout: mean      = per-stream patch mean -> concat -> (N, 2D)
             attentive = per-stream patch tokens -> concat on the token dim -> (N, 2*n_patch, D) [fp16 cache]
    """
    out = []
    for s in range(0, frames_prev.shape[0], batch):
        p = frames_prev[s:s + batch].to(device, non_blocking=True)  # (n, 3, H, W) [0,1]
        c = frames_curr[s:s + batch].to(device, non_blocking=True)
        p_t = model.preprocessing.compute_p_channel(p)
        tok_a = model._encode_p_unmasked(p_t)[:, 1:]                 # (n, n_patch, D) = P(t)
        # Pure single-stream readouts (for the factorization crossover analysis).
        if mode == "p_t_only":
            toks = [tok_a]
        elif mode == "m_only":
            m_chan = model.preprocessing.compute_m_channel(p, c)
            toks = [model._encode_m_unmasked(m_chan)[:, 1:]]         # (n, n_patch, D) = M(t,tk)
        else:
            if mode == "p_t_p_tk":
                p_tk = model.preprocessing.compute_p_channel(c)
                tok_b = model._encode_p_unmasked(p_tk)[:, 1:]
            elif mode == "p_t_m":
                m_chan = model.preprocessing.compute_m_channel(p, c)
                tok_b = model._encode_m_unmasked(m_chan)[:, 1:]
            else:
                raise ValueError(f"comp-mae mode: {mode}")
            toks = [tok_a, tok_b]
        if readout == "mean":
            tok = torch.cat([t.mean(dim=1) for t in toks], dim=-1)   # (n, len(toks)*D)
        elif readout == "attentive":
            tok = torch.cat(toks, dim=1).half()                      # (n, len(toks)*n_patch, D)
        else:
            raise ValueError(f"readout: {readout}")
        out.append(tok.cpu())
    return torch.cat(out, dim=0)


# ─────────────────────────────────────────────────────────────────────────
# VideoMAE baseline (token-level) -- loader omitted from this release.
# The encode function below is kept for reference: it produces both readouts
# from the same forward so that attentive <-> mean stay self-consistent.
# ─────────────────────────────────────────────────────────────────────────

def build_videomae_token_encoder(checkpoint: str, device: torch.device):
    raise NotImplementedError(
        "Baseline encoder loaders are omitted from this anonymized release. "
        "The headline CoMP-MAE / plain-control rows use the shipped checkpoints; see README."
    )


@torch.no_grad()
def encode_pairs_videomae_vla(
    encoder, frames_prev: torch.Tensor, frames_curr: torch.Tensor,
    device: torch.device, readout: str = "mean", batch: int = 64,
) -> torch.Tensor:
    """VideoMAE two-stream = frame_t / frame_tk (same-frame replicated forward).

    tubelet_size=2 binds two frames into a spatio-temporal patch, so a single-frame
    representation is obtained by feeding the frame twice. readout mean -> (N, 2D),
    attentive -> (N, 2*n_patch, D). Same output convention as the CoMP-MAE path.
    """
    out = []
    for s in range(0, frames_prev.shape[0], batch):
        p = frames_prev[s:s + batch].to(device, non_blocking=True)
        c = frames_curr[s:s + batch].to(device, non_blocking=True)
        tok_t = encoder(torch.cat([p, p], dim=1))    # (n, n_patch, D)
        tok_tk = encoder(torch.cat([c, c], dim=1))
        if readout == "mean":
            tok = torch.cat([tok_t.mean(dim=1), tok_tk.mean(dim=1)], dim=-1)
        elif readout == "attentive":
            tok = torch.cat([tok_t, tok_tk], dim=1).half()
        else:
            raise ValueError(f"readout: {readout}")
        out.append(tok.cpu())
    return torch.cat(out, dim=0)


def preprocess_frames(frames_uint8: np.ndarray, img_size: int) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, img_size, img_size) [0,1] float."""
    x = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x


def rotate_pair(prev: torch.Tensor, curr: torch.Tensor, device, seed_rng,
                max_shift: float = 0.2) -> tuple:
    """De-confound for the factorization analysis: apply the SAME random rotation +
    translation to both prev and curr of a pair. This breaks the position<->identity
    correlation -- appearance is preserved (P expected to survive) while absolute
    position / reach direction is randomized (M identity expected to drop to chance).
    The same transform within a pair is required (differing transforms would corrupt
    the motion channel |dL|). Both rotation (angle) and translation (radius) are
    randomized -- rotation about the center alone preserves radial distance and is
    empirically insufficient.
    max_shift: translation range in normalized (+/-1) coordinates. Too large pushes
    objects out of frame (P collapses too), hence 0.2."""
    n = prev.shape[0]
    ang = torch.from_numpy(seed_rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)).to(device)
    tx = torch.from_numpy(seed_rng.uniform(-max_shift, max_shift, size=n).astype(np.float32)).to(device)
    ty = torch.from_numpy(seed_rng.uniform(-max_shift, max_shift, size=n).astype(np.float32)).to(device)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
    out = []
    for x in (prev, curr):
        xd = x.to(device)
        grid = F.affine_grid(theta, xd.shape, align_corners=False)
        out.append(F.grid_sample(xd, grid, align_corners=False, padding_mode="zeros").cpu())
    return out[0], out[1]


def pos_covariate(eef_pos: np.ndarray, gap: int, kind: str) -> torch.Tensor:
    """Position covariate: pos = ee_pos(t) 3d / pos_delta = [ee_pos(t), delta_pos] 6d (identity only)."""
    p_t = eef_pos[:len(eef_pos) - gap].astype(np.float32)
    if kind == "pos_delta":
        p_t = np.concatenate([p_t, eef_pos[gap:].astype(np.float32) - p_t], axis=1)
    return torch.from_numpy(p_t)


def rff_expand(pos_z: torch.Tensor, seed: int, n_feat: int = 128) -> torch.Tensor:
    """z-scored ee_pos(t) -> random Fourier features (RBF-kernel approximation, fixed seed).
    Fairness of position partial-out: a linear head cannot read identity (a nonlinear
    boundary) from raw 3d coordinates, so the control ceiling would underfit. Uses a
    multi-scale lengthscale {0.5, 1, 2} (in z-units)."""
    rng = np.random.default_rng(seed)
    d = pos_z.shape[1]
    scales = np.repeat([0.5, 1.0, 2.0], n_feat // 3 + 1)[:n_feat]
    W = rng.normal(size=(d, n_feat)).astype(np.float32) / scales.astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, size=n_feat).astype(np.float32)
    proj = pos_z @ torch.from_numpy(W) + torch.from_numpy(b)
    return np.sqrt(2.0 / n_feat) * torch.cos(proj)


# ─────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int = ACTION_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, action_dim)

    def forward(self, x):
        return self.linear(x)


class AttentivePoolProbe(nn.Module):
    """Attentive-pooling probe. One learnable query per stream softmax-weights the
    patch tokens -> per-stream pooled concat -> linear head. Its capacity is matched
    to the linear probe to rule out a 'structure vs probe capacity' confound.
    Input x: [B, n_streams * n_patch, D] (the attentive output of encode_pairs_*).
    extra_dim > 0 concatenates a covariate (e.g. z-scored ee_pos(t)) after pooling --
    used for position partial-out."""
    def __init__(self, embed_dim: int, n_streams: int = 1, n_patch: int = 196,
                 action_dim: int = ACTION_DIM, extra_dim: int = 0):
        super().__init__()
        self.n_streams = n_streams
        self.n_patch = n_patch
        self.embed_dim = embed_dim
        self.extra_dim = extra_dim
        self.query = nn.Parameter(torch.randn(n_streams, embed_dim) * 0.02)
        self.head = nn.Linear(n_streams * embed_dim + extra_dim, action_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x, extra=None):
        B = x.shape[0]
        x = x.view(B, self.n_streams, self.n_patch, self.embed_dim)
        attn = torch.einsum("bsnd,sd->bsn", x, self.query) * self.scale
        attn = attn.softmax(dim=-1)
        pooled = torch.einsum("bsn,bsnd->bsd", attn, x)
        pooled = pooled.reshape(B, self.n_streams * self.embed_dim)
        if self.extra_dim:
            pooled = torch.cat([pooled, extra], dim=-1)
        return self.head(pooled)


def train_probe(
    train_emb, train_tgt, eval_emb, eval_tgt,
    epochs: int = 20, batch_size: int = 256, lr: float = 1e-3,
    device: str = "cuda",
    readout: str = "mean", n_streams: int = 1, weight_decay: float = 0.0,
    task: str = "regression", out_dim: int = ACTION_DIM,
    extra_train=None, extra_eval=None,
):
    """readout="mean": LinearProbe([N, D]) / readout="attentive": AttentivePoolProbe([N, S*n_patch, D]).

    task="regression": MSE loss, best = max r2_aggregate.
    task="classification": CE loss, out_dim=n_classes, best = max accuracy (factorization identity axis).
    Probe capacity (linear/attentive) is identical across tasks -- rules out a
    structure vs capacity confound. The attentive cache holds fp16 3D tokens, restored
    to float on the GPU for training and evaluated in batches (to avoid OOM).
    extra_*: attentive-only covariate ([N, E], for position partial-out concat). For the
    mean readout the covariate is concatenated at the call site instead.
    """
    is_cls = task == "classification"
    has_extra = extra_train is not None
    if readout == "attentive":
        n_patch = eval_emb.shape[1] // n_streams
        base_dim = eval_emb.shape[2]
        probe = AttentivePoolProbe(base_dim, n_streams=n_streams, n_patch=n_patch,
                                   action_dim=out_dim,
                                   extra_dim=(int(extra_train.shape[1]) if has_extra else 0)).to(device)
    else:
        assert not has_extra, "mean readout: concatenate the covariate at the call site"
        probe = LinearProbe(train_emb.shape[1], action_dim=out_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(
        TensorDataset(*((train_emb, extra_train, train_tgt) if has_extra
                        else (train_emb, train_tgt))),
        batch_size=batch_size, shuffle=True,
    )
    key = "accuracy" if is_cls else "r2"
    best = {key: -float("inf"), "epoch": 0}
    for ep in range(epochs):
        probe.train()
        for batch in train_loader:
            x = batch[0].to(device).float()  # restore float on GPU if fp16-cached
            y = batch[-1].to(device)
            pred = probe(x, batch[1].to(device).float()) if has_extra else probe(x)
            loss = F.cross_entropy(pred, y) if is_cls else F.mse_loss(pred, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        probe.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(eval_emb), batch_size):
                chunk = eval_emb[i:i + batch_size].to(device).float()
                if has_extra:
                    preds.append(probe(chunk, extra_eval[i:i + batch_size].to(device).float()).cpu())
                else:
                    preds.append(probe(chunk).cpu())
        pred = torch.cat(preds, dim=0)
        if is_cls:
            acc = float((pred.argmax(dim=1) == eval_tgt).float().mean())
            m = {"accuracy": acc, "n_classes": out_dim}
            if acc > best["accuracy"]:
                best = {"accuracy": acc, "epoch": ep + 1, "metrics": m}
        else:
            m = compute_metrics(pred.numpy(), eval_tgt.numpy())
            if m["r2_aggregate"] > best["r2"]:
                best = {"r2": m["r2_aggregate"], "epoch": ep + 1, "metrics": m}
    return best


def compute_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    """R^2 aggregate + per-dim, MSE, cosine."""
    ss_res = np.sum((tgt - pred) ** 2)
    ss_tot = np.sum((tgt - tgt.mean(axis=0)) ** 2)
    r2_agg = 1.0 - ss_res / (ss_tot + 1e-8)
    mse = float(np.mean((pred - tgt) ** 2))
    cos = float(np.mean(np.sum(pred * tgt, axis=1) /
                        (np.linalg.norm(pred, axis=1) * np.linalg.norm(tgt, axis=1) + 1e-8)))
    r2_per_dim = []
    for i in range(tgt.shape[1]):
        ss_r = np.sum((tgt[:, i] - pred[:, i]) ** 2)
        ss_t = np.sum((tgt[:, i] - tgt[:, i].mean()) ** 2)
        r2_per_dim.append(float(1.0 - ss_r / (ss_t + 1e-8)))
    return {
        "r2_aggregate": float(r2_agg),
        "r2_per_dim": r2_per_dim,
        "mse": mse,
        "cosine_sim": cos,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=SUPPORTED_ENCODERS)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--task-suite", required=True, choices=LIBERO_SUITES)
    parser.add_argument("--data-root", default=None, required=False,
                        help="Path to the LIBERO HDF5 dataset root, i.e. <LIBERO_DATA_ROOT> "
                             "containing <suite>/ subfolders. Required to run.")
    parser.add_argument("--view", default="agentview_rgb",
                        choices=["agentview_rgb", "eye_in_hand_rgb", "both"],
                        help="both = agentview + eye-in-hand feature-level concat")
    parser.add_argument("--gaps", type=int, nargs="+", default=DEFAULT_GAPS,
                        help="Frame gaps to evaluate (LIBERO 20Hz: 1=0.05s, 13=0.65s, 20=1s, 40=2s)")
    parser.add_argument("--max-length-percentile", type=float, default=99.0,
                        help="Exclude trajectories longer than this percentile (99 default keeps whole demos for the gap)")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-batch", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--videomae-mode", default="paired",
                        choices=["paired", "p_t_p_tk"],
                        help="paired = standard BC-T paired forward, p_t_p_tk = two single-frame forwards")
    # CoMP-MAE / VideoMAE readout axis.
    parser.add_argument("--readout", default="mean", choices=["mean", "attentive"],
                        help="mean = patch-mean concat (LinearProbe) / attentive = per-stream query pool (AttentivePoolProbe)")
    parser.add_argument("--stream-mode", default="p_t_p_tk",
                        choices=["p_t_p_tk", "p_t_m", "m_only", "p_t_only"],
                        help="comp-mae readout: p_t_p_tk(appearance)/p_t_m(P+M) combined | "
                             "m_only(M alone)/p_t_only(P(t) alone) = pure streams for the factorization crossover")
    parser.add_argument("--videomae-encoder", default="adapter", choices=["adapter", "vla"],
                        help="adapter = BC-T adapter (mean only) / vla = token-level encoder (mean+attentive self-consistent)")
    parser.add_argument("--probe-weight-decay", type=float, default=0.0,
                        help="AdamW weight decay (suppresses attentive P-appearance overfit). default 0")
    parser.add_argument("--target", default="action", choices=["action", "identity"],
                        help="action = pose-delta regression (R^2) | identity = task-id classification (top-1 acc, chance=1/n_task). "
                             "factorization crossover: M->motion high / identity chance, P->motion 0 / identity high")
    parser.add_argument("--rotate-aug", action="store_true",
                        help="de-confound: apply the same random rotation to both frames of a pair -> remove position<->identity correlation. "
                             "expectation: object suite keeps P identity, M->chance / spatial suite both collapse (position is the only cue).")
    parser.add_argument("--position-control", default="none", choices=["none", "only", "concat"],
                        help="position partial-out (statistical de-confound for factorization): "
                             "only = probe from ee_pos(t) 3d alone (position-only ceiling, no encoder/frames) / "
                             "concat = encoder feature + ee_pos(t) -> delta(concat-only) = contribution beyond position. "
                             "NOTE: the covariate is the t timestep only -- ee_pos(t+k) would leak the pose-delta target directly")
    parser.add_argument("--pos-covariate", default="pos", choices=["pos", "pos_delta"],
                        help="pos = ee_pos(t) 3d | pos_delta = [ee_pos(t), delta_pos(t->t+k)] 6d -- "
                             "isolates whether M-identity lies beyond geometry+realized motion (i.e. appearance leakage). "
                             "identity target only (for the action target delta_pos is the target itself -> leakage)")
    args = parser.parse_args()

    if args.readout == "attentive" and args.view == "both":
        raise ValueError("attentive readout supports a single view only (both = feature-concat breaks the token layout)")
    if args.position_control != "none" and args.view == "both":
        raise ValueError("--position-control supports a single view only")
    if args.position_control == "only":
        if args.rotate_aug:
            raise ValueError("--position-control only does not use frames -- combining with --rotate-aug is meaningless")
        if args.readout == "attentive":
            print("[probe_libero] position-control=only -> forcing readout=mean (LinearProbe for a 3d covariate)")
            args.readout = "mean"
    if args.pos_covariate == "pos_delta" and args.target != "identity":
        raise ValueError("--pos-covariate pos_delta is for the identity target only (leaks delta_pos into the action target)")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[probe_libero] encoder={args.encoder} suite={args.task_suite} gaps={args.gaps} device={device}")

    # ── Benchmark / HDF5 paths ───────────────────────────────────────────
    # The LIBERO benchmark package must be importable. If a vendored copy sits
    # alongside the supplement (external/LIBERO), add it to sys.path; otherwise
    # rely on an installed `libero` package.
    _libero_vendor = Path(__file__).resolve().parents[1] / "external" / "LIBERO"
    if _libero_vendor.exists():
        sys.path.insert(0, str(_libero_vendor))
    from libero.libero.benchmark import get_benchmark
    bm = get_benchmark(args.task_suite)(0)
    n_tasks = bm.n_tasks
    task_ids = args.task_ids or list(range(n_tasks))
    suite_dir = os.path.join(args.data_root, args.task_suite)
    hdf5_paths = [
        os.path.join(suite_dir, bm.get_task_demonstration(i).split("/")[-1])
        for i in task_ids
    ]
    print(f"  tasks: {len(task_ids)}/{n_tasks}, data_root={suite_dir}")

    # ── Length filter ────────────────────────────────────────────────────
    lens = []
    for p in hdf5_paths:
        with h5py.File(p, "r") as f:
            for k in sorted(f["data"].keys()):
                if k.startswith("demo_"):
                    lens.append(f[f"data/{k}/obs/ee_pos"].shape[0])
    lens = np.array(lens)
    cutoff = float(np.percentile(lens, args.max_length_percentile))
    n_kept = int((lens <= cutoff).sum())
    print(f"  length stats (n={len(lens)}): min={lens.min()} max={lens.max()} "
          f"mean={lens.mean():.1f} P50={np.percentile(lens,50):.0f} P95={np.percentile(lens,95):.0f} "
          f"P99={np.percentile(lens,99):.0f}  cutoff={cutoff:.0f} → kept {n_kept}/{len(lens)}")

    # ── Build encoder ────────────────────────────────────────────────────
    n_streams = 1
    if args.position_control == "only":
        encode_fn, img_size = None, None  # position control: ee_pos(t) is the feature -- no encoder needed
    elif args.encoder == "comp-mae":
        model = build_comp_mae_encoder(args.checkpoint, device)
        img_size = 224
        n_streams = 1 if args.stream_mode in ("m_only", "p_t_only") else 2

        def encode_fn(prev, curr):
            return encode_pairs_comp_mae(model, prev, curr, device,
                                      mode=args.stream_mode, readout=args.readout,
                                      batch=args.encode_batch)
    elif args.encoder == "videomae-ours" and args.videomae_encoder == "vla":
        # Baseline path -- loader is stubbed in this release.
        model = build_videomae_token_encoder(args.checkpoint, device)
        img_size = 224
        n_streams = 2

        def encode_fn(prev, curr):
            return encode_pairs_videomae_vla(model, prev, curr, device,
                                             readout=args.readout, batch=args.encode_batch)
    else:
        # Remaining baselines (videomae-ours adapter, siammae, dinov2, siglip, vc1)
        # route through build_standard_encoder, which is stubbed in this release.
        if args.readout == "attentive":
            raise ValueError(f"attentive readout supports comp-mae/videomae(vla) only (encoder={args.encoder})")
        adapter_kwargs = {}
        if args.encoder == "videomae-ours":
            adapter_kwargs["mode"] = args.videomae_mode
        adapter = build_standard_encoder(args.encoder, args.checkpoint, device, **adapter_kwargs)
        img_size = adapter.img_size

        def encode_fn(prev, curr):
            return encode_pairs_via_adapter(adapter, prev, curr, device, batch=args.encode_batch)

    print(f"  img_size={img_size}  readout={args.readout}  n_streams={n_streams}")

    # ── Demo-level train/test split ─────────────────────────────────────
    all_demo_keys = []  # list of (hdf5_path, demo_key, task_id)
    for tid, hp in zip(task_ids, hdf5_paths):
        demos = list_demos(hp)
        if args.max_demos_per_task is not None:
            demos = demos[:args.max_demos_per_task]
        for d in demos:
            with h5py.File(hp, "r") as f:
                T = f[f"data/{d}/obs/ee_pos"].shape[0]
            if T > cutoff:
                continue
            all_demo_keys.append((hp, d, tid))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_demo_keys))
    n_train = int(len(perm) * args.train_ratio)
    train_demos = [all_demo_keys[i] for i in perm[:n_train]]
    eval_demos = [all_demo_keys[i] for i in perm[n_train:]]
    print(f"  demos: train={len(train_demos)} / eval={len(eval_demos)}")

    # ── Target mode (action regression / identity classification) ──────────
    is_cls = args.target == "identity"
    n_classes = len(task_ids)
    if args.rotate_aug:
        if args.view == "both":
            raise ValueError("--rotate-aug supports a single view only (both = per-view rotation not wired)")
        rot_rng = np.random.default_rng(args.seed + 1)  # separate from the demo-split rng
        print("  rotate-aug ON: same-angle random rotation per pair (position<->identity de-confound)")
    if is_cls:
        # tid -> 0..n_classes-1 dense index (task_ids may not be contiguous)
        tid2cls = {tid: i for i, tid in enumerate(task_ids)}
        chance = 1.0 / n_classes
        print(f"  target=identity: {n_classes}-way task-id classification, chance={chance:.4f}")

    # ── Per-gap loop ─────────────────────────────────────────────────────
    cell_summaries = []
    os.makedirs(args.output_dir, exist_ok=True)

    for gap in args.gaps:
        print(f"\n=== gap={gap} ({gap/20:.2f}s @ LIBERO 20Hz) ===")
        t0 = time.time()

        # Streaming: inside the demo loop, load frames -> preprocess -> encoder
        # forward -> accumulate embeddings + targets only. Raw frames are discarded
        # immediately (to save memory).
        def collect_embed(demo_list, label):
            embed_chunks, tgt_chunks, demo_ids, pos_chunks = [], [], [], []
            for di, (hp, d, tid) in enumerate(demo_list):
                if args.position_control == "only":
                    # position control: no frames / encoder -- load ee/action only (fast on CPU)
                    with h5py.File(hp, "r") as f:
                        g = f[f"data/{d}"]
                        eef_pos = np.asarray(g["obs/ee_pos"])
                        ee_ori = np.asarray(g["obs/ee_ori"])
                        actions = np.asarray(g["actions"])
                    T = eef_pos.shape[0]
                elif args.view == "both":
                    # agentview + eye-in-hand feature-level concat
                    frames_av, eef_pos, ee_ori, actions = load_demo(hp, d, view="agentview_rgb")
                    frames_eih, _, _, _ = load_demo(hp, d, view="eye_in_hand_rgb")
                    T = frames_av.shape[0]
                else:
                    frames, eef_pos, ee_ori, actions = load_demo(hp, d, view=args.view)
                    T = frames.shape[0]
                if T <= gap + 1:
                    continue
                if is_cls:
                    tgts = np.full(T - gap, tid2cls[tid], dtype=np.int64)  # identity: task-id label
                else:
                    tgts = np.stack([
                        libero_action_target(eef_pos, ee_ori, actions, t, gap)
                        for t in range(T - gap)
                    ])
                if args.position_control == "only":
                    emb = pos_covariate(eef_pos, gap, args.pos_covariate)  # (T-gap, 3|6)
                elif args.view == "both":
                    prev_av = preprocess_frames(frames_av[:T - gap], img_size)
                    curr_av = preprocess_frames(frames_av[gap:], img_size)
                    prev_eih = preprocess_frames(frames_eih[:T - gap], img_size)
                    curr_eih = preprocess_frames(frames_eih[gap:], img_size)
                    emb_av = encode_fn(prev_av, curr_av)
                    emb_eih = encode_fn(prev_eih, curr_eih)
                    emb = torch.cat([emb_av, emb_eih], dim=-1)  # (T-gap, 2D)
                    del frames_av, frames_eih, prev_av, curr_av, prev_eih, curr_eih, emb_av, emb_eih
                else:
                    prev = preprocess_frames(frames[:T - gap], img_size)
                    curr = preprocess_frames(frames[gap:], img_size)
                    if args.rotate_aug:
                        prev, curr = rotate_pair(prev, curr, device, rot_rng)
                    emb = encode_fn(prev, curr)  # (T-gap, D), already on CPU
                    del frames, prev, curr
                if args.position_control == "concat":
                    pos_chunks.append(pos_covariate(eef_pos, gap, args.pos_covariate))
                embed_chunks.append(emb)
                tgt_chunks.append(tgts)
                demo_ids.extend([di] * (T - gap))
                del emb
            print(f"    [{label}] embedded {len(demo_ids)} pairs from {len(embed_chunks)} demos")
            return (
                torch.cat(embed_chunks, 0),
                torch.from_numpy(np.concatenate(tgt_chunks, 0)),
                np.array(demo_ids),
                torch.cat(pos_chunks, 0) if pos_chunks else None,
            )

        print(f"  encoding train (streaming per demo) ...")
        emb_tr, tgt_tr, demo_tr, pos_tr = collect_embed(train_demos, "train")
        print(f"  encoding eval  (streaming per demo) ...")
        emb_ev, tgt_ev, demo_ev, pos_ev = collect_embed(eval_demos, "eval")
        print(f"  pairs: train={len(tgt_tr)} eval={len(tgt_ev)}")

        # ── Position covariate: z-score (train stats) -> RFF expansion -- identical for only/concat ──
        extra_tr = extra_ev = None
        if args.position_control == "only":
            mu, sd = emb_tr.mean(0), emb_tr.std(0) + 1e-6
            emb_tr = rff_expand((emb_tr - mu) / sd, args.seed + 2)
            emb_ev = rff_expand((emb_ev - mu) / sd, args.seed + 2)
        elif args.position_control == "concat":
            mu, sd = pos_tr.mean(0), pos_tr.std(0) + 1e-6
            pos_tr = rff_expand((pos_tr - mu) / sd, args.seed + 2)
            pos_ev = rff_expand((pos_ev - mu) / sd, args.seed + 2)
            if args.readout == "mean":
                emb_tr = torch.cat([emb_tr, pos_tr], dim=-1)
                emb_ev = torch.cat([emb_ev, pos_ev], dim=-1)
            else:
                extra_tr, extra_ev = pos_tr, pos_ev  # attentive: concat after pooling (inside the probe)

        # Standardize target (MSE/R^2 scale-friendly).
        # NOTE: R^2 is scale-invariant, but this can help MSE / training stability -- skipped.
        # Linear probe
        print(f"  training probe (epoch={args.probe_epochs}, lr={args.probe_lr}, batch={args.probe_batch}) ...")
        best = train_probe(
            emb_tr, tgt_tr, emb_ev, tgt_ev,
            epochs=args.probe_epochs, batch_size=args.probe_batch,
            lr=args.probe_lr, device=str(device),
            readout=args.readout, n_streams=n_streams,
            weight_decay=args.probe_weight_decay,
            task=("classification" if is_cls else "regression"),
            out_dim=(n_classes if is_cls else ACTION_DIM),
            extra_train=extra_tr, extra_eval=extra_ev,
        )
        m = best["metrics"]
        elapsed = time.time() - t0

        if is_cls:
            # majority-class baseline: apply the train majority class to eval.
            # Under class imbalance this is a stricter "uninformative" reference than uniform chance (1/K).
            _tr, _ev = tgt_tr.numpy(), tgt_ev.numpy()
            maj_class = int(np.bincount(_tr, minlength=n_classes).argmax())
            majority_acc = float((_ev == maj_class).mean())
            print(f"  acc = {m['accuracy']:.4f} (chance={chance:.4f} / majority={majority_acc:.4f}, "
                  f"{n_classes}-way)  best_ep={best['epoch']}  ({elapsed:.0f}s)")
        else:
            print(f"  R² agg = {m['r2_aggregate']:+.4f}  per-dim = " +
                  " ".join(f"{r:+.3f}" for r in m["r2_per_dim"]) +
                  f"  cos = {m['cosine_sim']:+.3f}  best_ep={best['epoch']}  ({elapsed:.0f}s)")

        # Per-demo R^2 on eval set (diagnostic only; the paper table uses the aggregate).
        per_demo = []
        for di in np.unique(demo_ev):
            mask = demo_ev == di
            if mask.sum() < 5:
                continue
            ev_emb = emb_ev[mask].to(device)
            ev_tgt = tgt_ev[mask].numpy()
            with torch.no_grad():
                # Predictions from the best epoch are not cached here; per-demo is
                # skipped and only the aggregate is reported (paper table uses aggregate).
                pass

        cell_dir = Path(args.output_dir) / f"gap{gap}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "encoder": args.encoder,
            "checkpoint": args.checkpoint,
            "task_suite": args.task_suite,
            "view": args.view,
            "gap": gap,
            "gap_seconds": gap / 20.0,
            "readout": args.readout,
            "target": args.target,
            "position_control": args.position_control,
            "pos_covariate": args.pos_covariate,
            "stream_mode": args.stream_mode if args.encoder == "comp-mae" else None,
            "n_train_demos": len(train_demos),
            "n_eval_demos": len(eval_demos),
            "n_train_pairs": int(len(tgt_tr)),
            "n_eval_pairs": int(len(tgt_ev)),
            "best_epoch": best["epoch"],
            "elapsed_seconds": float(elapsed),
        }
        if is_cls:
            summary.update({"accuracy": m["accuracy"], "chance": chance,
                            "majority_acc": majority_acc, "n_classes": n_classes})
        else:
            summary.update({"r2_aggregate": m["r2_aggregate"], "r2_per_dim": m["r2_per_dim"],
                            "mse": m["mse"], "cosine_sim": m["cosine_sim"]})
        with open(cell_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        cell_summaries.append({"gap": gap, **({"accuracy": m["accuracy"], "majority_acc": majority_acc}
                                              if is_cls
                                              else {"r2_aggregate": m["r2_aggregate"],
                                                    "r2_per_dim": m["r2_per_dim"]})})
        del emb_tr, emb_ev, tgt_tr, tgt_ev

    # ── Aggregate CSV across gaps ────────────────────────────────────────
    csv_path = Path(args.output_dir) / "all_gaps.csv"
    with open(csv_path, "w", newline="") as f:
        if is_cls:
            w = csv.DictWriter(f, fieldnames=["gap", "accuracy", "majority_acc", "chance", "n_classes"])
            w.writeheader()
            for c in cell_summaries:
                w.writerow({"gap": c["gap"], "accuracy": c["accuracy"],
                            "majority_acc": c["majority_acc"], "chance": chance, "n_classes": n_classes})
        else:
            w = csv.DictWriter(f, fieldnames=["gap", "r2_aggregate"] + [f"r2_dim{i}" for i in range(ACTION_DIM)])
            w.writeheader()
            for c in cell_summaries:
                row = {"gap": c["gap"], "r2_aggregate": c["r2_aggregate"]}
                for i, r in enumerate(c["r2_per_dim"]):
                    row[f"r2_dim{i}"] = r
                w.writerow(row)
    print(f"\nAll gaps CSV: {csv_path}")
    print(f"\n=== Final summary (target={args.target}) ===")
    if is_cls:
        print(f"{'gap':>4}  {'acc':>8}  (chance={chance:.4f})")
        for c in cell_summaries:
            print(f"{c['gap']:>4d}  {c['accuracy']:>8.4f}")
    else:
        print(f"{'gap':>4}  {'r2_agg':>8}  per-dim r²")
        for c in cell_summaries:
            per = " ".join(f"{r:+.3f}" for r in c["r2_per_dim"])
            print(f"{c['gap']:>4d}  {c['r2_aggregate']:>+8.4f}  {per}")


if __name__ == "__main__":
    main()
