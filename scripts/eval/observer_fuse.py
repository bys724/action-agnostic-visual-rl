#!/usr/bin/env python
"""Layer-1 observer Fuse on a frozen CoMP encoder (STATUS.md 10-09 Vault decision).

Claim under test: "a stale P anchor + cheap M chunks keeps the state as well as re-encoding P+P".
  z_PP = Fuse({anchor P, current P})          (expensive: P encoder every step)
  z_PM = Fuse({anchor P, M chunk sequence})   (cheap: P once, then M only)

Frozen tokens are cached on a 5-frame grid (LIBERO 20Hz → 0.25s units):
  P[g]          = P(frame g)                    (n_patch, D)
  M[g, l-1]     = M(ΔL(frame g, frame g+5l))    l = 1..4 units (span 5..20 frames)
Training (mixed gaps): anchor age k ∈ {1..4} units, random composition of k into chunks.
  student input = PP or PM (random per batch, never all three → M no-op shortcut)
  loss = recon(P_t tokens) + recon(M(anchor→t) tokens) + ‖z_student − sg(EMA_teacher(PP))‖²
Evaluation (user 10-09): anchor age = 20 frames (1s) only. Attentive probe fit on z_PP(train) is
frozen and fed z_PM / anchor-only. Pass threshold (fixed 10-09): z_PM ≥ 90% of z_PP for
motion R² and identity acc (primary = 5-frame × 4 chunks).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "eval"))

from probe_action_libero import (  # noqa: E402  same-probe helpers (parity)
    AttentivePoolProbe, _suite_split, build_parvo_encoder, compute_metrics, encode_m_sparse,
    libero_action_target, load_demo, preprocess_frames,
)

UNIT = 5        # grid stride in frames (0.25s)
MAX_UNITS = 4   # max anchor age / chunk span = 20 frames (1s)
EVAL_SPLITS = {"pm_g5": [1, 1, 1, 1], "pm_g10": [2, 2], "pm_g20": [4]}


# ─────────────────────────────────────────────────────────────────────────
# Frozen token cache
# ─────────────────────────────────────────────────────────────────────────

class DinoP(torch.nn.Module):
    """Generic P baseline: frozen DINOv2-base. 196×196 input (ImageNet norm = DINO native) → 14×14 = 196
    patch tokens (768-d), grid-aligned with the 16-px ΔL patches on 224. M side must be raw ΔL."""

    def __init__(self, device):
        super().__init__()
        from transformers import AutoModel
        from src.models.common.preprocessing import TwoStreamPreprocessing
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        self.preprocessing = TwoStreamPreprocessing(use_sobel=False).to(device)  # ΔL only
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def p_tokens(self, x):  # x: (n, 3, 224, 224) [0,1]
        x = F.interpolate(x, size=(196, 196), mode="bilinear", align_corners=False)
        return self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 1:]


@torch.no_grad()
def build_cache(model, demos, device, batch=64, m_source="comp", m_sparse_tau=None):
    """→ dict(P (Ng,N,D) fp16, M (Ng,4,N,D) fp16, demo_start, n_grid, motion (Ng,7), tid (Ng,))
    motion[g] = libero_action_target(frame g − 20, 20) (same-probe gap20 target, ending at g).
    m_source: comp = encoder M tokens / raw = untrained ΔL 16×16 patches (256-d, zero-padded to D;
    same ΔL and pad rule as probe_action_libero raw-dl floor) — anchor P stays the encoder's.
    m_sparse_tau (opt-in, comp only): also store M_sp = sparse-M tokens (encoder on mean|ΔL|>tau patches
    only, mask token elsewhere) + nvis (Ng,4) visible counts, next to the full M (paired comparison)."""
    Ps, Ms, mot, tids, starts, ngrid, eep = [], [], [], [], [], [], []
    Msp, nvis = [], []
    off = 0
    for hp, d, tid in demos:
        frames, ee_pos, ee_ori, actions = load_demo(hp, d)
        T = frames.shape[0]
        grid = list(range(0, T, UNIT))
        x = preprocess_frames(frames, 224)
        p_tok, m_tok = [], []
        for s in range(0, len(grid), batch):
            g = grid[s:s + batch]
            if isinstance(model, DinoP):
                p_tok.append(model.p_tokens(x[g].to(device)).half())
            else:
                p = model.preprocessing.compute_p_channel(x[g].to(device))
                p_tok.append(model._encode_p_unmasked(p)[:, 1:].half())
            m_l, sp_l, nv_l = [], [], []
            for l in range(1, MAX_UNITS + 1):
                g2 = [min(i + UNIT * l, T - 1) for i in g]  # out-of-range spans never sampled
                m = model.preprocessing.compute_m_channel(x[g].to(device), x[g2].to(device))
                if m_source == "raw":
                    t = F.unfold(m, kernel_size=16, stride=16).transpose(1, 2)        # (n, N, 256)
                    m_l.append(F.pad(t, (0, p_tok[-1].shape[-1] - t.shape[-1])).half())
                else:
                    m_l.append(model._encode_m_unmasked(m)[:, 1:].half())
                    if m_sparse_tau is not None:
                        t_sp, nv = encode_m_sparse(model, m, m_sparse_tau)
                        sp_l.append(t_sp.half()); nv_l.append(nv.cpu())
            m_tok.append(torch.stack(m_l, 1))
            if sp_l:
                Msp.append(torch.stack(sp_l, 1)); nvis.append(torch.stack(nv_l, 1))
        Ps.append(torch.cat(p_tok)); Ms.append(torch.cat(m_tok))
        mot.append(np.stack([libero_action_target(ee_pos, ee_ori, actions, g - UNIT * MAX_UNITS,
                                                  UNIT * MAX_UNITS) if g >= UNIT * MAX_UNITS
                             else np.zeros(7, np.float32) for g in grid]))
        tids += [tid] * len(grid)
        eep.append(ee_pos[grid].astype(np.float32))
        starts.append(off); ngrid.append(len(grid)); off += len(grid)
    out = {"P": torch.cat(Ps), "M": torch.cat(Ms),
           "motion": torch.from_numpy(np.concatenate(mot)), "tid": torch.tensor(tids),
           "eepos": torch.from_numpy(np.concatenate(eep)),
           "start": np.array(starts), "n": np.array(ngrid)}
    if Msp:
        out["M_sp"], out["nvis"] = torch.cat(Msp), torch.cat(nvis)
    return out


def grid_index(cache, min_units):
    """Global grid indices g (current frame) whose anchor g − min_units stays in the same demo."""
    return np.concatenate([np.arange(s + min_units, s + n) for s, n in zip(cache["start"], cache["n"])])


# ─────────────────────────────────────────────────────────────────────────
# Fuse
# ─────────────────────────────────────────────────────────────────────────

class Fuse(nn.Module):
    """Perceiver-style: latent queries ← cross-attn(input tokens) → self-attn → z (n_lat tokens, no CLS).
    Input token = proj(token) + shared patch pos + type (anchorP/currentP/M) [+ M chunk start/len]."""

    def __init__(self, dim, n_patch=196, width=384, n_lat=32, depth=3, heads=6):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        self.pos = nn.Parameter(torch.randn(1, n_patch, width) * 0.02)
        self.tok_type = nn.Embedding(3, width)            # 0 anchor P / 1 current P / 2 M
        self.m_start = nn.Embedding(MAX_UNITS, width)  # chunk start (units after anchor)
        self.m_len = nn.Embedding(MAX_UNITS, width)    # chunk span − 1 (units)
        self.lat = nn.Parameter(torch.randn(1, n_lat, width) * 0.02)
        layer = dict(d_model=width, nhead=heads, dim_feedforward=4 * width, batch_first=True, norm_first=True)
        self.cross = nn.TransformerDecoderLayer(**layer)
        self.self_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(**layer), depth)
        self.norm = nn.LayerNorm(width)
        # decoder: per-target output queries (P_t tokens / M(anchor→t) tokens) ← cross-attn(z)
        self.q_type = nn.Embedding(2, width)
        self.dec = nn.TransformerDecoderLayer(**layer)
        self.out = nn.Linear(width, dim)

    def tok(self, x, typ, start=None, length=None):
        e = self.inp(x.float()) + self.pos + self.tok_type.weight[typ]
        if typ == 2:
            e = e + self.m_start.weight[start] + self.m_len.weight[length]
        return e

    def encode(self, anchor, current=None, chunks=()):
        """chunks: [(M tokens (B,N,D), start_unit, span_units)]"""
        toks = [self.tok(anchor, 0)]
        if current is not None:
            toks.append(self.tok(current, 1))
        toks += [self.tok(m, 2, s, l - 1) for m, s, l in chunks]
        x = torch.cat(toks, 1)
        z = self.cross(self.lat.expand(x.shape[0], -1, -1), x)
        return self.norm(self.self_attn(z))

    def decode(self, z):
        B = z.shape[0]
        q = torch.cat([self.pos + self.q_type.weight[0], self.pos + self.q_type.weight[1]], 1)
        y = self.out(self.dec(q.expand(B, -1, -1), z))
        n = self.pos.shape[1]
        return y[:, :n], y[:, n:]  # (P_t, M(anchor→t))


@torch.no_grad()
def token_stats(x, chunk=1024):
    """Per-(patch, dim) mean/std over the first axis (train cache) — chunked fp32 accumulation."""
    s = torch.zeros(x.shape[-2:], device=x.device); s2 = torch.zeros_like(s)
    flat = x.reshape(-1, *x.shape[-2:])
    for i in range(0, len(flat), chunk):
        c = flat[i:i + chunk].float(); s += c.sum(0); s2 += (c * c).sum(0)
    mu = s / len(flat)
    return mu, (s2 / len(flat) - mu * mu).clamp_min(1e-8).sqrt()


def _std(cache, key, t):
    """Standardize frozen tokens per (patch, dim) with train stats: the shared per-position component
    dominates the raw tokens (frame-to-frame std 0.06 ≪ |token|), so without this the Fuse/recon loss
    is solved by emitting the mean token and ignores its input (smoke 41466897/898)."""
    mu, sd = cache["stats"][key]
    return (t.float() - mu) / sd


def inputs(cache, g, k, mode, split=None):
    """mode: pp / pm / anchor. split = chunk spans (units) summing to k."""
    a = g - k
    anchor = _std(cache, "P", cache["P"][a])
    if mode == "pp":
        return dict(anchor=anchor, current=_std(cache, "P", cache["P"][g]))
    if mode == "anchor":
        return dict(anchor=anchor)
    chunks, s = [], 0
    for l in split:
        chunks.append((_std(cache, "M", cache["M"][a + s, l - 1]), s, l)); s += l
    return dict(anchor=anchor, chunks=chunks)


def targets(cache, g, k):
    return _std(cache, "P", cache["P"][g]), _std(cache, "M", cache["M"][g - k, k - 1])


def random_split(k, rng):
    out = []
    while k:
        l = int(rng.integers(1, k + 1)); out.append(l); k -= l
    return out


# ─────────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────────

def train_fuse(fuse, cache, args, device):
    """Train until convergence (fair across arms with different speeds — user 10-09).
    Stop: val loss (fixed held-out train demos, fixed samples) checked every `eval_every` steps; stop after
    `patience` checks without >`min_rel` relative improvement over best; restore best weights. LR = linear
    warmup then constant (no total-step-dependent schedule). Returns convergence/cost record."""
    rng = np.random.default_rng(args.seed)
    teacher = copy.deepcopy(fuse).requires_grad_(False).eval()  # targets without dropout
    opt = torch.optim.AdamW(fuse.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    # held-out val demos = last 10% of train demos (by demo, never the eval split)
    n_val = max(1, len(cache["start"]) // 10)
    val_demo = np.zeros(len(cache["P"]), bool)
    for st, n in zip(cache["start"][-n_val:], cache["n"][-n_val:]):
        val_demo[st:st + n] = True
    idx = {k: grid_index(cache, k) for k in range(1, MAX_UNITS + 1)}
    tr_idx = {k: v[~val_demo[v]] for k, v in idx.items()}
    vrng = np.random.default_rng(12345)  # fixed val batches, same for every arm
    val_batches = []
    for _ in range(16):
        k = int(vrng.integers(1, MAX_UNITS + 1))
        g = vrng.choice(idx[k][val_demo[idx[k]]], args.batch)
        mode = "pp" if args.pp_only or vrng.random() < 0.5 else "pm"
        val_batches.append((k, g, mode, random_split(k, vrng)))
    m_w = 0.0 if args.pp_only else args.m_weight

    def losses(model, k, g, mode, split):
        z = model.encode(**inputs(cache, g, k, mode, split))
        p_hat, m_hat = model.decode(z)
        p_tgt, m_tgt = targets(cache, g, k)
        with torch.no_grad():
            z_t = teacher.encode(**inputs(cache, g, k, "pp"))
        l_p, l_m, l_z = F.mse_loss(p_hat, p_tgt), F.mse_loss(m_hat, m_tgt), F.mse_loss(z, z_t)
        return l_p + m_w * l_m + args.align_weight * l_z, l_p, l_m, l_z

    @torch.no_grad()
    def val_loss():
        fuse.eval()
        v = np.mean([[t.item() for t in losses(fuse, k, torch.from_numpy(g).to(device), m, sp)]
                     for k, g, m, sp in val_batches], 0)
        fuse.train()
        return v

    t0 = time.time()
    best, best_step, bad, curve, best_state = float("inf"), 0, 0, [], None
    for step in range(1, args.steps + 1):
        k = int(rng.integers(1, MAX_UNITS + 1))                       # mixed anchor ages
        g = torch.from_numpy(rng.choice(tr_idx[k], args.batch)).to(device)
        mode = "pp" if args.pp_only or rng.random() < 0.5 else "pm"  # never PP+M together
        loss, l_p, l_m, l_z = losses(fuse, k, g, mode, random_split(k, rng))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for pt, ps in zip(teacher.parameters(), fuse.parameters()):
                pt.mul_(args.ema).add_(ps.detach(), alpha=1 - args.ema)
        if step % args.eval_every == 0:
            v = val_loss()
            curve.append([step, round(time.time() - t0, 1), *[float(x) for x in v]])
            print(f"  step {step:6d}  val total {v[0]:.4f}  P {v[1]:.4f}  M {v[2]:.4f}  z {v[3]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if v[0] < best * (1 - args.min_rel):
                best, best_step, bad = v[0], step, 0
                best_state = copy.deepcopy(fuse.state_dict())
            else:
                bad += 1
                if bad >= args.patience:
                    break
    fuse.load_state_dict(best_state)
    return {"stop_step": step, "best_step": best_step, "best_val": best, "train_seconds": round(time.time() - t0, 1),
            "converged": bad >= args.patience, "curve": curve,
            "curve_cols": ["step", "seconds", "val_total", "val_P", "val_M", "val_z"]}


@torch.no_grad()
def embed_all(fuse, cache, g_all, device, bs=256):
    """k = 20 frames. → {cond: (z (n,n_lat,W) fp16, P recon mse, M recon mse)}"""
    k = MAX_UNITS
    conds = {"pp": ("pp", None), "anchor": ("anchor", None), **{c: ("pm", s) for c, s in EVAL_SPLITS.items()}}
    out = {}
    for name, (mode, split) in conds.items():
        zs, ep, em = [], [], []
        for i in range(0, len(g_all), bs):
            g = torch.from_numpy(g_all[i:i + bs]).to(device)
            z = fuse.encode(**inputs(cache, g, k, mode, split))
            p_hat, m_hat = fuse.decode(z)
            p_tgt, m_tgt = targets(cache, g, k)
            zs.append(z.half().cpu())
            ep.append(((p_hat - p_tgt) ** 2).mean((1, 2)).cpu()); em.append(((m_hat - m_tgt) ** 2).mean((1, 2)).cpu())
        out[name] = (torch.cat(zs), float(torch.cat(ep).mean()), float(torch.cat(em).mean()))
    return out


def fit_probe(x, y, task, out_dim, args, device, seed=None, extra=None):
    """Same attentive probe as same-probe, fixed epochs, final weights (frozen for all conditions).
    extra = position covariate (RFF of anchor ee_pos), concatenated after pooling (same-probe concat)."""
    torch.manual_seed(args.seed if seed is None else seed)
    probe = AttentivePoolProbe(x.shape[2], n_streams=1, n_patch=x.shape[1], action_dim=out_dim,
                               extra_dim=0 if extra is None else extra.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    for _ in range(args.probe_epochs):
        for i in torch.randperm(len(x)).split(256):
            pred = probe(x[i].to(device).float(), None if extra is None else extra[i].to(device))
            yy = y[i].to(device)
            loss = F.cross_entropy(pred, yy) if task == "cls" else F.mse_loss(pred, yy)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe.eval()


def probe_curve(x, y, xe, ye, task, out_dim, args, device, seed, every=10):
    """Diagnostic (--probe-seeds): fit_probe loop with eval/train score every `every` epochs."""
    torch.manual_seed(seed)
    probe = AttentivePoolProbe(x.shape[2], n_streams=1, n_patch=x.shape[1], action_dim=out_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    curve = []
    for ep in range(args.probe_epochs):
        probe.train()
        for i in torch.randperm(len(x)).split(256):
            pred = probe(x[i].to(device).float())
            yy = y[i].to(device)
            loss = F.cross_entropy(pred, yy) if task == "cls" else F.mse_loss(pred, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % every == 0 or ep + 1 == args.probe_epochs:
            probe.eval()
            curve.append({"epoch": ep + 1, "eval": score(probe, xe, ye, task, device),
                          "train": score(probe, x, y, task, device)})
    return {"final_eval": curve[-1]["eval"], "final_train": curve[-1]["train"],
            "best_eval": max(c["eval"] for c in curve), "curve": curve}


@torch.no_grad()
def predict(probe, x, device, extra=None):
    return torch.cat([probe(x[i:i + 256].to(device).float(),
                            None if extra is None else extra[i:i + 256].to(device)).cpu()
                      for i in range(0, len(x), 256)])


@torch.no_grad()
def score(probe, x, y, task, device, extra=None):
    pred = predict(probe, x, device, extra)
    if task == "cls":
        return float((pred.argmax(1) == y).float().mean())
    return compute_metrics(pred.numpy(), y.numpy())["r2_aggregate"]


# ─────────────────────────────────────────────────────────────────────────
# Layer-2 hand-off (SC repo) · FLOPs
# ─────────────────────────────────────────────────────────────────────────

class Observer:
    """Inference entry point for downstream (layer 2). Loads a hand-off file written by --export-handoff
    (frozen encoder ckpt path + Fuse weights + train token stats + z stats). Frames = LIBERO agentview
    uint8 (H, W, 3) or a batch (n, H, W, 3). z = (n, n_lat, width) token set (no CLS).
      z_pp(anchor, current)                 — expensive path (P on both frames)
      z_pm(anchor, ends, spans)             — cheap path: ends[i] = frame at the end of chunk i,
                                              spans[i] = chunk length in 5-frame units (sum ≤ 4 → ≤ 1s)"""

    def __init__(self, handoff, device="cuda"):
        h = torch.load(handoff, map_location=device, weights_only=False)
        m = h["meta"]
        self.device, self.meta = torch.device(device), m
        self.enc = build_parvo_encoder(m["encoder_ckpt"], self.device)
        self.fuse = Fuse(m["dim"], n_patch=m["n_patch"], n_lat=m["n_lat"], depth=m["depth"]).to(self.device)
        self.fuse.load_state_dict(h["fuse"]); self.fuse.eval()
        self.cache = {"stats": h["input_stats"]}
        self.z_stats = h["z_stats"]  # (mu, sd) per (latent, dim) over train z_PP — optional for downstream

    def _x(self, f):
        f = np.asarray(f)
        return preprocess_frames(f[None] if f.ndim == 3 else f, 224).to(self.device)

    @torch.no_grad()
    def _p(self, f):
        # .half() = same rounding as the training cache (fp16 tokens) → z matches the training path
        return _std(self.cache, "P", self.enc._encode_p_unmasked(self.enc.preprocessing.compute_p_channel(self._x(f)))[:, 1:].half())

    @torch.no_grad()
    def _m(self, a, b):
        assert self.meta["m_source"] == "comp", "raw-ΔL hand-off not supported"
        mc = self.enc.preprocessing.compute_m_channel(self._x(a), self._x(b))
        return _std(self.cache, "M", self.enc._encode_m_unmasked(mc)[:, 1:].half())

    @torch.no_grad()
    def z_pp(self, anchor, current):
        return self.fuse.encode(anchor=self._p(anchor), current=self._p(current))

    @torch.no_grad()
    def z_pm(self, anchor, ends, spans):
        assert len(ends) == len(spans) and 0 < sum(spans) <= MAX_UNITS
        chunks, prev, s0 = [], anchor, 0
        for f, l in zip(ends, spans):
            chunks.append((self._m(prev, f), s0, l)); prev, s0 = f, s0 + l
        return self.fuse.encode(anchor=self._p(anchor), chunks=chunks)


def measure_flops(args, device):
    """Per-component forward FLOPs at batch 1 (torch FlopCounterMode; matmul/conv/attention only)."""
    from torch.utils.flop_counter import FlopCounterMode

    def fl(fn):
        with FlopCounterMode(display=False) as fc:
            fn()
        return int(fc.get_total_flops())

    enc = build_parvo_encoder(args.checkpoint, device)
    x = torch.rand(1, 3, 224, 224, device=device)
    D = enc.pos_embed_p.shape[-1]
    out = {"P_enc": fl(lambda: enc._encode_p_unmasked(enc.preprocessing.compute_p_channel(x))),
           "M_enc": fl(lambda: enc._encode_m_unmasked(enc.preprocessing.compute_m_channel(x, x)))}
    dino = DinoP(device)
    out["DINOv2_P_enc"] = fl(lambda: dino.p_tokens(x))
    fuse = Fuse(D, n_patch=196, n_lat=args.n_lat, depth=args.depth).to(device).eval()
    t = torch.rand(1, 196, D, device=device)
    # no torch.no_grad() here: FlopCounterMode's module tracker asserts on grad-less outputs (forward FLOPs unchanged)
    out["Fuse_PP"] = fl(lambda: fuse.encode(anchor=t, current=t))
    out["Fuse_anchor"] = fl(lambda: fuse.encode(anchor=t))
    for n, sp in EVAL_SPLITS.items():
        out[f"Fuse_{n}"] = fl(lambda: fuse.encode(anchor=t, chunks=[(t, i, l) for i, l in enumerate(sp)]))
    # per z update, 1s anchor (P refreshed every 20 frames = every 4 updates of 5 frames)
    out["update_PP"] = out["P_enc"] + out["Fuse_PP"]
    out["update_PM_g5"] = out["M_enc"] + out["Fuse_pm_g5"] + out["P_enc"] / 4
    out["note"] = ("update_PP = fresh P on current frame + Fuse(anchor,current). update_PM_g5 = one new 5-frame M chunk "
                   "+ Fuse(anchor, 4 chunks) + anchor P amortized over 4 updates. Fuse cost = worst case (4 chunks).")
    print(json.dumps(out, indent=1))
    os.makedirs(args.out_root, exist_ok=True)
    with open(f"{args.out_root}/flops.json", "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task-suite", default="libero_object")
    ap.add_argument("--data-root", default="/proj/external_group/mrg/datasets/libero")
    ap.add_argument("--m-source", default="comp", choices=["comp", "raw"])
    ap.add_argument("--p-source", default="ckpt", choices=["ckpt", "dino"], help="dino = DINOv2-base P (needs --m-source raw)")
    ap.add_argument("--max-demos", type=int, default=None, help="smoke: cap demos per split")
    ap.add_argument("--steps", type=int, default=60000, help="max steps (convergence stop usually earlier)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=4, help="val checks without improvement before stop")
    ap.add_argument("--min-rel", type=float, default=0.005, help="relative val improvement that counts")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--align-weight", type=float, default=1.0)
    ap.add_argument("--n-lat", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--probe-epochs", type=int, default=100)  # ≈ same-probe step count on full data
    ap.add_argument("--seed", type=int, default=42)
    # diagnostics (opt-in; defaults reproduce the original runs)
    ap.add_argument("--split-seed", type=int, default=42, help="demo split seed (fixed = same-probe split)")
    ap.add_argument("--pp-only", action="store_true",
                    help="common P+P ceiling: train on PP inputs only, no M recon (M-free denominator)")
    ap.add_argument("--m-weight", type=float, default=1.0, help="M-span recon loss weight")
    ap.add_argument("--load-fuse", default=None, help="skip training, load this fuse.pt")
    ap.add_argument("--probe-seeds", type=int, nargs="*", default=[], help="extra probe-only seed sweep (motion)")
    ap.add_argument("--flops", action="store_true", help="measure per-component FLOPs and exit")
    ap.add_argument("--export-handoff", default=None, help="write layer-2 hand-off file (fuse + stats) here")
    ap.add_argument("--stats-from", default=None,
                    help="transfer: take input standardization stats from this fuse.pt / hand-off (source suite)")
    ap.add_argument("--pos-control", action="store_true",
                    help="beyond-position motion probe (same-probe concat covariate, anchor ee_pos)")
    ap.add_argument("--diag", action="store_true", help="print cache hashes / standardized-token stats")
    ap.add_argument("--m-sparse-tau", type=float, default=None,
                    help="sparse-M check (needs --load-fuse --pos-control): eval M cache also built from patches with "
                         "mean|ΔL|>tau only; train cache / standardization stats stay full-token; paired demo bootstrap")
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "paper_artifacts" / "observer_fuse"))
    ap.add_argument("--ckpt-root", default="/proj/external_group/mrg/checkpoints/observer_fuse")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    t0 = time.time()
    if args.flops:
        return measure_flops(args, device)

    # same demo split as same-probe (99th-pct length cutoff, seed 42, 0.8)
    tr_demos, ev_demos = _suite_split(args.task_suite, args.data_root,
                                      args.seed if args.split_seed is None else args.split_seed, 0.8, 99.0)
    if args.max_demos:
        tr_demos, ev_demos = tr_demos[:args.max_demos], ev_demos[:max(2, args.max_demos // 4)]
    if args.p_source == "dino":
        assert args.m_source == "raw", "DINO P has no M stream → --m-source raw"
        enc = DinoP(device)
    else:
        enc = build_parvo_encoder(args.checkpoint, device)
    cache_tr = {k: (v.to(device) if k in ("P", "M") else v) for k, v in build_cache(enc, tr_demos, device, m_source=args.m_source).items()}
    cache_ev = {k: (v.to(device) if k in ("P", "M", "M_sp") else v)
                for k, v in build_cache(enc, ev_demos, device, m_source=args.m_source, m_sparse_tau=args.m_sparse_tau).items()}
    del enc
    stats = {"P": token_stats(cache_tr["P"]), "M": token_stats(cache_tr["M"])}  # train stats for both splits
    if args.stats_from:  # transfer (no training): use the loaded Fuse's own source-suite stats
        src = torch.load(args.stats_from, map_location=device, weights_only=False)["input_stats"]
        stats = {k: (mu.to(device), sd.to(device)) for k, (mu, sd) in src.items()}
    cache_tr["stats"] = cache_ev["stats"] = stats
    print(f"cache: train grid {len(cache_tr['P'])} / eval grid {len(cache_ev['P'])}  ({time.time()-t0:.0f}s)", flush=True)
    _p = cache_tr["P"][:2000].float()
    print(f"diag: P std across frames {_p.std(0).mean():.4f} / |P| {_p.abs().mean():.4f} · "
          f"tasks train {np.bincount(cache_tr['tid'].numpy()).tolist()} eval {np.bincount(cache_ev['tid'].numpy()).tolist()}", flush=True)
    del _p
    if args.diag:
        import hashlib
        for nm, c in (("train", cache_tr), ("eval", cache_ev)):
            for key in ("P", "M"):
                h = hashlib.sha1(c[key][:500].cpu().numpy().tobytes()).hexdigest()[:16]
                print(f"diag hash {nm} {key}[:500] {h}", flush=True)
        zm = torch.cat([_std(cache_tr, "M", cache_tr["M"][i:i + 256]).abs().flatten()[::97]
                        for i in range(0, len(cache_tr["M"]), 256)])
        zp = torch.cat([_std(cache_tr, "P", cache_tr["P"][i:i + 256]).abs().flatten()[::97]
                        for i in range(0, len(cache_tr["P"]), 256)])
        for nm, z in (("P", zp), ("M", zm)):
            q = torch.quantile(z[::max(1, len(z) // 2_000_000)].float(), torch.tensor([0.5, 0.99, 0.9999], device=z.device))
            print(f"diag std-{nm} |z|: median {q[0]:.3f} p99 {q[1]:.2f} p99.99 {q[2]:.1f} max {z.max():.1f} "
                  f"frac>10 {(z > 10).float().mean():.2e} mean z^2 {(z.float() ** 2).mean():.3f}", flush=True)
        del zm, zp

    fuse = Fuse(cache_tr["P"].shape[-1], n_patch=cache_tr["P"].shape[1], n_lat=args.n_lat, depth=args.depth).to(device)
    train_rec = None
    if args.load_fuse:
        fuse.load_state_dict(torch.load(args.load_fuse, map_location=device)["fuse"])
    else:
        train_rec = train_fuse(fuse, cache_tr, args, device)
    fuse.eval()
    if not args.load_fuse:
        os.makedirs(f"{args.ckpt_root}/{args.tag}", exist_ok=True)
        torch.save({"fuse": fuse.state_dict(), "args": vars(args),
                    "input_stats": {k: (mu.cpu(), sd.cpu()) for k, (mu, sd) in stats.items()}},
                   f"{args.ckpt_root}/{args.tag}/fuse.pt")

    g_tr, g_ev = grid_index(cache_tr, MAX_UNITS), grid_index(cache_ev, MAX_UNITS)
    if args.m_sparse_tau is not None:  # sparse twin of the eval cache (same P / stats / targets except M)
        assert args.load_fuse and args.pos_control and args.m_source == "comp"
        cache_ev_sp = {**cache_ev, "M": cache_ev.pop("M_sp")}
        emb_ev_sp = embed_all(fuse, cache_ev_sp, g_ev, device)
    emb_tr, emb_ev = embed_all(fuse, cache_tr, g_tr, device), embed_all(fuse, cache_ev, g_ev, device)
    if args.export_handoff:
        zt = emb_tr["pp"][0].float()
        os.makedirs(os.path.dirname(args.export_handoff), exist_ok=True)
        torch.save({"fuse": fuse.state_dict(),
                    "input_stats": {k: (mu.cpu(), sd.cpu()) for k, (mu, sd) in stats.items()},
                    "z_stats": (zt.mean(0), zt.std(0)),
                    "meta": {"encoder_ckpt": args.checkpoint, "m_source": args.m_source, "dim": cache_tr["P"].shape[-1],
                             "n_patch": cache_tr["P"].shape[1], "n_lat": args.n_lat, "depth": args.depth,
                             "unit_frames": UNIT, "max_units": MAX_UNITS, "task_suite": args.task_suite,
                             "split_seed": args.split_seed, "tag": args.tag,
                             "fuse_ckpt": args.load_fuse or f"{args.ckpt_root}/{args.tag}/fuse.pt",
                             "entry": "scripts/eval/observer_fuse.py:Observer"}}, args.export_handoff)
        print(f"hand-off written: {args.export_handoff}", flush=True)
    tid2cls = {t: i for i, t in enumerate(sorted(set(cache_tr["tid"].tolist()) | set(cache_ev["tid"].tolist())))}
    ys = {"motion": (cache_tr["motion"][g_tr], cache_ev["motion"][g_ev], "reg", 7),
          "identity": (torch.tensor([tid2cls[t] for t in cache_tr["tid"][g_tr].tolist()]),
                       torch.tensor([tid2cls[t] for t in cache_ev["tid"][g_ev].tolist()]), "cls", len(tid2cls))}

    res = {"args": vars(args), "n_train": len(g_tr), "n_eval": len(g_ev), "train": train_rec, "recon": {}, "probe": {}}
    for c, (_, ep, em) in emb_ev.items():
        res["recon"][c] = {"P_t_mse": ep, "M_span_mse": em}
    for name, (ytr, yev, task, od) in ys.items():
        probe = fit_probe(emb_tr["pp"][0], ytr, task, od, args, device)  # fit on fresh P+P only
        res["probe"][name] = {c: score(probe, emb_ev[c][0], yev, task, device) for c in emb_ev}
        # reference: same probe class directly on frozen [P(t−20), P(t)] tokens (Fuse bottleneck loss)
        x_tr = torch.cat([cache_tr["P"][g_tr - MAX_UNITS], cache_tr["P"][g_tr]], 1).cpu()
        x_ev = torch.cat([cache_ev["P"][g_ev - MAX_UNITS], cache_ev["P"][g_ev]], 1).cpu()
        pr = fit_probe(x_tr, ytr, task, od, args, device)
        res["probe"][name]["enc_pp_direct"] = score(pr, x_ev, yev, task, device)

    if args.probe_seeds:  # diagnostic: probe-only seed variance on the SAME inputs (motion)
        ytr, yev, task, od = ys["motion"]
        x_tr = torch.cat([cache_tr["P"][g_tr - MAX_UNITS], cache_tr["P"][g_tr]], 1).cpu()
        x_ev = torch.cat([cache_ev["P"][g_ev - MAX_UNITS], cache_ev["P"][g_ev]], 1).cpu()
        res["probe_seed_sweep"] = {
            src: {s: probe_curve(xt, ytr, xe, yev, task, od, args, device, s) for s in args.probe_seeds}
            for src, (xt, xe) in {"z_pp": (emb_tr["pp"][0], emb_ev["pp"][0]), "enc_pp_direct": (x_tr, x_ev)}.items()}
        for src, d in res["probe_seed_sweep"].items():
            for s, r in d.items():
                print(f"probe-seed {src} s{s}: final eval {r['final_eval']:.4f} best eval {r['best_eval']:.4f} "
                      f"final train {r['final_train']:.4f}", flush=True)
        del x_tr, x_ev

    if args.pos_control:  # same-probe beyond-position: covariate = ee_pos at anchor (t−20) → z-score → RFF
        from probe_action_libero import rff_expand
        ytr, yev, task, od = ys["motion"]
        p_tr, p_ev = cache_tr["eepos"][g_tr - MAX_UNITS], cache_ev["eepos"][g_ev - MAX_UNITS]
        mu, sd = p_tr.mean(0), p_tr.std(0) + 1e-6
        r_tr, r_ev = rff_expand((p_tr - mu) / sd, args.seed + 2), rff_expand((p_ev - mu) / sd, args.seed + 2)
        torch.manual_seed(args.seed)
        lin = nn.Linear(r_tr.shape[1], od).to(device)  # position only (same epochs/lr)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
        for _ in range(args.probe_epochs):
            for i in torch.randperm(len(r_tr)).split(256):
                loss = F.mse_loss(lin(r_tr[i].to(device)), ytr[i].to(device))
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pos_only = compute_metrics(lin(r_ev.to(device)).cpu().numpy(), yev.numpy())["r2_aggregate"]
        probe = fit_probe(emb_tr["pp"][0], ytr, task, od, args, device, extra=r_tr)  # fit on z_PP + pos only
        concat = {c: score(probe, emb_ev[c][0], yev, task, device, extra=r_ev) for c in emb_ev}
        res["probe_pos"] = {"pos_only": pos_only, "concat": concat,
                            "beyond": {c: v - pos_only for c, v in concat.items()}}
        print("pos-control:", json.dumps(res["probe_pos"]), flush=True)
        if args.m_sparse_tau is not None:
            # pre-fixed rule (종합 지시 ③): maintained ⇔ 95% CI of paired diff of beyond-position share contains 0.
            # pos-only term cancels → diff = R²(z_full+pos) − R²(z_sparse+pos), same fixed probe, demo bootstrap.
            from sparse_m import paired_bootstrap
            grp = np.searchsorted(cache_ev["start"], g_ev, side="right") - 1   # eval demo of each sample
            sp = {"tau": args.m_sparse_tau, "nvis_5frame": np.bincount(cache_ev["nvis"][:, 0].numpy(), minlength=197).tolist(),
                  "concat_sparse": {}, "beyond_sparse": {}, "probe_sparse": {}}
            for c in ("pm_g5", "pm_g10", "pm_g20"):
                pf = predict(probe, emb_ev[c][0], device, r_ev).numpy()
                ps = predict(probe, emb_ev_sp[c][0], device, r_ev).numpy()
                r2s = compute_metrics(ps, yev.numpy())["r2_aggregate"]
                sp["concat_sparse"][c], sp["beyond_sparse"][c] = r2s, r2s - pos_only
                sp.setdefault("paired_diff", {})[c] = paired_bootstrap(pf, ps, yev.numpy(), grp, "r2", n_boot=2000, seed=0)
            for name, (ytr_, yev_, task_, od_) in ys.items():   # plain (no-pos) probes, fit on z_PP as above
                pr_ = fit_probe(emb_tr["pp"][0], ytr_, task_, od_, args, device)
                sp["probe_sparse"][name] = {c: score(pr_, emb_ev_sp[c][0], yev_, task_, device) for c in EVAL_SPLITS}
                sp["probe_sparse"][name]["full_refit"] = {c: score(pr_, emb_ev[c][0], yev_, task_, device) for c in EVAL_SPLITS}
            d = sp["paired_diff"]["pm_g5"]
            sp["verdict_maintained_pm_g5"] = bool(d["ci_lo"] <= 0 <= d["ci_hi"])
            res["sparse_m"] = sp
            print("sparse-M:", json.dumps({k: v for k, v in sp.items() if k != "nvis_5frame"}), flush=True)

    # pre-fixed verdicts (STATUS 10-09): maintain = pm_g5 ≥ 0.9 × pp on both probes
    pp = {n: res["probe"][n]["pp"] for n in ys}
    res["ratio_to_pp"] = {n: {c: (res["probe"][n][c] / pp[n] if pp[n] > 0 else float("nan")) for c in ("anchor", *EVAL_SPLITS)} for n in ys}
    res["verdict_maintain_1s"] = all(res["ratio_to_pp"][n]["pm_g5"] >= 0.9 for n in ys)
    # pass check: cutting M (anchor-only) collapses M recon & motion only (collapse ≤ 0.5×, keep ≥ 0.9× of pm_g5)
    r, pm = res["recon"], res["probe"]
    res["pass_check"] = {
        "M_recon_collapses": r["anchor"]["M_span_mse"] >= 2 * r["pm_g5"]["M_span_mse"],
        "motion_collapses": pm["motion"]["anchor"] <= 0.5 * pm["motion"]["pm_g5"],
        "P_recon_kept": r["anchor"]["P_t_mse"] <= r["pm_g5"]["P_t_mse"] / 0.9,
        "identity_kept": pm["identity"]["anchor"] >= 0.9 * pm["identity"]["pm_g5"],
    }
    res["elapsed_s"] = time.time() - t0
    os.makedirs(args.out_root, exist_ok=True)
    with open(f"{args.out_root}/{args.tag}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("probe", "recon", "ratio_to_pp", "verdict_maintain_1s", "pass_check")}, indent=2))


if __name__ == "__main__":
    main()
