#!/usr/bin/env python
"""Observer v4 — per-step z (STATUS 10-10 user-confirmed consolidated instruction, item 2).

v3 (`observer_fuse.py`, kept frozen for reproducibility) fed only 5-frame-multiple chunks, so z could be
refreshed every 5 steps only. v4 removes that grid on the *current* side:
  anchor a  = frame on the 5-frame grid (deployment refreshes P every 20 frames = subset)
  current t = a + k, k ∈ 1..20 (uniform in training)
  M input   = complete 5-frame chunks a→a+5→… (cached) + ONE partial last chunk of 1–4 frames
              (encoded on the fly) · chunk embeddings = start unit (0..3) + length in frames (1..5)
  z_PP(t)   = Fuse(P(a), P(t)) — target via EMA teacher; recon targets = P(t) tokens + M(a→t) tokens
At k = 20 the input is the v3 "5×4" configuration (no layer-1 re-judgement); v4 adds k = 1..19.
Evaluation = anchor-age curve k = 1..20: per k, attentive probe fit on z_PP(train) + anchor EE position
(RFF, same-probe concat) → applied to z_PP / z_PM / anchor-only; beyond = R²(z+pos) − R²(pos only).
Motion target = EE pose change a→t (libero_action_target(a, k)); identity only at k = 20 (ceiling).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer_fuse import (  # noqa: E402  shared parts (v3)
    PROJECT_ROOT, Fuse, Observer, _std, _suite_split, build_parvo_encoder, compute_metrics, fit_probe,
    libero_action_target, load_demo, score, token_stats,
)
from probe_action_libero import rff_expand  # noqa: E402

UNIT, K_MAX = 5, 20


def make_fuse(dim, n_patch, args):
    fuse = Fuse(dim, n_patch=n_patch, n_lat=args.n_lat, depth=args.depth)
    fuse.m_len = nn.Embedding(UNIT, fuse.m_len.embedding_dim)  # chunk length in frames 1..5 (index len−1)
    return fuse


class Data:
    """Frozen-token cache for v4. All frames: raw uint8 + P tokens. Grid frames: complete 5-frame M chunk."""

    @torch.no_grad()
    def __init__(self, enc, demos, device, m_source, batch=128):
        self.enc, self.device, self.m_source = enc, device, m_source
        F_, P, M5, ee, ori, act, tid, start, n = [], [], [], [], [], [], [], [], []
        off = 0
        for hp, d, t_id in demos:
            frames, ee_pos, ee_ori, actions = load_demo(hp, d)
            T = len(frames)
            fr = torch.from_numpy(frames).to(device)
            F_.append(fr)
            for s in range(0, T, batch):
                P.append(self.p_tok(fr[s:s + batch]))
            ee.append(ee_pos); ori.append(ee_ori); act.append(actions)
            tid += [t_id] * T; start.append(off); n.append(T); off += T
        self.frames = torch.cat(F_)
        self.P = torch.cat(P)
        self.ee, self.ori, self.act = np.concatenate(ee), np.concatenate(ori), np.concatenate(act)
        self.tid = torch.tensor(tid)
        self.start, self.n = np.array(start), np.array(n)
        demo_of = np.repeat(np.arange(len(n)), n)
        self.demo_of, self.end = demo_of, (self.start + self.n)[demo_of]   # exclusive end per frame
        self.grid = np.concatenate([np.arange(s, s + m, UNIT) for s, m in zip(self.start, self.n)])
        # complete 5-frame chunk M(g, g+5) for every grid frame g with g+5 inside the demo
        self.m5_index = -np.ones(len(self.P), dtype=np.int64)
        ok = self.grid[self.grid + UNIT < self.end[self.grid]]
        self.m5_index[ok] = np.arange(len(ok))
        self.M5 = torch.cat([self.m_tok(torch.from_numpy(ok[i:i + batch]).to(device),
                                        torch.from_numpy(ok[i:i + batch] + UNIT).to(device))
                             for i in range(0, len(ok), batch)])

    @staticmethod
    def _f01(fr_uint8):  # same as preprocess_frames ([0,1] raw, bilinear to 224), on GPU
        return F.interpolate(fr_uint8.permute(0, 3, 1, 2).float().div_(255.0), size=(224, 224),
                             mode="bilinear", align_corners=False)

    def _x(self, idx):
        return self._f01(self.frames[idx])

    @torch.no_grad()
    def p_tok(self, fr_uint8):
        x = self._f01(fr_uint8)
        return self.enc._encode_p_unmasked(self.enc.preprocessing.compute_p_channel(x))[:, 1:].half()

    @torch.no_grad()
    def m_tok(self, i0, i1):
        """M tokens for ΔL(frame i0, frame i1) (global frame indices, same demo)."""
        m = self.enc.preprocessing.compute_m_channel(self._x(i0), self._x(i1))
        if self.m_source == "raw":  # untrained ΔL 16×16 patches, zero-padded to D (raw-dl floor rule)
            t = F.unfold(m, kernel_size=16, stride=16).transpose(1, 2)
            return F.pad(t, (0, self.P.shape[-1] - t.shape[-1])).half()
        return self.enc._encode_m_unmasked(m)[:, 1:].half()

    def anchors(self, k, demo_mask=None):
        """Grid anchors a with a+k inside the demo (optionally restricted to a demo subset)."""
        a = self.grid[self.grid + k < self.end[self.grid]]
        return a if demo_mask is None else a[demo_mask[self.demo_of[a]]]

    def inputs(self, a, k, mode):
        """a: (B,) global anchor frames (np), same k for the batch. mode: pp / pm / anchor."""
        dv = self.device
        at = torch.from_numpy(a).to(dv)
        out = {"anchor": _std(self.stats, "P", self.P[at])}
        if mode == "pp":
            out["current"] = _std(self.stats, "P", self.P[at + k])
        elif mode == "pm":
            q, r = divmod(k, UNIT)
            chunks = [(_std(self.stats, "M", self.M5[torch.from_numpy(self.m5_index[a + UNIT * i]).to(dv)]),
                       i, UNIT) for i in range(q)]
            if r:
                chunks.append((_std(self.stats, "M", self.m_tok(at + UNIT * q, at + k)), q, r))
            out["chunks"] = chunks
        return out

    def targets(self, a, k):
        at = torch.from_numpy(a).to(self.device)
        p, m = _std(self.stats, "P", self.P[at + k]), _std(self.stats, "M", self.m_tok(at, at + k))
        mode = getattr(self, "target_mode", "raw")
        if mode == "token":  # per-token normalization (own channel mean/std only; MAE norm-pix style) —
            # no dataset statistics; exact-zero tokens (static raw-ΔL patches) stay 0
            tn = lambda x: (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
            p, m = tn(p), tn(m)
        elif mode == "anchor_rel":  # P target relative to the anchor (per-sample difference, no statistics)
            p = p - _std(self.stats, "P", self.P[at])
        return p, m

    def motion(self, a, k):
        return torch.from_numpy(np.stack([libero_action_target(self.ee, self.ori, self.act, i, k) for i in a]))


def m_stats_sample(data, n=8192, seed=0):
    """M standardization stats over a mixed sample of spans 1..20 (partial chunks + span targets share them)."""
    rng = np.random.default_rng(seed)
    toks = []
    for _ in range(n // 256):
        k = int(rng.integers(1, K_MAX + 1))
        a = rng.choice(data.anchors(k), 256)
        at = torch.from_numpy(a).to(data.device)
        toks.append(data.m_tok(at, at + k))
    return token_stats(torch.cat(toks))


def train(fuse, data, args):
    """v3 train_fuse with v4 sampling: k ~ U{1..20} per batch, PP/PM random, convergence stop on held-out
    train demos (last 10%), best-val weights restored."""
    rng = np.random.default_rng(args.seed)
    teacher = copy.deepcopy(fuse).requires_grad_(False).eval()
    opt = torch.optim.AdamW(fuse.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    n_val = max(1, len(data.n) // 10)
    val_mask = np.zeros(len(data.n), bool); val_mask[-n_val:] = True
    pools = {k: (data.anchors(k, ~val_mask), data.anchors(k, val_mask)) for k in range(1, K_MAX + 1)}
    m_w = 0.0 if args.pp_only else 1.0
    vrng = np.random.default_rng(12345)
    val_batches = []
    for _ in range(16):
        k = int(vrng.integers(1, K_MAX + 1))
        val_batches.append((k, vrng.choice(pools[k][1], args.batch), "pp" if args.pp_only or vrng.random() < 0.5 else "pm"))

    def losses(model, k, a, mode):
        z = model.encode(**data.inputs(a, k, mode))
        p_hat, m_hat = model.decode(z)
        p_tgt, m_tgt = data.targets(a, k)
        with torch.no_grad():
            z_t = teacher.encode(**data.inputs(a, k, "pp"))
        l_p = F.mse_loss(p_hat, p_tgt) / data.loss_scale["P"]  # scale = 1 (std) or one global scalar (ln)
        l_m = F.mse_loss(m_hat, m_tgt) / data.loss_scale["M"]
        l_z = F.mse_loss(z, z_t)
        return l_p + m_w * l_m + l_z, l_p, l_m, l_z

    t0 = time.time()
    best, best_step, bad, curve, best_state = float("inf"), 0, 0, [], None
    for step in range(1, args.steps + 1):
        k = int(rng.integers(1, K_MAX + 1))
        a = rng.choice(pools[k][0], args.batch)
        mode = "pp" if args.pp_only or rng.random() < 0.5 else "pm"
        loss = losses(fuse, k, a, mode)[0]
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for pt, ps in zip(teacher.parameters(), fuse.parameters()):
                pt.mul_(args.ema).add_(ps.detach(), alpha=1 - args.ema)
        if step % args.eval_every == 0:
            fuse.eval()
            with torch.no_grad():
                v = np.mean([[t.item() for t in losses(fuse, k_, a_, m_)] for k_, a_, m_ in val_batches], 0)
                # collapse monitor: z spread across samples relative to |z| (≈0 → z ignores its input)
                k_, a_, _ = val_batches[0]
                zs = {m: fuse.encode(**data.inputs(a_, k_, m)) for m in ("pp", "pm")}
                spread = (zs["pp"].std(0).mean() / (zs["pp"].abs().mean() + 1e-8)).item()
                pp_pm = ((zs["pp"] - zs["pm"]).norm(dim=-1).mean() / (zs["pp"].norm(dim=-1).mean() + 1e-8)).item()
            fuse.train()
            curve.append([step, round(time.time() - t0, 1), *[float(x) for x in v], spread, pp_pm])
            print(f"  step {step:6d}  val total {v[0]:.4f}  P {v[1]:.4f}  M {v[2]:.4f}  z {v[3]:.4f}  "
                  f"z-spread {spread:.3f}  |pp-pm|/|pp| {pp_pm:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            if v[0] < best * (1 - args.min_rel):
                best, best_step, bad, best_state = v[0], step, 0, copy.deepcopy(fuse.state_dict())
            else:
                bad += 1
                if bad >= args.patience:
                    break
    fuse.load_state_dict(best_state)
    return {"stop_step": step, "best_step": best_step, "best_val": best, "train_seconds": round(time.time() - t0, 1),
            "converged": bad >= args.patience, "curve": curve,
            "curve_cols": ["step", "seconds", "val_total", "val_P", "val_M", "val_z", "z_spread", "pp_pm_rel"]}


@torch.no_grad()
def embed(fuse, data, a, k, mode, bs=256):
    return torch.cat([fuse.encode(**data.inputs(a[i:i + bs], k, mode)).half().cpu() for i in range(0, len(a), bs)])


def lin_probe(x_tr, y_tr, x_ev, y_ev, args, device):
    torch.manual_seed(args.seed)
    lin = nn.Linear(x_tr.shape[1], y_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    for _ in range(args.probe_epochs):
        for i in torch.randperm(len(x_tr)).split(256):
            loss = F.mse_loss(lin(x_tr[i].to(device)), y_tr[i].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return compute_metrics(lin(x_ev.to(device)).cpu().numpy(), y_ev.numpy())["r2_aggregate"]


def evaluate(fuse, tr, ev, args, device):
    """Anchor-age curve k = 1..20 (or --eval-ks). Per k: probe on z_PP(train)+pos, applied to all modes."""
    fuse.eval()
    out = {}
    for k in args.eval_ks:
        a_tr, a_ev = tr.anchors(k), ev.anchors(k)
        y_tr, y_ev = tr.motion(a_tr, k), ev.motion(a_ev, k)
        p_tr, p_ev = torch.from_numpy(tr.ee[a_tr]).float(), torch.from_numpy(ev.ee[a_ev]).float()
        mu, sd = p_tr.mean(0), p_tr.std(0) + 1e-6
        r_tr, r_ev = rff_expand((p_tr - mu) / sd, args.seed + 2), rff_expand((p_ev - mu) / sd, args.seed + 2)
        pos_only = lin_probe(r_tr, y_tr, r_ev, y_ev, args, device)
        z_tr = embed(fuse, tr, a_tr, k, "pp")
        z_ev = {m: embed(fuse, ev, a_ev, k, m) for m in ("pp", "pm", "anchor")}
        pr = fit_probe(z_tr, y_tr, "reg", 7, args, device, extra=r_tr)
        concat = {m: score(pr, z_ev[m], y_ev, "reg", device, extra=r_ev) for m in z_ev}
        pr0 = fit_probe(z_tr, y_tr, "reg", 7, args, device)
        raw = {m: score(pr0, z_ev[m], y_ev, "reg", device) for m in z_ev}
        out[k] = {"n_train": len(a_tr), "n_eval": len(a_ev), "pos_only": pos_only, "raw": raw, "concat": concat,
                  "beyond": {m: v - pos_only for m, v in concat.items()}}
        if k == K_MAX:  # identity (ceiling) at 1s only
            cls = {t: i for i, t in enumerate(sorted(set(tr.tid.tolist()) | set(ev.tid.tolist())))}
            c_tr = torch.tensor([cls[t] for t in tr.tid[a_tr].tolist()])
            c_ev = torch.tensor([cls[t] for t in ev.tid[a_ev].tolist()])
            pi = fit_probe(z_tr, c_tr, "cls", len(cls), args, device)
            out[k]["identity"] = {m: score(pi, z_ev[m], c_ev, "cls", device) for m in z_ev}
        print(f"  k={k:2d}  pos {pos_only:.3f}  beyond pp {out[k]['beyond']['pp']:+.3f}  pm {out[k]['beyond']['pm']:+.3f}"
              f"  anchor {out[k]['beyond']['anchor']:+.3f}", flush=True)
    return out


class ObserverV4(Observer):
    """Layer-2 entry point for v4 hand-off. z_seq(frames): frames = [f_a, f_a+1, …, f_t] (uint8 H×W×3,
    k = len−1 ∈ 1..20, f_a = P anchor). Builds complete 5-frame chunks + one partial last chunk."""

    def __init__(self, handoff, device="cuda"):
        h = torch.load(handoff, map_location=device, weights_only=False)
        m = h["meta"]
        assert m.get("version") in (4, 5)  # 5 = v5 (norm ln, same interface)
        self.device, self.meta = torch.device(device), m
        self.enc = build_parvo_encoder(m["encoder_ckpt"], self.device)
        args = argparse.Namespace(n_lat=m["n_lat"], depth=m["depth"])
        self.fuse = make_fuse(m["dim"], m["n_patch"], args).to(self.device)
        self.fuse.load_state_dict(h["fuse"]); self.fuse.eval()
        self.cache = {"stats": h["input_stats"]}
        self.z_stats = h.get("z_stats")  # None for v5 (no z standardization)

    @torch.no_grad()
    def z_seq(self, frames):
        k = len(frames) - 1
        assert 1 <= k <= K_MAX
        q, r = divmod(k, UNIT)
        chunks = [(self._m(frames[UNIT * i], frames[UNIT * (i + 1)]), i, UNIT) for i in range(q)]
        if r:
            chunks.append((self._m(frames[UNIT * q], frames[k]), q, r))
        return self.fuse.encode(anchor=self._p(frames[0]), chunks=chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task-suite", default="libero_goal")
    ap.add_argument("--data-root", default="/proj/external_group/mrg/datasets/libero")
    ap.add_argument("--m-source", default="comp", choices=["comp", "raw"])
    ap.add_argument("--pp-only", action="store_true", help="common P+P reference (no M input/recon)")
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-rel", type=float, default=0.005)
    ap.add_argument("--n-lat", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--probe-epochs", type=int, default=100)
    ap.add_argument("--eval-ks", type=int, nargs="+", default=list(range(1, K_MAX + 1)))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--export-handoff", default=None)
    ap.add_argument("--norm", default="std", choices=["std", "ln"],
                    help="std = v4/analysis 1 (per-(patch,dim) train stats) · ln = v5 (no dataset-stat normalization)")
    ap.add_argument("--load-fuse", default=None, help="skip training, re-read this fuse.pt")
    ap.add_argument("--target-mode", default="raw", choices=["raw", "token", "anchor_rel"],
                    help="recon targets (with --norm ln): raw tokens · per-token normalized · P relative to anchor")
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "paper_artifacts" / "observer_fuse"))
    ap.add_argument("--ckpt-root", default="/proj/external_group/mrg/checkpoints/observer_fuse")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    t0 = time.time()

    tr_d, ev_d = _suite_split(args.task_suite, args.data_root, args.split_seed, 0.8, 99.0)
    if args.max_demos:
        tr_d, ev_d = tr_d[:args.max_demos], ev_d[:max(2, args.max_demos // 4)]
    enc = build_parvo_encoder(args.checkpoint, device)
    tr, ev = Data(enc, tr_d, device, args.m_source), Data(enc, ev_d, device, args.m_source)
    tr.target_mode = ev.target_mode = args.target_mode
    if args.norm == "std":  # v4 = analysis 1 (per-(patch,dim) train-stat standardization), kept for reproducibility
        stats = {"P": token_stats(tr.P), "M": None}
        tr.stats = ev.stats = {"stats": stats}
        stats["M"] = m_stats_sample(tr)
        tr.loss_scale = ev.loss_scale = {"P": 1.0, "M": 1.0}
    else:  # v5 (Vault 10-10 3rd): raw tokens → Fuse's token-wise LayerNorm (inp[0]); raw recon targets;
        # loss scale = ONE global scalar per target stream (mean squared token value), no per-patch/dim stats
        stats = None
        tr.stats = ev.stats = {"stats": None}
        with torch.no_grad():
            sp = float(sum(tr.P[i:i + 4096].float().pow(2).mean() * len(tr.P[i:i + 4096])
                           for i in range(0, len(tr.P), 4096)) / len(tr.P))
            rng = np.random.default_rng(0); sm = []
            for _ in range(32):
                k = int(rng.integers(1, K_MAX + 1)); a = torch.from_numpy(rng.choice(tr.anchors(k), 256)).to(device)
                sm.append(tr.m_tok(a, a + k).float().pow(2).mean().item())
        if args.target_mode == "token":
            sp, sm = 1.0, [1.0]
        elif args.target_mode == "anchor_rel":  # global scalar of the relative target
            rng = np.random.default_rng(1); acc = []
            for _ in range(32):
                k = int(rng.integers(1, K_MAX + 1)); a = torch.from_numpy(rng.choice(tr.anchors(k), 256)).to(device)
                acc.append((tr.P[a + k].float() - tr.P[a].float()).pow(2).mean().item())
            sp = float(np.mean(acc))
        tr.loss_scale = ev.loss_scale = {"P": sp, "M": float(np.mean(sm))}
        print(f"v5 loss scales (global scalars): {tr.loss_scale}", flush=True)
    print(f"cache: train frames {len(tr.P)} grid {len(tr.grid)} / eval frames {len(ev.P)}  ({time.time()-t0:.0f}s)", flush=True)

    fuse = make_fuse(tr.P.shape[-1], tr.P.shape[1], args).to(device)
    stats_cpu = None if stats is None else {k: (mu.cpu(), sd.cpu()) for k, (mu, sd) in stats.items()}
    if args.load_fuse:  # re-read only (e.g. diagnostic: other-suite Fuse with this suite's stats)
        fuse.load_state_dict(torch.load(args.load_fuse, map_location=device, weights_only=False)["fuse"])
        train_rec = None
    else:
        train_rec = train(fuse, tr, args)
        os.makedirs(f"{args.ckpt_root}/{args.tag}", exist_ok=True)
        torch.save({"fuse": fuse.state_dict(), "args": vars(args), "input_stats": stats_cpu,
                    "loss_scale": tr.loss_scale}, f"{args.ckpt_root}/{args.tag}/fuse.pt")

    curve = evaluate(fuse, tr, ev, args, device)
    if args.export_handoff:
        zt = embed(fuse, tr, tr.anchors(K_MAX), K_MAX, "pp").float()
        os.makedirs(os.path.dirname(args.export_handoff), exist_ok=True)
        z_stats = (zt.mean(0), zt.std(0)) if args.norm == "std" else None  # v5: no z standardization
        torch.save({"fuse": fuse.state_dict(), "input_stats": stats_cpu, "z_stats": z_stats,
                    "meta": {"version": 4 if args.norm == "std" else 5, "norm": args.norm, "loss_scale": tr.loss_scale,
                             "z_scale_ref": {"mean_abs": zt.abs().mean().item(), "token_norm": zt.norm(dim=-1).mean().item()}, "encoder_ckpt": args.checkpoint, "m_source": args.m_source,
                             "dim": tr.P.shape[-1], "n_patch": tr.P.shape[1], "n_lat": args.n_lat, "depth": args.depth,
                             "unit_frames": UNIT, "k_max": K_MAX, "task_suite": args.task_suite,
                             "split_seed": args.split_seed, "tag": args.tag, "fuse_ckpt": f"{args.ckpt_root}/{args.tag}/fuse.pt",
                             "entry": "scripts/eval/observer_fuse_v4.py:ObserverV4 (z_seq for P+M, z_pp for P+P)"}},
                   args.export_handoff)
        print(f"hand-off written: {args.export_handoff}", flush=True)
    res = {"args": vars(args), "version": 4, "train": train_rec, "curve": curve, "elapsed_s": time.time() - t0}
    os.makedirs(args.out_root, exist_ok=True)
    with open(f"{args.out_root}/{args.tag}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
