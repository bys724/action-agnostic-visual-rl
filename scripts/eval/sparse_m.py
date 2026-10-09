#!/usr/bin/env python
"""Sparse-M measurement for the layer-1 observer (STATUS 종합 지시 10-10 ③).

Question: can the cheap path (anchor P + M chunks) get cheaper by running the frozen M encoder only on
patches that actually changed?  τ = 1/255 on per-patch mean|ΔL| (ΔL = model's compute_m_channel on [0,1]
frames resized to 224, 16×16 patches → 196).  Empty chunk rule (fixed before results): keep the single
patch with the largest mean|ΔL| (≥ 1 token).  Sparse tokens = encode_m_sparse (probe_action_libero).

  --mode ratio : (a) non-zero patch count histograms — 5-frame grid chunks, partial chunks 1–4 frames,
                 20-frame same-probe pairs — for libero_object / libero_goal → sparse_ratio.json
  --mode flops : (c) M-encoder FLOPs vs visible count + expected cost per update (v3) / per step (v4)
                 using the ratio histograms → flops_sparse.json
  --mode probe : (b1) same-probe M (m_only · attentive · gap 20 · action + identity), probe fit on full
                 encodings, applied to full vs sparse eval encodings, paired demo bootstrap
                 → sparse_b1_<suite>.json
(b2) lives in observer_fuse.py --m-sparse-tau (paired_bootstrap below is shared).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "eval"))

from probe_action_libero import (  # noqa: E402
    _suite_split, build_parvo_encoder, compute_metrics, encode_m_sparse, encode_pairs_parvo,
    libero_action_target, load_demo, m_patch_absdl, preprocess_frames, train_probe,
)

TAU = 1.0 / 255.0
N_PATCH = 196
OUT = PROJECT_ROOT / "paper_artifacts" / "observer_fuse"
C1DN = ("/proj/external_group/mrg/checkpoints/two_stream_v15b_refine_comp_s_denoise_augtgt/"
        "20260927_235906/checkpoint_epoch0010.pt")


# ─────────────────────────────────────────────────────────────────────────
# Paired demo bootstrap (shared with observer_fuse.py --m-sparse-tau)
# ─────────────────────────────────────────────────────────────────────────

def _metric(pred, y, kind):
    if kind == "acc":
        return float((pred.argmax(1) == y).mean())
    res = ((y - pred) ** 2).sum(0)
    tot = ((y - y.mean(0)) ** 2).sum(0)
    if kind == "r2":       # = compute_metrics r2_aggregate
        return float(1 - res.sum() / (tot.sum() + 1e-8))
    if kind == "r2pos3":   # mean per-dim R² over the 3 position dims (STEP 1 table convention)
        return float(np.mean(1 - res[:3] / (tot[:3] + 1e-8)))
    raise ValueError(kind)


def paired_bootstrap(pred_a, pred_b, y, groups, kind, n_boot=2000, seed=0):
    """metric(a) − metric(b) on the same samples, resampling eval demos with replacement.
    → point estimates, 95% percentile CI of the difference."""
    pred_a, pred_b, y, groups = map(np.asarray, (pred_a, pred_b, y, groups))
    demos = np.unique(groups)
    idx_of = [np.nonzero(groups == d)[0] for d in demos]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([idx_of[i] for i in rng.integers(0, len(demos), len(demos))])
        diffs[b] = _metric(pred_a[idx], y[idx], kind) - _metric(pred_b[idx], y[idx], kind)
    a, b_ = _metric(pred_a, y, kind), _metric(pred_b, y, kind)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"metric": kind, "a": a, "b": b_, "diff": a - b_, "ci_lo": float(lo), "ci_hi": float(hi),
            "boot_mean": float(diffs.mean()), "n_demos": int(len(demos)), "n_samples": int(len(y)),
            "n_boot": n_boot, "contains_0": bool(lo <= 0 <= hi)}


# ─────────────────────────────────────────────────────────────────────────
# (a) non-zero patch ratio
# ─────────────────────────────────────────────────────────────────────────

def _hist(counts):
    return np.bincount(np.asarray(counts), minlength=N_PATCH + 1).tolist()


def hist_summary(h):
    h = np.asarray(h, float)
    n = np.arange(len(h))
    cdf = np.cumsum(h) / h.sum()
    q = {f"q{int(p * 100)}": int(np.searchsorted(cdf, p)) for p in (0.1, 0.5, 0.9)}
    return {"n_chunks": int(h.sum()), "mean_count": float((h * n).sum() / h.sum()),
            "mean_ratio": float((h * n).sum() / h.sum() / N_PATCH),
            **{k: v for k, v in q.items()}, **{f"{k}_ratio": v / N_PATCH for k, v in q.items()},
            "frac_zero": float(h[0] / h.sum())}


@torch.no_grad()
def run_ratio(args, device):
    from src.models.common.preprocessing import TwoStreamPreprocessing
    pre = TwoStreamPreprocessing(use_sobel=False).to(device)   # same compute_m_channel as the model
    res = {"tau": args.tau, "rule": "patch non-zero iff mean|ΔL| over its 16x16 pixels > tau", "suites": {}}
    for suite in args.suites:
        tr, ev = _suite_split(suite, args.data_root, 42, 0.8, 99.0)
        demos = tr + ev
        if args.max_demos:
            demos = demos[:args.max_demos]
        cnt = {"chunk5_grid": [], **{f"partial{L}": [] for L in range(1, 5)}, "pair20": []}
        t0 = time.time()
        for hp, d, _ in demos:
            frames = load_demo(hp, d)[0]
            T = len(frames)
            x = preprocess_frames(frames, 224).to(device)
            grid = np.arange(0, T, 5)

            def nz(a, b):
                return (m_patch_absdl(pre.compute_m_channel(x[a], x[b])) > args.tau).sum(1).cpu().numpy()
            g5 = grid[grid + 5 <= T - 1]
            cnt["chunk5_grid"].append(nz(g5, g5 + 5))
            for L in range(1, 5):
                gL = grid[grid + L <= T - 1]
                cnt[f"partial{L}"].append(nz(gL, gL + L))
            t = np.arange(0, T - 20)
            cnt["pair20"].append(np.concatenate([nz(t[i:i + 256], t[i:i + 256] + 20) for i in range(0, len(t), 256)]))
        hs = {k: _hist(np.concatenate(v)) for k, v in cnt.items()}
        res["suites"][suite] = {"n_demos": len(demos), "hist": hs, "summary": {k: hist_summary(h) for k, h in hs.items()}}
        print(f"[{suite}] {len(demos)} demos ({time.time() - t0:.0f}s)")
        for k, s in res["suites"][suite]["summary"].items():
            print(f"  {k:12s} n={s['n_chunks']:6d} mean {s['mean_ratio']:.3f} q10/50/90 {s['q10']}/{s['q50']}/{s['q90']} "
                  f"zero {s['frac_zero']:.3f}")
    with open(OUT / f"sparse_ratio{args.suffix}.json", "w") as f:
        json.dump(res, f, indent=1)


# ─────────────────────────────────────────────────────────────────────────
# (c) FLOPs
# ─────────────────────────────────────────────────────────────────────────

def run_flops(args, device):
    from torch.utils.flop_counter import FlopCounterMode
    from observer_fuse import Fuse

    def fl(fn):  # no torch.no_grad(): FlopCounterMode module tracker asserts on grad-less outputs
        with FlopCounterMode(display=False) as fc:
            fn()
        return int(fc.get_total_flops())

    enc = build_parvo_encoder(args.checkpoint, device)
    x = torch.rand(1, 3, 224, 224, device=device)
    m = enc.preprocessing.compute_m_channel(x, torch.rand_like(x))
    D = enc.pos_embed_p.shape[-1]
    P = fl(lambda: enc._encode_p_unmasked(enc.preprocessing.compute_p_channel(x)))
    M_dense = fl(lambda: enc._encode_m_unmasked(m))

    def m_sparse(n):  # n visible patch tokens (+CLS) through blocks_m, mask-token fill back to 196
        mask = torch.ones(1, N_PATCH, dtype=torch.bool, device=device)
        mask[0, :n] = False
        return fl(lambda: enc._build_full_seq_m(enc._encode_m_masked(m, mask), mask))
    M_n = [0] + [m_sparse(n) for n in range(1, N_PATCH + 1)]
    fuse = Fuse(D, n_patch=N_PATCH).to(device).eval()
    t = torch.rand(1, N_PATCH, D, device=device)
    F_pp = fl(lambda: fuse.encode(anchor=t, current=t))
    F_c = {c: fl(lambda: fuse.encode(anchor=t, chunks=[(t, j, 1) for j in range(c)])) for c in range(1, 5)}

    ratio = json.load(open(OUT / f"sparse_ratio{args.suffix}.json"))
    out = {"P_enc": P, "M_enc_dense": M_dense, "M_enc_sparse_by_n": M_n, "Fuse_PP": F_pp,
           "Fuse_PM_by_chunks": F_c, "tau": ratio["tau"], "suites": {}}

    def exp_m(h):  # expected M cost; empty chunk → 1 token (pre-fixed rule)
        h = np.asarray(h, float)
        return float(sum(h[n] * M_n[max(n, 1)] for n in range(len(h))) / h.sum())

    for suite, r in ratio["suites"].items():
        h = r["hist"]
        E = {L: exp_m(h[f"partial{L}"]) for L in range(1, 5)}
        E[5] = exp_m(h["chunk5_grid"])
        # (i) v3 per update: one new 5-frame chunk + Fuse(anchor, 4 chunks) + anchor P / 4 updates
        upd_pp = P + F_pp
        upd_dense = M_dense + F_c[4] + P / 4
        upd_sparse = E[5] + F_c[4] + P / 4
        # (ii) v4 per step: anchor age k = 1..20 (uniform), anchor P refreshed every 20 steps.
        #      new M work at age k = the chunk ending at k: length k mod 5 (partial) or 5 (completed chunk).
        #      Fuse sees ceil(k/5) chunks.
        ks = range(1, 21)
        L = lambda k: k % 5 or 5
        step_pp = P + F_pp
        step_dense = np.mean([M_dense + F_c[-(-k // 5)] for k in ks]) + P / 20
        step_sparse = np.mean([E[L(k)] + F_c[-(-k // 5)] for k in ks]) + P / 20
        out["suites"][suite] = {
            "E_M_sparse_by_len": E,
            "per_update_v3": {"PP": upd_pp, "PM_dense": upd_dense, "PM_sparse": upd_sparse,
                              "saving_vs_PP_dense": 1 - upd_dense / upd_pp, "saving_vs_PP_sparse": 1 - upd_sparse / upd_pp},
            "per_step_v4": {"PP": step_pp, "PM_dense": float(step_dense), "PM_sparse": float(step_sparse),
                            "saving_vs_PP_dense": float(1 - step_dense / step_pp),
                            "saving_vs_PP_sparse": float(1 - step_sparse / step_pp)},
        }
    out["note"] = ("Forward FLOPs at batch 1 (FlopCounterMode: matmul/conv/attention). Sparse M = patch-embed conv on "
                   "all patches + blocks_m on n visible tokens + CLS; ΔL / patch-selection / mask-token fill are "
                   "elementwise and not counted. Empty chunk counted as 1 token. per_update_v3 = new 5-frame chunk + "
                   "Fuse(4 chunks) + P/4. per_step_v4 = mean over anchor age k=1..20 of [M(chunk ending at k, length "
                   "k mod 5 or 5) + Fuse(ceil(k/5) chunks)] + P/20; P+P = P + Fuse_PP.")
    print(json.dumps({k: v for k, v in out.items() if k != "M_enc_sparse_by_n"}, indent=1))
    print("M_enc by n (GFLOPs) 1/10/50/98/196:", [round(M_n[n] / 1e9, 3) for n in (1, 10, 50, 98, 196)])
    with open(OUT / f"flops_sparse{args.suffix}.json", "w") as f:
        json.dump(out, f, indent=1)


# ─────────────────────────────────────────────────────────────────────────
# (b1) same-probe M, full vs sparse
# ─────────────────────────────────────────────────────────────────────────

def run_probe(args, device):
    np.random.seed(42)
    torch.manual_seed(42)
    gap = 20
    model = build_parvo_encoder(args.checkpoint, device)
    rng_state = torch.get_rng_state()   # = RNG state at train_probe in the original same-probe runs
    tr, ev = _suite_split(args.suite, args.data_root, 42, 0.8, 99.0)
    if args.max_demos:
        tr, ev = tr[:args.max_demos], ev[:max(2, args.max_demos // 4)]

    def embed(demos):
        E, Es, Y, I, G, NV = [], [], [], [], [], []
        for di, (hp, d, tid) in enumerate(demos):
            frames, eef_pos, ee_ori, actions = load_demo(hp, d)
            T = frames.shape[0]
            if T <= gap + 1:
                continue
            Y.append(np.stack([libero_action_target(eef_pos, ee_ori, actions, t, gap) for t in range(T - gap)]))
            prev, curr = preprocess_frames(frames[:T - gap], 224), preprocess_frames(frames[gap:], 224)
            E.append(encode_pairs_parvo(model, prev, curr, device, mode="m_only", readout="attentive"))
            Es.append(encode_pairs_parvo(model, prev, curr, device, mode="m_only", readout="attentive",
                                         m_sparse_tau=args.tau))
            with torch.no_grad():
                mc = model.preprocessing.compute_m_channel(prev.to(device), curr.to(device))
                NV.append((m_patch_absdl(mc) > args.tau).sum(1).cpu().numpy())
            I.append(np.full(T - gap, tid, np.int64)); G.append(np.full(T - gap, di))
        cat = lambda v: np.concatenate(v)
        return (torch.cat(E), torch.cat(Es), torch.from_numpy(cat(Y)), torch.from_numpy(cat(I)), cat(G), cat(NV))

    t0 = time.time()
    E_tr, Es_tr, Y_tr, I_tr, _, nv_tr = embed(tr)
    E_ev, Es_ev, Y_ev, I_ev, G_ev, nv_ev = embed(ev)
    print(f"pairs train {len(Y_tr)} eval {len(Y_ev)} ({time.time() - t0:.0f}s)", flush=True)
    # sanity: sparse with all patches visible must equal dense
    with torch.no_grad():
        x = preprocess_frames(load_demo(*ev[0][:2])[0][:2], 224).to(device)
        mc = model.preprocessing.compute_m_channel(x[:1], x[1:])
        dense = model._encode_m_unmasked(mc)[:, 1:]
        allvis = encode_m_sparse(model, mc, -1.0)[0]
    sanity = float((dense - allvis).abs().max())
    print(f"sanity max|dense − sparse(all visible)| = {sanity:.2e}", flush=True)

    n_cls = int(max(I_tr.max(), I_ev.max()) + 1)
    res = {"suite": args.suite, "gap": gap, "tau": args.tau, "checkpoint": args.checkpoint,
           "empty_rule": "keep argmax mean|ΔL| patch (>=1 token)", "sanity_allvis_maxabs": sanity,
           "n_train": len(Y_tr), "n_eval": len(Y_ev), "n_eval_demos": int(len(np.unique(G_ev))),
           "nvis_train": hist_summary(_hist(nv_tr)), "nvis_eval": hist_summary(_hist(nv_ev)), "targets": {}}

    def fit(Xtr, ytr, Xev, yev, task, od):
        torch.set_rng_state(rng_state)
        return train_probe(Xtr, ytr, Xev, yev, epochs=20, batch_size=256, lr=1e-3, device=str(device),
                           readout="attentive", n_streams=1, task=task, out_dim=od, return_probe=True)

    @torch.no_grad()
    def pred(probe, X):
        return torch.cat([probe(X[i:i + 256].to(device).float()).cpu() for i in range(0, len(X), 256)]).numpy()

    for name, (ytr, yev, task, od, kinds) in {
            "action": (Y_tr, Y_ev, "regression", 7, ("r2pos3", "r2")),
            "identity": (I_tr, I_ev, "classification", n_cls, ("acc",))}.items():
        b = fit(E_tr, ytr, E_ev, yev, task, od)            # protocol: fit on full, best epoch on full eval
        pf, ps = pred(b["probe"], E_ev), pred(b["probe"], Es_ev)
        bs = fit(Es_tr, ytr, Es_ev, yev, task, od)         # secondary: fit on sparse
        pss = pred(bs["probe"], Es_ev)
        y = yev.numpy()
        res["targets"][name] = {
            "best_epoch_full": b["epoch"], "best_epoch_sparse_fit": bs["epoch"],
            "primary_full_probe": {k: paired_bootstrap(pf, ps, y, G_ev, k, args.n_boot) for k in kinds},
            "secondary_sparse_fit": {k: paired_bootstrap(pf, pss, y, G_ev, k, args.n_boot) for k in kinds},
        }
        for k in kinds:
            p = res["targets"][name]["primary_full_probe"][k]
            s = res["targets"][name]["secondary_sparse_fit"][k]
            print(f"[{name}/{k}] full {p['a']:.4f} sparse {p['b']:.4f} diff {p['diff']:+.4f} CI [{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}]"
                  f" | sparse-fit {s['b']:.4f} diff {s['diff']:+.4f} CI [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]", flush=True)
    res["elapsed_s"] = time.time() - t0
    with open(OUT / f"sparse_b1_{args.suite}{args.suffix}.json", "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["ratio", "flops", "probe"])
    ap.add_argument("--checkpoint", default=C1DN)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--suites", nargs="+", default=["libero_object", "libero_goal"])
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--data-root", default="/proj/external_group/mrg/datasets/libero")
    ap.add_argument("--max-demos", type=int, default=None, help="smoke")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--suffix", default="", help="output file suffix (smoke)")
    args = ap.parse_args()
    device = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)
    {"ratio": run_ratio, "flops": run_flops, "probe": run_probe}[args.mode](args, device)


if __name__ == "__main__":
    main()
