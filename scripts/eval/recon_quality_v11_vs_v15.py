"""C6 per-sample reconstruction quality v11 vs v15 (paper_experiments_plan §6).

목적: v11은 reconstruction path 비대칭 (frame_t direct, frame_tk motion-routing
거침), v15는 같은 path → quality 대칭이라는 design intent를 정량 입증.

각 model × N test sample:
- v11: (frame_t, frame_tk) → pred_t, pred_tk → per-sample MSE
- v15: (frame_t, frame_n, frame_tk) → pred_t, pred_tk → per-sample MSE
같은 EgoDex video pool에서 동일 seed로 sample.

출력: JSON with per-sample MSE arrays + mean ± std + (pred_tk / pred_t) ratio.

Usage:
    python scripts/eval/recon_quality_v11_vs_v15.py \\
        --v11-ckpt /proj/external_group/mrg/checkpoints/two_stream_v11/20260426_014333/checkpoint_epoch0044.pt \\
        --v15-ckpt /proj/external_group/mrg/checkpoints/two_stream_v15/20260511_045319/latest.pt \\
        --egodex-frames-root /proj/external_group/mrg/datasets/egodex/frames \\
        --output-dir paper_artifacts/recon_quality \\
        --num-samples 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.egodex import EgoDexDataset
from src.models import TwoStreamV11Model, TwoStreamV15Model


@torch.no_grad()
def per_sample_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) × 2 → (B,) per-sample MSE."""
    return ((pred - gt) ** 2).mean(dim=(1, 2, 3))


def load_v11(ckpt_path: str, device: str) -> TwoStreamV11Model:
    model = TwoStreamV11Model(
        image_size=224, patch_size=16, embed_dim=768,
        p_depth=12, m_depth=6, mask_ratio_p=0.75, mask_ratio_m=0.3,
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model_state_dict", state)
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  v11 load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()
    return model


def load_v15(ckpt_path: str, device: str) -> TwoStreamV15Model:
    model = TwoStreamV15Model(
        image_size=224, patch_size=16, embed_dim=768,
        p_depth=12, m_depth=6, mask_ratio_p=0.75, mask_ratio_m_jepa=0.5,
        routing_mode="v_from_p",
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model_state_dict", state)
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  v15 load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v11-ckpt", required=True)
    ap.add_argument("--v15-ckpt", required=True)
    ap.add_argument("--egodex-frames-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-gap", type=int, default=30)
    ap.add_argument("--sample-center", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # v15 학습 분포 (triple, max_gap=30, sample_center=15, triangular) 그대로
    print(f"\n=== Loading EgoDex {args.split} (return_triple=True) ===")
    ds = EgoDexDataset(
        data_root=args.egodex_frames_root,
        split=args.split, img_size=224, train=False,
        max_gap=args.max_gap, sample_dist="triangular",
        sample_center=args.sample_center,
        return_triple=True,
    )
    print(f"  {len(ds)} videos available; sampling {args.num_samples} pairs")

    # Sample N samples by index iteration (each idx → random gap)
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(len(ds), size=args.num_samples, replace=False).tolist()

    print(f"\n=== Loading models ===")
    t0 = time.time()
    v11 = load_v11(args.v11_ckpt, device)
    v15 = load_v15(args.v15_ckpt, device)
    print(f"  models loaded in {time.time()-t0:.1f}s")

    v11_mse_t, v11_mse_tk = [], []
    v15_mse_t, v15_mse_tk = [], []

    print(f"\n=== Forward + per-sample MSE (batch={args.batch_size}) ===")
    t0 = time.time()
    for b_start in range(0, args.num_samples, args.batch_size):
        b_idx = sample_indices[b_start: b_start + args.batch_size]
        items = [ds[i] for i in b_idx]  # each: (img_t, img_t_n, img_t_m, gap_n, gap_m)
        img_t = torch.stack([it[0] for it in items]).to(device)
        img_n = torch.stack([it[1] for it in items]).to(device)
        img_tk = torch.stack([it[2] for it in items]).to(device)

        # v11: (img_t, img_tk) only (skip intermediate)
        out_v11 = v11(img_t, img_tk)
        m_t = per_sample_mse(out_v11["pred_t"], img_t).cpu().numpy()
        m_tk = per_sample_mse(out_v11["pred_tk"], img_tk).cpu().numpy()
        v11_mse_t.extend(m_t.tolist())
        v11_mse_tk.extend(m_tk.tolist())

        # v15: 3-frame triple
        out_v15 = v15(img_t, img_n, img_tk)
        m_t = per_sample_mse(out_v15["pred_t"], img_t).cpu().numpy()
        m_tk = per_sample_mse(out_v15["pred_tk"], img_tk).cpu().numpy()
        v15_mse_t.extend(m_t.tolist())
        v15_mse_tk.extend(m_tk.tolist())

        if b_start % (10 * args.batch_size) == 0:
            print(f"  [{b_start}/{args.num_samples}] {time.time()-t0:.1f}s")

    print(f"\n=== Done in {time.time()-t0:.1f}s ===")

    def stats(arr):
        a = np.array(arr)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
                "p75": float(np.percentile(a, 75))}

    v11_t_s, v11_tk_s = stats(v11_mse_t), stats(v11_mse_tk)
    v15_t_s, v15_tk_s = stats(v15_mse_t), stats(v15_mse_tk)

    summary = {
        "config": {
            "v11_ckpt": args.v11_ckpt, "v15_ckpt": args.v15_ckpt,
            "num_samples": args.num_samples, "seed": args.seed,
            "max_gap": args.max_gap, "sample_center": args.sample_center,
        },
        "v11": {
            "mse_t": v11_t_s, "mse_tk": v11_tk_s,
            "ratio_tk_over_t_mean": v11_tk_s["mean"] / v11_t_s["mean"],
        },
        "v15": {
            "mse_t": v15_t_s, "mse_tk": v15_tk_s,
            "ratio_tk_over_t_mean": v15_tk_s["mean"] / v15_t_s["mean"],
        },
        "per_sample": {
            "v11_mse_t": v11_mse_t, "v11_mse_tk": v11_mse_tk,
            "v15_mse_t": v15_mse_t, "v15_mse_tk": v15_mse_tk,
        },
    }

    out_path = output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")
    print(f"\n  v11 mse_t   = {v11_t_s['mean']:.4f} ± {v11_t_s['std']:.4f}")
    print(f"  v11 mse_tk  = {v11_tk_s['mean']:.4f} ± {v11_tk_s['std']:.4f}")
    print(f"  v11 ratio (tk/t) = {summary['v11']['ratio_tk_over_t_mean']:.3f}")
    print(f"  v15 mse_t   = {v15_t_s['mean']:.4f} ± {v15_t_s['std']:.4f}")
    print(f"  v15 mse_tk  = {v15_tk_s['mean']:.4f} ± {v15_tk_s['std']:.4f}")
    print(f"  v15 ratio (tk/t) = {summary['v15']['ratio_tk_over_t_mean']:.3f}")


if __name__ == "__main__":
    main()
