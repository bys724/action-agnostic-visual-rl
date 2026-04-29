"""Export probing results from raw JSON files into clean CSV summaries.

The raw probe runs in `data/probing_results/` accumulate over time and contain
duplicate / failed measurements. This script consolidates them into per-model
CSV files that the paper repo can consume directly without re-implementing the
filename parser or duplicate-handling logic.

Usage:
    python scripts/analysis/export_probing_summary.py
"""
import csv
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "data" / "probing_results"
OUT_DIR = REPO / "paper_artifacts" / "probing"

# Same v11 modes as in probe_action_v11.py CLS_MODES_ALL
V11_MODES = {
    "patch_mean_m_enc",
    "patch_mean_p_enc",
    "patch_mean_p_state_after_routing",
    "patch_mean_p_features_tk",
    "patch_mean_concat_enc_only",
    "patch_mean_concat_enc_phase3",
    "patch_mean_concat_enc_d_prime",
    "patch_mean_concat_p_enc_d_prime",
    "patch_mean_concat_all",
    "cls_m_enc",
    "cls_p_enc",
    "cls_concat_enc",
}

# Probe runs with R² below this floor are considered failed (probe did not
# converge — worse than predicting the mean) and dropped.
R2_FLOOR = -1.0


def _select_best(records):
    """Among duplicate (epoch, mode) probes, keep latest by mtime above floor."""
    best = {}
    for rec in records:
        if rec["r2"] < R2_FLOOR:
            continue
        key = (rec["epoch"], rec["mode"])
        prev = best.get(key)
        if prev is None or rec["mtime"] > prev["mtime"]:
            best[key] = rec
    return sorted(best.values(), key=lambda r: (r["epoch"], r["mode"]))


def export_v11_egodex():
    """v11 × EgoDex within-domain (gap=10, test split)."""
    pattern = re.compile(
        r"probe_v11_(checkpoint_epoch(\d+)|latest)_(.+?)_gap10_test_\d+_\d+\.json$"
    )
    records = []
    for path in PROBE_DIR.glob("probe_v11_*_gap10_test_*.json"):
        m = pattern.match(path.name)
        if m is None:
            continue
        epoch = 50 if m.group(1) == "latest" else int(m.group(2))
        mode = m.group(3)
        if mode not in V11_MODES:
            continue
        with open(path) as f:
            d = json.load(f)
        records.append({
            "epoch": epoch,
            "mode": mode,
            "r2": d["r2"],
            "mse": d.get("mse"),
            "cosine_sim": d.get("cosine_sim"),
            "train_samples": d.get("train_samples"),
            "eval_samples": d.get("eval_samples"),
            "embed_dim": d.get("embed_dim"),
            "source_file": path.name,
            "mtime": path.stat().st_mtime,
        })

    rows = _select_best(records)

    out = OUT_DIR / "v11_egodex_summary.csv"
    cols = ["epoch", "mode", "r2", "mse", "cosine_sim",
            "train_samples", "eval_samples", "embed_dim", "source_file"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in cols})
    print(f"[v11 egodex] {len(rows)} rows  ->  {out.relative_to(REPO)}")
    return rows


def export_v11_droid():
    """v11 × DROID cross-domain (multiple gaps, test split)."""
    pattern = re.compile(
        r"probe_droid_v11_(checkpoint_epoch(\d+)|latest)_(.+?)_gap(\d+)_\d+_\d+\.json$"
    )
    records = []
    for path in PROBE_DIR.glob("probe_droid_v11_*.json"):
        m = pattern.match(path.name)
        if m is None:
            continue
        epoch = 50 if m.group(1) == "latest" else int(m.group(2))
        mode = m.group(3)
        gap = int(m.group(4))
        with open(path) as f:
            d = json.load(f)
        records.append({
            "epoch": epoch,
            "mode": mode,
            "gap": gap,
            "r2": d["r2"],
            "mse": d.get("mse"),
            "cosine_sim": d.get("cosine_sim"),
            "train_samples": d.get("train_samples"),
            "eval_samples": d.get("eval_samples"),
            "source_file": path.name,
            "mtime": path.stat().st_mtime,
        })

    # Dedupe per (epoch, mode, gap)
    best = {}
    for rec in records:
        if rec["r2"] < R2_FLOOR:
            continue
        key = (rec["epoch"], rec["mode"], rec["gap"])
        prev = best.get(key)
        if prev is None or rec["mtime"] > prev["mtime"]:
            best[key] = rec
    rows = sorted(best.values(), key=lambda r: (r["epoch"], r["gap"], r["mode"]))

    if not rows:
        print("[v11 droid] no records found, skipping")
        return []

    out = OUT_DIR / "v11_droid_summary.csv"
    cols = ["epoch", "mode", "gap", "r2", "mse", "cosine_sim",
            "train_samples", "eval_samples", "source_file"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in cols})
    print(f"[v11 droid] {len(rows)} rows  ->  {out.relative_to(REPO)}")
    return rows


def export_videomae_droid():
    """VideoMAE-ours × DROID cross-domain (baseline for v11 comparison)."""
    pattern = re.compile(
        r"probe_droid_videomae_(.+?)_gap(\d+)_\d+_\d+\.json$"
    )
    records = []
    for path in PROBE_DIR.glob("probe_droid_videomae_*.json"):
        m = pattern.match(path.name)
        if m is None:
            continue
        mode = m.group(1)
        gap = int(m.group(2))
        with open(path) as f:
            d = json.load(f)
        records.append({
            "mode": mode,
            "gap": gap,
            "r2": d["r2"],
            "mse": d.get("mse"),
            "cosine_sim": d.get("cosine_sim"),
            "source_file": path.name,
            "mtime": path.stat().st_mtime,
        })

    best = {}
    for rec in records:
        if rec["r2"] < R2_FLOOR:
            continue
        key = (rec["mode"], rec["gap"])
        prev = best.get(key)
        if prev is None or rec["mtime"] > prev["mtime"]:
            best[key] = rec
    rows = sorted(best.values(), key=lambda r: (r["gap"], r["mode"]))

    if not rows:
        print("[videomae droid] no records, skipping")
        return []

    out = OUT_DIR / "videomae_droid_summary.csv"
    cols = ["mode", "gap", "r2", "mse", "cosine_sim", "source_file"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in cols})
    print(f"[videomae droid] {len(rows)} rows  ->  {out.relative_to(REPO)}")
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_v11_egodex()
    export_v11_droid()
    export_videomae_droid()


if __name__ == "__main__":
    main()
