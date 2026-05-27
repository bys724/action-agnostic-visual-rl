"""Case 3 후속: CALVIN/LIBERO per-dim R² 통합 + 비교 plot.

Aggregate R²이 binary gripper에 의해 dominate되는 현상을 분리해 보고하기 위함.
- CALVIN cross-folder gap=30 = paper §C10 main fair (segment-based, jobs 34869586~590)
- LIBERO spatial gap=20 = paper §C7 catalyst (jobs 34367612 외)
- 산출: per_dim_r2.csv + per_dim_r2.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
CALVIN = PROJECT / "paper_artifacts/calvin_action_probing"
LIBERO = PROJECT / "paper_artifacts/libero_action_probing"
TAB2 = PROJECT / "paper_artifacts/tables/tab2_probing/libero_per_cell"
OUT_DIR = Path(__file__).parent

DIM_NAMES = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "gripper"]

CALVIN_RUNS = {
    "v15":          CALVIN / "two-stream-v11_training_20260526_213639_gapsweep",
    "videomae-ours":CALVIN / "videomae-ours_training_20260526_213639_gapsweep",
    "vc1":          CALVIN / "vc1_training_20260526_213639_gapsweep",
    "dinov2":       CALVIN / "dinov2_training_20260526_213639_gapsweep",
    "siglip":       CALVIN / "siglip_training_20260526_213639_gapsweep",
}
CALVIN_GAP = "gap30"

LIBERO_RUNS = {
    "v15":          LIBERO / "two-stream-v11_libero_spatial_20260519_085921_v15ep50_av",
    "videomae-ours":LIBERO / "videomae-ours_libero_spatial_20260515_162346_ptptk",
    "vc1":          LIBERO / "vc1_libero_spatial_20260519_091516_av",
    "dinov2":       TAB2    / "dinov2_libero_spatial_20260430_164335",
    "siglip":       LIBERO / "siglip_libero_spatial_20260519_085920_av",
}
LIBERO_GAP = "gap20"


def load_row(run_dir: Path, gap: str, encoder: str, bench: str) -> dict:
    s = json.loads((run_dir / gap / "summary.json").read_text())
    row = {"bench": bench, "encoder": encoder, "agg": round(s["r2_aggregate"], 4)}
    for name, r2 in zip(DIM_NAMES, s["r2_per_dim"]):
        row[name] = round(r2, 4)
    return row


def main():
    rows = []
    for enc, d in CALVIN_RUNS.items():
        rows.append(load_row(d, CALVIN_GAP, enc, "CALVIN_xfolder_gap30"))
    for enc, d in LIBERO_RUNS.items():
        rows.append(load_row(d, LIBERO_GAP, enc, "LIBERO_spatial_gap20"))

    csv_path = OUT_DIR / "per_dim_r2.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "encoder"] + DIM_NAMES + ["agg"])
        w.writeheader()
        w.writerows(rows)
    print(f"CSV -> {csv_path}")

    # Console pretty
    for bench in ["CALVIN_xfolder_gap30", "LIBERO_spatial_gap20"]:
        print(f"\n=== {bench} ===")
        print(f"{'encoder':15s} " + " ".join(f"{d:>7s}" for d in DIM_NAMES) + f" {'agg':>7s}")
        for r in [x for x in rows if x["bench"] == bench]:
            print(f"{r['encoder']:15s} " + " ".join(f"{r[d]:+7.3f}" for d in DIM_NAMES) + f" {r['agg']:+7.3f}")

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encoders = ["v15", "videomae-ours", "vc1", "dinov2", "siglip"]
    colors = {"v15":"#d62728","videomae-ours":"#ff7f0e","vc1":"#2ca02c","dinov2":"#1f77b4","siglip":"#9467bd"}

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, bench, title in [
        (axes[0], "CALVIN_xfolder_gap30",  "CALVIN cross-folder OOD, gap=30 (1.0s) — segment-based fair protocol"),
        (axes[1], "LIBERO_spatial_gap20", "LIBERO spatial, gap=20 (1.0s)"),
    ]:
        rmap = {r["encoder"]: r for r in rows if r["bench"] == bench}
        n_enc = len(encoders)
        bar_w = 0.16
        x = list(range(len(DIM_NAMES) + 1))  # +1 = aggregate
        labels = DIM_NAMES + ["aggregate"]
        for i, enc in enumerate(encoders):
            r = rmap[enc]
            ys = [r[d] for d in DIM_NAMES] + [r["agg"]]
            xs = [xi + (i - (n_enc - 1) / 2) * bar_w for xi in x]
            ax.bar(xs, ys, bar_w, label=enc, color=colors[enc], edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(len(DIM_NAMES) - 0.5, color="gray", lw=0.5, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("R²"); ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.18), fontsize=9, frameon=False)
    fig.tight_layout()
    png = OUT_DIR / "per_dim_r2.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    print(f"PNG -> {png}")


if __name__ == "__main__":
    main()
