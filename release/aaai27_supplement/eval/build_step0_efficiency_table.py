#!/usr/bin/env python3
"""Render the OOD action-probing efficiency table (paper headline table).

Pipeline:
  1. `probe_action_libero.py` / `probe_action_calvin.py` fit a linear/attentive
     probe on frozen encoder features and write a per-run summary.json with the
     per-dimension action-prediction R^2 (and n_eval).
  2. This script aggregates those per-run results into the position-R^2
     efficiency table (translational dims 0-2).

For convenience, the aggregated per-(encoder x benchmark) results of the
*reported* runs are shipped as `artifacts/probing_efficiency.csv` (a tidy
long-format file derived from the summary.json outputs, no hand transcription),
so the table reproduces out of the box:

    python3 eval/build_step0_efficiency_table.py

To regenerate the underlying numbers from scratch, run the probe scripts on the
released checkpoints + probing data and re-aggregate.

The table reports position R^2 (mean over translational dims 0-2), higher = better.
CoMP-MAE-S (~32M, EgoDex part1 subset) is the headline; Plain xMAE-S is the
matched control (same params/data/schedule, M-recon off). Baselines: a same-data
video model (VideoMAE-ours, 86M) and internet-scale frozen ViT-B encoders
(VC-1 / DINOv2 / SigLIP, 86M each).
"""
import csv
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts" / "probing_efficiency.csv"

# Column order for the rendered table (benchmark key -> header label).
BENCHMARKS = [
    ("calvin_xfold_gap30", "CALVIN"),
    ("libero_spatial_gap20", "LIB_spatial"),
    ("libero_object_gap20", "LIB_object"),
    ("libero_goal_gap20", "LIB_goal"),
]
# Parity anchors: the evaluation-pair count per benchmark must be identical across
# encoders (same probing split). A mismatch flags an inconsistent source run.
PARITY_ANCHORS = {
    "calvin_xfold_gap30": 32183,
    "libero_spatial_gap20": 9690,
    "libero_object_gap20": 12710,
    "libero_goal_gap20": 11100,
}
# Encoder render order (stable, matches the paper table).
ENCODER_ORDER = [
    "CoMP-MAE-S  P_t⊕M  (mean)", "CoMP-MAE-S  P_t⊕M  (attn)",
    "CoMP-MAE-S  P_t⊕P_tk (mean)", "CoMP-MAE-S  P_t⊕P_tk (attn)",
    "CoMP-MAE-B-full  P_t⊕M  (mean)", "CoMP-MAE-B-full  P_t⊕M  (attn)",
    "Plain xMAE-S  P_t⊕M  (mean)", "Plain xMAE-S  P_t⊕M  (attn)",
    "VideoMAE-ours (mean, vla)", "VideoMAE-ours (attn, vla)",
    "VC-1 (frozen)", "DINOv2 (frozen)", "SigLIP (frozen)",
]


def load():
    """Read the tidy long-format CSV into {encoder: {benchmark: (r2, n_eval)}}."""
    table, meta = {}, {}
    with open(ART) as f:
        for row in csv.DictReader(f):
            enc = row["encoder"]
            table.setdefault(enc, {})[row["benchmark"]] = (
                float(row["pos_r2"]),
                int(row["n_eval"]) if row["n_eval"] else None,
            )
            meta[enc] = (row["params_M"], row["pretrain_data"])
    return table, meta


def main():
    table, meta = load()
    warn = []

    # Parity check across all encoders.
    for enc, cells in table.items():
        for bkey, (_, n) in cells.items():
            anchor = PARITY_ANCHORS.get(bkey)
            if n is not None and anchor is not None and n != anchor:
                warn.append(f"{enc}: {bkey} n_eval {n} != anchor {anchor}")

    # Rendered table.
    labels = [lab for _, lab in BENCHMARKS]
    hdr = f"\n{'encoder':32s} {'params':>7s}" + "".join(f"{lab:>12s}" for lab in labels)
    print(hdr)
    for enc in ENCODER_ORDER:
        if enc not in table:
            continue
        params, _ = meta[enc]
        line = f"{enc:32s} {float(params):6.1f}M"
        for bkey, _ in BENCHMARKS:
            cell = table[enc].get(bkey)
            line += f"{cell[0]:+12.4f}" if cell else f"{'—':>12s}"
        print(line)

    anchors = " / ".join(f"{lab} {PARITY_ANCHORS[k]}" for k, lab in BENCHMARKS)
    print(f"\nparity anchors (n_eval): {anchors}")
    if warn:
        print("\nPARITY WARNINGS:")
        for w in warn:
            print("  " + w)
    else:
        print("parity OK (all encoders with known n_eval match the anchors)")


if __name__ == "__main__":
    main()
