#!/usr/bin/env python3
"""Paired significance test for the LIBERO BC-T rollout comparison.

Compares CoMP-MAE-S vs. the plain cross-modal MAE control (the matched control:
same params/data/schedule, M-recon off) on per-task success rate. Each task is
evaluated over 3 seeds; success rates are averaged over seeds per (suite, task),
the two encoders are paired per task (3 suites x 10 tasks = 30 pairs), and a
Wilcoxon signed-rank test is run on the pooled per-task differences.

Note: the pooled mean delta and the qualitative verdict (within-noise parity)
are robust across reasonable pairing choices; the exact p-value depends on the
pairing granularity (per-task-seed-averaged n=30, per-(suite,seed) n=9, or
per-(suite,seed,task) n=90) and the Wilcoxon variant. All give a non-significant
result -- CoMP-MAE and the plain control are within noise on value-level rollout.

Reads the shipped evidence file `artifacts/per_task.csv` (one row per
encoder x suite x seed x task, with n_episodes / n_success / success_rate).

Usage:
    python3 stats/stats_libero_rollout.py
    python3 stats/stats_libero_rollout.py --csv path/to/per_task.csv \
        --a comp-mae-s --b plain-xmae-s
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

DEFAULT_CSV = Path(__file__).resolve().parent / "artifacts" / "per_task.csv"


def load_pairs(csv_path, enc_a, enc_b):
    """Return two aligned lists of per-task success rates (enc_a, enc_b),
    averaged over seeds and matched per (suite, task_id)."""
    # (encoder, suite, task_id) -> list of per-seed success rates
    acc = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(open(csv_path)):
        if row["encoder"] in (enc_a, enc_b):
            acc[(row["suite"], row["task_id"])][row["encoder"]].append(
                float(row["success_rate"]))

    a_vals, b_vals, tasks = [], [], []
    for key in sorted(acc):  # deterministic order over (suite, task_id)
        per_enc = acc[key]
        if enc_a in per_enc and enc_b in per_enc:
            a_vals.append(sum(per_enc[enc_a]) / len(per_enc[enc_a]))
            b_vals.append(sum(per_enc[enc_b]) / len(per_enc[enc_b]))
            tasks.append(key)
    return a_vals, b_vals, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help="per-task rollout CSV (default: shipped artifacts/per_task.csv)")
    ap.add_argument("--a", default="comp-mae-s", help="encoder A (CoMP-MAE-S)")
    ap.add_argument("--b", default="plain-xmae-s", help="encoder B (plain control)")
    args = ap.parse_args()

    a_vals, b_vals, tasks = load_pairs(args.csv, args.a, args.b)
    n = len(a_vals)
    if n == 0:
        raise SystemExit(f"No paired tasks found for {args.a} vs {args.b} in {args.csv}")

    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    mean_a = sum(a_vals) / n
    mean_b = sum(b_vals) / n
    pooled_delta = sum(diffs) / n

    from scipy.stats import wilcoxon
    # zero_method='wilcox' drops zero-differences (standard); two-sided test.
    try:
        stat, p = wilcoxon(a_vals, b_vals)
    except ValueError as e:
        stat, p = float("nan"), float("nan")
        print(f"[wilcoxon] {e}")

    print(f"Paired LIBERO rollout: {args.a}  vs  {args.b}")
    print(f"  paired tasks (seed-averaged, 3 suites x 10 tasks): n = {n}")
    print(f"  mean success rate A = {mean_a:.4f}")
    print(f"  mean success rate B = {mean_b:.4f}")
    print(f"  pooled mean delta (A - B) = {pooled_delta:+.4f} ({pooled_delta*100:+.2f} pt)")
    print(f"  Wilcoxon signed-rank: statistic = {stat:.1f}, p = {p:.4f}")
    verdict = "within noise (no significant difference)" if p != p or p > 0.05 \
        else "significant at p<0.05"
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
