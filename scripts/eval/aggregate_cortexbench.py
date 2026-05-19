#!/usr/bin/env python3
"""Aggregate CortexBench §C11 BC results.

Walks `<root>/<encoder>/<task>/seed_<n>/` directories and produces:
  - per_run.csv:  encoder, suite, task, seed, status, best_success, best_score, final_loss, n_epochs
  - per_task.csv: encoder, suite, task, n_seeds, mean_success, std_success, mean_score, std_score
  - summary.csv:  encoder, suite, n_tasks, mean_success, mean_score

`status`:
  - DONE   : wandb-summary.json에 eval/highest_success 키 존재
  - RUNNING: run_dir 존재 + wandb dir 있으나 summary 미완료
  - MISSING: run_dir 없음 (skipped or yet-to-launch)

학습 진행 중에도 부분 결과 집계용. 실행: `python3 scripts/eval/aggregate_cortexbench.py --root paper_artifacts/cortexbench`
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ENCODERS = ["v15_p_only", "videomae_ours", "siglip_base", "dinov2_base", "vc1_vitb"]
TASKS_ADROIT = ["pen-v0", "relocate-v0"]
TASKS_METAWORLD = [
    "assembly-v2-goal-observable",
    "bin-picking-v2-goal-observable",
    "button-press-topdown-v2-goal-observable",
    "drawer-open-v2-goal-observable",
    "hammer-v2-goal-observable",
]
TASK_TO_SUITE = {t: "adroit" for t in TASKS_ADROIT}
TASK_TO_SUITE.update({t: "metaworld" for t in TASKS_METAWORLD})
SEEDS = [100, 200, 300]


def parse_metrics(run_dir: Path) -> dict | None:
    """학습 stdout 로그에서 final 메트릭 파싱. 미완료면 None.

    Stdout pattern (train_loop.py tabulate output, 매 eval epoch마다):
        eval/highest_score   -7.7231
        eval/highest_success  0.0
        train/highest_score  -4.65495
        epoch_loss            0.00787
    """
    if not (run_dir / "_DONE").exists():
        return None
    # 학습 로그는 paper_artifacts/cortexbench/_logs/<enc>_<task>_seed<n>.log
    enc = run_dir.parent.parent.name
    task = run_dir.parent.name
    seed = run_dir.name.replace("seed_", "")
    log_path = run_dir.parent.parent.parent / "_logs" / f"{enc}_{task}_seed{seed}.log"
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text()
    except Exception:
        return None
    # 마지막 epoch_loss / eval/highest_score / eval/highest_success 추출
    out = {}
    for line in text.splitlines():
        line = line.strip()
        for key in ("epoch_loss", "eval/highest_score", "eval/highest_success", "eval/epoch"):
            prefix = key
            if line.startswith(prefix):
                rest = line[len(prefix):].strip()
                try:
                    out[key] = float(rest)
                except ValueError:
                    continue
    if "eval/highest_success" not in out:
        return None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("paper_artifacts/cortexbench"))
    args = ap.parse_args()

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    per_run_rows = []
    by_task: dict[tuple[str, str, str], list[dict]] = {}  # (enc, suite, task) → [row]

    for enc in ENCODERS:
        for task, suite in TASK_TO_SUITE.items():
            for seed in SEEDS:
                run_dir = root / enc / task / f"seed_{seed}"
                row = {
                    "encoder": enc,
                    "suite": suite,
                    "task": task,
                    "seed": seed,
                    "status": "MISSING",
                    "best_success": "",
                    "best_score": "",
                    "final_loss": "",
                    "n_epochs": "",
                }
                if run_dir.exists():
                    metrics = parse_metrics(run_dir)
                    if metrics is not None:
                        row["status"] = "DONE"
                        row["best_success"] = metrics.get("eval/highest_success", "")
                        row["best_score"] = metrics.get("eval/highest_score", "")
                        row["final_loss"] = metrics.get("epoch_loss", "")
                        row["n_epochs"] = metrics.get("eval/epoch", "")
                        by_task.setdefault((enc, suite, task), []).append(row)
                    else:
                        row["status"] = "RUNNING"
                per_run_rows.append(row)

    # per_run.csv
    per_run_csv = root / "per_run.csv"
    with open(per_run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        w.writeheader()
        for r in per_run_rows:
            w.writerow(r)

    # per_task.csv (seed 평균)
    per_task_rows = []
    for (enc, suite, task), rows in sorted(by_task.items()):
        succ = [float(r["best_success"]) for r in rows]
        score = [float(r["best_score"]) for r in rows]
        per_task_rows.append({
            "encoder": enc,
            "suite": suite,
            "task": task,
            "n_seeds": len(rows),
            "mean_success": round(statistics.mean(succ), 4),
            "std_success": round(statistics.stdev(succ), 4) if len(succ) > 1 else 0.0,
            "mean_score": round(statistics.mean(score), 4),
            "std_score": round(statistics.stdev(score), 4) if len(score) > 1 else 0.0,
        })

    per_task_csv = root / "per_task.csv"
    if per_task_rows:
        with open(per_task_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_task_rows[0].keys()))
            w.writeheader()
            for r in per_task_rows:
                w.writerow(r)

    # summary.csv (encoder × suite, task 평균)
    by_enc_suite: dict[tuple[str, str], list[dict]] = {}
    for r in per_task_rows:
        by_enc_suite.setdefault((r["encoder"], r["suite"]), []).append(r)
    summary_rows = []
    for (enc, suite), rows in sorted(by_enc_suite.items()):
        succ = [r["mean_success"] for r in rows]
        score = [r["mean_score"] for r in rows]
        summary_rows.append({
            "encoder": enc,
            "suite": suite,
            "n_tasks": len(rows),
            "mean_success": round(statistics.mean(succ), 4),
            "mean_score": round(statistics.mean(score), 4),
        })

    summary_csv = root / "summary.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    # stdout progress
    n_done = sum(1 for r in per_run_rows if r["status"] == "DONE")
    n_run = sum(1 for r in per_run_rows if r["status"] == "RUNNING")
    n_miss = sum(1 for r in per_run_rows if r["status"] == "MISSING")
    print(f"per_run.csv:  {len(per_run_rows)} rows (DONE={n_done}, RUNNING={n_run}, MISSING={n_miss})")
    print(f"per_task.csv: {len(per_task_rows)} rows")
    print(f"summary.csv:  {len(summary_rows)} rows")

    if summary_rows:
        print("\n=== Per-suite mean (encoder × suite) ===")
        print(f"{'encoder':18s} {'suite':10s} {'n_tasks':>8s} {'mean_success':>13s} {'mean_score':>12s}")
        for r in summary_rows:
            print(f"{r['encoder']:18s} {r['suite']:10s} {r['n_tasks']:>8d} "
                  f"{r['mean_success']:>13.4f} {r['mean_score']:>12.4f}")


if __name__ == "__main__":
    main()
