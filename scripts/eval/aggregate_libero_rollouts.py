#!/usr/bin/env python
"""LIBERO BC-T rollout JSON 결과를 paper_artifacts/ CSV로 집계.

`src/eval_libero.py` 산출물 (`data/libero/results/**/*.json`)을 모아
3 종류 long/wide-format CSV를 생성. 논문 표·그래프에서 직접 사용.

사용법:
    python scripts/eval/aggregate_libero_rollouts.py \
        --input-dir data/libero/results \
        --output-dir paper_artifacts/libero_rollout

산출:
    episodes.csv       per-episode (long format) — 가장 유연
    per_task.csv       per-(encoder, suite, seed, task) — 표준 LIBERO breakdown
    summary.csv        per-(encoder, suite) — seeds 평균/표편, paper main table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
from collections import defaultdict


# ckpt path → seed 추출용 (디렉명: vc1_libero_spatial_seed0_20260505_042336_v3)
SEED_RE = re.compile(r"_seed(\d+)_")


def extract_seed_from_ckpt(ckpt_path: str) -> int | None:
    m = SEED_RE.search(ckpt_path)
    return int(m.group(1)) if m else None


def load_jsons(
    input_dir: pathlib.Path, exclude: list[str], include_legacy: bool,
) -> list[dict]:
    rows = []
    for p in sorted(input_dir.rglob("*.json")):
        if any(token in str(p) for token in exclude):
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue
        if "metadata" not in data or "task_results" not in data:
            print(f"[skip] {p}: missing metadata/task_results")
            continue
        md = data["metadata"]
        # 구형: encoder_type 누락 = pre-v3 broken format (use_joint=False 등)
        if not include_legacy and "encoder_type" not in md:
            continue
        md.setdefault("task_suite", data.get("task_suite", "unknown"))
        if "checkpoint" not in md:
            md["checkpoint"] = ""
        rows.append((p, data))
    return rows


def write_episodes_csv(jsons: list, out_path: pathlib.Path) -> int:
    fields = [
        "encoder", "suite", "seed", "task_id", "ep_id",
        "success", "steps_to_done", "errored",
        "task_description", "ckpt", "result_json", "timestamp",
    ]
    n = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for json_path, data in jsons:
            md = data["metadata"]
            ckpt_seed = extract_seed_from_ckpt(md.get("checkpoint", ""))
            seed = ckpt_seed if ckpt_seed is not None else md.get("seed")
            for tr in data["task_results"]:
                eps = tr.get("episode_records")
                if not eps:
                    # 구버전 JSON은 per-episode 없음 — 집계만 가능, episodes.csv 제외
                    continue
                for ep in eps:
                    w.writerow({
                        "encoder": md.get("encoder_type", "unknown"),
                        "suite": md["task_suite"],
                        "seed": seed,
                        "task_id": tr["task_id"],
                        "ep_id": ep["ep_id"],
                        "success": int(ep["success"]),
                        "steps_to_done": ep["steps_to_done"],
                        "errored": int(ep.get("errored", False)),
                        "task_description": tr["task_description"],
                        "ckpt": md.get("checkpoint", ""),
                        "result_json": str(json_path),
                        "timestamp": md.get("timestamp", ""),
                    })
                    n += 1
    return n


def write_per_task_csv(jsons: list, out_path: pathlib.Path) -> int:
    fields = [
        "encoder", "suite", "seed", "task_id", "task_description",
        "n_episodes", "n_success", "success_rate",
    ]
    n = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for _, data in jsons:
            md = data["metadata"]
            ckpt_seed = extract_seed_from_ckpt(md.get("checkpoint", ""))
            seed = ckpt_seed if ckpt_seed is not None else md.get("seed")
            for tr in data["task_results"]:
                w.writerow({
                    "encoder": md.get("encoder_type", "unknown"),
                    "suite": md["task_suite"],
                    "seed": seed,
                    "task_id": tr["task_id"],
                    "task_description": tr["task_description"],
                    "n_episodes": tr["episodes"],
                    "n_success": tr["successes"],
                    "success_rate": round(tr["success_rate"], 4),
                })
                n += 1
    return n


def write_summary_csv(jsons: list, out_path: pathlib.Path) -> int:
    """(encoder, suite) 단위로 seed-level mean ± std."""
    # (encoder, suite, seed) → suite-level SR (10-task 평균)
    seed_sr = defaultdict(dict)  # (enc, suite)[seed] = sr
    seed_eps = defaultdict(dict)  # (enc, suite)[seed] = total_eps
    for _, data in jsons:
        md = data["metadata"]
        ckpt_seed = extract_seed_from_ckpt(md["checkpoint"])
        seed = ckpt_seed if ckpt_seed is not None else md.get("seed")
        key = (md.get("encoder_type", "unknown"), md["task_suite"])
        seed_sr[key][seed] = data["overall_success_rate"]
        seed_eps[key][seed] = data["total_episodes"]

    fields = [
        "encoder", "suite", "n_seeds", "seeds",
        "mean_success_rate", "std_success_rate", "se_success_rate",
        "n_episodes_per_seed", "n_episodes_total",
    ]
    n = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (enc, suite), seed_vals in sorted(seed_sr.items()):
            seeds = sorted(seed_vals.keys())
            srs = [seed_vals[s] for s in seeds]
            mean = sum(srs) / len(srs)
            var = sum((x - mean) ** 2 for x in srs) / max(1, len(srs) - 1)
            std = math.sqrt(var) if len(srs) > 1 else 0.0
            se = std / math.sqrt(len(srs)) if len(srs) > 1 else 0.0
            eps_per_seed = list(seed_eps[(enc, suite)].values())
            w.writerow({
                "encoder": enc,
                "suite": suite,
                "n_seeds": len(seeds),
                "seeds": ",".join(str(s) for s in seeds),
                "mean_success_rate": round(mean, 4),
                "std_success_rate": round(std, 4),
                "se_success_rate": round(se, 4),
                "n_episodes_per_seed": eps_per_seed[0] if eps_per_seed else 0,
                "n_episodes_total": sum(eps_per_seed),
            })
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/libero/results")
    ap.add_argument("--output-dir", default="paper_artifacts/libero_rollout")
    ap.add_argument("--exclude", nargs="*", default=["_timing", "_sanity", "_archive"],
                    help="path 안에 이 토큰이 들어가면 스킵")
    ap.add_argument("--include-legacy", action="store_true",
                    help="encoder_type 메타 누락된 옛 JSON도 포함 (기본 제외)")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsons = load_jsons(in_dir, args.exclude, args.include_legacy)
    print(f"Loaded {len(jsons)} result JSON(s) from {in_dir} (excluded: {args.exclude})")

    n_ep = write_episodes_csv(jsons, out_dir / "episodes.csv")
    n_task = write_per_task_csv(jsons, out_dir / "per_task.csv")
    n_sum = write_summary_csv(jsons, out_dir / "summary.csv")

    print(f"  episodes.csv : {n_ep:>6} rows  → {out_dir / 'episodes.csv'}")
    print(f"  per_task.csv : {n_task:>6} rows  → {out_dir / 'per_task.csv'}")
    print(f"  summary.csv  : {n_sum:>6} rows  → {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
