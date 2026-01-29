#!/usr/bin/env python
"""
LIBERO Benchmark Results Comparison Script

여러 모델의 LIBERO 평가 결과를 비교하고 시각화합니다.
"""

import argparse
import json
import pathlib
from typing import Dict, List, Optional


def load_result(path: str) -> Dict:
    """Load evaluation result from JSON file."""
    with open(path) as f:
        return json.load(f)


def find_latest_results(
    results_dir: str,
    models: List[str],
    task_suite: str
) -> Dict[str, Dict]:
    """Find the latest result file for each model."""
    results_path = pathlib.Path(results_dir)
    results = {}

    for model in models:
        pattern = f"{model}_{task_suite}_*.json"
        files = sorted(results_path.glob(pattern), reverse=True)

        if files:
            results[model] = load_result(files[0])
            results[model]["_file"] = str(files[0])
        else:
            print(f"Warning: No results found for {model} on {task_suite}")

    return results


def compare_results(results: Dict[str, Dict]) -> None:
    """Print comparison table of results."""
    if not results:
        print("No results to compare")
        return

    # Header
    print("\n" + "=" * 80)
    print("LIBERO Benchmark Comparison")
    print("=" * 80)

    # Get task suite from first result
    first_result = list(results.values())[0]
    task_suite = first_result.get("task_suite", "unknown")
    print(f"\nTask Suite: {task_suite}")

    # Overall comparison
    print("\n" + "-" * 80)
    print("Overall Results")
    print("-" * 80)
    print(f"{'Model':<15} {'Success Rate':>15} {'Successes':>12} {'Episodes':>12}")
    print("-" * 80)

    for model, result in sorted(results.items()):
        rate = result.get("overall_success_rate", 0) * 100
        successes = result.get("total_successes", 0)
        episodes = result.get("total_episodes", 0)
        print(f"{model:<15} {rate:>14.1f}% {successes:>12} {episodes:>12}")

    # Per-task comparison
    print("\n" + "-" * 80)
    print("Per-Task Success Rates (%)")
    print("-" * 80)

    # Get all task descriptions
    task_results_list = []
    for model, result in results.items():
        task_results_list.append(result.get("task_results", []))

    if not task_results_list or not task_results_list[0]:
        return

    num_tasks = len(task_results_list[0])

    # Print header
    model_names = sorted(results.keys())
    header = f"{'Task':<5} "
    for name in model_names:
        header += f"{name:>12} "
    print(header)
    print("-" * 80)

    # Print per-task results
    for task_id in range(num_tasks):
        row = f"{task_id:<5} "
        for model in model_names:
            task_results = results[model].get("task_results", [])
            if task_id < len(task_results):
                rate = task_results[task_id].get("success_rate", 0) * 100
                row += f"{rate:>11.1f}% "
            else:
                row += f"{'N/A':>12} "
        print(row)

    # Print task descriptions
    print("\n" + "-" * 80)
    print("Task Descriptions")
    print("-" * 80)
    first_task_results = task_results_list[0]
    for i, task in enumerate(first_task_results):
        desc = task.get("task_description", "N/A")
        print(f"Task {i}: {desc[:70]}{'...' if len(desc) > 70 else ''}")

    # Metadata comparison
    print("\n" + "-" * 80)
    print("Evaluation Settings")
    print("-" * 80)

    for model, result in sorted(results.items()):
        metadata = result.get("metadata", {})
        if metadata:
            print(f"\n{model}:")
            print(f"  - Seed: {metadata.get('seed', 'N/A')}")
            print(f"  - Replan steps: {metadata.get('replan_steps', 'N/A')}")
            print(f"  - Trials/task: {metadata.get('num_trials_per_task', 'N/A')}")
            print(f"  - Max steps: {metadata.get('max_steps', 'N/A')}")
            print(f"  - Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"  - File: {result.get('_file', 'N/A')}")

    print("\n" + "=" * 80)


def generate_markdown_report(
    results: Dict[str, Dict],
    output_path: Optional[str] = None
) -> str:
    """Generate markdown report for results comparison."""
    if not results:
        return "No results to compare"

    lines = []

    # Header
    first_result = list(results.values())[0]
    task_suite = first_result.get("task_suite", "unknown")

    lines.append(f"# LIBERO Benchmark Comparison: {task_suite}")
    lines.append("")

    # Overall comparison table
    lines.append("## Overall Results")
    lines.append("")
    lines.append("| Model | Success Rate | Successes | Episodes |")
    lines.append("|-------|-------------|-----------|----------|")

    for model, result in sorted(results.items(),
                                 key=lambda x: x[1].get("overall_success_rate", 0),
                                 reverse=True):
        rate = result.get("overall_success_rate", 0) * 100
        successes = result.get("total_successes", 0)
        episodes = result.get("total_episodes", 0)
        lines.append(f"| {model} | {rate:.1f}% | {successes} | {episodes} |")

    lines.append("")

    # Per-task comparison
    lines.append("## Per-Task Success Rates")
    lines.append("")

    model_names = sorted(results.keys())
    header = "| Task | Description | " + " | ".join(model_names) + " |"
    separator = "|------|-------------|" + "|".join(["-----" for _ in model_names]) + "|"

    lines.append(header)
    lines.append(separator)

    first_task_results = list(results.values())[0].get("task_results", [])
    for task_id, task in enumerate(first_task_results):
        desc = task.get("task_description", "N/A")[:40]
        row = f"| {task_id} | {desc}... |"
        for model in model_names:
            task_results = results[model].get("task_results", [])
            if task_id < len(task_results):
                rate = task_results[task_id].get("success_rate", 0) * 100
                row += f" {rate:.0f}% |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("")

    # Settings
    lines.append("## Evaluation Settings")
    lines.append("")

    for model, result in sorted(results.items()):
        metadata = result.get("metadata", {})
        lines.append(f"### {model}")
        lines.append(f"- Seed: {metadata.get('seed', 'N/A')}")
        lines.append(f"- Replan steps: {metadata.get('replan_steps', 'N/A')}")
        lines.append(f"- Trials per task: {metadata.get('num_trials_per_task', 'N/A')}")
        lines.append(f"- Max steps: {metadata.get('max_steps', 'N/A')}")
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Compare LIBERO evaluation results")

    parser.add_argument("--results-dir", type=str, default="data/libero/results",
                        help="Directory containing result JSON files")
    parser.add_argument("--models", type=str, nargs="+", default=["openvla", "pi0"],
                        help="Models to compare")
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        help="Task suite to compare")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                        help="Specific result files to compare (overrides auto-discovery)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for markdown report")

    args = parser.parse_args()

    # Load results
    if args.files:
        results = {}
        for f in args.files:
            result = load_result(f)
            # Extract model name from filename
            name = pathlib.Path(f).stem.split("_")[0]
            results[name] = result
            results[name]["_file"] = f
    else:
        results = find_latest_results(args.results_dir, args.models, args.task_suite)

    # Print comparison
    compare_results(results)

    # Generate markdown report if requested
    if args.output:
        generate_markdown_report(results, args.output)


if __name__ == "__main__":
    main()
