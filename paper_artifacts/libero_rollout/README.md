# LIBERO Rollout — Closed-loop Evaluation

Rollout runs on the local workstation (mujoco GPU rendering, `libero-eval`
container) using BC-T ckpts trained on the cluster.

## Files

| File | Granularity | Use |
|------|-------------|-----|
| `episodes.csv` | One row per episode | Most flexible. Re-aggregate any way for paper figures. |
| `per_task.csv` | One row per (encoder, suite, seed, task) | Per-task SR breakdown — failure analysis figures. |
| `summary.csv` | One row per (encoder, suite) | Paper main table — mean ± std across seeds. |

## How to (re)generate

After running rollouts via `src/eval_libero.py` (JSONs land in
`data/libero/results/`):

```bash
python scripts/eval/aggregate_libero_rollouts.py \
    --input-dir data/libero/results \
    --output-dir paper_artifacts/libero_rollout
```

This rebuilds all 3 CSVs from scratch by scanning the JSON results dir.
`_timing`/`_sanity` paths are excluded by default.

## Schemas

### `episodes.csv`
`encoder, suite, seed, task_id, ep_id, success, steps_to_done, errored,
task_description, ckpt, result_json, timestamp`

`success` ∈ {0, 1}, `steps_to_done` is the t at which `done=True` (success)
or `max_steps + num_steps_wait` (failure to converge).

### `per_task.csv`
`encoder, suite, seed, task_id, task_description, n_episodes, n_success,
success_rate`

### `summary.csv`
`encoder, suite, n_seeds, seeds, mean_success_rate, std_success_rate,
se_success_rate, n_episodes_per_seed, n_episodes_total`

`std`/`se` are computed across **seeds** (suite-level SR, averaged over
10 tasks per seed). For per-task uncertainty, derive from `episodes.csv`
directly (binomial CI on the binary `success` column).

## Why this is the headline metric

- Probing R² and BC val MSE both measure **prediction**, not control.
- LIBERO closed-loop success rate is what the paper claims about: whether the
  representation enables a robot to actually accomplish the manipulation
  task in simulation.
- Open-loop val MSE can be misleading — small per-step errors compound in
  closed-loop rollout, so success rate is the rigorous test.

## Evaluation protocol

- Each (encoder × suite × seed) BC ckpt is rolled out for **20 episodes per
  task × 10 tasks per suite = 200 episodes**.
- Episode success defined by the LIBERO task spec (e.g. object placed in
  correct receptacle, bowl on plate within tolerance).
- Episodes use stochastic action sampling at policy temperature 1.0 (default
  LIBERO BC-T eval setting).

## Source code

- Rollout driver: [`src/eval_libero.py`](../../src/eval_libero.py)
- Container: `libero` Docker compose service (mujoco-EGL).
- BC ckpts source: cluster `/proj/external_group/mrg/checkpoints/libero_bct/`,
  transferred to local `/mnt/data/checkpoints/libero_bct/`.

## Notes for paper repo

- Success rate is reported as percentage. Round to 1 decimal place.
- Standard error across seeds is the primary uncertainty estimate. Report
  mean ± std over 3 seeds.
- For per-task breakdown (often useful for failure analysis figures), use
  `success_rate.csv` rather than the aggregated summary.
