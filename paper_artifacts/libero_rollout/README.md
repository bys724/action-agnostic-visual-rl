# LIBERO Rollout — Closed-loop Evaluation

**Status: pending**. Rollout will be performed on the local workstation
(mujoco GPU rendering) after BC ckpts complete on the cluster and are
transferred. This directory will be populated once rollouts run.

## Planned files

| File | Description |
|------|-------------|
| `success_rate.csv` | Per-(encoder × suite × seed × task) episode-level success / failure. |
| `summary.csv` | Aggregated success rate per (encoder × suite), averaged over seeds and tasks. |

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
