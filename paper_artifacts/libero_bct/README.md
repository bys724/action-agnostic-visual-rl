# LIBERO Behavior Cloning (BC-T)

**Status: pending**. BC training is in progress as of 2026-04-29. This
directory will be populated once jobs complete.

## Planned files

| File | Description |
|------|-------------|
| `summary.csv` | Per-(encoder × suite × seed) val MSE / NLL trajectory and best epoch. |
| `nll_curves.csv` | Full epoch-by-epoch eval NLL trajectory for plotting. |

## Experimental matrix (Phase 3 main BC table)

5 encoders × 3 task suites × 3 seeds = **45 BC runs**.

| Encoder | Notes |
|---------|-------|
| `two-stream-v11` (ours) | ckpt ep44 (`A+D'` mode used by `finetune_libero_bct.py`). |
| `videomae-ours` | ckpt best (controlled comparison — same EgoDex pre-training data). |
| `dinov2-base` | Public weights (Meta DINOv2). |
| `siglip-base` | Public weights (Google SigLIP). |
| `vc1-base` | Public weights (VC-1 from `vc_models` package). |

Suites: `libero_spatial`, `libero_object`, `libero_goal`. Seeds: 0, 1, 2.

V-JEPA 2.1 was originally planned but excluded — 16-frame × 384² input makes
50-epoch BC infeasible (estimated 30–100 days). Probing-only inclusion in the
paper.

## Source code

- Driver: [`scripts/eval/finetune_libero_bct.py`](../../scripts/eval/finetune_libero_bct.py)
- Encoder adapters: [`src/encoders/adapters/single_frame.py`](../../src/encoders/adapters/single_frame.py)
- sbatch: [`scripts/cluster/finetune_libero_bct.sbatch`](../../scripts/cluster/finetune_libero_bct.sbatch)
- Raw output dir: `/proj/external_group/mrg/checkpoints/libero_bct/<encoder>_<suite>_seed<N>_<timestamp>/`

## Notes for paper repo

- BC val MSE is reported in normalized action space (robomimic convention).
  Lower is better.
- The metric of interest is **rollout success rate** (closed-loop in mujoco),
  not val MSE. See `libero_rollout/` for that.
- BC-T is the official LIBERO benchmark protocol, including the BasePolicy +
  spatial softmax + RNN head; the encoder is frozen during BC fine-tune.
