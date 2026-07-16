# CoMP-MAE — Code & Data Supplement

Anonymous code and data supplement for a double-blind AAAI submission.

**CoMP-MAE** (Compositional Motion-Prior MAE) is an action-agnostic visual
representation pretrained on unlabeled first-person video (EgoDex) via a
**symmetric cross-reconstruction** two-stream masked autoencoder. Two streams are
derived from each frame pair `(t, t+k)`:

- **P** (appearance) = RGB of frame `t` — a semantic backbone (ViT depth 12).
- **M** (motion) = `|ΔL|`, the absolute luminance difference — a small motion
  sensor (ViT depth 6).

A shared motion-routing decoder reconstructs masked patches of **both** streams
using **value-ownership routing**: the routing pattern (Q/K) comes from one
stream while the reconstructed values (V) are owned by the other. CoMP-MAE's
**M-recon** branch mirrors the P-recon branch so the motion encoder is grounded
in its own reconstruction target. The matched control (**plain cross-modal MAE**)
removes the M-recon branch and uses standard cross-attention.

This supplement contains the model definition, training objective, training
script + hyperparameter configs, the evaluation harness (action probing + BC-T
finetuning/rollout), the aggregated evaluation artifacts, and the statistical
test with its evidence CSV. The trained weights are released upon publication
(the checklist commits the source code); the headline efficiency table and the
rollout significance test reproduce from the shipped artifacts without them.

---

## Repository structure

```
model/                         Model definition + training objective
  comp_mae.py            CoMP-MAE model (comp_mae) + plain control (pixel_pred)
  two_stream_base.py            Shared base (encoders, masking, decoder modules)
  common/blocks.py             TransformerBlock, MotionRoutingBlock (value-ownership routing)
  common/preprocessing.py      M (|ΔL|) / P (RGB) channel preprocessing
train/
  pretrain.py                  Single-file pretraining script (EgoDex frame pairs)
  config_comp_mae_s.yaml       CoMP-MAE-S hyperparameters (reported run)
  config_plain_control.yaml    Plain control hyperparameters (reported run)
eval/
  probe_action_libero.py       Action probing on LIBERO (linear/attentive probe R^2)
  probe_action_calvin.py       Action probing on CALVIN (cross-folder OOD)
  build_step0_efficiency_table.py   Aggregate probing R^2 into the efficiency table
  finetune_libero_bct.py       BC-Transformer finetuning (frozen encoder + policy)
  eval_libero.py               LIBERO rollout evaluation (success rate)
  aggregate_libero_rollouts.py Aggregate per-seed rollouts into summary tables
  adapters/                    Encoder -> BC-T token adapter (CoMP-MAE = 'comp-mae')
  datasets/calvin.py           CALVIN action-probing dataset loader
  policies/bc_transformer_adapted.py   BC-Transformer policy wrapper
  artifacts/probing_efficiency.csv     Aggregated probing R^2 of the reported runs
stats/
  stats_libero_rollout.py      Paired Wilcoxon test: CoMP-MAE vs plain control
  artifacts/per_task.csv       Per-task rollout success rates (evidence)
checkpoints/                   README (weights released upon publication) + SHA-256 + run metadata
requirements.txt               Core dependencies
LICENSE                        MIT
```

### Naming note

- Model class `CoMPMAE` (in `model/comp_mae.py`) implements both objectives:
  `comp_mae=True` selects CoMP-MAE (ours); `pixel_pred=True` selects the plain
  cross-modal MAE control.
- In the eval CLI, the CoMP-MAE encoder is selected with `--encoder comp-mae`.
- The two-stream P (appearance) / M (motion) split is motivated by the
  parvocellular / magnocellular visual pathways.

---

## Installation

```bash
pip install -r requirements.txt
```

BC-Transformer finetuning and LIBERO rollout additionally require the LIBERO
benchmark + simulator (robosuite / MuJoCo); CALVIN probing requires the CALVIN
dataset dependencies. Install those separately per their official repositories
(see `requirements.txt`). Action probing and the aggregation/stats scripts run
with the core dependencies only.

### Data

The datasets are **not** redistributed here — download them from their official
sources and point the scripts at your local copies:

- **EgoDex** (pretraining) — pre-extract frames to
  `<frames_root>/<task>/<video>/frame_000000.jpg ...` (center-cropped square,
  256x256).
- **LIBERO** (probing + BC rollout) and **CALVIN** (probing).

### Preprocessing parity (required)

Inputs are raw RGB in `[0, 1]` (`/255.0` only). **Do not** apply ImageNet
mean/std normalization — the encoders were trained without it, and evaluating
with normalized inputs is out-of-distribution.

---

## Pretrained checkpoints

The two headline ViT-S encoders (CoMP-MAE-S, plain control; 32.3M params each,
50 epochs on EgoDex part1) are **released upon publication** — this anonymized
supplement ships the full source, configs, and evaluation artifacts, not the
weight files. See `checkpoints/README.md` for the training cost (~110 GPU-hours
per encoder), SHA-256 sums for the upon-publication release, and loading.

The headline **efficiency table** and the **rollout significance test** below
reproduce *without* the weights, from the shipped aggregated artifacts. The
per-encoder probing / BC steps need the weights (pretrain them with
`train/pretrain.py`, or use the upon-publication release).

---

## Reproducing the headline results

### 1. OOD action-probing efficiency table (no weights, no data needed)

The aggregated probing R^2 of the reported runs is shipped
(`eval/artifacts/probing_efficiency.csv`), so the table renders directly:

```bash
python3 eval/build_step0_efficiency_table.py
```

Expected (position R^2, translational dims 0-2; higher = better):

| Encoder | Params | CALVIN | LIB-spatial | LIB-object | LIB-goal |
|---|---:|---:|---:|---:|---:|
| **CoMP-MAE-S  P⊕M (attn)** | 32.3M | +0.487 | +0.814 | +0.851 | +0.751 |
| Plain xMAE-S  P⊕M (attn) *control* | 32.3M | +0.030 | +0.127 | +0.109 | +0.059 |
| VideoMAE-ours (attn) *same-data 86M* | 86.0M | +0.610 | +0.879 | +0.903 | +0.830 |
| VC-1 / DINOv2 / SigLIP (frozen ViT-B) | 86.0M | see table | | | |

CoMP-MAE-S (~32M, narrow unlabeled home video) matches or edges the
internet-scale frozen ViT-B encoders and approaches a larger same-data video
model, while the matched plain control collapses — the efficiency is a product of
the CoMP-MAE mechanism, not of params/data.

### 2. Action probing from scratch (needs weights + LIBERO / CALVIN)

Pretrain an encoder first (see Pretraining below) or use the upon-publication
weights, then:

```bash
python3 eval/probe_action_libero.py --encoder comp-mae \
    --checkpoint <CHECKPOINT_DIR>/comp_mae_s.pt \
    --task-suite libero_spatial --data-root <LIBERO_DATA_ROOT> \
    --readout attentive --stream-mode p_t_m --output-dir outputs/probe_spatial

python3 eval/probe_action_calvin.py --encoder comp-mae \
    --checkpoint <CHECKPOINT_DIR>/comp_mae_s.pt \
    --data-root <CALVIN_DATA_ROOT> --output-dir outputs/probe_calvin
```

Each run writes a `summary.json` (per-dim R^2, n_eval); re-aggregate with
`build_step0_efficiency_table.py`. Baseline encoder loaders (VideoMAE, DINOv2,
SigLIP, VC-1) are omitted from this anonymized release — the shipped rows use the
two released checkpoints.

### 3. BC-Transformer finetuning + LIBERO rollout (needs LIBERO)

```bash
python3 eval/finetune_libero_bct.py --encoder comp-mae \
    --checkpoint <CHECKPOINT_DIR>/comp_mae_s.pt --task-suite libero_spatial \
    --seed 0 --output-dir outputs/bct_comp_spatial_s0

python3 eval/eval_libero.py --checkpoint <CHECKPOINT_DIR>/best.pt \
    --task-suite libero_spatial --n-eval 50

python3 eval/aggregate_libero_rollouts.py --input-dir outputs/ \
    --output-dir outputs/libero_rollout
```

### 4. Rollout significance (CoMP-MAE vs plain control; no data needed)

```bash
python3 stats/stats_libero_rollout.py
```

Expected:

```
Paired LIBERO rollout: comp-mae-s  vs  plain-xmae-s
  paired tasks (seed-averaged, 3 suites x 10 tasks): n = 30
  pooled mean delta (A - B) = -0.0091 (-0.91 pt)
  Wilcoxon signed-rank: statistic = 186.0, p = 0.6986
  -> within noise (no significant difference)
```

CoMP-MAE's representation-level signature (the probing gap above) does not
translate into a value-level rollout advantage over the matched control — the two
are within noise on LIBERO success rate.

---

## Pretraining (needs EgoDex)

```bash
# CoMP-MAE-S (ours)
python3 train/pretrain.py --frames-root <EGODEX_FRAMES_ROOT> --comp-mae \
    --embed-dim 384 --num-heads 6 --m-depth 6 --epochs 50 \
    --batch-size 128 --lr 2.8e-4 --routing-mode v_from_p --rotation-aug

# Plain cross-modal MAE (matched control)
python3 train/pretrain.py --frames-root <EGODEX_FRAMES_ROOT> --pixel-pred \
    --embed-dim 384 --num-heads 6 --m-depth 6 --epochs 50 \
    --batch-size 128 --lr 2.8e-4 --routing-mode v_from_m --masked-anchor \
    --rotation-aug --independent-rotation-prob 0.0
```

The reported runs used 8x H100 with DistributedDataParallel (global batch 1024),
costing roughly **~13.7 h wall-clock (~110 GPU-hours)** per encoder. Full
hyperparameters are in `train/config_{comp_mae_s,plain_control}.yaml`.

---

## License

MIT (see `LICENSE`). Datasets are subject to their own licenses.
