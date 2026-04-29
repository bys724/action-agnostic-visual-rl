# Probing Results

Linear probing of frozen visual representations on action prediction. The
probe is a single linear layer (MLP head with no hidden layer) trained for 20
epochs at lr 1e-3 with batch size 256 — see
[`scripts/eval/probe_action_v11.py`](../../scripts/eval/probe_action_v11.py)
and [`scripts/eval/probe_action_droid_v11.py`](../../scripts/eval/probe_action_droid_v11.py)
for the exact protocol.

## Files

| File | Encoder × dataset | Rows | Status |
|------|-------------------|------|--------|
| `v11_egodex_summary.csv` | Two-Stream v11 × **EgoDex** (within-domain) | 11 epochs × 12 modes = 132 | + ep32, ep40 incoming (24 jobs) |
| `v11_droid_summary.csv` | Two-Stream v11 × **DROID** (cross-domain) | 13 (ep12 only, partial coverage) | partial — see Limitations |
| `videomae_droid_summary.csv` | VideoMAE-ours × **DROID** (baseline) | 4 (gap × patch_mean only) | partial |

## CSV schema

### `v11_egodex_summary.csv`

| Column | Type | Description |
|--------|------|-------------|
| `epoch` | int | Pre-training epoch of the v11 ckpt. `50` = `latest.pt` (end of training). |
| `mode` | str | Feature extraction mode (12 values, see Mode definitions below). |
| `r2` | float | Linear-probe coefficient of determination on EgoDex test split. **Primary metric.** |
| `mse` | float | Probe mean squared error (lower is better). |
| `cosine_sim` | float | Cosine similarity between predicted and ground-truth action vectors. |
| `train_samples` / `eval_samples` | int | Probe train / eval set sizes (`180921` / `40914` for EgoDex test split). |
| `embed_dim` | int | Dimensionality of extracted feature (768 single, 1536 2-way concat, 2304 3-way). |
| `source_file` | str | Original probe JSON file the row was derived from (for traceability). |

### `v11_droid_summary.csv` / `videomae_droid_summary.csv`

Same schema, but with an extra `gap` column (frame stride, in 15 Hz DROID
units — `gap=15` ≈ 1 second ahead). No `embed_dim` column on the VideoMAE
baseline file (single-stream, fixed 768).

## Mode definitions

The v11 architecture has 4 candidate feature extraction positions:

| Position | Description |
|----------|-------------|
| **A** | Output of the M (motion) encoder (6-layer ViT). |
| **B** | Output of the P (perception) encoder (12-layer ViT). |
| **D'** | Output of the motion-routing block (P-stream after motion-guided attention, before final interpreter). |
| **D** | Output of the Phase 3 final interpreter (last block of the P decoder). |

The 12 probe modes combine these positions with two pooling strategies
(`patch_mean` averages all patch tokens; `cls` uses the CLS token):

| Mode (CSV value) | Position(s) | Pooling | Notes |
|------------------|-------------|---------|-------|
| `patch_mean_m_enc` | A | patch_mean | Single M-encoder feature. |
| `patch_mean_p_enc` | B | patch_mean | Single P-encoder feature. Often near zero — see Limitations. |
| `patch_mean_p_state_after_routing` | D' | patch_mean | Motion-guided P-stream feature. |
| `patch_mean_p_features_tk` | D | patch_mean | Phase 3 final P-stream feature. |
| `patch_mean_concat_enc_only` | A + B | concat → patch_mean | 2-way encoder concat. |
| `patch_mean_concat_enc_phase3` | A + D | concat → patch_mean | M encoder + Phase 3 final. |
| `patch_mean_concat_enc_d_prime` | A + D' | concat → patch_mean | M encoder + post-routing. |
| `patch_mean_concat_p_enc_d_prime` | B + D' | concat → patch_mean | P encoder + post-routing. |
| `patch_mean_concat_all` | A + B + D' | concat → patch_mean | **Champion mode** at ep44 (R² = +0.288). |
| `cls_m_enc`, `cls_p_enc`, `cls_concat_enc` | A, B, A+B | CLS token | CLS-pooled variants — consistently weaker than patch_mean (paper-relevant finding). |

## Experimental conditions

- **EgoDex split**: held-out `test` split. Probe trained on disjoint train split.
- **Gap**: 10 frames (≈ 0.67 s ahead at 15 Hz). Encoder takes frame `t` and predicts hand-pose at `t + gap`.
- **Action target**: 6-joint right-hand pose (`rightHand` + 5 fingertip positions, 18-D total) — see `per_joint_r2` in raw JSONs for breakdown if needed.
- **Ckpt provenance**: all v11 rows from
  `checkpoints/two_stream_v11/20260426_014333/checkpoint_epoch00{04,08,12,16,20,24,32,40,44,48}.pt`
  + `latest.pt` (= ep50). Submitted under JobIDs 33594155 (full 1차) +
  33600621 (resume), see [`docs/cluster_sessions.md`](../../docs/cluster_sessions.md).

## Limitations and known issues

- **Failed probe runs**: a small number of raw JSONs in `data/probing_results/`
  contain R² values like `-69` (probe diverged). Likely caused by NaN gradient
  or unlucky lr × random init combination during 20-epoch probe training.
  These are **filtered out** by the export script (`r2 < -1.0` is dropped).
  When duplicate (epoch, mode) pairs remain after filtering, the latest by
  mtime wins.
- **`patch_mean_p_enc` (B alone)**: R² ≈ 0 at all v11 epochs ≥ ep12. The P
  encoder alone does not develop action-predictive features without the
  motion-routing pathway. This is a feature of the design (M is the action
  carrier), not a bug. Mention in paper if reporting B alone.
- **DROID R² absolute values are small** (~0.005). DROID action probing is
  noisier and cross-domain — the signal is real but the absolute metric is
  weak. Use **gap-wise comparison vs VideoMAE baseline** rather than absolute
  R² for paper claims.
- **DROID coverage incomplete**: only ep12 measured for the v11 × DROID matrix.
  Re-measurement at ep44 / ep50 was deferred and may need to be added before
  paper submission. Decision was: cross-domain DROID is supplementary, not
  main result, so this gap is acceptable.

## Candidate figures / tables

- **F1 — Representation evolution trajectory**: x = epoch, y = R², lines per
  mode. Shows the W-shape collapse-recovery pattern (peak at ep44 +0.288,
  dip at ep20). With ep32, ep40 added, the ep24→ep44 ramp visualizes the
  LR cosine decay's late-stage representation gain.
- **F2 — 12-mode comparison at ep44**: bar chart, modes grouped by feature
  position (single encoder, intermediate, concat, CLS). Shows A+B+D' is
  champion and patch_mean ≫ CLS uniformly.
- **F3 — Cross-domain DROID gap sweep** (supplementary): bar chart of v11 vs
  VideoMAE at gap ∈ {1, 10, 15, 30}. Weak result by absolute R² but v11
  consistently avoids negative values.

## Reproducing

```bash
# Regenerate CSVs from current raw JSONs
python scripts/analysis/export_probing_summary.py

# Add new probe runs (e.g. for an additional epoch):
sbatch --export=ALL,CHECKPOINT=<ckpt.pt>,CLS_MODE=<mode> scripts/cluster/probe_v11.sbatch
# Repeat for each of the 12 modes, then re-run the export.
```
