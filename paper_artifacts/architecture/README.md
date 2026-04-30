# Architecture and Visualization Artifacts

Model diagrams (Mermaid + rendered PNG) and supporting visualizations for the
Two-Stream v11 architecture.

## Files in this directory

| File | Purpose | Editable? | Paper position |
|------|---------|-----------|----------------|
| `two_stream_v11_architecture.mmd` | **Mermaid source** of the v11 model diagram (M / P encoders, motion-routing block, decoder phases, dual-target loss). Single source of truth — edit this. | ✅ via Mermaid MCP on the paper workstation, or any text editor | Method section, main figure. |
| `two_stream_v11_architecture.png` | Rendered Mermaid output. Regenerate from the .mmd whenever the diagram changes. | ❌ regenerate, do not edit by hand | Same. |

### Editing workflow

The Mermaid source is here as the single source of truth. The paper-side
workstation has Mermaid MCP tooling configured; the cluster does not. Edit
the .mmd there, re-render to .png, commit both, push. The cluster picks up
the new diagram on the next `git pull`.

To re-render manually if you don't use MCP:
```bash
mmdc -i two_stream_v11_architecture.mmd -o two_stream_v11_architecture.png \
     -t neutral -b transparent
```

## Files in `docs/architecture/` (NOT moved here)

The following remain in [`docs/architecture/`](../../docs/architecture/) because
multiple cluster-side documents reference them by that path
(`docs/cluster_sessions.md`, `docs/artifacts.md`, `CLAUDE.md`). They are
generated artifacts (model output visualizations), not editable sources.

| Source path | Purpose | Paper position |
|-------------|---------|----------------|
| `attn_v11_ep44_nomask.png` | Motion-routing attention at the champion epoch (ep44). 4 rows × 8 cols (col 0/1 raw frames, col 2-5 attention overlays, col 6/7 reconstructions). **Updated 2026-04-30**: Sobel-edge background with **dual-frame composite** (past frame in light blue ghost + current frame in sharp dark) for direct motion visualization, motion-peak anchor (auto-selects end-effector patch via frame-diff). | Method or qualitative section, main figure candidate. |
| `attn_v11_ep{4,8,12,16,20,24,48,50}*_nomask.png` | Attention progression across training. | Supplementary, optional. |
| `attn_v11_ep4.png`, `attn_v11_ep48.png`, `attn_v11_ep50.png` | Variants with masked-input attention (training-distribution view). Less common in paper figures than `_nomask` variants. | Supplementary. |
| `rotation_v10_ep*`, `rotation_v6_baseline` | Rotation augmentation samples (data viz). | Supplementary, optional. |
| `sample_detail/` | Single-sample breakdown: M vs P channel + attention overlay. | Supplementary, qualitative. |

The paper repo can pull these directly via `docs/architecture/<filename>` —
they are stable PNGs, not under active editing. If the paper repo needs to
copy or symlink any of them into its own figure directory, do so per-file
rather than syncing the whole folder (most viz are exploratory and won't go
into the paper).

## Naming convention (for the attention viz)

- `attn_v11_ep<N>_nomask.png`: attention at inference with masking **disabled**
  — visualizes what the trained model attends to on full natural input.
  Usually the right choice for paper figures.
- `attn_v11_ep<N>.png` (no `_nomask` suffix): attention with the same 75% MAE
  masking pattern used in training — shows learned recovery behavior on the
  artificial training distribution.

## Source code

- Model: [`src/models/two_stream_v11.py`](../../src/models/two_stream_v11.py)
- Attention extraction: [`scripts/eval/visualize_attn_v11.py`](../../scripts/eval/visualize_attn_v11.py)

## Notes for the paper repo

- The model has **250.9 M total parameters** but only **204 M is the downstream
  encoder** (M + P encoders + motion-routing). The 47 M decoder is discarded
  for downstream use. State this clearly when reporting param count.
- v11 vs the standard cross-attention ablation (`v11-VfromM`) has **identical**
  param count (208.33 M) — single-variable comparison. The ablation is
  pre-training in JobID 33615395 (running at the time of writing).
- v11 was trained with `mask_p=0.75` (P-stream MAE) and `mask_m=0.5` (M-stream
  light masking). The asymmetric masking is core to the design — see
  CLAUDE.md memory `feedback_two_stream_masking_philosophy` for the
  rationale.
