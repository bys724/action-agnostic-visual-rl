# Architecture and Visualization Artifacts

Diagrams and visualizations of the Two-Stream v11 model. The paper repo can
either copy these into its figure directory or reference them in place.

## Files (planned)

These should be **copied from [`docs/architecture/`](../../docs/architecture/)
when the design stabilizes**. Not yet copied — paper repo can pull directly
from `docs/architecture/` for now, or sync once the diagrams are paper-final.

| Source path (in `docs/architecture/`) | Purpose | Paper position |
|---------------------------------------|---------|----------------|
| `two_stream_v11_architecture.png` (+ `.mmd` source) | Full v11 model diagram (M / P encoders, motion-routing block, dual-target loss). | Method section, main figure. |
| `attn_v11_ep44_nomask.png` | Motion-routing attention heatmaps at the champion epoch (ep44). Visualizes what M-stream attends to in P-stream. | Method or qualitative section. |
| `attn_v11_ep4_nomask.png` ... `attn_v11_ep50.png` | Attention progression through training. | Supplementary, optional. |
| `rotation_*` | Rotation augmentation samples (training data viz). | Supplementary, optional. |
| `sample_detail/` | Single-sample breakdown: M vs P channel + attention overlay. | Supplementary, qualitative. |

## Naming convention

`attn_v11_ep<N>_nomask.png`: attention with masking disabled at inference
(visualizes what the trained model attends to on full unmasked frames).
`attn_v11_ep<N>.png`: attention with the same 75% masking pattern used in
training (shows learned recovery behavior).

The "nomask" variants are usually the right choice for paper figures — they
show the model's perception of full natural input, not the artificial
training distribution.

## Source code

- Model: [`src/models/two_stream_v11.py`](../../src/models/two_stream_v11.py)
- Attention extraction: [`scripts/eval/visualize_attn_v11.py`](../../scripts/eval/visualize_attn_v11.py)
- Mermaid source for the architecture diagram is checked in alongside the PNG
  (`two_stream_v11_architecture.mmd`). Re-render with mermaid-cli if the
  design changes.

## Notes for paper repo

- The model has 250.9 M total parameters but only 204 M is the downstream
  encoder (M + P encoders + motion-routing). The 47 M decoder is discarded
  for downstream use. State this clearly when reporting param count.
- v11 vs the standard cross-attention ablation (`v11-VfromM`) has identical
  param count (208.33 M) — single-variable comparison. The ablation is
  pre-training in JobID 33615395 (running at the time of writing).
