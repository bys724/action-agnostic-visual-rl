# Refinement floor — does the learned M stream refine raw ΔL? (2026-09-25/26)

Response to AAAI-27 reviews ("no raw-ΔL floor"). Plan and pre-registered criteria:
[`docs/refinement_floor_plan.md`](../../../docs/refinement_floor_plan.md) (§5 arms/tests, §6 criteria).
Evaluation protocol for the perturbation test: `docs/eval_protocols.md` §4-b.

## Setup

- **C1** = submitted CoMP-S (C0) retrained 50 ep with per-frame independent brightness
  augmentation (`--bright-aug`: gain U[0.8,1.2] + ramp p0.5 amp≤0.15; M input augmented ΔL,
  M-recon target clean ΔL). ckpt `two_stream_v15b_refine_comp_s_bright/20260925_163504/latest.pt`.
- **Floors** (no pretraining): raw ΔL 16×16 patches (F1), fixed random projection to 384-d (F1′),
  raw with brightness aug on probe-train split only (F1-aug), hand-normalized ΔL (F2),
  random-init CoMP-S M (F3).
- Readout: one attentive-pool probe (1 query + linear) for every arm; target = end-effector
  position Δ (pos R² = mean of dims 0–2). Probe seeds {42, 1, 2}; 95% CI = mean ± 4.303·sd/√3.
  Representation training n = 1 per model.

## Results (pos R², mean over 3 probe seeds)

| Arm | CALVIN clean (i) | LIBERO in-suite (i) | LIBERO transfer, 6-dir (s) | CALVIN shadow 0.6 (s) | CALVIN noise σ0.01 (s) | CALVIN 5% labels |
|---|---:|---:|---:|---:|---:|---:|
| C1 M | 0.46 | 0.71 | −0.33 | −2.71 | −1.43 | 0.39 |
| C0 M | 0.49 | 0.74 | −0.23 | −2.59 | −2.57 | 0.36 |
| raw ΔL (F1) | 0.22 | 0.16 | −0.32 | −0.32 | +0.22 | 0.22 |
| proj raw (F1′) | 0.24 | 0.17 | −0.29 | −0.63 | +0.24 | 0.19 |
| aug raw (F1-aug) | 0.05 | 0.09 | −0.22 | +0.04 | +0.05 | 0.05 |
| norm ΔL (F2) | 0.13 | 0.32 | −0.31 | −0.92 | −0.26 | 0.00 |
| random-init M (F3) | 0.25 | 0.33 | −0.73 | −4.93 | +0.24 | 0.20 |
| P_t ⊕ C1 M | 0.49 | 0.80 | −1.29 | −1.99 | −1.28 | 0.33 |
| P_t ⊕ raw ΔL | 0.22 | 0.54 | −10.15 | −3.75 | +0.22 | 0.19 |

(i) = same-distribution reference; (s) = shift test (verdict axis). P_t = C1's P stream, shared.

## Reading

- **Same distribution:** the learned M makes action 2–4× more linearly accessible than raw ΔL;
  random-init M ≈ raw, so the gain comes from trained weights.
- **Under shift:** raw ΔL alone is the most robust configuration on both shift tests. The
  P_t⊕C1 M > P_t⊕raw result (transfer, shadow) is driven largely by P_t⊕raw being worse than
  raw alone (probe latches onto P appearance; CALVIN best epoch 2–3), not by C1 being robust.
  Transfer failures concentrate on the object suite (different camera: Floor scene).
- **Brightness augmentation (C1 vs C0):** fixes ramp only; no gain in gain/shadow/noise
  robustness. Noise fragility is learned (random-init M is noise-invariant; CALVIN static ΔL is
  exactly 0 for 73% of pixels and σ0.01 noise removes that).
- **§6 verdict (M-alone column):** (A) transfer not met, (B) label efficiency not met (CI
  overlap), (C) uncomputable (F1-aug never halves) → stop signal. Floor expansion on the P_t⊕X
  column deferred (2026-09-26).

## Files

| File | Content |
|---|---|
| `verdict_inputs_20260926.txt` | §6 inputs, all 7 M-alone arms × (A)(B)(C), with CIs (`agg_verdict.py`) |
| `pt_x_20260926.txt` | P_t⊕C1 M vs P_t⊕raw ΔL (`agg_pt.py`) |
| `calvin_m_alone_floor.csv` | CALVIN same-distribution floor table + 60-ep convergence check |
| `calvin_perturb_seed3.csv`, `calvin_perturb_pilot.csv`, `calvin_perturb_grid.png` | perturbation curves, pilot, visual check |
| `calvin_min_cell.csv` | first minimal cell + probe-seed variance |
| `jobs_*.txt` | Slurm job lists (test, arm, seed, job id) |

Re-aggregate: `python3 paper_artifacts/tables/refinement_floor/agg_verdict.py` / `agg_pt.py`.
