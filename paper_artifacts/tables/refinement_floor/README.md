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

## Round 2 (2026-09-26/27, probe-only, plan §9)

Three follow-up questions, no representation training. Mean ± 95% CI over probe seeds {42,1,2}; representation n = 1.
Aggregate: `round2_agg.txt` (`agg_round2.py`); exact-zero diagnostic: `round2_exact_zero.txt`.

| Question | Test | Result |
|---|---|---|
| Q1 — is M's noise collapse an exact-zero artifact? | R2-4: fraction of ΔL == 0 (1 s gap) | CALVIN 77% · LIBERO 80–85% · EgoDex 2.3% (0% all-zero patches). C1's Case A input is not exactly zero (independent brightness gains), so the pretraining exact-zero chain does not apply to C1 |
| | R2-3: pixel noise on probe-train **and** test | Same-distribution gap survives: CALVIN σ0.01 C1 M 0.44±0.08 vs raw 0.22±0.06 (σ0.02: 0.41 vs 0.22); LIBERO in-suite 0.66 vs 0.16 (σ0.02: 0.64 vs 0.16). F1′, F3 ≈ raw. C0 ≥ C1. → Round-1 noise collapse = clean-probe/noisy-test mismatch. Transfer stays negative for all arms (C1 −0.37 vs raw −0.32) |
| Q2 — is P_t⊕raw's collapse a zero-pad artifact? | R2-2: learned 256→384 projection in the probe | No: transfer −7.8±3.8 (zero-pad −10.2), shadow 0.6 −2.9, gain 1.3 −17.4, best epoch still 3. Adding P makes the readout latch onto appearance |
| Q3 — does the deployed P (two frames) hold up? | R2-1: P_t⊕P_tk on all shift tests | C1's P: noise-invariant (0.33 at σ0.01–0.04), shadow 0.6 −0.13±0.37 (raw −0.32±0.58, C1 M −2.71), gain 1.3 −4.4±5.6 (raw −3.9, C1 M −15.8); LIBERO in-suite 0.80, transfer −0.56±0.24 (raw −0.32±0.01). C0's P transfers worse (−1.94±0.36). → Photometric fragility is local to M; deployed P ≈ raw on shift tests |

## §10 pilot — sensor-noise / illumination pretraining (2026-09-27/28, plan §10)

C1 + 10 epochs (`--init-from` C1, LR 2.8e-5, 3×H100, effective batch 1023): M input gets per-pixel RGB Gaussian
noise σ~U[0, 0.01] (P input clean) and a pair-shared scene gain U[0.73, 1.37] (±1 stop); M-recon target =
brightness-preserved, **noise-free** ΔL (`--bright-target aug --m-noise-max 0.01 --bright-scene-gain-range 0.73 1.37`).
Checkpoint `two_stream_v15b_refine_comp_s_denoise_augtgt/20260927_235906/checkpoint_epoch0010.pt` (= **C1-DN**, adopted as the paper model 09-28).
Probe fit on clean CALVIN, tested on held-out noise types (never used in training); probe seeds {42,1,2}.

| Test | C1-DN (10 ep) | C1 M | raw ΔL |
|---|---:|---:|---:|
| clean | 0.43 | 0.46 | 0.22 |
| shot noise 0.01 / 0.005 | 0.21 / 0.40 | −1.47 / 0.07 | 0.22 / 0.22 |
| correlated noise 0.01 / 0.005 | 0.00 / 0.34 | −1.40 / 0.05 | 0.22 / 0.22 |
| JPEG q75 (keeps exact zeros; reference only) | 0.40 | 0.44 | 0.22 |
| shadow 0.6 / gain 1.3 | −2.9 / −12.2 | −2.7 / −15.8 | −0.3 / −3.9 |
| LIBERO in-suite / transfer | 0.68 / −0.32 | 0.71 / −0.33 | 0.16 / −0.32 |

Pre-registered verdict: ① (≥ raw on both shot and corr at 0.01) **not met** — collapse removed down to raw parity,
not beyond; ② CALVIN 0.43 (≥ 0.40, just under 2× raw), LIBERO 0.68 → no 50-ep main training. Epoch-5 checkpoint gave
the same picture (`pilot10_ep5_20260927.txt`). Files: `pilot10_ep10_20260928.txt`.

## Factorization re-measure on C1-DN (2026-09-28)

STEP 1 same-probe protocol unchanged (LIBERO-object, attentive readout, gap 20, one probe run per cell; motion = mean R² of the
3 position dims, identity = 10-way task accuracy, chance 0.10; Δ = (feature ⊕ ee_pos) − ee_pos alone, controls 0.579 / 0.314).
C0 raw cells re-run on current code reproduce the 07-02 values exactly (0.835 / 0.526 / 0.547 / 0.999) → no probe drift.
Jobs 40321645–656; dirs `paper_artifacts/libero_action_probing/parvo_libero_object_20260928_*_f{dn,c0}_*`.

| cell (raw / Δ) | C0 | C1-DN |
|---|---|---|
| M motion | 0.835 / +0.338 | 0.785 / +0.320 |
| M identity | 0.526 / +0.307 | 0.448 / +0.241 |
| P_t motion | 0.547 / +0.126 | 0.621 / +0.209 |
| P_t identity | 0.999 | 1.000 |

Directional double dissociation holds (M ≫ P on beyond-position motion, P at identity ceiling, M identity residual lower),
but the M–P motion gap narrows (Δ ratio 2.7× → 1.5×) because P_t carries more motion. Single probe run; C1 vs denoise
contribution not separated.

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

## E0 — external RGB encoders fed the ΔL image (2026-10-02, `docs/claim_spine_v2.md` §3)

Reviewer UnGc W1. DINOv2-B/14, SigLIP-B/16, VC-1-B get a single ΔL image (same BT.709 luminance difference as the
M input), replicated to 3 channels: **signed** = 0.5 + 0.5·ΔL, **abs** = |ΔL|; each encoder's own normalization;
all final-norm patch tokens → the same attentive probe as round 1. Probe seeds {42,1,2}; mean ± 95% CI; encoders
frozen (n = 1 each). Run on MIG-1g (round 1 on V100; one cell checked identical). Aggregate: `e0_agg.txt` (`agg_e0.py`).

| Arm | CALVIN clean (i) | LIBERO in-suite (i) | LIBERO transfer, 6-dir (s) |
|---|---:|---:|---:|
| VC-1 + signed ΔL | 0.51 ± 0.04 | **0.79 ± 0.01** | **+0.06 ± 0.04** |
| VC-1 + \|ΔL\| | 0.12 ± 0.02 | 0.67 ± 0.01 | −0.45 ± 0.09 |
| DINOv2 + signed / \|ΔL\| | −0.19 / −0.54 | 0.71 / 0.63 | −0.11 / −0.38 |
| SigLIP + signed / \|ΔL\| | −0.26 / −0.62 | 0.66 / 0.60 | −0.19 / −0.35 |
| C0 M (round 1) | 0.47 ± 0.12 | 0.74 ± 0.03 | −0.23 ± 0.10 |
| raw ΔL F1 (round 1) | 0.22 | 0.16 | −0.32 |

Pre-registered reading (best external arm = VC-1 signed vs CoMP M CI): CALVIN overlaps (parity); LIBERO in-suite and
transfer exceed C0 M's CI upper bound → "our encoder is special" premise **not supported**. DINOv2/SigLIP fall below
raw ΔL on CALVIN, so the result is encoder-specific (VC-1 is an MAE-pretrained egocentric-video ViT, 86M vs CoMP-S M).
