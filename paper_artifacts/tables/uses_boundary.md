# Uses boundary — change channel × use (writing input · 2026-10-23)

Plan: `docs/paper_finalization_plan.md` §4. Numbers are quoted from the source READMEs / exports, not recomputed
(SC numbers quoted as exported, contract v2). Machine-readable: `uses_boundary.csv`.

**Model roles (state in the table caption):** the mechanism (claim 1) is shown on the submitted recipe **C0**;
the use boundaries are measured on the augmented recipe **C1-DN** (C0 → C1 brightness aug 50 ep → +10 ep sensor-noise
denoising), which keeps the directional M/P double dissociation (M motion Δ +0.320 vs P +0.209, P identity 1.000;
`refinement_floor/README.md` "Factorization re-measure on C1-DN") but has no plain control. Columns A and C use C0;
B uses both; D and E use C1-DN. No unifying retrain.

## Verdict grid

Rows = what carries the change signal. Verdict of CoMP M is against raw ΔL; verdict of the external-encoder row is
against CoMP M.

| change channel | A. readout, same distribution | B. readout under shift | C. policy input (BC) | D. judgment on a stale anchor | E. prediction target (world model) |
|---|---|---|---|---|---|
| raw ΔL (no encoder) | floor | **most robust** | not run | floor | floor |
| external encoder + ΔL (E0, VC-1) | **≥ CoMP M** (CALVIN parity, LIBERO above CI) | **above CoMP M** (transfer) | not run | not run | not run |
| CoMP M | **holds** (2–4× raw) | **negative** (≈ raw on transfer, far below on photometric shift) | **negative** (copycat) | **holds vs raw, loses to a fresh P at equal compute** | **readable, not more predictive** |

One-line explanation shared by C–E: M is a trace of the agent's own action (a_(t−1) R² 0.82–0.88 from the same
features, α 4-b) — useful where reading recent motion is the task (A, D vs raw), harmful or neutral where the answer
must come from the current state or the world (C, D vs fresh P, E).

## A. Readout, same distribution

Attentive probe → end-effector position Δ R² (mean of dims 0–2). Probe seeds 3, 95% CI; representation n = 1.

| channel | model | CALVIN clean | LIBERO in-suite | source |
|---|---|---:|---:|---|
| raw ΔL (F1) | — | 0.22 | 0.16 | `tables/refinement_floor/README.md` Results |
| random-init M (F3) | — | 0.25 | 0.33 | same |
| CoMP M | C0 | 0.47 ± 0.12 | 0.74 ± 0.03 | same, E0 section |
| CoMP M | C1-DN | 0.43 | 0.68 | same, §10 pilot |
| VC-1 + signed ΔL | external, n = 1 | 0.51 ± 0.04 | **0.79 ± 0.01** | same, E0 section |
| DINOv2 / SigLIP + signed ΔL | external | −0.19 / −0.26 | 0.71 / 0.66 | same, E0 section |

Holds with a probe trained under noise too (R2-3: CALVIN σ0.01 C1 M 0.44 vs raw 0.22). Bound: E0 — "ΔL needs
encoding; joint training is one way", not "our encoder is special" (VC-1 n = 1; DINOv2/SigLIP fall below raw on CALVIN).

## B. Readout under shift

Same probe; (s) = shift tests. Probe seeds 3; representation n = 1.

| channel | model | LIBERO transfer 6-dir | CALVIN shadow 0.6 | CALVIN shot noise 0.01 | source |
|---|---|---:|---:|---:|---|
| raw ΔL | — | −0.32 | −0.32 | 0.22 | `tables/refinement_floor/README.md` |
| CoMP M | C0 | −0.23 ± 0.10 | −2.59 | −2.57 | same |
| CoMP M | C1-DN | −0.32 | −2.9 | 0.21 | same, §10 pilot |
| VC-1 + signed ΔL | external | **+0.06 ± 0.04** | — | — | same, E0 section |

Pre-registered refinement verdict (§6): transfer and label efficiency not met → efficiency (refinement) claim dropped.
Photometric fragility is local to M; the deployed P is ≈ raw on shift (R2-1).

## C. Policy input (behaviour cloning)

LIBERO-object, frozen encoder, BC-Transformer, 500 rollouts per arm. **Seed 1 each.**

| arm | model | success | source |
|---|---|---:|---|
| current P only | C0 | 47.2% | `tables/e3/README.md` stage 1 |
| P(t−g), P(t) | C0 | 39.0% | same |
| current P + CoMP M | C0 | **3.4%** | same |
| same, M zeroed at rollout | C0 | 0.6% | same |
| ★ every-step P / anchor P only (20) / anchor P + M (20) | C1-DN | 0.81 / 0.93 / **0.61** | `docs/STATUS.md` α pilot row (10 tasks × 10 trials) |

Raw ΔL and E0 arms not run (E3 remaining cells deferred, then closed). Reading: adding M lowers success and the
policy depends on it (zeroing is worse) → copycat [inferred from videos + 4-b].

## D. Judgment on a stale anchor (α, and analysis 1)

α: frozen C1-DN, attentive readout of remaining steps to success on SC export `alpha_v1` (450 demos + 149 successful
recoveries), anchor age 1..20 as input, seeds 3 × 5 folds; seed ranges non-overlapping where "holds".
Source `paper_artifacts/alpha_value/README.md` Rounds 2–3.

| measure | anchor P + CoMP M | anchor P + raw ΔL | anchor P only | reference | verdict |
|---|---:|---:|---:|---:|---|
| remaining-step MAE (all frames, steps) | **15.70** | 16.33 | 16.42 | every-step P 16.65 | M > raw (J2 MAE); "keeps" ★ level only on this average |
| event registration, anchor before event (age 11–20) | **0.098** | −0.027 | −0.048 | every-step P 0.365 | M > raw (J1, J2) · about ¼ of a fresh P |
| same, DINOv2 anchor P (J4) | **0.067** | −0.006 | −0.032 | — | M > raw on an external P too |
| cost-matched (§4-b): P every 2 steps 4.60 GF vs anchor P + M 5.04 GF | 0.141 | — | **0.333** (period-2 P) | — | **FAIL** — fresh P wins at equal compute → α cost claim dropped |

**Second operating point (alpha §4-c, 10-25):** shrinking M to 192×2 (0.43 GF, below the break-even X = 0.562 GF) loses to periodic re-encoding of the same P at similar cost (registration 0.083 vs period-9 P 0.097; current M 0.141; raw 0.037) — pre-registered FAIL, ladder stopped. Wording (Vault 10-25, confirmed): M must cost below X = 0.562 GF to beat periodic re-encoding, and the M built below X does not retain registration (0.141 → 0.083), so no M size between the two operating points tested (4.58 and 0.43 GF) wins. "Capacity floor" holds only for this recipe (frozen C1-DN P, M-only loss, 10 ep — not the joint 60-ep recipe); capacity vs training not separated.

Analysis 1 (observer, `observer_fuse/README.md` v3/v4, C1-DN P, beyond-position motion share): CoMP M 0.279 ± 0.012
vs raw ΔL 0.245 ± 0.008 (object, Fuse seeds 3, non-overlapping); goal 0.196 vs 0.144 (seed 1); the Fuse module was
retired (10-10) — report as an observation.

## E. Prediction target (world model)

Same A v4 predictor and training, only input/target swapped; ridge probe trained on observed tokens, applied to
predicted tokens. **Seed 1.** Executed in SC, exported to `paper_artifacts/worldmodel/pilot/`.

| arm | pilot (1): object R² k19 | pilot (2) cross-fit probe in event windows |
|---|---:|---:|
| (a) keep P, predict CoMP M | 0.735 | **0.12** |
| (b) DINOv2, predict full | **0.746** | 0.26–0.27 |
| (c) same-encoder P, predict full | 0.734 | 0.26–0.27 |
| (d) keep P, predict raw ΔL | 0.665 | 0.26–0.27 |

Sources: pilot (1) `worldmodel/pilot/pilot_s1.md`; pilot (2) cross-fit values are from SC `docs/STATUS.md`
(10-22 14:00 entry) — not in the export `pilot2_events_s1.*`, whose primary probe is invalid (ceiling R² on observed
tokens −0.30 to +0.04). Reading (`docs/worldmodel_plan.md` decision log 10-22 (2)): (a) − (d) +0.070 equals the
token-readability gap (health probe 0.072) → M tokens are easier to read, not more predictable; no format support on
either ruler; track closed.

## F. World-model residual as an event signal (SC §3 · 2026-10-27)

Residual = observed CoMP M tokens − A v4 predicted M (self-caused change subtracted). Event vs paired undisturbed
windows, front camera, W 5, tf1 probe, AUC, **probe seeds 3** (seed range in brackets). Executed and judged in SC
(Vault (43) §3); quoted from the export `paper_artifacts/worldmodel/sc_event_read/event_read_C.md`, not recomputed.

| event · region | raw ΔL | CoMP M | residual | P_t |
|---|---:|---:|---:|---:|
| drop (pertC) · object — **primary** | 0.728 | **0.919** [0.914, 0.922] | 0.908 [0.907, 0.909] | 0.882 |
| drop (pertC) · background | 0.623 | 0.899 | 0.910 | 0.836 |
| push · object | 0.851 | 0.858 | 0.914 | **0.918** |
| push · background | 0.660 | 0.820 | 0.904 | **0.923** |
| arm push (block) · all | 0.686 | 0.859 | 0.871 | 0.567 |

Reading: subtracting the predicted self-caused change does not beat M itself on the primary cell (SC pre-fixed reading
"subtraction is the bottleneck", gap 0.005); it helps where M is weakest (push: 0.86 → 0.91) but only up to P_t, and
the residual is **not localized to the object** — background AUC equals object AUC for every event. Note that M and P are nearly as non-localized here (background ≈ object for every token channel; P_t is the most
object-specific), so this is a property of the probe setting, not of the residual alone. Closed loop (residual as a policy
input): SC arms that rely on the residual (dependence 0.4–0.7) show no success gain (JHp − J −0.030 [−0.100, +0.037], SC `docs/STATUS.md`
10-27 19:00). Consistent with the shared line above: M is mostly self-caused change; removing it leaves a
global "something happened" signal, not a task-region map. Whether the residual is useful as a label-free training-weight
signal marking event onsets is SC's open test (SC Vault instruction 2026-10-27), not a CoMP claim.

## Gaps to state in the caption

- seed 1: C, E, analysis 1 goal. Representation n = 1 everywhere.
- External encoder + ΔL measured only in A/B (E0).
- C0 / C1-DN split as above; plain control only on C0 (and see `tables/plain_convergence/README.md`: the plain run
  did not converge after warm-up).
