# α v2 — remaining-steps readout (docs/alpha_plan.md §3–4)

Data = SC export `alpha_v1` (demos 450 + successful recovery 149; 13 timed-out recoveries unlabeled → excluded) ·
frozen C1-DN · attentive readout (target = remaining steps / 100, Huber δ 0.1, q + task one-hot covariates) ·
5 folds pooled · seeds 1/2/3 · anchor age 1..20. Jobs 42458909–42458983 (2026-10-19).
Reproduce: `python scripts/eval/alpha_readout.py --summarize --target remaining`.

| arm | input (front + wrist) | pooled MAE, steps (age mean) [seed range] |
|---|---|---|
| star | P_t every step | 16.67 [16.62, 16.72] |
| comp | anchor P_a + CoMP M(a→t) | **15.87** [15.81, 15.96] |
| raw | anchor P_a + raw ΔL(a→t) | 16.58 [16.42, 16.82] |
| none | anchor P_a only | 16.95 [16.86, 17.01] |
| none_noq | anchor P_a only, no q | 17.18 [17.16, 17.21] |

- J2 (MAE half): comp < raw, seed ranges disjoint → PASS. Event-window half pending the J1 definition.
- J3: comp/star = 0.946–0.957 (seed max 0.957; episode-bootstrap 95% upper ≤ 0.973) ≤ 1.2 → PASS under both CI readings.
- J1 / J2 event window (rule fixed before opening, `docs/alpha_plan.md` §4, 2026-10-19): jump registration = (V̂(t_inj+10) − V̂(t_inj−1) + 11) / (regrasp_end − t_inj), both ends read at the same anchor age a, judged on a = 11..20 (anchor precedes the injection) · 149 labeled recovery episodes · episode mean → age mean · `scripts/eval/alpha_j1.py` → `j1_event_window.json`.

  | arm | a = 11..20 (judged) [seed range] | a = 1..10 (reference: anchor after injection) |
  |---|---|---|
  | star | 0.367 [0.366, 0.368] (age-independent) | 0.367 |
  | comp | **0.076** [0.067, 0.085] | 0.171 |
  | raw | −0.023 [−0.025, −0.022] | 0.108 |
  | none | −0.017 [−0.023, −0.009] | 0.093 |
  | none_noq | −0.116 [−0.118, −0.113] | 0.088 |

  - J1: comp > none, seed ranges disjoint → **PASS**. J2 event half: comp > raw, disjoint → **PASS** (with the MAE half, J2 PASS).
  - Size: with the anchor before the event, comp registers ~8% of the event's time cost vs ~37% for a fresh P every step (≈ 1/5 of star); raw ΔL and anchor-only register none (≤ 0). "Maintained" (J3) holds for pooled MAE, not for the event window.
- Age curve (seed mean, ages 1/5/10/15/20): comp 17.31/16.27/15.64/15.35/15.58 · none 18.27/17.22/16.46/16.49/17.37 · raw 17.28/16.95/16.42/16.17/16.57 · star 16.67 (age-independent).

## Caveats (dev, 2026-10-19)

- Frame-set check: ages are evaluated on t ≥ age, so age sets differ. Re-scored on a common set (t ≥ 20 for every age): star [16.73, 16.83] · comp [15.89, 16.07] · raw [16.58, 16.98] · none [17.01, 17.17] · none_noq [17.27, 17.33] — same ordering.
- comp < star is **not** claimed: anchor arms see two time points (anchor + current), star sees one frame, so the gap may be temporal context rather than M. The claim stops at "maintained" (J3).
- Mixed-age training costs accuracy at small ages: `none` at age 1 has nearly the star input yet 18.3–18.6 vs 16.7.
- 13 timed-out recoveries are unlabeled and excluded → event data is biased toward successful recoveries.
- Seed ranges reflect readout initialisation (and age sampling) only; one suite (libero_goal tasks 1/3/4/8 for events), one event type (grasp slip).
- Per-frame predictions (`value_*_preds.npz`, git-ignored, 93 MB) stay on the cluster at `paper_artifacts/alpha_value/`; J1 can be computed from them without rerunning.

## Round 2 — anchor age as readout input (Vault 10-19 decision, 2026-10-20)

Same data, folds, seeds and rules as round 1; the readout gets anchor age / 20 as an extra covariate (star = 0).
DINOv2 block: front-camera P = DINOv2-base (wrist stays C1-DN, M stays CoMP M), tokens zero-padded to 768 → compared only within the block.
Jobs 42471947–48, 42471981–42472113 (135 cells). Outputs `r2/`. Reproduce:
`python scripts/eval/alpha_readout.py --target remaining --summarize --age-input` · `python scripts/eval/alpha_j1.py paper_artifacts/alpha_value/r2`.

| arm | pooled MAE r2 [seed range] | (r1) | event window a = 11..20 (judged) | a = 1..10 (ref) | (r1 judged / ref) |
|---|---|---|---|---|---|
| star | 16.65 [16.64, 16.67] | 16.67 | 0.365 [0.361, 0.368] | 0.365 | 0.367 / 0.367 |
| comp | **15.70** [15.61, 15.81] | 15.87 | **0.098** [0.090, 0.109] | 0.184 | 0.076 / 0.171 |
| raw | 16.33 [16.26, 16.41] | 16.58 | −0.027 [−0.033, −0.021] | 0.100 | −0.023 / 0.108 |
| none | 16.42 [16.38, 16.49] | 16.95 | −0.048 [−0.055, −0.040] | 0.086 | −0.017 / 0.093 |
| none_noq | 16.60 [16.54, 16.68] | 17.18 | −0.119 [−0.120, −0.118] | 0.086 | −0.116 / 0.088 |
| dino_star | 17.02 [16.94, 17.14] | — | 0.355 [0.342, 0.365] | 0.355 | — |
| dino_comp | **16.53** [16.34, 16.71] | — | **0.067** [0.058, 0.072] | 0.191 | — |
| dino_raw | 17.23 [17.12, 17.39] | — | −0.006 [−0.009, −0.001] | 0.093 | — |
| dino_none | 17.08 [17.01, 17.16] | — | −0.032 [−0.034, −0.028] | 0.058 | — |

- J1 (comp > none, a = 11..20) PASS · J2 (comp < raw MAE, comp > raw event) PASS · J3 comp/star seed max 0.950, bootstrap upper ≤ 0.968 → PASS.
- J4 (DINOv2 block): J1 PASS · J2 PASS (MAE + event) · J3 dino_comp/dino_star seed max 0.975, bootstrap upper ≤ 1.001 → PASS.
- Cross-block (reported, not judged): C1-DN P beats DINOv2 P on this task (star 16.65 vs 17.02; anchor-only 16.42 vs 17.08).
- Age curve (seed mean, ages 1/5/10/15/20): comp 17.21/16.09/15.48/15.22/15.28 · none 16.98/16.67/16.35/16.18/16.17 · raw 17.02/16.65/16.24/16.01/16.02 · star 16.65 · dino_comp 17.46/16.78/16.41/16.24/16.28 · dino_none 17.39/17.20/17.04/16.96/16.98 · dino_star 17.02.

Caveats (dev, 2026-10-20):
- Age input fixed the small-age MAE penalty (none at age 1: 18.3 → 17.0 vs star 16.65) but **not** the reference-band event gap (anchor-only a = 1..10: 0.093 → 0.086 vs star 0.365). The 10-19 attribution of that gap to the ruler is not supported. Per-age registration (seed mean, ages 1..20) shows it is anchor staleness: anchor-only 0.33/0.32/0.27/0.24/0.18/0.08/−0.09 at ages 1–7 (star 0.36) — a fresh anchor registers the event like star, and the 1..10 band average hides that. comp 0.39/0.40/0.35/0.28/0.21/0.14/0.01 · raw ≈ anchor-only at every age (raw ΔL adds no event registration).
- All arms drift upward over a = 11..20 (anchor-only −0.14 → +0.09 without seeing the event), so absolute registration carries an age-dependent baseline; J1/J2 compare arms at the same age, where it cancels (comp − anchor-only ≈ +0.18 at a = 11, +0.09 at a = 20).
- Anchor-only arms now *improve* with age and reach below star at age 20 (16.17 vs 16.65): an old image + current q + age carries two-time-point information star lacks. Same reason comp < star is not claimed.
- comp's MAE margin over anchor-only narrowed (r1 1.08 → r2 0.72 steps) as age input helped the M-less arms most; still disjoint.

## Round 3 — cost-matched contrast (docs/alpha_plan.md §4-b, 2026-10-20) — **FAIL → α closes as a boundary result**

Question: at equal compute, does anchor P + M (period 20: 0.46 + 4.58 = 5.04 GF/step) beat simply refreshing P more often
(period-2 P = anchor P only, 4.60 GF/step)? Period-2 arm = anchor-P-only readout trained and evaluated on ages {1,2} only
(same capacity, age input, q, task covariates; jobs 42476703, 42476734–747; outputs `r3/`). Deployment average (fixed before results):
P+M = uniform over ages 1..20, period-2 = mean of ages 1, 2. Primary = event-window registration, secondary = pooled MAE.
Reproduce: `python scripts/eval/alpha_j1.py --deploy paper_artifacts/alpha_value/r2 paper_artifacts/alpha_value/r3` → `r3/j1_deploy.json`.

| arm | GF/step | deployment event registration [seed range] | deployment MAE [seed range] |
|---|---|---|---|
| anchor P + CoMP M (period 20) | 5.04 | 0.141 [0.136, 0.145] | **15.70** [15.61, 15.81] |
| **period-2 P** (dedicated readout) | 4.60 | **0.333** [0.326, 0.337] | 16.64 [16.60, 16.67] |
| (ref) r2 anchor-only at ages 1–2, mixed-age readout | 4.60 | 0.324 [0.315, 0.336] | 16.93 [16.91, 16.98] |
| (ref) star, P every step | 9.20 | 0.365 [0.361, 0.368] | 16.65 [16.64, 16.67] |

- Primary: P+M > period-2 P with disjoint seed ranges → **FAIL** (period-2 registers 2.4× more). Pre-registered disposition: the α cost claim is dropped; cost table, application cell (§5) and the alpha_v2 push-event grid are retired.
- Secondary (reported): P+M has lower MAE (15.70 vs 16.64, disjoint) — consistent with the two-time-point effect noted in Round 2, not with event detection.
- Expectation "period-2 wins the primary" was recorded as partially post-hoc (Round 2 per-age curves were seen before the rule was fixed); the rule itself was fixed before the dedicated readout was trained.
- What remains (alpha_plan §4-b): ① learned M > raw ΔL (J2 · J4) ② event detection depends on anchor freshness more than on M ③ anchor P + M retains average accuracy (pooled MAE) only.

## Round 4 — M-size ladder, stage 0 (2026-10-24, zero training · `docs/alpha_plan.md` §4-c)

Second operating point of the cost boundary: how small must M be for anchor P + M (period 20) to beat periodic re-encoding of the same P?

**Break-even X.** The r2 anchor-P-only arm (age-input readout) re-scored as a period-k P deployment average (ages 1..k uniform), `alpha_j1.py --deploy r2 r3` → `r3/j1_deploy.json` key `period_k_none`.
P+M (period 20) = 0.1405 [0.1362, 0.1450].

| period k | 1 | 2 | 4 | 6 | 7 | **8** | **9** | 10 | 12 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| registration (seed mean) | 0.331 | 0.324 | 0.288 | 0.235 | 0.189 | 0.1414 | 0.110 | 0.086 | 0.050 | 0.019 |

k* = 9 (first k at or below 0.1405) → **X = 9.20·(1/9 − 1/20) = 0.562 GF**. Borderline: k = 8 misses by 0.0009 with overlapping seed ranges (if k* = 8, X = 0.690).
Caveat: this readout was trained on mixed ages; the dedicated period-2 readout (r3) scored higher (0.333 vs 0.324), so dedicated period-k readouts would likely move k* up and X down [inferred].

**Candidate M GFLOPs** (batch 1; CPU FlopCounterMode + analytic attention 4·N²·d per block, N = 197, because CPU SDPA is not counted; reproduces §4-b: M 384×6 = 4.581, P = 9.197). `ladder/m_ladder_flops.{py,json}`.

| M (width × depth) | 384×6 (current) | 384×3 | 384×2 | 384×1 | 256×2 | 192×4 | 192×3 | **192×2** | **128×3** | **192×1** | **128×2** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GF | 4.58 | 2.31 | 1.55 | 0.80 | 0.73 | 0.84 | 0.63 | **0.43** | **0.31** | **0.23** | **0.21** |
| params (M) | 10.8 | 5.5 | 3.7 | 1.95 | 1.70 | 1.87 | 1.42 | 0.98 | 0.65 | 0.53 | 0.46 |

Bold = below X (0.562). None of the example rungs in §4-c (384×3, 192×6 / 384×2, 192×3) is below X; every width-384 M is above X (384×1 = 0.80), so a depth-only retrain (`m_depth`, option (b)) cannot reach X.
