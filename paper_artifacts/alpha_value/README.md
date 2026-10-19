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
