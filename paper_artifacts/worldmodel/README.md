# World-model track — change-channel prediction (docs/worldmodel_plan.md v0)

Claim (primary, format): keeping the state channel P and predicting only the change channel M under action beats
predicting the full next state of a single latent. Encoder sub-claim only if arm (d) allows it.
Execution = SC repo (A v4 pipeline, caches, rulers are SC canonical); SC exports tables/json to `pilot/` (contract v2).

## Pilot (= SC instruction (31) + arm (d) · seed 1 · branch decision, not a paper judgment)

Predictor, training setup, data and loss = A v4 fixed; only input tokens and target change.
Ruler = ridge probes trained on observed tokens, applied to k-step predicted tokens → hand (EE) / object position R², k ∈ {1, 5, 10, 20}.

| arm | input | target | hand R² k1/5/10/19 | **object R² k19** (primary) | object health (observed) k19 | object context-only k19 |
|---|---|---|---|---|---|---|
| (a) current A v4 | C1-DN anchor P + cumulative M + command | step-coordinate M | 0.967 / 0.962 / 0.948 / 0.920 | **0.735** | 0.774 | 0.709 |
| (b) DINOv2 full | DINOv2-base patches | next DINOv2 tokens | 0.950 / 0.950 / 0.943 / 0.926 | **0.746** | 0.782 | 0.692 |
| (c) C1-DN P full (format control) | C1-DN P only | next P tokens | 0.949 / 0.943 / 0.933 / 0.910 | **0.734** | 0.766 | 0.674 |
| (d) raw ΔL (encoder control) | C1-DN anchor P + raw ΔL tokens | next raw ΔL tokens | 0.955 / 0.944 / 0.918 / 0.858 | **0.665** | 0.702 | 0.648 |

Object = body with the longest in-episode path (check the per-task body distribution before reading).
Reading (fixed before results, primary = object R² at k 19): (a) ≯ (c) → no format support (back to Vault) · (a) > (c) ∧ (a) ≈ (d) → format only ·
(a) > (c) ∧ (a) > (d) → format + encoder sub-claim → register the main grid · fourth-loss rule (plan §1).

Token entry points (CoMP-owned, `scripts/eval/token_provider.py:TokenProvider`): `p_tokens` · `m_tokens` ·
`raw_tokens` (raw ΔL 16×16 patch pixels, 256 → zero-padded to D = α raw-arm / analysis-1 rule; verified identical, max|diff| 0.0).

### Result (2026-10-21 · seed 1 · SC jobs 42519191–193 + probe chain · 400 validation rollouts, n = 2420 at k 19 · `pilot/pilot_s1.{json,md}`)

- Object label check: the "longest-path body" is body 0 in 382/400 episodes; in sampled rollouts body 0 travels 0.3–0.7 m per episode while other bodies stay near 0 and its distance to the EE drops to ~0.05 m while carried → it is the manipulated object, not a robot-attached body.
- Primary (object R² at k 19): (a) 0.7348 · (b) 0.7456 · (c) 0.7340 · (d) 0.6650. **(a) − (c) = +0.0008** — the pre-registered rule reads "(a) > (c)" without a margin, so it passes literally while the gap is far below any plausible seed-to-seed spread (unmeasured; one seed). Reported as a rule problem, not read as format support.
- (a) − (d) = +0.070 (learned-M target clearly above raw-ΔL target) · (b) DINOv2 full is the best arm (+0.011 over (a)).
- What prediction adds beyond its own context (pred − context-only, object k 19): (a) +0.026 · (b) +0.053 · (c) +0.060 · (d) +0.017; hand: (a) +0.044 · (b) +0.103 · (c) +0.091 · (d) +0.047.
- Pre-result expectation "hand: (a), (d) beat (b), (c) easily" did not hold (hand k 19: b 0.926 ≥ a 0.920 > c 0.910 > d 0.858).
