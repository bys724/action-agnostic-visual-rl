# World-model track — change-channel prediction (docs/worldmodel_plan.md v0)

Claim (primary, format): keeping the state channel P and predicting only the change channel M under action beats
predicting the full next state of a single latent. Encoder sub-claim only if arm (d) allows it.
Execution = SC repo (A v4 pipeline, caches, rulers are SC canonical); SC exports tables/json to `pilot/` (contract v2).

## Pilot (= SC instruction (31) + arm (d) · seed 1 · branch decision, not a paper judgment)

Predictor, training setup, data and loss = A v4 fixed; only input tokens and target change.
Ruler = ridge probes trained on observed tokens, applied to k-step predicted tokens → hand (EE) / object position R², k ∈ {1, 5, 10, 20}.

| arm | input | target | hand R² k1/5/10/20 | **object R² k20** (primary) | probe health (observed tokens) |
|---|---|---|---|---|---|
| (a) current A v4 | C1-DN anchor P + cumulative M + command | step-coordinate M | — | — | — |
| (b) DINOv2 full | DINOv2-base patches | next DINOv2 tokens | — | — | — |
| (c) C1-DN P full (format control) | C1-DN P only | next P tokens | — | — | — |
| (d) raw ΔL (encoder control) | C1-DN anchor P + raw ΔL tokens | next raw ΔL tokens | — | — | — |

Reading (fixed before results): (a) ≯ (c) → no format support (back to Vault) · (a) > (c) ∧ (a) ≈ (d) → format only ·
(a) > (c) ∧ (a) > (d) → format + encoder sub-claim → register the main grid · fourth-loss rule (plan §1).

Token entry points (CoMP-owned, `scripts/eval/token_provider.py:TokenProvider`): `p_tokens` · `m_tokens` ·
`raw_tokens` (raw ΔL 16×16 patch pixels, 256 → zero-padded to D = α raw-arm / analysis-1 rule; verified identical, max|diff| 0.0).
