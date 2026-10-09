# Observer Fuse — 1s anchor maintenance (LIBERO-object)

Tool: `scripts/eval/observer_fuse.py` · frozen encoder · Fuse trained with mixed anchor ages (5–20 frames) ·
probe (attentive, same-probe class) fit on z_PP(train) and frozen · eval at anchor age 20 frames (1s).
motion = R² of EE pose change over the last 1s (same-probe gap20 target, no position control) ·
identity = 10-way object/task acc (ceiling: anchor P alone already ≈1.0 — reported, not discriminative).
Single Fuse seed (42), single probe seed. Only `v3_*`, `v3pos_*` JSONs are tracked in git; v1 (`c1dn_s42` …), `diag_*`, `smoke*` stay on the cluster copy of this folder (superseded / pipeline checks).

| arm (TAG) | anchor P | M source | motion P+P | motion P+M 5×4 (ratio) | 10×2 | 20×1 | anchor only | identity P+P / P+M 5×4 / anchor |
|---|---|---|---|---|---|---|---|---|
| `c1dn_s42` | C1-DN | C1-DN M | 0.713 | 0.664 (93.2%) | 0.644 | 0.621 | 0.116 | 1.000 / 1.000 / 0.999 |
| `c0_s42` | C0 | C0 M | 0.720 | 0.671 (93.3%) | 0.670 | 0.660 | 0.262 | 0.998 / 0.998 / 0.998 |
| `c1dn_rawdl_s42` | C1-DN | raw ΔL patches | 0.565 | 0.553 (97.8%) | 0.545 | 0.556 | 0.399 | 1.000 / 1.000 / 1.000 |
| `dino_rawdl_s42` | DINOv2-base | raw ΔL patches | 0.631 | 0.595 (94.3%) | 0.596 | 0.599 | 0.346 | 1.000 / 1.000 / 1.000 |

Reference: probe directly on frozen [P(t−20), P(t)] tokens — motion 0.393 (C1-DN) / 0.232 (C0), identity 1.0.

Caveats: (1) the P+P denominator is each arm's own Fuse, so ratios are not comparable across arms
(raw arm's z_PP is lower; its eval raw-ΔL recon error 2.47 is a standardization artifact — near-zero-std
pixels in train stats — not a training failure; train loss was normal).
(2) No position control: anchor P alone reaches 0.40 in the raw arm and direct tokens 0.39.

**Seed diagnosis (10-09, `diag_*.json`, split seed fixed 42, Fuse/probe seed varied, 4 runs/arm):**
z_PP motion C1-DN 0.716±0.003 vs raw 0.588±0.048 (real gap). C1-DN P+M 5×4 ratio 0.911±0.067 (seed 2 = 0.811 → single-seed PASS above is NOT robust).
With M-recon weight 0, z_PP drops to 0.510 (C1-DN) / 0.542 (raw): the M-side loss leaks motion into z_PP, so per-arm P+P is not a P-only ceiling.
Probe on fixed inputs is stable (probe-seed spread ≤0.005 direct, ≤0.018 on z). Fuse training is not bit-reproducible on GPU; EMA teacher was deep-copied in train mode (dropout on).

## v3 (10-09, final protocol) — teacher eval-mode fix · split seed 42 · convergence stop · common M-free denominator · 3 seeds

Denominator = Fuse trained on P+P only (`--pp-only`, no M input/recon), per P encoder. Ratio = mean P+M 5×4 / mean ref.
Pre-registered rule (cluster_sessions 10-09 v2): maintain if ratio ≥ 0.90.

| arm | P | M source | P+M 5×4 motion | ratio (min seed) | own P+P | anchor only | stop step | train min (HW) |
|---|---|---|---|---|---|---|---|---|
| `v3_c1dn` | C1-DN | C1-DN M | 0.780±0.006 | 1.16 (1.16) | 0.770 | 0.35±0.10 | 28–38k | 47–64 (MIG-3g) |
| `v3_c0` | C0 | C0 M | 0.774±0.007 | 1.19 (1.17) | 0.757 | 0.45±0.03 | 30–33k | 50–56 (MIG-3g) |
| `v3_c1dnraw` | C1-DN | raw ΔL | 0.717±0.012 | 1.07 (1.06) | 0.747 | 0.57±0.03 | 20–31k | 34–52 (MIG-3g) |
| `v3_dinoraw` | DINOv2-base | raw ΔL | 0.655±0.019 | 0.99 (0.97) | 0.673 | 0.28±0.02 | 16–20k | 14–18 (H100) |
| ref `v3_ref_c1dn` | C1-DN | — (P+P only) | — | — | 0.670±0.021 | — | 51.5k–60k ⚠ 2/3 hit cap | 85–99 (MIG-3g) |
| ref `v3_ref_c0` | C0 | — | — | — | 0.652±0.018 | — | 37–60k ⚠ 1/3 hit cap | 61–100 (MIG-3g) |
| ref `v3_ref_dino` | DINOv2-base | — | — | — | 0.660±0.021 | — | 33–46k | 28–39 (H100) |

identity = 1.000 in every condition (ceiling; anchor P alone suffices) — listed per user request.
Caveats: no position control (anchor-only reaches 0.57 in the raw arm) · CoMP refs partly unconverged at the 60k cap
(denominator may be understated) · DINO ran on full H100, CoMP arms on MIG-3g (minutes not directly comparable).

## v3 position-controlled re-probe (10-10, `v3pos_*.json`, same Fuse ckpts, no retraining)

Covariate = anchor (t−20) EE position → z-score → RFF 128, concatenated after pooling (same-probe `concat`).
beyond = R²(z + pos) − R²(pos only); pos only = 0.513±0.014 (same for all arms, varies with probe seed). Consistency: no-covariate scores equal v3 to 4 decimals (21/21).

| arm | beyond P+P | beyond P+M 5×4 | 10×2 | 20×1 | anchor only | ratio P+M 5×4 ÷ ref P+P |
|---|---|---|---|---|---|---|
| C1-DN + CoMP M | 0.270±0.013 | **0.279±0.012** | 0.278 | 0.271 | −0.09 | 1.05 |
| C0 + CoMP M | 0.261±0.014 | 0.278±0.010 | 0.280 | 0.269 | −0.04 | 1.08 |
| C1-DN + raw ΔL | 0.264±0.016 | **0.245±0.008** | 0.246 | 0.243 | 0.10 | 0.92 |
| DINOv2 + raw ΔL | 0.258±0.015 | 0.228±0.009 | 0.226 | 0.223 | 0.01 | 0.90 |
| ref P+P only (C1-DN / C0 / DINO) | 0.265 / 0.257 / 0.253 | (out-of-distribution input) | | | | — |

Pre-registered (10-10): CoMP M vs raw ΔL on the same C1-DN P, P+M 5×4 beyond-position, seed ranges 0.270–0.293 vs 0.236–0.252 → **advantage holds**.
The M-recon boost of the P+P path seen without position control (A′) mostly vanishes here (0.270 vs 0.265).

## v3 on libero_goal (10-10, seed 2 only, position-controlled) + transfer reference

Same v3 protocol (convergence stop, P+P-only reference, anchor-position RFF). pos only = 0.598. Seed 1 → observation, not a verdict.

| arm | beyond P+P | beyond P+M 5×4 | anchor only | identity P+P | stop / min (MIG-3g) |
|---|---|---|---|---|---|
| C1-DN + CoMP M (`goal_c1dn_s2`, **SC layer-2 hand-off for plumbing test**) | 0.197 | **0.196** | −0.005 | 0.952 | 26k / 44 |
| C1-DN + raw ΔL (`goal_c1dnraw_s2`) | 0.166 | **0.144** | 0.040 | 0.938 | 21k / 36 |
| P+P only ref (`goal_ref_c1dn_s2`) | 0.163 | — | — | 0.936 | 35k / 58 |

Ratio P+M 5×4 ÷ ref P+P: CoMP 1.20 · raw ΔL 0.88 (object: 1.05 / 0.92). Direction matches object; gap CoMP − raw +0.05 (object +0.03).

Transfer (no training; Fuse + its own source-suite standardization stats; probe fit on target z_PP) — reference only:

| | beyond P+P | beyond P+M 5×4 | P recon (std. MSE) | native beyond P+P |
|---|---|---|---|---|
| object Fuse → goal | 0.087 | −0.156 | 130.7 | 0.197 |
| goal Fuse → object | 0.202 | −0.945 | 36.2 | 0.272 |

The Fuse is suite-specific: P+M collapses off-suite. The exploding P recon error suggests part of this is the
per-(patch,dim) standardization (near-zero-std dims in the source suite) rather than the Fuse itself [inferred].
