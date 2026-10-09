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

## Sparse M (10-10, consolidated instruction ③) — `sparse_ratio.json` · `sparse_b1_libero_object.json` · `sparse_c1dn_s2.json` · `flops_sparse.json`

Frozen M encoder run only on patches with mean|ΔL| > τ = 1/255 (16×16 patches on 224, ΔL = model's `compute_m_channel`),
positions kept (APE added before dropping), learned `mask_token_m` re-inserted elsewhere → 196 tokens
(`probe_action_libero.encode_m_sparse`; all-visible sparse == dense bit-exact). Empty-chunk rule (fixed before results):
keep the max-|ΔL| patch (it fired only 4 times, all 1-frame chunks). C1-DN, seed 2 Fuse (`v3_c1dn_s2`), full-token
standardization stats. Pre-fixed rule: **maintained ⇔ 95% demo-bootstrap CI (2000) of the paired diff (full − sparse) contains 0**.
M was pre-trained with masked M-recon at a fixed 50% random mask (98 visible) plus full 196-token passes — the
~12–19% motion-selected visible sets here are outside that range.

(a) non-zero patches / 196, all kept demos (495 per suite)

| suite | chunk | mean | q10 / q50 / q90 (patches) | zero chunks |
|---|---|---|---|---|
| object | 5-frame grid | 0.148 | 22 / 30 / 34 | 0 |
| object | partial 1 / 2 / 3 / 4 frames | 0.124 / 0.133 / 0.139 / 0.144 | q50 25 / 27 / 28 / 29 | 3 (1-frame) |
| object | 20-frame same-probe pair | 0.191 | 31 / 37 / 45 | 0 |
| goal | 5-frame grid | 0.170 | 21 / 32 / 47 | 0 |
| goal | partial 1 / 2 / 3 / 4 frames | 0.145 / 0.154 / 0.160 / 0.165 | q50 28 / 29 / 31 / 32 | 1 (1-frame) |
| goal | 20-frame same-probe pair | 0.214 | 29 / 40 / 58 | 0 |

(b) readout with sparse vs full M tokens (libero_object). Full-token numbers reproduce the earlier runs exactly
(same-probe M 0.785 / 0.448; v3pos P+M 5×4 0.7873).

| readout | full | sparse | diff (95% CI) | maintained |
|---|---|---|---|---|
| b1 same-probe M motion (pos-3 R², probe fit on full) | 0.785 | −0.535 | +1.320 [+1.289, +1.353] | no |
| b1 same-probe M identity (acc, probe fit on full) | 0.448 | 0.052 | +0.396 [+0.353, +0.441] | no |
| b1 secondary: probe refit on sparse — motion pos-3 R² / agg R² | 0.785 / 0.696 | 0.821 / 0.694 | −0.036 [−0.045, −0.028] / +0.002 [−0.008, +0.012] | — |
| b1 secondary: probe refit on sparse — identity | 0.448 | 0.590 | −0.142 [−0.167, −0.118] | — |
| **b2 Fuse P+M 5×4, z + anchor pos (R²; beyond-pos share)** | 0.787 (+0.275) | 0.319 (−0.194) | +0.469 [+0.428, +0.510] | **no** |
| b2 P+M 10×2 / 20×1 (same) | 0.786 / 0.784 | 0.508 / 0.540 | +0.278 / +0.244 (CI excl. 0) | no |

Readers trained on full tokens break on sparse tokens (incl. the frozen Fuse). A probe refit on sparse tokens recovers motion
(so the motion information survives) but gains identity (0.45 → 0.59): the visible/mask pattern itself encodes where things
move, i.e. position/appearance leaks into M [inferred]. A Fuse trained on sparse M was not tested.

(c) forward GFLOPs (batch 1). M encoder vs visible tokens: 1 → 0.08 · 10 → 0.27 · 50 → 1.15 · 98 → 2.23 · 196 → 4.58.

| | P+P | P+M dense | P+M sparse (object / goal) | saving vs P+P, dense → sparse |
|---|---|---|---|---|
| per update (v3: 5-frame chunk + Fuse 4 chunks + P/4) | 10.04 | 8.27 | 4.38 / 4.47 | −18% → −56% / −55% |
| per step (v4: chunk ending at age k, len k mod 5 or 5, + Fuse ⌈k/5⌉ chunks, k=1..20, + P/20) | 10.04 | 6.16 | 2.22 / 2.31 | −39% → −78% / −77% |

Expected sparse M per chunk ≈ 0.58–0.69 GFLOPs (object), 0.67–0.78 (goal); with sparse M the remaining cost is mostly
anchor P/4 (2.30) + Fuse (0.84–1.39) per update. FLOPs only — not wall-clock (variable token counts need grouping);
and the savings apply only if a reader trained on sparse M keeps the readout, which (b) does not yet show.

## v4 — per-step z, anchor-age curve k = 1..20 (10-10, `v4_*.json`, seed 2, position-controlled)

`scripts/eval/observer_fuse_v4.py`: complete 5-frame chunks + one partial last chunk (1–4 frames, encoded on the fly);
training k ~ U{1..20}; per k a fresh probe on z_PP(train) + anchor EE position RFF; target = EE change anchor→t.
k = 20 reproduces v3 (goal 0.194 vs 0.196, object 0.274 vs 0.275). Single Fuse seed → descriptive curve, not a verdict.

Beyond-position share (R²(z+pos) − R²(pos)), P+M = cheap path:

| suite | k | pos only | CoMP P+M | CoMP P+P | raw ΔL P+M | P+P-only ref |
|---|---|---|---|---|---|---|
| goal | 1 | 0.413 | **0.271** | 0.287 | 0.149 | 0.243 |
| goal | 5 | 0.492 | **0.319** | 0.328 | 0.149 | 0.222 |
| goal | 10 | 0.571 | **0.245** | 0.244 | 0.123 | 0.180 |
| goal | 20 | 0.598 | **0.194** | 0.195 | 0.089 | 0.165 |
| object | 1 | 0.385 | **0.352** | 0.395 | 0.334 | 0.385 |
| object | 5 | 0.454 | **0.380** | 0.399 | 0.330 | 0.365 |
| object | 10 | 0.518 | **0.326** | 0.317 | 0.295 | 0.308 |
| object | 20 | 0.513 | **0.274** | 0.263 | 0.236 | 0.271 |

CoMP P+M − raw ΔL P+M over all k = 1..20: goal +0.095…+0.192 (mean +0.133), object +0.014…+0.049 (mean +0.032); raw never higher.
identity at k=20: goal CoMP P+M 0.946 vs raw 0.828 (P+P 0.950 / 0.914); object 1.0 everywhere.
Training: CoMP 36k/28.5k steps (95/76 min), raw 13k/32.5k (24/60 min), P+P-only refs hit the 60k cap (unconverged, 138 min) — MIG-3g.
Hand-offs (git-ignored, cluster path): `ckpt/goal_c1dn_v4_handoff.pt`, `ckpt/object_c1dn_v4_handoff.pt`; entry `observer_fuse_v4.py:ObserverV4.z_seq(frames)`.
