# Fig 6 — Motion-Routing Source Ablation (V from P vs V from M)

**Paper position**: §5.1 Architecture ablation (★ main paper)
**Status**: 🟡 probing 완료 (2026-05-19), BC SR rollout 로컬 진행 후 통합

## 가설

v15의 motion-routing 핵심 design choice는 `softmax(Q_M K_M^T) @ V_P` — **V from P** (M의 self-similarity graph로 P value re-route). 이 vs **V from M** (standard cross-attention, Q from P, K/V from M) 비교가 paper §5.1 main ablation.

- **v15 main** (V from P, paper main 모델): 343 GPU·h × 50ep, ckpt `two_stream_v15/20260511_045319/latest.pt`
- **v15-vfromm** (V from M, ablation 모델): 420 GPU·h × 50ep, ckpt `two_stream_v15_vfromm/20260515_153323/checkpoint_epoch{04..50}.pt`
- 다른 hyperparameter 모두 동일 — 단일 변인 = motion-routing direction

## 1. Probing fair head-to-head (3 dataset × p_t_p_tk mode)

[`vfromm_probing_sweep.csv`](vfromm_probing_sweep.csv) (32 cells).

### EgoDex (within-domain, gap=10)

| ckpt | R² | vs v15 main +0.405 |
|------|-----|---------------------|
| **v15 main ep50** ★ | **+0.405** | (reference) |
| v15-vfromm ep28 | +0.413 | +0.008 |
| v15-vfromm ep32 (fair pair w/ main ep32 +0.390) | +0.380 | −0.010 |
| v15-vfromm ep36 | +0.402 | −0.003 |
| v15-vfromm ep40 | +0.405 | 0.000 |
| v15-vfromm ep44 ★ | **+0.415** | +0.010 |

→ EgoDex within-domain: **두 routing 사실상 동률** (Δ ±0.01, probing 변동성 범위).

### LIBERO spatial (cross-embodiment robot arm, p_t_p_tk × 4 gap)

| ckpt | gap=1 | gap=13 | gap=20 ★ | gap=40 |
|------|-------|--------|----------|--------|
| v15 main ep32 (cluster_sessions 34367612) | +0.401 | +0.576 | **+0.584** | +0.379 |
| v15-vfromm ep28 | +0.437 | +0.603 | **+0.600** | +0.364 |
| v15-vfromm ep32 (fair pair) | +0.383 | +0.563 | +0.565 | +0.329 |
| v15-vfromm ep36 | +0.378 | +0.564 | +0.569 | +0.337 |
| v15-vfromm ep40 | +0.368 | +0.549 | +0.569 | +0.338 |
| v15-vfromm ep44 | +0.387 | +0.564 | +0.576 | +0.331 |

→ LIBERO cross-embodiment: vfromm ep32 vs v15 main ep32 = **−0.019 @ gap=20** (probing 변동성 범위). vfromm ep28만 가장 좋음 (+0.600 > main +0.584, +0.016 우위) — vfromm 학습 중반이 LIBERO에 적합.

### DROID (cross-domain real robot arm, p_t_p_tk gap=15)

| ckpt | R² |
|------|-----|
| v15 main ep32 | −0.006 |
| v15-vfromm ep28 | −0.014 |
| v15-vfromm ep32 (fair) | −0.016 |
| v15-vfromm ep36 | −0.005 |
| v15-vfromm ep40 ★ | **−0.003** |
| v15-vfromm ep44 | −0.009 |

→ DROID: 모두 noise level (|R²| < 0.02). routing 차이 무의미.

## 2. 종합 결론 (paper §5.1 C1 ablation 핵심)

**Δ(vfromm ep32 − main ep32) per dataset** (fair pair):

| Dataset | Δ R² (vfromm − main) |
|---------|----------------------|
| EgoDex within-domain | **−0.010** |
| LIBERO cross-embodiment (spatial gap=20) | **−0.019** |
| DROID cross-domain (gap=15) | **−0.010** (noise) |

→ 세 도메인 모두 **probing 변동성 범위 내 (±0.02)**. "V-from-P (ours) 우위" 강한 claim 불가.

**5/19 사용자 결정 framing**: "**Motion-routing이 핵심이지 어느 stream의 value를 쓰느냐는 부차적**" — robustness evidence로 활용. paper §5.1에서 "routing direction is not critical to representation quality, validating motion-routing's architectural simplicity."

**부가 관찰** (vfromm 학습 trajectory):
- vfromm는 ep28이 cross-embodiment(LIBERO) champion (+0.600), 학습 진행될수록 약간 하락
- EgoDex within-domain은 ep44가 champion (+0.415) → late-stage divergence가 generalization 손상 시그널
- v15 main은 ep50까지 안정 학습 가능, vfromm는 ep45-50 train loss 0.37→0.77 + eval 0.43→0.92 폭증 → **stability 측면에서 v15 main (V-from-P) 우위**

## 3. BC fine-tuning (paper §5.1 C3, 진행 중)

BC-T 18잡 (ptptk × 9 + mp × 9, ep32 fair pair):
- vfromm mp 9잡 ✅ COMPLETED 22h02m~22h18m (2026-05-19)
- vfromm ptptk 9잡 🔄 RUNNING 26h+ (2026-05-19 19:28 기준)

BC ckpt → 로컬 워크스테이션 전송 후 LIBERO rollout (각 ckpt × 500 episodes) → SR per (encoder, suite, seed) → [`../fig3_bc_main/summary.csv`](../fig3_bc_main/summary.csv)에 추가.

**Hypothesis**: probing에서 fair pair −0.01~−0.02였으니 BC SR도 비슷한 작은 격차 예상.

## Source

- Probing CSV: [`vfromm_probing_sweep.csv`](vfromm_probing_sweep.csv) (32 cells)
- Cluster jobs:
  - vfromm 5 ckpt EgoDex p_t_p_tk: 34579946~951 (5/18, 14m25s each)
  - vfromm 5 ckpt DROID p_t_p_tk gap=15: 34592055~060 (5/18, ~2분 each)
  - vfromm 5 ckpt LIBERO spatial 4 gap: 34592050~054 (5/18, 17m each)
  - vfromm BC-T 18잡: 34595206~227 (5/18 제출, 5/19 진행)
- v15 main reference: cluster_sessions §2026-05-12 (34367238/34367577/34367612) + [`../probing/v15_egodex_summary.csv`](../probing/v15_egodex_summary.csv)
- v15-vfromm 학습 잡: cluster_sessions §2026-05-15 (34464714 sanity, 34464715 본 학습 52.55h × 8 GPU = 420 GPU·h)

## TODO

- [ ] **vfromm BC-T 18잡 완료 → 로컬 rollout → SR 통합** (`fig3_bc_main/summary.csv`에 `two-stream-v15-vfromm-{ptptk,mp}` row 추가)
- [ ] **fig6 schematic** — v15 vs vfromm architecture 비교 diagram (paper repo TikZ)
- [ ] **fig6 bar chart** — probing Δ + BC SR Δ 3-row × 3-dataset
- [ ] **LaTeX 표** — `tab3_ablation`에 통합
