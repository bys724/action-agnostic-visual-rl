# Paper Experiments Plan — v15 CoRL 2026 main table 완성

> **Status**: 2026-05-14 plan 작성, 2026-05-18 갱신 (CALVIN + CortexBench + view-sensitivity 추가).
> **Scope**: Cluster + 로컬 워크스테이션 (rollout) 실험 전부.
> **Source decision**: Vault `Projects/Action-Agnostic Paper/3. Experiments.md` + `7. Outline.md` + paper §5 ablation plan.

## 0. Critical path 요약

```
C1 (V from M 학습, 50ep) ────┬── C2 (probing)
                              └── C3 (BC-T 18잡 + 로컬 rollout)

C4, C5, C6 (v15 ep50 추가 + recon quality) — 의존성 X, 즉시 병렬 가능
C7 (VideoMAE-ours P_t+P_tk) — 의존성 X, sanity 수준 (1 GPU·h)
C8, C9 — 시간 되면

[★ 신규 2026-05-18 — evaluation 확장]
C10 (CALVIN action probing, 5 enc × 4 ABCD × static view) — DROID 보강 (대체 X)
C11 (CortexBench manipulation subset, Adroit + Meta-World) — vision encoder generic capability
C12 (LIBERO view-sensitivity, eye_in_hand × 5 enc × 3 suite × 4 gap) — §5 ¶6 sub-analysis
```

**합계 (필수 C1-C6, C10-C12)**: ~147-149 GPU·h ≈ 6-7일
**선택 (C7-C9)**: +22 GPU·h ≈ 1일

---

## 1. C1 — v15 cross-attention variant 학습 (★ paper §5.1 main ablation)

### Motivation

v15의 motion-routing 핵심 design choice는 `softmax(Q_M K_M^T) @ V_P` — **V from P** (M의 self-similarity graph로 P value re-route). 이 vs **V from M** (standard cross-attention, Q from P, K/V from M) 비교가 paper §5.1 main ablation. 사용자 5/14 결정.

**v15 본 학습 (34288968)** = V from P. **새 variant (C1)** = V from M, 다른 모든 hyperparameter 동일.

### Architecture diff (구현 stub)

```python
# 기존 v15 motion_routing (V from P, in p_motion_decoder)
# Q, K = M_encoder unmasked output → softmax(Q K^T)
# V = teacher_P_encoder output → attn @ V_P → p_state update

# C1 variant motion_routing (V from M, standard cross-attention)
# Q = teacher_P_encoder output (or p_state) → from P space
# K, V = M_encoder unmasked output → from M space
# attn @ V_M → output in M space → W_V_bridge (M → P space) → residual to p_state

# TODO (dev session에서 구현):
# 1. src/models/two_stream_v15.py — motion_routing variant 추가
#    - 기존 routing: Q_M_proj, K_M_proj, V_P_proj
#    - C1 routing: Q_P_proj, K_M_proj, V_M_proj, W_V_bridge (M→P)
# 2. CLI flag --motion-routing-variant {v_from_p, v_from_m}
# 3. src/training/pretrain.py — 동일 학습 loop, variant만 다름
```

### Critical guards

- **Parameter count 동일** 유지 (motion_routing projection 차원 같게)
- **다른 모든 hyperparameter v15 본 학습과 동일**:
  - 50ep, EgoDex part1-5, batch 32/GPU global 256, num_workers=8
  - λ_pred=λ_m_jepa=λ_compose=1.0 with warmup 10ep 0.01→1.0
  - composition_mode=linear_residual
  - EMA 0.999→0.9999, mask_p=0.75, mask_m_jepa=0.5, max_gap=30 sample_center=15
- **EMA teacher P/M encoder 동일하게 사용** — V source가 student vs teacher 비교 아님, 두 variant 다 predictor-only (V source는 teacher)

### sbatch 예시

```bash
# scripts/cluster/pretrain.sbatch에 추가
MODEL=two-stream-v15-vfromm  # 또는 ENV 변수 MOTION_ROUTING_VARIANT=v_from_m
PARTITION=AIP_long
NODES=2
GPUS=4
TIME=10-00:00:00
```

### 비용

- ~50 GPU·h (v15 본 학습 343 GPU·h / 50ep와 비교 시 동일 시간 예상)
- AIP_long 2노드 × 4 H100 × 42.91h ≈ 343 GPU·h (v15 ep50 reference)
- 학습 ~1.5-2일

### 결과 ckpt

- `/proj/external_group/mrg/checkpoints/two_stream_v15_vfromm/<timestamp>/checkpoint_epoch_*.pt`
- ep4/8/12/16/20/24/28/32/36/40/44/48 + latest.pt(=ep50) 보존

---

## 2. C2 — C1 ckpt action probing (probing fair pair)

### 목적

C1 (V from M) ckpt를 v15 본 학습 (V from P)과 같은 mode들로 probing → fair head-to-head 비교.

### 측정 항목

각 ckpt (ep32, ep50) × 3 dataset:

| Dataset | Suite | Probing modes |
|---|---|---|
| EgoDex | within-domain | `patch_mean_p_enc`, `patch_mean_p_enc_tk`, `patch_mean_m_enc`, **`patch_mean_concat_p_t_p_tk`** (champion), `patch_mean_concat_enc_only`, `patch_mean_concat_p_tk_m`, `patch_mean_concat_all` (A+B+D') |
| DROID | cross-domain | 같은 7 mode × gap=15 (또는 4 gap) |
| LIBERO | spatial | `p_t_p_tk` × gap=1/13/20/40 |

### 비용

- 7 mode × 1 ckpt × ~13min = ~1.5 GPU·h (EgoDex)
- DROID 7 mode × ~1min = trivial
- LIBERO spatial 4 gap × ~17min = ~1 GPU·h
- **총 ~3-5 GPU·h**

### 의존성

C1 학습 완료 (또는 ep32+ ckpt 보존 시점).

### 결과 위치

`paper_artifacts/probing/v15_vfromm_*.csv`, `paper_artifacts/libero_action_probing/two-stream-v15-vfromm_libero_spatial_<timestamp>/`

---

## 3. C3 — C1 ckpt BC-T (paper §5.1 BC ablation)

### 목적

v15 (V from P) vs C1 (V from M) BC-T 학습 → closed-loop manipulation 비교.

### 매트릭스

```
Adapter: two-stream-v15-vfromm-ptptk + two-stream-v15-vfromm-mp (2)
Suite:   libero_spatial, libero_object, libero_goal (3)
Seed:    0, 1, 2 (3)
= 18 BC-T jobs
```

신규 adapter 작성 필요 (dev session):
- `src/encoders/adapters/two_stream_v15_vfromm_pt_ptk.py` — v15-ptptk와 동일 흐름, encoder 가 vfromm ckpt
- `src/encoders/adapters/two_stream_v15_vfromm_mp.py` — 동일

또는 기존 adapter에 ckpt path만 다르게 dispatch (단순화 권장).

### 비용

- 18 잡 × AIP 1×1 H100 × ~2-3일 per job
- 클러스터: ~36-54 GPU·h 학습
- 로컬 rollout: 각 ckpt 500 episodes × 18 = 9000 episodes

### 의존성

C1 ckpt (ep50 권장).

### 결과 통합

`paper_artifacts/libero_rollout/summary.csv`에 행 추가:
- `two-stream-v15-vfromm-ptptk` × 3 suite
- `two-stream-v15-vfromm-mp` × 3 suite

---

## 4. C4 — v15 ep50 LIBERO action probing 완료

### 현재 상태

`docs/cluster_sessions.md` 2026-05-13 entry:
- v15 ep50 spatial `p_t_p_tk` (4 gaps) — ✅ COMPLETED
- v15 ep50 object — ❌ CANCELLED
- v15 ep50 goal — ❌ CANCELLED

### 재수행

object, goal suite × `p_t_p_tk` mode × 4 gap (1, 13, 20, 40).

### sbatch

```bash
# scripts/cluster/probe_action_libero.sbatch
SUITE=libero_object  # 또는 libero_goal
MODE=patch_mean_concat_p_t_p_tk
GAPS=1,13,20,40
MODEL_CKPT=two_stream_v15/<timestamp>/checkpoint_latest.pt  # ep50
```

### 비용

- 2 suite × ~17min = ~1 GPU·h
- 또는 `--all-gaps` 옵션 사용 시 1 잡으로 통합

### Paper 위치

§4 supplementary table (paper main BC가 LIBERO이므로 probing은 supplementary로).

---

## 5. C5 — v15 ep50 DROID action probing 완료

### 현재 상태

- v15 ep50 DROID `patch_mean_concat_p_t_p_tk` gap=15 — ✅ COMPLETED
- v15 ep50 DROID `patch_mean_concat_enc_only` (A+B) gap=15 — ✅ COMPLETED
- gap 1, 10, 30 — 미수행

### 재수행

모든 gap (1, 10, 15, 30) × best mode (`p_t_p_tk` + `concat_enc_only`).

### 비용

- 3 gap × 2 mode × ~1min = ~15min (trivial)
- gap=15 재측정 fair 비교 위해 포함 시 +5min

### Paper 위치

§4 cross-domain probing, Fig 5 (DROID action probing R² across gaps line plot).

---

## 6. C6 — frame_t vs frame_tk reconstruction quality 정량

### 목적

사용자 5/13 분석: v11은 reconstruction path 비대칭 (frame_t direct, frame_tk motion-routing 거침), v15는 같은 path → quality 대칭.

### Fig 4 design intent realization 증거

각 ckpt × test sample (예: 100-200개) × 두 frame reconstruction:
- v11 ckpt ep44: pred_t (interpreter_1 → recon_head), pred_tk (interpreter_1 → motion-routing → interpreter_2 → recon_head)
- v15 ckpt ep32/ep50: pred_t (interpreter_1 → recon_head), pred_tk (interpreter_1 → recon_head, same path)

### 측정 metric

- **per-sample MSE**: pred vs GT, per-pixel L2
- **FID** (optional): batch-level distribution similarity
- **LPIPS** (optional): perceptual similarity

### 결과 visualization

- Side-by-side: v11 frame_t vs frame_tk quality dispersion (large)
- v15 frame_t vs frame_tk quality dispersion (small)
- Quantitative: mean ± std of per-sample MSE ratio (frame_tk / frame_t)

### 비용

- v11, v15 각각 100 sample × forward = ~1 GPU·h
- Visualization script: ~30min CPU

### 의존성

기존 ckpt만 필요 (학습 X).

### Paper 위치

§3 method, Fig 4 — v11 → v15 design intent realization의 결정적 figure.

---

## 7. C7 — VideoMAE-ours P_t+P_tk probing (선택, catalyst 인과 보강)

### 목적

DINOv2 controlled (+0.006 vs v15 +0.390)의 짝. 2-frame MAE SSL (VideoMAE-ours)도 P_t+P_tk concat에서 motion 정보 emerge 안 함 입증 → catalyst hypothesis 추가 인과 증거.

### 측정

VideoMAE-ours encoder × 두 frame:
- EgoDex `patch_mean_concat_p_t_p_tk` (gap=10) 또는 equivalent (VideoMAE는 단일 encoder이므로 같은 frame_t와 frame_tk 각각 forward 후 concat)
- LIBERO spatial `p_t_p_tk` × gap=20

### 예상 결과

DINOv2 (+0.006)와 유사한 near-zero. 다른 가능성: VideoMAE-ours가 2-frame 학습이라 약간 양수일 수도 (단 v15 +0.390에 비해 격차 큼).

### 비용

- 2 dataset × ~13min = ~30min

### Paper 위치

§5 Table 4 (mechanism evidence): v15 +0.390 vs VideoMAE-ours +X.XX vs DINOv2 +0.006.

---

## 8. C8 — DROID main table 완성 (선택)

### 목적

DROID cross-domain main table에 5 encoder × 4 gap × best mode 완성.

### 현재 누락

paper_artifacts/probing/에 v11_droid_summary.csv, videomae_droid_summary.csv 있음. dinov2, siglip, vc1 DROID probing 미수행.

### 매트릭스

- 5 encoder (v15, VideoMAE-ours, DINOv2, SigLIP, VC-1) × 4 gap (1, 10, 15, 30) × best mode
- = 20 cells

### 비용

- 각 cell ~1min × 20 = ~30min total

### Paper 위치

§4 cross-domain table (Fig 5 line plot의 data source).

---

## 9. C9 — Causal Future Prediction probing 완성 (선택)

### 현재 상태

cluster_sessions 2026-05-13 entry:
- target_mode=future option 추가 (probe_action.py)
- v15 ep32 + v11 ep44 × 7 mode = 14잡 fair pair 비교
- 일부만 완료 (cluster_sessions에서 4 entries만 보임)

### 재수행

남은 mode 측정. 또는 v15 ep50으로 옮겨서 14잡 새로 시작.

### 비용

- 14 cells × ~13min = ~3 GPU·h

### Paper 위치

§4 또는 supplementary — "역량 1 (변화 인지) + 역량 2 (미래 예측)" 두 capability framing.

---

## ★ C10 — CALVIN action probing (DROID 보강, 2026-05-18 신규)

### Motivation

DROID는 image-only feature setting의 본질적 어려움으로 모든 encoder 절대값 R²≈0 (정직 보고함). 격차 패턴 (gap=15 +0.040)은 보이지만 cleaner ground가 필요. **CALVIN**은 tabletop language-conditioned manipulation으로 action signal 정제도 ↑, CoRL reviewer 친숙도 ↑.

**DROID 대체 X, CALVIN은 추가**. 두 dataset 모두 cross-domain probing axis로 paper §4 ¶2에 등장 — 격차 패턴의 dataset cross-validation 효과.

### Dataset prep (구현 TODO)

```python
# scripts/preprocessing/extract_calvin.py (신규 작성 예정)
# CALVIN dataset (http://calvin.cs.uni-freiburg.de/)
# - 4 splits: A, B, C, D (각 ~6h trajectories)
# - Static RGB-D + gripper RGB-D 둘 다 제공 (paper main은 static만)
# - Action: 7-DoF EE delta + gripper open/close
# TODO (dev session):
# 1. Download CALVIN ABCD splits → /proj/external_group/mrg/datasets/calvin/
# 2. Frame extraction (static view only, 224×224 unified)
# 3. Action label dump (pose Δ, gripper binary)
# 4. Probing dataset loader (src/probing/datasets/calvin_dataset.py)
```

### 측정 항목

- 5 encoder (v15, VideoMAE-ours, DINOv2, SigLIP, VC-1) × 4 ABCD splits × static view
- Probing mode: encoder별 best mode (v15는 `patch_mean_concat_p_t_p_tk`)
- gap: CALVIN trajectories 표준 frame rate에 맞춰 결정 (1, 5, 10 정도)
- Target: end-effector pose Δ + gripper state

### Critical guards

- **CALVIN gripper view는 본 paper main에서 미사용** — §5 ¶6 view-sensitivity sub-analysis는 LIBERO 한정. CALVIN gripper view는 future work.
- **Frozen encoder**: 모든 encoder downstream learn X
- **각 encoder native input distribution**: 1-frame encoder는 `(img_{t-1}, img_t)` 각각 인코딩 후 concat
- **DROID protocol과 동일 origin**: 5 encoder × probing mode set 일치
- **A→B→C→D OOD setting 활용 옵션**: CALVIN 본래 ABC→D OOD가 design intent — 단순 within-split 측정 외에 OOD generalization 추가 측정 가능 (선택)

### 비용

- 5 encoder × 4 splits × ~30분 (encoder forward + linear probe) = ~10 GPU·h
- Frame extraction (CPU 큐) ~1일 별도 (dataset prep)

### 결과 위치

- `paper_artifacts/calvin_action_probing/<encoder>_<split>_<mode>/summary.json`
- Paper §4 ¶2 (iii) — DROID와 같은 cross-domain table에 통합

### Paper 위치

§4 ¶2 action probing across domains (iii) CALVIN row 추가. 정직 보고: "consistent direction with DROID, with larger absolute R² thanks to cleaner action signal."

### 의존성

- 없음 (data prep 후 즉시 launch 가능)

---

## ★ C11 — CortexBench manipulation subset (vision encoder generic capability, 2026-05-18 신규)

### Motivation

Action probing은 본 paper main axis지만, vision encoder의 generic capability를 별도 ground에서 검증할 필요. **CortexBench** (Majumdar et al. 2024, VC-1 paper standard)는 17 task × 7 domain 표준 — 본 paper에서는 **manipulation subset (Adroit + Meta-World)** 만 측정 (full 17 task는 비용 부담, supplementary 또는 future work).

Frozen encoder + linear policy 평가 — robotics-specific representation 비교의 standard ground.

### Dataset/code prep (구현 TODO)

```python
# CortexBench codebase: https://github.com/facebookresearch/eai-vc
# - Adroit (pen-v0, relocate-v0 — 공식 eai-vc expert demo zip이 2 task만 제공)
# - Meta-World (assembly, bin-picking, button-press-topdown, drawer-open, hammer — 5 task)
# - Frozen visual encoder → linear policy head → BC training
# TODO (dev session):
# 1. Clone eai-vc + 의존성 설치
# 2. 본 paper의 5 encoder를 CortexBench encoder API에 wrap (v15 forward + adapter)
# 3. Adroit + Meta-World linear policy training launch
# 4. Standard CortexBench eval protocol (seed 평균)
```

### 측정 항목

- 5 encoder × {Adroit 2 task, Meta-World 5 task} = **7 task** × 3 seed = **105 BC training**
- Linear policy head, frozen encoder
- Eval metric: CortexBench 표준 success rate (per task → 평균)

### Critical guards

- **CortexBench 표준 protocol 엄격 준수** — fair comparison 위해 hyperparameter 변경 금지
- **본 paper의 5 encoder만** — full 17 task baseline은 paper 표 footnote에서 reference만 (VC-1 paper에서 가져옴)
- **v15 adapter는 main paper와 동일** — separate adapter 학습 X
- **시간 부족 시 Plan B**: paper supplementary 또는 CoRL rebuttal로 이관 — main claim은 LIBERO BC + CALVIN/DROID probing만으로도 성립

### 비용

- ~21 GPU·h (7 task × 3 seed × ~1 GPU·h per linear policy + eval)
- 시간 빠듯 — C10/C12 우선, C11은 병행

### 결과 위치

- `paper_artifacts/cortexbench/<encoder>/<task>/seed_<n>/summary.json`
- Paper §4 ¶4 (신규) — "Vision encoder generic capability" 표

### Paper 위치

§4 ¶4 신규 paragraph + 단독 sub-table. 또는 시간 부족 시 Appendix.

### 의존성

- 없음 (eai-vc 설치 + encoder wrap 후 launch)

---

## ★ C12 — LIBERO view-sensitivity sub-analysis (av_only vs av+eih combined, 2026-05-18 신규 / 2026-05-19 framing 재설계)

### Motivation

Paper §5 ¶6 sub-analysis: "v15 advantage scales with view information availability." Motion-routing의 architectural prediction — agentview에 wrist eye_in_hand view가 추가됐을 때 (motion-rich complementary view), v15가 두 view 정보를 가장 효과적으로 통합 → Δ(av+eih − av_only)가 baseline 대비 ↑ 가설.

**Framing 재설계 근거 (2026-05-19 사용자 비판)**: 초기 plan은 eye_in_hand 단독 vs agentview 단독 비교 = realistic robot system은 두 view 동시 사용하므로 single-view comparison은 의미 해석 제한적. 실제 paper main claim에 맞는 framing은 "agentview에 wrist view를 추가했을 때 우리 모델이 가장 큰 격차" — monotonic, 직관적, practical relevance ↑. eye_in_hand 단독 결과 9잡은 sunk cost (Tab 7 footnote/appendix 가능).

**Scope**: standard main protocol은 agentview (변경 없음). 본 sub-analysis는 controlled view comparison, **3 encoder (v15-ptptk + LIBERO BC 상위 2 baseline siglip/vc1)**, probing only (BC 제외).

**비교군 축소 근거 (2026-05-18 사용자 결정)**: 5 enc 전체보다 LIBERO BC SR 상위 2 baseline (siglip 0.855, vc1 0.821, [fig3_bc_main/summary.csv](paper_artifacts/fig3_bc_main/summary.csv))과 v15-ptptk만 비교. dinov2 (0.811), videomae-ours는 제외 — main paper에서 SR 격차가 명확한 top 2가 view-sensitivity sub-analysis의 fair contrast 제공.

### Dataset prep

LIBERO eye_in_hand 데이터는 이미 LIBERO 표준 구성에 포함 (HDF5 `obs/eye_in_hand_rgb` shape=(T,128,128,3)) — 추가 데이터 다운로드 불필요. `probe_action_libero.py:251`에 `--view {agentview_rgb, eye_in_hand_rgb, both}` flag 구현 완료 (2026-05-19 추가).

- `view=both`: load_demo()를 두 view 각각 호출 → encode_fn()을 두 번 (prev_av/curr_av, prev_eih/curr_eih) → feature-level concat → linear probe (2× embed_dim)
- av_only baseline: `view=agentview_rgb` 동일 protocol

### 측정 항목

**2 condition × 3 encoder × 3 suite × 4 gap = 72 cells (= 18 sbatch jobs, 각 잡 내부 4 gap sweep)**

| Condition | View input | Probe input dim (v15 p_t_p_tk 기준) |
|-----------|-----------|-----|
| A. **av_only** | agentview 단독 | 1536 |
| B. **av+eih combined** | agentview encode ⊕ eye_in_hand encode | 3072 |

- 3 encoder = v15-ptptk, SigLIP, VC-1
- 비교 metric: **Δ(B − A) per (enc, suite, gap)** → 가설: v15 Δ 최대
- gaps = 1/13/20/40 (LIBERO 20Hz에서 0.05/0.65/1.00/2.00s)

### Critical guards

- **본 paper main protocol은 agentview 유지** — view-sensitivity 결과는 §5 ¶6 + Tab 7 appendix 한정
- **3 encoder (v15-ptptk + top-2 BC baselines)** — main paper top performer set과 일관성. dinov2/videomae-ours는 footnote에서 reference
- **Probe input dim 차이 (1536 vs 3072) 자체는 fair** — 각 encoder가 동일 dim 변화 적용. over-fit 위험은 controlled comparison에서 동일하게 적용
- **결과 분기 시나리오 3가지 framing 미리 준비** (Vault `우려사항 및 대응방안.md` §16.6 참조):
  1. v15 Δ 최대 → motion-routing이 wrist의 motion-rich 정보 통합 능력 직접 증명 (paper main claim 보강)
  2. v15 Δ baseline과 비슷 → multi-view aggregation 능력은 universal (장점 reframe)
  3. v15 Δ baseline보다 작음 → M-stream wrist saturation (§6 future work)
- **"Wrist view 추가" framing은 monotonic** — 격차 부호 명확 (combined > av_only가 좋음). single-view comparison의 양방향 해석 모호함 회피
- **eye_in_hand 단독 9잡 (구 설계)** = sunk cost. Tab 7 footnote 또는 supplementary 인용 가능

### 비용

- av+eih combined 9잡 (3 enc × 3 suite × view=both, 각 잡 4 gap sweep, ~16분/잡) ≈ **2.5 GPU·h**
- av_only baseline 7잡 (siglip + vc1 × 3 suite + v15ep50 spatial. v15ep50 object/goal는 기존 결과 재활용) ≈ **2.0 GPU·h**
- 합계 **~4.5 GPU·h** (eye_in_hand 단독 sunk cost 2.5 GPU·h 별도)

### 결과 위치

- `paper_artifacts/libero_action_probing/<encoder>_libero_<suite>_<ckpt>_eye_in_hand/<gap>/summary.json`
- Vault `7. Outline.md` Tab 7 (appendix) 채움

### Paper 위치

§5 ¶6 (한 줄 본문 + Tab 7 cite) + Tab 7 (appendix 60-cell table).

### 의존성

- 없음 (코드 1줄 추가 + 기존 5 encoder ckpt 그대로 사용)

---

## 10. 작업 흐름 권장

### Day 0 (즉시)

```
# C1 sbatch launch (가장 critical, 가장 오래 걸림)
sbatch scripts/cluster/pretrain.sbatch \
  --model two-stream-v15 \
  --motion-routing-variant v_from_m \
  --time 10-00:00:00

# C6 frame quality 측정 (학습 X, 즉시)
# C4, C5 v15 ep50 추가 probing (즉시)
# C7 VideoMAE-ours P_t+P_tk (1 GPU·h, 즉시)
```

### Day 1-2

```
C4, C5, C6, C7 결과 수합 + paper artifact 정리
C1 학습 진행 모니터링 (ep1-12 진단)
```

### Day 3-4

```
C1 ep32+ ckpt 도착 → C2 (probing) 즉시
C1 ep50 도착 → C3 (BC-T 18잡 launch)
```

### Day 5-6

```
C3 BC-T 학습 → 로컬 워크스테이션 rollout
C2 결과 + Tab 3 작성
```

### Day 7+

```
C8, C9 시간 되면
Paper figure / table 생성
```

---

## 11. 결과 통합 위치

### `paper_artifacts/`

- `libero_rollout/{summary,per_task,episodes}.csv` — C3 v15-vfromm 행 추가
- `libero_action_probing/<ckpt>/<mode>/<gap>/summary.json` — C2, C4
- `probing/` — C5 (DROID), C7 (VideoMAE P_t+P_tk)
- `recon_quality/` (신규) — C6 정량 + viz

### `docs/`

- `cluster_sessions.md` — 모든 C1-C9 잡 추적
- `RESEARCH_PLAN.md` — phase 진행 update
- `paper_experiments_plan.md` (본 문서) — overview + completion check

---

## 12. Cross-references

- Vault: [[Action-Agnostic Paper/Evolution#3. Post-v11 Lineage (v12-v15) — Paradigm Conflict 진단 + Separation]] (v15 §)
- Vault: [[Action-Agnostic Paper/v15 - Layered Specialization (Future)]] (v15 design detail)
- 본 dev repo: `docs/RESEARCH_PLAN.md` (전반 plan), `docs/cluster_sessions.md` (잡 진행)
- Paper repo (`action-agnostic-paper`): §3 method, §4 experiments, §5 ablation 작성 시 본 문서 reference
