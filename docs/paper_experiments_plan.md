# Paper Experiments Plan — v15 NeurIPS 2026 main table 완성

> **Status**: 2026-05-14 plan 작성. 5/12 v15 only main 전략 결정 후 필요한 실험 list.
> **Scope**: Cluster + 로컬 워크스테이션 (rollout) 실험 전부.
> **Source decision**: Vault `Projects/Action-Agnostic Paper/Evolution.md` §3 v15 + paper §5 ablation plan.

## 0. Critical path 요약

```
C1 (V from M 학습, 50ep) ────┬── C2 (probing)
                              └── C3 (BC-T 18잡 + 로컬 rollout)

C4, C5, C6 (v15 ep50 추가 + recon quality) — 의존성 X, 즉시 병렬 가능
C7 (VideoMAE-ours P_t+P_tk) — 의존성 X, sanity 수준 (1 GPU·h)
C8, C9 — 시간 되면
```

**합계 (필수 C1-C6)**: ~104 GPU·h ≈ 4-5일
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
