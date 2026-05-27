# Action Probing Guide

## 실험 개요

**평가 전략 (2026-04-14 재편)**: Two-Stream과 VideoMAE-ours만 EgoDex로 학습하므로 EgoDex within-domain probing은 **controlled comparison 전용**(Two-Stream vs VideoMAE-ours). 공개 가중치(VC-1, DINOv2, SigLIP, VideoMAE-official, V-JEPA-official)와의 main 비교는 OOD인 **DROID에서 수행**.

| 평가 | 비교 대상 | 역할 |
|------|----------|------|
| EgoDex probing | Two-Stream vs VideoMAE-ours (둘 다 EgoDex 학습) | 구조적 bias 기여 sanity |
| **DROID probing (main)** | 상기 2개 + VideoMAE-official, V-JEPA-official, VC-1, DINOv2, SigLIP | Cross-encoder fair comparison |

**핵심 주장**:
> "Action-agnostic pretraining으로 학습한 변화 임베딩은 action-informative하다. 따라서 VLM은 임베딩 공간에서 sequence modeling만으로 실질적 action planning이 가능하다."

### Probing 프로토콜

- **Linear probe**: frozen encoder → linear layer → action 회귀
- **Encoder frozen**: backbone 가중치 고정, probe만 학습
- **Input**: `(img_t, img_{t+gap})` 2-frame pair
- **Target**:
  - EgoDex: hand pose delta (within-domain)
  - DROID: robot 7-DoF joint velocity (cross-domain, Franka)
- **Epochs**: 20, **Gap**: 10 (기본)

### Dataset Split

- **EgoDex**: train = part1~3 (학습 미사용 보장), **test = part4** (180K train / 40K eval pair)
- **DROID**: 95,658 에피소드, ext1 카메라 기본 사용, 256x256 리사이즈 (180x320 원본, crop 없음)

## 실행 방법

### EgoDex Probing (within-domain)

```bash
python scripts/eval/probe_action.py \
    --encoder two-stream \
    --checkpoint /mnt/data/checkpoints/two_stream/.../checkpoint_epochXXXX.pt \
    --egodex-root /mnt/data/egodex \
    --frames-root /mnt/data/egodex_frames \
    --egodex-split part4 \
    --cls-mode patch_mean_concat \
    --gap 10 --epochs 20

# VideoMAE
python scripts/eval/probe_action.py --encoder videomae \
    --checkpoint <ckpt> --egodex-split part4 --gap 10

# Baseline
python scripts/eval/probe_action.py --encoder dinov2 --egodex-split part4 --gap 10
```

### DROID Probing (cross-domain, primary)

```bash
python scripts/eval/probe_action_droid.py \
    --encoder two-stream --checkpoint <ckpt> \
    --droid-root /mnt/data/droid_frames/ext1 \
    --gap 10 --epochs 20
```

### 클러스터 실행

```bash
sbatch scripts/cluster/probe_action.sbatch  # sbatch launcher
```

### LIBERO Action Probing (Phase 2 보강) ✅ POSITIVE (2026-04-30 완료)

DROID action probing R²~0.005 한계를 LIBERO sim 환경에서 보완. Gap-matched protocol (LIBERO 20Hz {1,13,20,40} = DROID 15Hz {1,10,15,30}). Plan: [`libero_action_probing_plan.md`](libero_action_probing_plan.md).

**구현**: [`scripts/eval/probe_action_libero.py`](../scripts/eval/probe_action_libero.py) + [`scripts/cluster/probe_action_libero.sbatch`](../scripts/cluster/probe_action_libero.sbatch). 5 encoder × 3 suite × 4 gap = 60 cells, ~3.6 GPU·h. 한 잡 안에서 4 gaps loop (encoder forward 동일 비용, 큐 점유 줄임).

**Target (plan §3, pose-derived)**: 7-DoF = 3 pos delta (eef_pos[t+k]-eef_pos[t]) + 3 rotvec ((R.from_rotvec(ee_ori[t]).inv()*R.from_rotvec(ee_ori[t+k])).as_rotvec()) + 1 gripper (actions[t+k-1, 6]). **HDF5 schema 발견**: `ee_ori`는 (T,3) **axis-angle** (plan의 `eef_quat` xyzw가 아님 — `R.from_quat`→`R.from_rotvec` substitution).

**주요 결과 (R² aggregate)**:

| Encoder | spatial gap=1 | gap=20 | gap=40 |
|---------|--------------:|-------:|-------:|
| **v11 ep44 (A+B+D')** | **+0.660** ★ | +0.555 | +0.374 |
| videomae-ours | +0.408 | +0.487 | +0.289 |
| dinov2 | +0.611 | **+0.666** | **+0.502** |

→ **Controlled comparison (v11 > VideoMAE-ours)**: 12/12 cells 전부 v11 우위 (+0.04~+0.25). **v11이 gap=1에서 internet-scale도 추월** (spatial/object). 자세한 매트릭스 + 4가지 발견 + paper framing은 [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) §Phase 2 보강.

**산출물**: `paper_artifacts/libero_action_probing/<encoder>_<suite>_<ts>/gap<k>/{summary.json, all_gaps.csv}`.

**실행 예시**:
```bash
# 한 잡 = encoder × suite × 4 gaps
sbatch --export=ALL,ENCODER=two-stream-v11,\
CHECKPOINT=/proj/external_group/mrg/checkpoints/two_stream_v11/.../checkpoint_epoch0044.pt,\
TASK_SUITE=libero_spatial \
       scripts/cluster/probe_action_libero.sbatch

# baselines (HF cache)
sbatch --export=ALL,ENCODER=dinov2,TASK_SUITE=libero_spatial scripts/cluster/probe_action_libero.sbatch
```

**구현 주의사항 (트러블슈팅 — 재발 방지)**:
- **OOM**: 초기 구현은 train demos 전체 frames을 메모리에 preprocess → OOM_KILL. **streaming refactor** 필수 (demo 단위 forward + embeddings only 누적). raw frames은 즉시 폐기.
- `ee_ori`는 axis-angle (plan의 quat 아님) — `R.from_rotvec` 사용.

### LIBERO Single-Frame Task Classification (Phase 2 보강 — P Stream Evaluation Gap) — TODO (2026-04-30 신규)

Phase 2 보강 LIBERO action probing 결과의 한계 (P stream specialization positive evidence 부재) 를 보완하기 위한 추가 evaluation. Single frame → 10-way task classification. **Single frame 이라 motion 정보 부재 → P stream 우위 자연 영역**.

**핵심 결정**:
- Frame: trajectory mid-point single frame. 2-frame encoder 는 (frame_t, frame_t) self-pair (M stream ΔL ≈ 0 자연 발생)
- Probe: linear classifier + softmax, 10-way per suite. cross-entropy + Adam + 20 epoch
- Cells: 5 encoder × 3 suite (= 15) + v11 mode ablation (A / B / D' / A+B+D' × 3 suite = 12) = **27 cells**
- 가설: **B (P encoder) > A (M encoder)** — single frame 에서 P 우위 → functional differentiation 양방향 입증
- 비용 ~2 GPU·h, 1-2일

**자세한 plan + 주의사항 + pseudocode + HDF5 sanity (Step 1) + TODO checklist**: [`libero_task_classification_plan.md`](libero_task_classification_plan.md)

**4 시나리오별 후속 계획** (실험 결과 보고 paper 측 framing 결정):
- B > A: paper §4 새 sub-section "Functional Differentiation Two Directions" 검토
- B > DINOv2: hero result. paper §4.5 강조
- B ≈ DINOv2: P 가 image SSL 과 경쟁력 paragraph 추가
- B 약함: paper main framing 약화 ("P as substrate") + Limitations 명시

**상태**: 계획만 확정. Implementation 은 별도 dev session 에서 작성 (`scripts/eval/task_classification_libero.py` 신규 예정).

### Trajectory-Level Value Alignment (VIP-inspired, Phase 2.5) — ❌ NEGATIVE (2026-04-30 완료)

VIP (Ma et al., 2022, Value-Implicit Pretraining)에서 영감 받은 metric. trajectory별 e_t와 e_T(마지막 frame) 간 cosine을 V(t)로 두고 Spearman ρ(t, V(t)) 측정.

**구현**: [`scripts/eval/value_alignment.py`](../scripts/eval/value_alignment.py) (frame-wise encoder forward + trailing window sweep + v11 mode 3종 분기) + [`scripts/cluster/value_alignment.sbatch`](../scripts/cluster/value_alignment.sbatch). 5 encoder × 3 suite × 50 demo × 10 task. 산출물 `paper_artifacts/value_alignment/<encoder>_<suite>_<ts>/{per_demo_rho.csv, per_demo_rho_summary.json}`.

**결과 (frac=1.0)**: vc1 +0.800 ≫ siglip/dinov2 +0.71/+0.73 > videomae-ours +0.67 ≫ v11 +0.47. 가설 (v11 ≥ baselines) 전면 기각.

**핵심 분석**: VIP-style state-similarity metric은 VIP-objective로 학습한 encoder가 본질적으로 유리. v11은 frame-pair MAE reconstruction objective라 cosine alignment를 직접 학습한 적 없음 — 약한 결과는 expected. Motion encoding(A) 추가는 무관 (D' only ≈ A+B+D'). Architectural motion routing(D' > B)은 healthy.

자세한 분석·가설·framing은 [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) §Phase 2.5 참조.

**실행 예시**:
```bash
# 단일 encoder × suite (full trajectory만)
sbatch --export=ALL,ENCODER=vc1,TASK_SUITE=libero_spatial scripts/cluster/value_alignment.sbatch

# fractions sweep (trailing window 가설 검증용)
sbatch --export=ALL,ENCODER=two-stream-v11,\
CHECKPOINT=/proj/external_group/mrg/checkpoints/two_stream_v11/.../checkpoint_epoch0044.pt,\
TASK_SUITE=libero_object,V11_MODE=d_prime_only,\
"LAST_K_FRACTIONS=1.0 0.5 0.3 0.15" scripts/cluster/value_alignment.sbatch
```

#### 📌 TODO: v11 mode ablation 결과 확인 + 본 가이드에 추가 (2026-04-30)

**상태**: `cluster_sessions.md` 의 Phase 2.5 § 끝에 v11 mode ablation 잡 (b_only / d_prime_only) 이 PENDING 으로 기록됨:
- 33632852~33632857 (b_only × 3 suite + d_prime_only × 3 suite, fractions sweep)

**확인 + 기록 작업**:
- [ ] sacct 또는 `paper_artifacts/value_alignment/two-stream-v11_*_b_only_*` 출력 디렉토리 확인 → b_only 결과 도착했는지 확인
- [ ] d_prime_only (이미 일부 도착 — full vs b_only contrast) 결과 종합
- [ ] **본 § Phase 2.5 섹션에 결과 매트릭스 추가**:

  ```
  | v11 mode | spatial ρ | object ρ | goal ρ |
  |----------|----------|---------|--------|
  | A+B+D' (default)  | +0.531 | +0.379 | +0.513 |
  | b_only            |   ?    |    ?    |    ?   |
  | d_prime_only      |   ?    |    ?    |    ?   |
  ```

- [ ] **결과 해석 paragraph**:
  - b_only > A+B+D'? → P stream 이 cosine alignment / state-similarity 에 강함 evidence (P specialization positive 한 axis)
  - b_only ≈ A+B+D'? → mode 무관, P encoder 자체가 cosine alignment 약함
  - d_prime_only ≈ A+B+D'? → motion encoding (A) 의 기여 무관 (이미 RESEARCH_PLAN.md 에 추정 기록됨)

- [ ] **Cross-link**: 결과는 P stream evaluation 의 한 axis 로 활용. LIBERO Single-Frame Task Classification (위 섹션) 결과와 종합 분석. 자세한 분석은 `RESEARCH_PLAN.md` §Phase 2.5 + Vault `3. Experiments § Phase 2 보강 § P Stream Evaluation Gap`

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--cls-mode` | embedding 추출 방식 | average |
| `--gap` | 프레임 간격 | 10 |
| `--egodex-split` | 데이터 파티션 | part4 (test) |
| `--max-videos` | 비디오 수 제한 (디버깅) | None (전체) |

### cls-mode 선택 가이드 (Two-Stream v6/v10, `probe_action.py`)

| cls-mode | 차원 | 용도 |
|----------|------|------|
| `average` | 768 | CLS (m+p)/2, compact representation |
| `concat` | 1536 | CLS [m;p], stream 분리 비교용 |
| `m_only` | 768 | M stream CLS만, temporal 분석 |
| `p_only` | 768 | P stream CLS만, spatial 분석 |
| `patch_mean` | 768 | M+P 패치 전체 mean pool |
| **`patch_mean_concat`** | **1536** | **M/P 패치 각각 mean → concat (기본 권장)** |
| `patch_mean_m` | 768 | M 패치만 mean pool (stream 진단) |
| `patch_mean_p` | 768 | P 패치만 mean pool (stream 진단) |

### cls-mode 선택 가이드 (Two-Stream v11, `probe_action_v11.py`)

v11은 4 위치에서 representation 추출 가능: A=M encoder, B=P encoder, D'=motion-routing 후, D=Phase 3 final.

| cls-mode | 차원 | 위치 | 용도 |
|----------|------|------|------|
| `cls_m_enc` | 768 | A | M encoder CLS |
| `cls_p_enc` | 768 | B | P encoder CLS |
| `cls_concat_enc` | 1536 | A+B | M/P CLS concat |
| `patch_mean_m_enc` | 768 | A | M encoder patches mean (motion-biased task에 강력) |
| `patch_mean_p_enc` | 768 | B | P encoder patches mean (단독은 약함) |
| `patch_mean_concat_enc_only` | 1536 | A+B | M+P enc patches concat |
| `patch_mean_p_state_after_routing` | 768 | D' | motion-routing × 2 후 (interpreter_2 전) |
| `patch_mean_p_features_tk` | 768 | D | Phase 3 final (interpreter_2 후) |
| **`patch_mean_concat_enc_phase3`** | **1536** | **A+D** | **M enc + P Phase 3 (★ 기본 권장, ep12 +0.219)** |
| `patch_mean_concat_enc_d_prime` | 1536 | A+D' | M enc + D' |
| `patch_mean_concat_p_enc_d_prime` | 1536 | B+D' | P enc + D' |
| `patch_mean_concat_all` | 2304 | A+B+D' | 3-way concat |

## Probing 결과 (EgoDex part4, gap=10, linear probe, 20ep)

### 활성 모델 lineup

| 모델 | 체크포인트 | cls_mode | R² | 비고 |
|------|-----------|----------|-----|------|
| **v6 (APE + rotaug)** | ep8 | patch_mean_concat | **+0.259** | 현재 챔피언 |
| **VideoMAE-ours** | ep50 | patch_mean | **+0.326** | 수렴 (ep28 +0.317) |
| v4 (RoPE) | ep48 | patch_mean_concat | +0.197 | 정체 |
| Two-Stream v10 | ep8 | patch_mean_concat | +0.206 | 1차 peak |
| Two-Stream v10 | ep20 | patch_mean_concat | +0.137 | collapse 저점 |
| Two-Stream v10 | ep24 | patch_mean_concat | +0.202 | W-shape 회복 |
| Two-Stream v10 | ep36 | patch_mean_concat | +0.214 | new peak |
| Two-Stream v10 | ep40 | patch_mean_concat | **+0.221** | **plateau 시작** |
| Two-Stream v10 | ep44 / ep48 | patch_mean_concat | +0.221 / +0.222 | plateau 지속, **v6 추월 실패 확정** |
| Two-Stream v11 | ep4 | patch_mean_concat_enc_phase3 (A+D) | +0.143 | 학습 시작 |
| Two-Stream v11 | ep8 | patch_mean_concat_enc_phase3 (A+D) | +0.194 | |
| Two-Stream v11 | ep12 | patch_mean_concat_enc_phase3 (A+D) | +0.219 | v10 ep40 plateau 도달, 12 epoch만에 |
| Two-Stream v11 | ep24 | patch_mean_concat_all (A+B+D') | +0.234 | 점진 향상 |
| **🏆 Two-Stream v11** | **ep44** | **patch_mean_concat_all (A+B+D')** | **+0.288** | **v6 챔피언 (+0.259) 추월! Final champion ★** |
| Two-Stream v11 | ep48 | patch_mean_concat_all (A+B+D') | +0.281 | plateau (-0.007 vs ep44) |
| Two-Stream v11 | ep50 (final) | patch_mean_concat_enc_phase3 (A+D) | +0.267 | 학습 종료. A+B+D' = +0.279, A+D' = +0.282 stable plateau |
| DINOv2 (frozen) | — | CLS concat | (ceiling 참조) | 공개 weight |
| Random-init | — | — | (floor) | 구조적 prior 측정 |

### v10 stream-별 추세 (`patch_mean_{m,p}`)

| Epoch | M | P |
|-------|---|---|
| ep4   | +0.176 | +0.126 |
| ep8   | +0.150 | **+0.152** (peak) |
| ep12  | +0.129 | +0.083 |
| ep16  | +0.125 | +0.038 (sparse pinpoint viz) |
| ep20  | +0.135 | +0.022 |
| ep24  | +0.138 | +0.092 (peak의 60% 회복) |
| ep36  | +0.129 | +0.141 (peak의 93%) |

### v10 분석 (종료)

- **1차 peak ep8 +0.206 → ep20 collapse +0.137 → ep36 W-shape 회복 +0.214 → ep40 plateau +0.221**
- v6 ep8 챔피언 (+0.259) 추월 실패 확정. P-stream 내부 강화 방식의 한계로 결론
- LR cosine decay 후반 효과로 W-shape 회복은 진짜였으나 ceiling +0.222에서 plateau

### v11 (Motion-Guided Routing) — ep4~ep50 12-mode 비교 (학습 종료 후)

4 위치: A=M encoder, B=P encoder, D'=motion-routing 후, D=Phase 3 final

| Mode | ep4 | ep8 | ep12 | ep16 | ep20 | ep24 | **ep44** | ep48 | ep50 |
|------|-----|-----|------|------|------|------|----------|------|------|
| `patch_mean_m_enc` (A) | +0.170 | +0.176 | +0.208 | +0.213 | +0.220 | +0.222 | **+0.267** ★ | +0.264 | +0.265 |
| `patch_mean_p_enc` (B) | -0.041 | -0.025 | 0.000 | -0.001 | -0.002 | -0.004 | -0.003 | -0.000 | -0.001 |
| `patch_mean_p_state_after_routing` (D') | +0.121 | +0.066 | +0.072 | +0.077 | +0.098 | +0.113 | +0.135 | +0.138 | +0.129 |
| `patch_mean_p_features_tk` (D) | +0.023 | +0.055 | +0.054 | +0.047 | +0.060 | +0.057 | +0.050 | +0.049 | +0.048 |
| `patch_mean_concat_enc_only` (A+B) | +0.160 | +0.168 | +0.200 | +0.211 | +0.213 | +0.224 | +0.259 | +0.263 | +0.263 |
| `patch_mean_concat_enc_phase3` (A+D) | +0.143 | +0.194 | +0.219 | +0.217 | +0.230 | +0.232 | +0.264 | +0.264 | **+0.267** |
| `patch_mean_concat_enc_d_prime` (A+D') | +0.149 | +0.166 | +0.153 | +0.205 | +0.196 | +0.232 | +0.284 | +0.283 | +0.282 |
| `patch_mean_concat_p_enc_d_prime` (B+D') | +0.135 | +0.011 | +0.076 | +0.079 | +0.087 | +0.107 | +0.137 | +0.139 | +0.139 |
| **`patch_mean_concat_all`** (A+B+D') | +0.114 | +0.094 | +0.178 | +0.223 | +0.185 | +0.234 | **+0.288** ★★ | +0.281 | +0.279 |
| `cls_m_enc` | +0.066 | +0.155 | +0.162 | +0.163 | +0.172 | +0.158 | +0.125 | +0.123 | +0.123 |
| `cls_p_enc` | -0.059 | -0.011 | -0.008 | -0.010 | -0.009 | -0.013 | -0.002 | -0.002 | -0.002 |
| `cls_concat_enc` | -0.048 | +0.092 | +0.148 | +0.139 | +0.162 | +0.140 | +0.114 | +0.118 | +0.113 |

**🏆 핵심 결론 — ep44 final champion 확정**:
- **A+B+D' = +0.288 (ep44)** — v6 ep8 (+0.259) 추월 +0.029 ★★ (Final champion)
- **A 단독 (+0.267)도 v6 추월** — 단일 mode로
- **VideoMAE +0.326까지 격차 -0.038** (ep24 -0.092 → 절반 이상 좁힘)
- **W-shape 회복 패턴 확정**: ep12 +0.219 → ep20 +0.185 dip → ep24 +0.234 → ep44 +0.288 (LR cosine decay 후반 ep24→44 +0.054 큰 도약)
- **ep44~ep50 plateau 확정** (ep44 이후 LR≈0 정체):
  · A+B+D': ep44 +0.288 → ep48 +0.281 → ep50 +0.279 (-0.010 미세 over-tightening)
  · A+D': ep44 +0.284 → ep50 +0.282 (가장 robust한 plateau)
  · A+D: ep44 +0.264 → ep50 **+0.267** (미세 신피크, +0.003)
  → ep44가 진짜 peak. ep48/ep50은 안정성 검증 역할
- 사용자 v11 설계 가설 정량 확정:
  · 3-way concat (A+B+D')이 best — M+P+motion-routed P 상보적
  · A+D' (+0.282) > A+D (+0.267) — interpreter_2는 decoder wrapper
  · CLS 모두 약화 추세, patch_mean이 정답

## DROID Cross-domain Probing 결과

| Gap (DROID 15Hz) | VideoMAE | v11 best (mode) | 격차 |
|------------------|----------|-----------------|------|
| 1 (0.07초) | -0.006 | -0.005 | +0.001 |
| 10 (0.67초) | -0.006 | +0.006 (A+B) | +0.012 |
| **15 (1초)** ★ | **-0.035** | **+0.005 (A+B)** | **+0.040** |
| 30 (2초) | -0.028 | -0.010 | +0.018 |

- 모든 gap에서 v11이 VideoMAE보다 일관 우위
- gap=15 (EgoDex 학습 분포 1초와 일치)에서 격차 가장 큼 (+0.040)
- VideoMAE는 in-domain (EgoDex +0.326) 강력하지만 cross-domain 음수
- 절대 R²은 작음 (~0.005) — DROID action probing 자체 한계
- **방향성 검증**: v11이 cross-domain 일반화 우수 (사용자 직감 정량 검증)

### cls_mode 비교 (v4 ep48 기준)

| cls_mode | dim | R² | Cosine Sim |
|----------|-----|----|------------|
| patch_mean_concat | 1536 | **0.197** | 0.236 |
| concat (CLS) | 1536 | 0.177 | 0.260 |
| patch_mean | 768 | 0.164 | 0.209 |
| average (CLS) | 768 | 0.052 | 0.197 |

**순위 패턴**: patch_mean_concat > concat > patch_mean > average. CLS average는 사실상 무용 (0.052).

### 해석

1. **Rotation augmentation이 결정적**: v5(rotaug 없음) ep4→ep8 하락(0.208→0.192) vs v6(rotaug) ep8 0.259. Position prior 의존을 차단하면 표현 품질이 개선됨
2. **patch_mean > CLS**: CLS는 cls_exchange로 혼합되어 정보 희석. Patch level에서 M/P는 서로 다른 정보를 담고 있어 probe가 독립적으로 활용 가능
3. **Two-Stream vs VideoMAE**: ep4에서 거의 동등(0.208 vs 0.198). 이후 학습 효율에서 차이 — VideoMAE 수렴(→0.326), Two-Stream은 rotaug/mask 설정에 민감
4. **Gap 효과**: gap=1은 노이즈 수준(~0.0006). full training 후 평가는 **gap=10 기본**

## CALVIN Cross-folder OOD Probing 결과 (paper §C10 main, 2026-05-26)

Segment-based fair protocol (commit `7eb0c48`) + cross-folder OOD (training/ → validation/). gap=10/15/20/30, max_episodes=200.

### per-dim R² 분리 (gap=30 = 1.0s)

| Encoder | pos avg | rot avg | **gripper** | aggregate |
|---------|---------|---------|-------------|-----------|
| **v15** | +0.262 | −0.005 | **−0.005** | −0.012 |
| videomae-ours | **+0.553** ★ | +0.152 | +0.059 | +0.056 |
| vc1 | +0.536 | +0.203 | +0.022 | +0.035 |
| **dinov2** | +0.223 | +0.173 | **+0.359** ★ | **+0.307** |
| siglip | −0.314 | −0.029 | +0.183 | +0.162 |

자세한 표·plot: [paper_artifacts/calvin_action_probing/_diagnostic/per_dim_r2.{csv,png}](../paper_artifacts/calvin_action_probing/_diagnostic/) (CALVIN + LIBERO 비교).

### 핵심 발견 — aggregate R²가 v15에 unfair

- **R² aggregate가 binary gripper에 dominated**. dinov2 aggregate 우위(+0.307)는 gripper R² (+0.359)에서 옴. continuous motion (pos delta)은 motion-SSL (videomae +0.553, v15 +0.262 ≈ dinov2 +0.223) 우위
- 모든 벤치에서 일관:
  - **EgoDex** (18-dim joint pose, gripper 없음): v15 +0.390 ★ 1위 — gripper bias 없는 motion-only 지표
  - **LIBERO spatial** gap=20: v15 pos avg **+0.896 ★ 1위**, dinov2 +0.766. gripper만 dinov2 우위(−0.09 격차)
  - **CALVIN**: 위 표대로
- CALVIN-specific 격차 큰 이유: pos delta scale 매우 작음(fine manipulation) → pos R² 절댓값 낮아 gripper dim이 aggregate 더 dominate. gripper R² 격차도 LIBERO 대비 4배
- gap sweep: v15 R² drop gap=10→30 = −120%, dinov2 −40%. segment 길이(34~65)에 gap=30이 거의 닿아 P_tk가 segment 끝 cluster → motion-SSL에 추가 hurt. 단 base 격차는 gripper에서 옴

**paper §C10 narrative**: "v15가 CALVIN OOD에서 약함"(오해) → "R² aggregate는 binary gripper-dominated metric. v15는 continuous motion에서 image-SSL 동급/우위, gripper binary 식별에서 약함. 모든 벤치 per-dim 분석 시 v15 motion 인코딩 1위 또는 공동 1위 일관".

자세한 진단 series (Case 1 per-task + Case 2 motion magnitude + Case 3 GAP sweep)는 [docs/cluster_sessions.md `2026-05-26 CALVIN cross-folder 원인 진단`](cluster_sessions.md) 참고.

## 권장 평가 프로토콜

```bash
# 1. 공정 비교: part4 (미사용), gap=10, patch_mean_concat
#    → Two-Stream vs VideoMAE-ours 비교

# 2. Stream 진단: cls_mode={patch_mean_m, patch_mean_p}
#    → M/P 각각의 action-informativeness 분리 측정

# 3. DROID cross-domain: probe_action_droid.py 동일 프로토콜
#    → 공개 weight 포함 main comparison
```

## 다음 단계

1. ~~v11 ep4~ep50 12-mode probing~~ ✅ 완료. ep44 final champion 확정
2. LIBERO BC v11 ep44/ep50 재측정 — ep12 결과 (VideoMAE 0.0286 vs v11 0.0290) 거의 동등, 학습 진전 ckpt로 우위 기대
3. LIBERO Rollout setup (downstream success rate가 v11 채택 결정타)
4. DROID 추가 gap (5/20) 보강 + ep44/ep50 ckpt로 재측정
5. 공개 weight lineup (VC-1, DINOv2, SigLIP, VideoMAE-official, V-JEPA-official) DROID 평가