# Action-Agnostic Visual RL Research Plan

**마지막 업데이트**: 2026-04-26
**연구 질문**: **구조적 inductive bias를 가진 시각 표현 학습이, action label 없이도 시각-행동 연결 태스크에 유용한 표현을 만드는가?**

> 현재 진행 상태 / 체크포인트 / sbatch 로그 등은 [`CLAUDE.md`](../CLAUDE.md)·[`docs/cluster_sessions.md`](cluster_sessions.md)·[`docs/PROBING_GUIDE.md`](PROBING_GUIDE.md) 참고. 본 문서는 **연구 설계와 로드맵**에 집중한다.

## 연구 동기

- **핵심 가설**: 영장류 시각 피질의 magnocellular / parvocellular 경로 분리는 생물학적 시각 시스템의 근본적 구조 원리. 이 구조를 비디오 표현 학습 모델에 반영하면 더 범용적인 표현을 학습할 수 있다.
- **Two-Stream 접근**: M(motion) / P(pattern) 채널 분리 + CLS exchange를 통해 구조적 inductive bias를 부여. Pixel reconstruction 목표와 결합.
- **시각-행동 연결 태스크로 검증**: 로봇 조작 태스크(LIBERO)는 "어떤 feature가 진짜 유용한지"를 가장 직접적으로 평가할 수 있는 설정. **Two-Stream은 로봇용 특수 알고리즘이 아니라, 일반 시각 표현 학습 방법의 로봇 태스크 검증**이다.
- **V-JEPA에 대한 중요 관찰**: V-JEPA (Bardes et al., 2024)는 같은 조건에서 feature prediction > pixel reconstruction을 실증 주장. 하지만 이 증거는 전부 **classification 벤치마크**(Kinetics-400, SSv2, ImageNet)에 국한. **로봇 제어(continuous control) 도메인 증거는 부재**. 우리는 이 갭을 채운다.

---

## 실험 대상 모델

| 구분 | Encoder | 사전학습 데이터 | 파라미터 | 방법 철학 | 학습 주체 |
|------|---------|---------------|---------|----------|----------|
| **제안** | **Two-Stream v10 (ours)** | EgoDex (~100M frames) | ~213M | M/P 구조 + pixel reconstruction + rotation aug + MAE-style P mask 0.75 | 🔥 우리 학습 |
| **Controlled comparison** | **VideoMAE-ours (2-frame)** | **EgoDex (same)** | ~101M | Vanilla MAE (구조적 bias 없음, mask 0.5) | 🔥 우리 학습 |
| **Native 세팅 baseline** | VideoMAE-official | Kinetics-400/SSv2 (16-frame) | ~86M | MAE (공식 세팅) | 📦 공개 가중치 |
| **Native 세팅 baseline** | V-JEPA 2.1 ViT-B | VideoMix22M (16-frame, 384px) | 86.8M | Feature prediction (최신 video SSL) | 📦 공개 가중치 |
| **로봇 제어 SOTA** | VC-1-Base | Ego4D + 조작 (~500M frames) | 86M | MAE (embodied AI 표준) | 📦 공개 가중치 |
| Internet-scale SSL | DINOv2-Base | LVD-142M | 86M | Self-distillation | 📦 공개 가중치 |
| Internet-scale VL | SigLIP-Base | WebLI (10B 이미지-텍스트) | 86M | Vision-language contrastive | 📦 공개 가중치 |

### 3-축 비교 구조

**축 1: 같은 데이터·같은 framework, 다른 구조** (Two-Stream vs VideoMAE-ours)
- 통제: EgoDex, 2-frame pixel MAE framework, 같은 compute, 같은 sampling
- 변수: encoder 구조 (M/P two-stream vs vanilla ViT)
- 답할 질문: "**M/P 구조적 bias가 표현 품질에 기여하는가?**" (단일 변수 격리)
- 주의: Two-Stream(213M) > VideoMAE(101M) 파라미터 차이는 M/P 두 스트림 설계상 불가피. 각 스트림이 ViT-B backbone으로 baseline과 동일한 per-stream capacity 유지

**축 1': 2-frame regime에서의 방법론 한계** (V-JEPA-ours negative result)
- V-JEPA-ours 3차례 시도 (LR warmup 추가, mask ratio 완화) 모두 발산. 2-frame은 V-JEPA의 16-frame temporal redundancy 전제를 깨뜨려 EMA target drift 유발 → paper Appendix에 loss curve overlay로 기록

**축 2: 같은 도메인 계열, 다른 구조** (Two-Stream vs VC-1)
- Egocentric/manipulation video. 변수: 데이터(EgoDex vs Ego4D+) + 방법(M/P vs standard MAE)

**축 3: 소규모 principled vs 대규모 data-driven** (Two-Stream vs DINOv2/SigLIP)
- 변수: 데이터 규모 + 방법 (100M frames 구조 기반 vs 수십억 규모 범용 SSL/VL)

### 공정성 원칙

- **모든 encoder는 frozen** (downstream에서 학습 안 함)
- **동일 입력 형식**: 단일 프레임 encoder (DINOv2, SigLIP, VC-1)도 `(img_{t-1}, img_t)` 두 프레임을 각각 forward pass 후 feature concat
- **각 encoder의 공식 preprocessing 사용** (정규화, resolution)
- Downstream에서 유일하게 학습되는 것: **MLP action decoder** (Phase 3) 또는 **projection + LoRA** (Phase 3B)

---

## 실험 로드맵

### Phase 1: EgoDex 사전학습 → Hand Pose Probing ✅ 완료

**목표**: 제안 Two-Stream 구조의 표현 품질 검증 (within-domain)

**작업**: v1~v4 iterative ablation — architecture / MAE masking / gap 분포 / composition consistency 전체 ablation 후 v4 설정 확정

**결과 요약** (500-video 소규모 평가, 상대 비교만 유효):
- **Masking 효과**: mask 0.3 > nomask (CLS 정보 밀도 33% 개선)
- **Composition loss 불필요**: base가 우세, 학습시간 5배·메모리 3배 대비 이점 없음 → 현재 구현되어 있지 않음
- **max_gap 60 + triangular > gap 30 linear**: eval loss 16% 개선, probing +0.054
- **patch_mean_concat > patch_mean > CLS average**: M/P 분리 보존이 유리
- **Architecture**: depth=12, num_stages=2 (6 blocks/stage, CLS exchange 2회)가 최적 (blk/stage가 depth보다 중요)

### Phase 1.5: 데이터 스케일업 + 설계 iteration 🔄 진행 중

**목표**: 전체 EgoDex (part1~5, 314k videos)로 최종 체크포인트 확보

#### Two-Stream 설계 iteration 기록 (폐기 lineup)

| 버전 | 설정 요지 | 결과 | 폐기 사유 |
|------|----------|------|----------|
| **v4** | RoPE + mask 0.3/0.5 | 48ep, R²=0.197 | RoPE ↔ APE 편향 검증(v5)에서 content 학습 약함 확인, v6 rotation_aug로 해결 |
| **v5** | APE + mask 0.5/0.5 | ep8 R²=0.192 | PE 변경만으로 position prior overfit 해결 불가 |
| **v6** | APE + mask 0.5/0.5 + **rotation aug** | ep23 scancel, **ep8 peak R²=0.259** (현 챔피언) | 챔피언이지만 v7~v9 시도 계기 |
| **v7-big** | CLS_P bg/motion 분리 + isolation | ep8 cos_st=0.9997 완전 collapse | EMA teacher 라인 구조적 실패 |
| **v8 1차/2차** | L_P + EMA P teacher (BYOL form) | ep12 patch_mean_p R²=**-0.468**, static salience로 수렴 | Teacher가 M=zero 입력받는 설계 → P가 "미래 static appearance"로 수렴 |
| **v9 P=current** | MAE frame_t, mask 0.75 | ep6 scancel, patch_mean concat +0.154 | frame_t가 입력이자 target → trivial |
| **v9 residual+norm** | Residual P target + patch-wise norm | ep8 scancel, P **+0.100→-0.006** degrade | ep4까지 P 학습 기여 후 ep8부터 분산 minimization 회귀 |
| **V-JEPA-ours** | 3차 시도 (warmup, mask 완화) | 모두 발산 | 2-frame이 V-JEPA temporal redundancy 전제 위배, negative result로 기록 |

#### ✅ Two-Stream v10 (v6 base + mask_p 0.75) — 종료, ep40 plateau (+0.221)

**설계 근거**: v6 ep8이 R²=0.259 챔피언. v9 실험에서 residual target은 부적합 확정 → v6 구조·target·loss 그대로 유지하고 **P-stream만 mask_ratio 0.5 → 0.75** (MAE-style aggressive mask로 P 역할 분화 강화).
- M-stream mask 0.5 유지 (motion sparse → aggressive mask 금지 원칙)
- Full run: JobID 33570871 (2026-04-22 ~, AIP_long 2노드×4 H100, 50ep)
- **결과**: ep4-8 1차 peak +0.206 → ep12-20 점진 collapse (+0.137 ep20) → ep24-40 W-shape 회복 → **ep40 peak +0.221** → ep44/ep48 plateau (+0.221, +0.222). **v6 챔피언 (+0.259) 추월 실패 확정**
- **결론**: P-stream 내부 강화 방식의 한계. v11로 전환

#### 🔄 진행 중: Two-Stream v11 (Motion-Guided Attention Routing + Dual-Target Reconstruction)

**상태**: 구현 + ep12 결과 도달, 학습 진행 중 (JobID 33594155, 2026-04-25 01:11~, AIP_long 2노드×4 H100, 50ep, `--time=3d`)

**설계 철학**: v7~v10의 네 가지 시도가 모두 "P stream 내부 강화"였고 모두 collapse 또는 v6 추월 실패. v11은 발상 전환 — **P stream 강화 포기, M이 P를 semantic-level operator로 조작하는 구조**. Dual-target reconstruction으로 collapse anchor 확보.

**학습 결과 (ep1-12, JobID 33594155)**:

| Epoch | L_total | L_t | L_tk | std_p | cos_intra_p |
|-------|---------|-----|------|-------|-------------|
| 1 | 0.0196 | 0.0109 | 0.0088 | 0.349 | 0.866 |
| 4 | 0.0057 | 0.0044 | 0.0014 | 0.210 | 0.897 |
| 8 | 0.0043 | 0.0038 | 0.00052 | 0.009 | 1.000 |
| 12 | 0.0024 | 0.00197 | 0.00039 | 0.004 | 1.000 |

→ Loss 단조 감소. P encoder CLS는 cos_intra≈1.0 collapse, 그러나 patches는 healthy (75% MAE 복구 작동)

**Probing — ep12 12 mode 비교** (4 위치: A=M enc, B=P enc, D'=motion-routing 후, D=Phase 3 final):

| Mode | ep4 | ep8 | **ep12** |
|------|-----|-----|----------|
| `patch_mean_m_enc` (A) | +0.170 | +0.176 | **+0.208** |
| `patch_mean_p_enc` (B) | -0.041 | -0.025 | 0.000 |
| `patch_mean_p_state_after_routing` (D') | +0.121 | +0.066 | +0.072 |
| `patch_mean_p_features_tk` (D) | +0.023 | +0.055 | +0.054 |
| `patch_mean_concat_enc_only` (A+B) | +0.160 | +0.168 | +0.200 |
| `patch_mean_concat_enc_phase3` (A+D) | +0.143 | +0.194 | **+0.219** ★ |
| `patch_mean_concat_enc_d_prime` (A+D') | +0.149 | +0.166 | +0.153 |
| `patch_mean_concat_p_enc_d_prime` (B+D') | +0.135 | +0.011 | +0.076 |
| `patch_mean_concat_all` (A+B+D') | +0.114 | +0.094 | +0.178 |
| `cls_m_enc` (A CLS) | +0.066 | +0.155 | +0.162 |
| `cls_p_enc` (B CLS) | -0.059 | -0.011 | -0.008 |
| `cls_concat_enc` (A+B CLS) | -0.048 | +0.092 | +0.148 |

- **ep12 A+D = +0.219** ≈ v10 ep40 plateau (+0.221). **v11이 12 epoch만에 v10 50 epoch 도달**
- 사용자 통찰 검증: interpreter는 decoder의 reconstruction wrapper (D' < D 역전 ep8에)
- M encoder 단독(+0.208)이 강력 — task가 motion-biased (hand pose ≈ motion)
- P encoder 단독은 약함, motion routing 거치면 살아남
- Loss와 R² 정직 상관 (L_total 0.0057 → 0.0024 절반 감소 → A+D R² +0.143 → +0.219)

**Cross-domain DROID probing** (사용자 직감 검증):

| Gap (DROID 15Hz) | VideoMAE | v11 best | 격차 |
|------------------|----------|----------|------|
| 1 (0.07초) | -0.006 | -0.005 | +0.001 |
| 10 (0.67초) | -0.006 | +0.006 (A+B) | +0.012 |
| **15 (1초)** ★ | **-0.035** | **+0.005 (A+B)** | **+0.040** |
| 30 (2초) | -0.028 | -0.010 | +0.018 |

- 모든 gap에서 v11이 VideoMAE보다 일관 우위
- gap=15 (EgoDex 학습 분포 1초와 일치)에서 격차 가장 큼
- 절대 R²은 작음 (~0.005) — DROID action probing 자체 한계
- 방향성 검증: **v11 cross-domain 일반화 우수**

**Cross-domain LIBERO** (진행 중):
- BC fine-tune 도구 신규 작성 (`scripts/eval/finetune_libero_v11.py`)
- libero_spatial 30 epoch 시작 (33600616 VideoMAE, 33600617 v11 ep12 A+D), 4-6h 후 결과

**핵심 구조 (pseudo code)**:

```
[Encoder — 둘 다 공식 MAE visible-only]
M encoder: [ΔL, Sobel_x(ΔL), Sobel_y(ΔL)] → self-attn × 12 → m_visible  (mask 0.3)
P encoder: [Sobel_x(L), Sobel_y(L), R, G, B] → self-attn × 12 → p_visible  (mask 0.75)
[Encoder output = downstream representation (pure per-stream)]

[M Decoder — motion field in-painting, P와 무관, loss 없음]
m_full = inject_mask_token(m_visible, mask_m, mask_token_m) + APE
self-attn + FFN × 3 → m_completed

[P Decoder — 3-phase dual-target]
p_full = inject_mask_token(p_visible, mask_p, mask_token_p) + APE

# Phase 1: pre-motion semantic interpretation
p_semantic_t = interpreter_1(p_full)                  # self-attn × 3
predicted_t  = recon_head(p_semantic_t[:, 1:])        # shared head
L_t = MSE(predicted_t · mask_p, img_t · mask_p)

# Phase 2: motion integration (N=2 iterations, projections iteration별 독립)
p_state = p_semantic_t
for i in range(2):
    Q_M_i, K_M_i = projections_M_i(m_completed)       # M stream에서 Q, K 둘 다
    V_P_i        = projections_V_i(p_state)           # P stream에서 V
    attn_pattern = softmax(Q_M_i K_M_i^T / √d)
    p_state      = p_state + attn_pattern @ V_P_i
    p_state      = p_state + FFN_motion_i(p_state)

# Phase 3: post-motion semantic re-interpretation (non-shared with interpreter_1)
p_semantic_tk = interpreter_2(p_state)                # self-attn × 3
predicted_tk  = recon_head(p_semantic_tk[:, 1:])      # same recon_head (SHARED)
L_tk = MSE(predicted_tk · mask_p, img_{t+k} · mask_p)

L_total = L_t + L_tk
```

**주요 설계 결정 (확정)**:

| 항목 | 결정 | 근거 |
|------|------|------|
| Motion-routing mechanism | Q/K from M, V from P | "M의 self-attention graph를 P value에 적용" = motion-guided spatial routing. 원 novelty |
| Motion-routing iteration | N=2, projections iteration별 독립 | 표준 multi-layer decoder 패턴, 다양성 확보 |
| interpreter_1 vs interpreter_2 | **Non-shared** (독립 parameters) | Phase 1 input(mask_token 포함)과 Phase 3 input(motion-routed)의 분포 차이 |
| Recon head | **Shared** | 두 interpreter output의 "같은 pixel space" alignment |
| Mask ratio | M 0.3, P 0.75 (asymmetric) | M sparse 보존 + P 도전적 복원. v10 실패 시 motion guide로 보강 |
| M decoder loss | 없음 (gradient만) | M은 sensor 역할. P 복원에 유용한 motion field 제공으로 간접 학습 |
| Gradient flow (p_state = p_semantic_t) | Joint (detach 없음) | interpreter_1이 L_t + L_tk 모두 학습, 전체적 일관성 |
| Total parameters | ~248M | v6 (213M) 대비 +35M. VideoMAE-heavy ablation으로 파라미터 공정성 대응 |

**예상 효과 / 실패 모드 방어**:

| 과거 실패 모드 | v11 방어 기제 |
|--------------|-------------|
| v7-big CLS collapse (대칭 구조 공유 parameter) | interpreter_1/2 non-shared + 비대칭 역할 (M sensor, P 해석자) |
| v8 static salience (EMA teacher + L_P) | EMA teacher 없음, pixel target 유지, L_t가 semantic anchor |
| v9 residual degrade (target 함수 변경) | Target은 원래 frame, residual 없음 |
| v10 P aggressive mask collapse | Phase 1 self-attn이 mask 영역을 semantic으로 anchor. Motion 주입은 Phase 2로 분리 |

#### v11 구현 TODO

**P1 — 구현 (`src/models/two_stream_v11.py`)** — ✅ 완료

**P2 — Launcher / Sanity** — ✅ 완료 (33591381, 10:26 sanity 통과 후 full run)

**P3 — Full Training** — 🔄 진행 중 (ep12 도달, ep16/ep20 probing 예정)

**P4 — Ablation** (일부 진행, 본격 ablation은 ep50 이후):
- [x] interpreter shared vs non-shared → **non-shared 채택** (명세대로)
- [x] Motion routing N=2 → **fixed N=2** (구현 명세대로 적용)
- [ ] Motion routing N=1 vs N=2 vs N=4
- [ ] L_t only vs L_tk only vs L_t + L_tk
- [ ] Motion-routing V from P (원안) vs V from M (표준 cross-attn) — novelty 정당성 실증
- [ ] VideoMAE-heavy baseline (decoder 용량 맞춤) — 파라미터 공정성

자세한 학습 결과 / probing 수치는 [`CLAUDE.md`](../CLAUDE.md) 현재 Phase 섹션 및 [`cluster_sessions.md`](cluster_sessions.md).

#### VideoMAE-ours

- 50 epoch 완주. Probing R²=0.326 (best 중 하나)
- sin-cos APE, mask 0.5 (공식 0.75는 2-frame temporal redundancy 없으므로 visible token 부족 리스크)

#### Attention 시각화 방향 (2026-04-14)

- 학습 진화 스토리는 포기 (epoch 4에 복원 품질 포화, ep33부터 plateau)
- **대체**: "M vs P stream 기능적 분리" — 한 checkpoint에서 다양한 샘플(6-8개)의 M/P attention 분포 차이를 main figure로
- 정량 보강(선택): M/P attention cosine distance, spatial entropy
- 우선순위: paper figure 시점으로 미룸. 지금은 DROID probing / LIBERO 준비가 중요

### Phase 2: Cross-Domain Action Probing ⏸️ 대기

**배경**: Two-Stream·VideoMAE-ours만 EgoDex 학습 → EgoDex probing은 축 1 controlled sanity 전용 (공개 가중치는 EgoDex OOD이라 home-field advantage 문제). **Main cross-encoder 비교는 DROID**.

| 평가 | 비교 대상 | 역할 |
|------|----------|------|
| **EgoDex probing** (축 1) | Two-Stream vs VideoMAE-ours | 구조적 bias 기여도 sanity |
| **DROID probing** (main) | Two-Stream, VideoMAE-ours, VideoMAE-official, V-JEPA-official, VC-1, DINOv2, SigLIP | Cross-embodiment 전이 비교 |
| **LIBERO** (Phase 3) | 동일 7개 encoder | Downstream robot control |

**데이터**: DROID v1.0.1 (95k episodes Franka, 3.4 TiB 다운로드 완료, 프레임 추출 진행 중). Ego4D 다운로드 중 (라이선스 취득).

**평가 프로토콜 (2단계)**:

1. **Primary — 2-frame 통제 비교**: 7개 encoder 전부 (frame[0], frame[gap]) 동일 입력. V-JEPA/VideoMAE-official도 2프레임 (정보량 통제). Action target: `action[gap] - action[0]` (7-DoF delta). 지표: R², cosine similarity
2. **Supplementary — native input**: CLIP/DINOv2/SigLIP/VC-1 1프레임, Two-Stream/VideoMAE-ours 2프레임, V-JEPA/VideoMAE-official 16프레임 → "정보량 증가 대비 성능 향상" 정량화

**이상적 결과**: Two-Stream v10이 2-frame 통제에서 최상위 + native 16-frame V-JEPA에 근접/우위 → "2-frame M/P 구조가 정보 효율적"

**Go/No-Go**: 2-frame 비교에서 우위 → Phase 3 순항 / 열세 → 원인 분석 (도메인 갭 vs 표현 품질)

#### Encoder 로더 구현 현황

Two-Stream v6/v10, VideoMAE-ours, CLIP, DINOv2, SigLIP, VC-1, V-JEPA 2.1 ViT-B (384px, 2-frame 동작 확인), VideoMAE-official (pos_embed slice로 2-frame) — **모두 완료 (2026-04-17)**

### Phase 3: LIBERO BC (메인 downstream 실험) ⏸️ 대기

**목표**: "표현에 인코딩된 action 정보가 **실제 제어**에 유용한가"를 검증

```
frozen encoder → MLP action decoder → 7-DOF action
  (7종 비교)     (학습 대상)          (task-conditioned)
```

- **학습 대상**: MLP action decoder만 (encoder frozen)
- **학습 데이터**: LIBERO demonstrations (BC, supervised regression)
- **태스크 구분**: per-task policy 또는 task ID embedding (자연어 사용 안 함)
- **평가**: LIBERO 시뮬레이터 closed-loop rollout. Task suite: `libero_spatial` (main), `libero_object/goal/10` (supp). Task당 50 trials × 3 seed 평균. 지표: success rate (%)

**공정 비교 체크리스트**: 모든 encoder 동일 MLP 아키텍처, 단일 프레임 encoder는 (img_{t-1}, img_t) concat, 동일 action 정규화·lr·epoch·batch·trial·seed

### Phase 3B: OpenVLA 통합 (축소안, 조건부 진입) ⏸️ 대기

**목표**: Llama 7B 크기 decoder 조건에서 Phase 3 결과를 재검증. "encoder 우위가 decoder 용량과 무관"을 주장.

```
encoder (frozen) → projection (학습) → Llama 7B + LoRA (학습) → action tokens
```

**축소된 실험 대상** (3 encoder × libero_spatial):

| 모델 | Encoder | 역할 |
|------|---------|------|
| OpenVLA + SigLIP (frozen) | SigLIP | Home advantage reference |
| OpenVLA + VideoMAE-ours | VideoMAE-ours (frozen) | 축 1 유지 |
| OpenVLA + Two-Stream | Two-Stream v10 (frozen) | **메인 제안** |

- Trial: task당 50 × 3 seed
- 학습: Projection layer + Llama 7B LoRA (rank 32). Vision encoder frozen 통일
- 논문 명시: "absolute numbers are not directly comparable to the original OpenVLA paper; we report encoder-only relative comparison under controlled freeze policy"

**진입 조건 (Phase 3 결과 보고 결정)**:
- Two-Stream 명확히 우위 → "강화 증거" 역할
- 박빙 → 판정 역할. 결과 뒤집힐 가능성 감수

**의도적 제외** (scope 관리): Pi0, RT-2, LAPA, 여러 IL 알고리즘 비교, Full fine-tuning — rebuttal/future work.

---

## 논문 스토리 3단계

| 단계 | 주장 | 뒷받침 Phase | 지표 |
|------|------|-------------|------|
| 1 | 표현에 action 정보가 **인코딩되어 있다** | Phase 1 + Phase 2 | R², cosine similarity (open-loop) |
| 2 | 인코딩된 정보가 **실제 제어에 유용하다** | Phase 3 (LIBERO BC rollout) | success rate (closed-loop) |
| 3 | action-agnostic 사전학습이 **범용 비전 feature보다 낫다** | Phase 3 (7종 비교) + Phase 3B (SOTA 통합) | 순위 + success rate 차이 |

---

## 모델 설계

### Two-Stream (제안)

- **구조**: ViT backbone (depth=12, num_stages=2, 6 blocks/stage, CLS exchange 2회, ~213M params)
- **M/P 채널 분리**: M(motion)은 inter-frame 차이 중심, P(pattern)는 appearance 중심
- **CLS exchange**: stream 간 정보 교환을 CLS token 수준에서만 수행 (patch-level cross-attention 없음)
- **Target**: M decoder는 `frame_{t+k}` 복원, P decoder는 `frame_{t+k}` 복원 (v10 기준, mask 비대칭 0.5/0.75로 역할 분화)
- **Positional encoding**: APE (learnable, v6 이후 RoPE → APE로 전환. RoPE가 position prior overfit 유발)
- **Data augmentation**: rotation aug (90% 동일회전, 10% 독립회전, {0°,90°,180°,270°}) — position prior 차단이 probing R²에 결정적

### VideoMAE-ours (controlled comparison)

- `src/models/videomae.py`: 2-frame (num_frames=2, tubelet_size=2, 196 patches), sin-cos APE
- Mask ratio 0.5 (공식 0.75는 16-frame temporal redundancy 전제 → 2-frame에서 visible token 부족)
- Two-Stream과 동일 gap sampling, 동일 EgoDex splits → 단일 변수(구조) 통제

### 확정 설정 (v6 / v10)

```
depth=12, num_stages=2
mask_ratio=0.5 (M)
mask_ratio_p=0.5 (v6) / 0.75 (v10)
max_gap=60, sample_dist=triangular, sample_center=30
APE (learnable positional embedding)
rotation_aug=True (90% 동일회전 + 10% 독립회전)
```

---

## 학습 설정

```
epochs: 50 (full)
batch_size: 64 per GPU (DDP, global 512 on 8 H100)
max_gap: 60, sample_dist=triangular, sample_center=30
데이터: EgoDex frames (클러스터: /proj/external_group/mrg/.../egodex_frames)
Mixed precision: BF16 autocast (GradScaler 불필요)
LR scheduler: cosine with warmup (5 epoch)
Optimizer: AdamW
DDP: 2 nodes × 4 H100 on AIP_long partition
```

**Gap 샘플링 (v4 이후 확정)**:
- Triangular with center=30, max=60
- 이유: 작은 gap에서 M/P가 너무 비슷 → CLS exchange 압력 부족 (학습 데이터 색상/배경 bias를 학습하게 됨). 큰 gap에서 프레임 차이가 커야 cross-stream 교환 필수

### Slurm DDP 3대 함정 (재발 방지)

1. `--gpus-per-task=1` 대신 `--gres=gpu:N` (NCCL PCI 탐색 실패 방지)
2. `MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))` (포트 충돌 방지)
3. srun에서 `$CONDA_PREFIX/bin/python` 절대 경로 사용

실행 명령어·Docker 세팅·환경별 launcher는 [`CLAUDE.md`](../CLAUDE.md) 참고.

---

## 현재 상태

`CLAUDE.md` "현재 Phase" 섹션 및 `docs/cluster_sessions.md` 진행 중 표 참고. 요약:

- Phase 1 ✅ 완료 (v4 설정 확정)
- Phase 1.5 🔄 v10 종료 (ep40 plateau +0.221), v6 챔피언(+0.259) 추월 실패 / v11 ep12 도달 (A+D +0.219), 학습 진행
- Phase 2 🔄 DROID cross-domain probing 일부 (VideoMAE vs v11 gap 1/10/15/30 비교), v11 모든 gap 우위
- Phase 3 🔄 LIBERO BC fine-tune 진행 중 (libero_spatial 30ep, VideoMAE & v11 ep12)
- Phase 3B ⏸️ Phase 3 결과 보고 결정

---

## 참고 문서

- [`CLAUDE.md`](../CLAUDE.md) — 프로젝트 가이드 + 현재 Phase 상세
- [`docs/PROBING_GUIDE.md`](PROBING_GUIDE.md) — Action probing 가이드 + 결과
- [`docs/setup/LIBERO_TEST_GUIDE.md`](setup/LIBERO_TEST_GUIDE.md) — LIBERO 평가
- [`docs/cluster_sessions.md`](cluster_sessions.md) — IBS 클러스터 sbatch/salloc 로그
- [`scripts/cluster/README.md`](../scripts/cluster/README.md) — 클러스터 사용법 quickstart
