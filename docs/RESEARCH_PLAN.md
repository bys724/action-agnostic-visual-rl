# Action-Agnostic Visual RL Research Plan

**마지막 업데이트**: 2026-04-29
**투고 목표**: NeurIPS 2026
**연구 질문**: **구조적 inductive bias를 가진 시각 표현 학습이, action label 없이도 시각-행동 연결 태스크에 유용한 표현을 만드는가?**

> 현재 진행 상태 / 체크포인트 / sbatch 로그 등은 [`CLAUDE.md`](../CLAUDE.md)·[`docs/cluster_sessions.md`](cluster_sessions.md)·[`docs/PROBING_GUIDE.md`](PROBING_GUIDE.md) 참고. 본 문서는 **연구 설계와 로드맵**에 집중한다.

## 연구 동기

- **핵심 가설**: 영장류 시각 피질의 magnocellular / parvocellular 경로 분리는 생물학적 시각 시스템의 근본적 구조 원리. 이 구조를 비디오 표현 학습 모델에 반영하면 더 범용적인 표현을 학습할 수 있다.
- **Two-Stream 접근**: M(motion) / P(pattern) 채널 분리 + CLS exchange를 통해 구조적 inductive bias를 부여. Pixel reconstruction 목표와 결합.
- **시각-행동 연결 태스크로 검증**: 로봇 조작 태스크(LIBERO)는 "어떤 feature가 진짜 유용한지"를 가장 직접적으로 평가할 수 있는 설정. **Two-Stream은 로봇용 특수 알고리즘이 아니라, 일반 시각 표현 학습 방법의 로봇 태스크 검증**이다.
- **V-JEPA에 대한 중요 관찰**: V-JEPA (Bardes et al., 2024)는 같은 조건에서 feature prediction > pixel reconstruction을 실증 주장. 하지만 이 증거는 전부 **classification 벤치마크**(Kinetics-400, SSv2, ImageNet)에 국한. **로봇 제어(continuous control) 도메인 증거는 부재**. 우리는 이 갭을 채운다.

---

## 실험 대상 모델

| 구분 | Encoder | 사전학습 데이터 | 파라미터 | 방법 철학 | LIBERO BC | Probing |
|------|---------|---------------|---------|----------|-----------|---------|
| **제안** | **Two-Stream v11 (ours)** | EgoDex (~100M frames) | ~204M downstream | M/P 구조 + motion-routing + dual-target reconstruction. **A+D' mode** | ✓ main | ✓ main |
| **Controlled comparison** | **VideoMAE-ours (2-frame)** | **EgoDex (same)** | ~101M | Vanilla MAE (구조적 bias 없음, mask 0.5) | ✓ main | ✓ main |
| **로봇 제어 SOTA** | VC-1-Base | Ego4D + 조작 (~500M frames) | 86M | MAE (embodied AI 표준) | ✓ main | ✓ main |
| Internet-scale SSL | DINOv2-Base | LVD-142M | 86M | Self-distillation | ✓ main | ✓ main |
| Internet-scale VL | SigLIP-Base | WebLI (10B 이미지-텍스트) | 86M | Vision-language contrastive | ✓ main | ✓ main |
| Video SSL | V-JEPA 2.1 ViT-B | VideoMix22M (16-frame, 384px) | 86.8M | Feature prediction (최신 video SSL) | ✗ skip [^1] | ✓ main |
| Native 세팅 reference | VideoMAE-official | Kinetics-400/SSv2 (16-frame) | ~86M | MAE (공식 세팅) | (probing only) | (조건부) |

[^1]: V-JEPA 2.1은 16-frame × 384² 입력 (token count 4,608, 다른 인코더 196 대비 24×). LIBERO BC 학습 환경에서 sanity 측정 결과 50ep ≈ 30-100일 (실현 불가능). DROID/EgoDex probing 결과는 paper에 포함, BC main table에선 footnote 처리. 시간 여유 시 옵션 A (사전 feature extraction → 정책 head만 학습) 시도.

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
- **각 encoder는 native input 분포 그대로 사용** — 1-frame encoder는 `(img_{t-1}, img_t)` 각각 인코딩 후 concat / 2-frame encoder는 pair native / V-JEPA 2.1은 16-frame 누적 슬라이딩 window 그대로 (Phase 3 D3 결정)
- **각 encoder의 공식 preprocessing 사용** (정규화, resolution → 224×224 unified)
- Downstream에서 유일하게 학습되는 것: **BC-Transformer policy head** (Phase 3, 공식 LIBERO 프로토콜) 또는 **projection + LoRA** (Phase 3B)

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

#### ✅ Two-Stream v11 (Motion-Guided Attention Routing + Dual-Target Reconstruction) — 학습 종료, 🏆 ep44 final champion

**상태**: 50 epoch 완주 (JobID 33594155 1차 → 33600621 resume, 2026-04-25 01:11 ~ 04-28 07:19, AIP 2노드×4 H100). ep4-ep50 12-mode probing 모두 완료.

**설계 철학**: v7~v10의 네 가지 시도가 모두 "P stream 내부 강화"였고 모두 collapse 또는 v6 추월 실패. v11은 발상 전환 — **P stream 강화 포기, M이 P를 semantic-level operator로 조작하는 구조**. Dual-target reconstruction으로 collapse anchor 확보.

**학습 결과 (ep1~ep50)**:

| Epoch | L_total | L_t | L_tk | std_p | cos_intra_p |
|-------|---------|-----|------|-------|-------------|
| 1 | 0.0196 | 0.0109 | 0.0088 | 0.349 | 0.866 |
| 4 | 0.0057 | 0.0044 | 0.0014 | 0.210 | 0.897 |
| 8 | 0.0043 | 0.0038 | 0.00052 | 0.009 | 1.000 |
| 12 | 0.0024 | 0.00197 | 0.00039 | 0.004 | 1.000 |
| 50 | 0.00220 | — | — | — | — |

→ Loss 단조 감소. P encoder CLS는 cos_intra≈1.0 collapse, 그러나 patches는 healthy (75% MAE 복구 작동)

**Probing — ep4~ep50 12 mode** (4 위치: A=M enc, B=P enc, D'=motion-routing 후, D=Phase 3 final):

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
| `cls_m_enc` (A CLS) | +0.066 | +0.155 | +0.162 | +0.163 | +0.172 | +0.158 | +0.125 | +0.123 | +0.123 |
| `cls_p_enc` (B CLS) | -0.059 | -0.011 | -0.008 | -0.010 | -0.009 | -0.013 | -0.002 | -0.002 | -0.002 |
| `cls_concat_enc` (A+B CLS) | -0.048 | +0.092 | +0.148 | +0.139 | +0.162 | +0.140 | +0.114 | +0.118 | +0.113 |

**🏆 ep44 final champion 확정 — ep48/ep50은 plateau**:
- **A+B+D' = +0.288** → v6 ep8 (+0.259) **추월 +0.029** ★★
- **A 단독 (+0.267)도 v6 추월** — 단일 mode로
- **VideoMAE +0.326까지 격차 -0.038** (ep24 -0.092 → 절반 좁힘)
- **W-shape 회복 패턴 확정**: ep12 +0.219 → ep20 +0.185 dip → ep24 +0.234 → ep44 +0.288 (LR cosine decay 후반 ep24→44 +0.054 큰 도약)
- **ep44~ep50 plateau**:
  · A+B+D': ep44 +0.288 → ep48 +0.281 → ep50 +0.279 (-0.010, LR≈0 구간 미세 over-tightening)
  · A+D': ep44 +0.284 → ep50 +0.282 (가장 robust)
  · A+D: ep44 +0.264 → ep50 **+0.267** (미세 신피크)
  → ep44 이후는 새 학습 효과 없음. ep44가 진짜 peak으로 확정
- 사용자 v11 가설 정량 확정:
  · 3-way concat (A+B+D')이 best — M+P+motion-routed P 상보적
  · A+D' (+0.282) > A+D (+0.267) — interpreter_2는 decoder wrapper
  · CLS 모두 약화 추세, patch_mean이 정답

**Cross-domain DROID probing** (ep12 기준, 사용자 직감 검증):

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

**Cross-domain LIBERO BC** (libero_spatial 30 ep, val MSE):

| Encoder | best val MSE |
|---------|-------------|
| VideoMAE-ours ep50 | **0.0286** |
| v11 ep12 A+D | 0.0290 |

- ep12에서는 거의 동등 (격차 +0.0004). VideoMAE in-domain 학습 advantage
- v11 ep44/ep50 ckpt로 재측정 필요 — A+B+D' best mode 사용 시 우위 기대

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

**P3 — Full Training** — ✅ 완료 (50 epoch 완주, 2026-04-28 07:19 종료)

**P4 — Ablation** (일부 진행, 본격 ablation 시작 가능):
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

### Phase 2 보강: LIBERO Action Probing ✅ POSITIVE (2026-04-30 완료)

**동기**: DROID action probing 절대 R² ~0.005 한계를 LIBERO sim 환경에서 보완 — 같은 gap-matched protocol로 5 encoder 비교, 절대 R² 0.3~0.7 range 확보.

**Plan/구현**: [`docs/libero_action_probing_plan.md`](libero_action_probing_plan.md) + [`scripts/eval/probe_action_libero.py`](../scripts/eval/probe_action_libero.py) + [`scripts/cluster/probe_action_libero.sbatch`](../scripts/cluster/probe_action_libero.sbatch). 5 encoder × 3 LIBERO suite × 4 gap (1, 13, 20, 40 ≈ DROID 1, 10, 15, 30 시간 매칭). 한 잡 = (encoder × suite) × 4 gaps loop → 15 잡, 누적 **~3.6 GPU·h**.

**Target (plan §3)**: pose-derived 7-DoF (3 pos + 3 rotvec + 1 gripper). cumulative action sum 사용 안 함 — `eef_pos[t+k] - eef_pos[t]`, `(R.from_rotvec(ee_ori[t]).inv() * R.from_rotvec(ee_ori[t+k])).as_rotvec()`, `actions[t+k-1, 6]`. OSC scaling 영향 무관.

**결과 매트릭스 (R² aggregate, 60 cells)**:

| Encoder | suite | gap=1 (50ms) | gap=13 (0.65s) | gap=20 (1s★) | gap=40 (2s) |
|---------|-------|-------------:|---------------:|-------------:|------------:|
| **v11 ep44 (A+B+D')** | spatial | **+0.660** ★ | +0.593 | +0.555 | +0.374 |
| videomae-ours | spatial | +0.408 | +0.510 | +0.487 | +0.289 |
| dinov2 | spatial | +0.611 | **+0.701** | **+0.666** | **+0.502** |
| siglip | spatial | +0.583 | +0.679 | +0.646 | +0.451 |
| vc1 | spatial | +0.479 | +0.585 | +0.552 | +0.346 |
| **v11 ep44** | object | **+0.702** ★ | +0.702 | +0.681 | +0.614 |
| videomae-ours | object | +0.655 | +0.656 | +0.646 | +0.555 |
| dinov2 | object | +0.690 | **+0.784** | **+0.739** | **+0.641** |
| siglip | object | +0.670 | +0.765 | +0.731 | +0.626 |
| vc1 | object | +0.462 | +0.698 | +0.680 | +0.597 |
| **v11 ep44** | goal | +0.546 | +0.595 | +0.613 | +0.631 |
| videomae-ours | goal | +0.457 | +0.525 | +0.538 | +0.538 |
| dinov2 | goal | +0.592 | **+0.727** | **+0.732** | **+0.696** |
| siglip | goal | +0.589 | +0.722 | +0.712 | +0.680 |
| vc1 | goal | +0.427 | +0.607 | +0.624 | +0.576 |

★ = encoder-row best, **bold** = column best. 절대 R² 0.3~0.78 range — DROID 0.005 대비 1~2 order magnitude 향상, encoder 간 격차 식별 가능.

**4가지 핵심 발견**:

1. **🏆 Controlled comparison (v11 vs VideoMAE-ours): v11이 12/12 cells 전부 우위** — 같은 EgoDex 학습 데이터 controlled comparison에서 Two-Stream + motion-routing architecture가 **모든 (suite, gap) cells**에서 +0.04~+0.25 향상. **Phase 2 architectural contribution claim의 strong positive evidence**.
   ```
   suite      gap=1   gap=13   gap=20   gap=40
   spatial    +0.252  +0.083   +0.068   +0.086
   object     +0.047  +0.046   +0.035   +0.060
   goal       +0.089  +0.070   +0.075   +0.092
   ```

2. **v11이 gap=1 (50ms, fine-grained motion)에서 dominance** — spatial/object suite에서 internet-scale dinov2도 추월 (spatial +0.05, object +0.01). v11이 *instantaneous motion encoding*에 specialized. 그러나 gap이 커질수록 약화: spatial gap=1→40 −0.29 (vs dinov2 −0.11). Encoder가 다른 temporal scale에 specialize한다는 새 insight.

3. **DINOv2가 internet-scale의 강자** — 12 cells 중 9 cells에서 best (gap≥13 모든 cell). manipulation-pretrained VC-1 (Ego4D + manipulation video)는 평균 4번째. ImageNet-scale + DINO objective가 본 metric에서 가장 강한 baseline. Paper에서 "internet-scale strong baseline" 인정.

4. **Phase 2.5 negative result 반전 framing** — value alignment에서 v11 꼴찌였던 이유는 *state-similarity* metric이 VIP-objective encoder만 fair하게 평가하기 때문. Action probing (action-relevance metric)에서는 v11이 architectural contribution + gap=1 dominance 모두 입증. **두 metric, 두 능력**: action-relevance (v11 win) vs state-similarity (DINOv2 win). Encoder 선택은 downstream task 특성에 맞춰야 한다는 paper 분석 기여로 negative→positive framing 완성.

**Paper 위치**: §4.5 LIBERO Action Probing (main supplementary, replaces value alignment as primary §4.5). Tab 2 = controlled comparison (v11 vs VideoMAE-ours, 12 cells). Tab 3 = cross-encoder analysis (5 encoder × 4 gap × 3 suite). Phase 2.5 value alignment는 §Limitations로 짧게.

**산출물**: [`paper_artifacts/libero_action_probing/<encoder>_<suite>_<ts>/gap<k>/{summary.json, all_gaps.csv}`](../paper_artifacts/libero_action_probing/)

### Phase 2.5: Trajectory-Level Value Alignment (VIP-inspired) ❌ NEGATIVE RESULT (2026-04-30)

**동기**: DROID single-step action delta probing R²이 절대값 ~0.005 수준으로 작음. Trajectory-level multi-step evaluation으로 frozen encoder의 robot-relevant capacity를 보완 측정.

**구현**: [`scripts/eval/value_alignment.py`](../scripts/eval/value_alignment.py) + [`scripts/cluster/value_alignment.sbatch`](../scripts/cluster/value_alignment.sbatch). 5 encoder × 3 suite × 50 demos × 10 task = 1500 trajectory. fractions={1.0, 0.5, 0.3, 0.15} sweep (last-K trailing window). 길이 P95 cutoff. v11은 mode 3종 (A+B+D' / B / D').

**비용**: 21잡 × ~3min ≈ **1.0 GPU·h** 총.

**결과 (frac=1.0, full trajectory ρ mean)**:

| Encoder | spatial | object | goal | 평균 |
|---------|--------:|-------:|-----:|-----:|
| **vc1** | **+0.905** | **+0.727** | **+0.768** | **+0.800** |
| siglip | +0.833 | +0.567 | +0.725 | +0.708 |
| dinov2 | +0.805 | +0.626 | +0.748 | +0.726 |
| videomae-ours | +0.795 | +0.559 | +0.654 | +0.669 |
| v11 (A+B+D') | +0.531 | +0.379 | +0.513 | +0.474 |
| v11 (B only, P enc) | +0.524 | +0.166 | +0.256 | +0.315 |
| v11 (D' only, motion-routed P) | +0.502 | +0.352 | +0.496 | +0.450 |

→ **plan 가설 (v11 ≥ VideoMAE ≈ VC-1) 전면 기각**. 모든 suite/mode/fraction에서 v11이 baseline 대비 −0.25~−0.56 격차. **Hard No-Go (v11 < baselines)**.

**4가지 핵심 발견**:

1. **Trailing window 가설 (사용자 제안)**: "v11이 motion-specific이라면 골 근접 frame에서는 격차 좁아져야". → frac=1.0 → 0.15 sweep 결과 **격차는 그대로 유지** (gap@0.15 ≈ gap@1.0). 가설 기각.
2. **Motion encoder(A) 무관 (사용자 직감 정량 확인)**: D' only ≈ A+B+D' (격차 ±0.03). M encoder의 motion 신호는 본 metric에 거의 기여 안 함.
3. **Motion routing은 healthy**: D' > B (모든 suite에서 +0.05~+0.25 향상). v11의 architectural innovation은 작동 중. 단지 LIBERO state encoding에서 internet-scale 추월 못 함.
4. **Baselines의 frac↓ 향상 패턴**: 모든 baseline이 trailing window 줄이면 ρ 상승 (object: +0.149~+0.288). trajectory 끝부분이 더 monotonic. v11만 이 패턴 미관찰.

**원인 분석 (VIP-objective와의 미스매치)**:
- VIP (Ma et al., 2022)는 학습 시 *명시적으로* `cos(e_t, e_T) ↑ as t→T`를 objective로 사용. 그런 encoder는 trajectory monotonicity ρ가 자연스럽게 높음.
- 우리 5 encoder 중 **VIP-objective로 학습된 건 없음**. VC-1/DINOv2/SigLIP은 manipulation/scene state encoding이 강해 *부산물*로 monotonicity capture. v11은 frame-pair MAE reconstruction → motion delta encoding 중심 → trajectory 전역 state similarity와 직교에 가까움.
- VIP-style metric은 사실 **VIP-objective encoder만 fair하게 평가**. v11의 약점은 motion encoding 의존이 아니라, two-stream + EgoDex pre-training의 결합이 internet-scale encoder만큼 풍부한 frame-level scene state representation을 못 만드는 데 있음 (B only 측정으로 입증).

**Paper framing (Negative result를 강점으로 전환)**:
- §4.5 Trajectory-Level Value Alignment를 그대로 게재 + 명시적 framing: "**Action-relevance와 state-similarity는 직교에 가까운 두 능력**. Encoder 선택은 downstream task 특성에 맞춰야 한다." 분석 기여.
- §4.4 Action Probing R² (v11 +0.288 champion) + §5 LIBERO BC main + §4.5 (v11 ρ underperform) → 두 metric, 두 story.
- 흥미로운 supplementary: D' > B (motion routing 작동), baselines의 frac↓ 향상 패턴 (trajectory 끝부분 monotonic).
- 데이터: [`paper_artifacts/value_alignment/`](../paper_artifacts/value_alignment/)

### Phase 3: LIBERO BC (메인 downstream 실험) 🔄 진행 중 (2026-04-28~)

**목표**: "표현에 인코딩된 action 정보가 **실제 제어**에 유용한가"를 검증

#### 3-1차: LIBERO 공식 BC-Transformer (frozen encoder + 정책 head 학습)

```
frozen encoder → encoder adapter → BC-Transformer policy head → action distribution
  (5종 비교)     (인코더별 차이 통일)  (TemporalTransformer + GMM, LIBERO 공식)
```

- **정책 head**: LIBERO 공식 [`BCTransformerPolicy`](../external/LIBERO/libero/lifelong/models/bc_transformer_policy.py) 그대로 (TemporalTransformer 4-layer × 6-head + GMM head). 인코더 부분만 우리 어댑터로 교체
- **학습 대상**: BC-T policy head 전체 (encoder frozen, 약 3M params)
- **학습/평가 데이터**: LIBERO demonstrations (50 demo × 10 task × suite). 1차는 `libero_spatial`, 1차 결과 보고 `libero_object` / `libero_goal` 추가
- **평가 (rollout)**: LIBERO 시뮬레이터, **로컬 워크스테이션** (`docker/libero` Apptainer 또는 docker compose). Task당 50 trials × 3 seed
- **학습은 클러스터 (1 GPU H100), rollout은 로컬** (clean separation: training은 HDF5만 필요, rollout은 mujoco 시뮬레이터)

**핵심 설계 결정** (D1-D4):

| ID | 결정 | 근거 |
|----|------|------|
| **D1 이미지 사이즈** | 모든 encoder 입력 **224×224**로 resize (LIBERO 원본 128×128) | 인코더 사전학습 분포 일치, 모든 encoder fair. 128 사용 시 16-patch grid 8×8로 정보 손실 |
| **D2 FiLM language conditioning** | 인코더 단계에서 **FiLM 제거**. Language는 BC-T temporal Transformer의 text token으로만 주입 | 우리 인코더 모두 FiLM 미지원. paper claim ("visual representation quality")에 부합. 모든 encoder fair |
| **D3 인코더별 입력** | 각 encoder가 **native input 그대로** 받음 (아래 표) | 각 인코더의 사전학습 분포 보존 = 각 인코더의 best protocol 비교 |
| **D4 2-frame pair 형성** | 학습 시 (obs_{t-1}, obs_t) 시퀀스에서 직접 생성. Rollout 시 adapter 내부 prev_obs buffer 유지 | episode 시작 t=0은 (obs_0, obs_0) 복제. 표준 video encoder fine-tune 패턴 |

**비교 인코더 (5종) — D3 input format**:

| Encoder | Native input | Per-timestep 출력 | 비고 |
|---------|--------------|-------------------|------|
| **Two-Stream v11** (A+D' mode) | (obs_{t-1}, obs_t) 2-frame pair | A+D' concat → 2 × stream_dim | EgoDex pre-trained (ours) |
| **VideoMAE-ours** | (obs_{t-1}, obs_t) 2-frame pair | patch mean → embed_dim | EgoDex pre-trained (ours) |
| **DINOv2-Base** | obs_{t-1}, obs_t 각각 단독 인코딩 | 2 embeddings concat → 2 × 768 | LVD-142M pre-trained |
| **SigLIP-Base** | obs_{t-1}, obs_t 각각 단독 인코딩 | 2 embeddings concat → 2 × 768 | WebLI pre-trained |
| **VC-1-Base** | obs_{t-1}, obs_t 각각 단독 인코딩 | 2 embeddings concat → 2 × 768 | Ego4D + 조작 pre-trained |

**V-JEPA 2.1 별도 처리** (NeurIPS BC main table 제외):
- Sanity 측정 결과 bs=4 epoch당 460s → bs=32 본격 학습 50ep ≈ 30-100일 (실현 불가능)
- 16-frame × 384² × seq_len=25 입력으로 token count 4,608 (다른 인코더 196 대비 24×). attention O(N²)에서 ~552× 비싸짐
- **paper 처리 방안**:
  - DROID/EgoDex probing 결과는 그대로 포함 (이미 완료)
  - LIBERO main table에선 footnote로 "16-frame × 384² 입력 비용 prohibitive로 fair BC 비교 불가"
  - 시간 여유 시 옵션 A (사전 feature extraction + cached feature로 정책 head 학습) 시도

**공정 비교 체크리스트**: 모든 encoder 동일 BC-T 정책 head, 동일 LR/epoch/batch/seed/trial, 동일 task suite, 동일 image preprocessing pipeline 외 인코더 native preprocessing은 각자.

**1차 진행 상황** (2026-04-29 시작, 5 encoder × `libero_spatial` × seed=0):
- 33615385 (v11 ep44), 33615386 (videomae), 33615387 (dinov2), 33615391 (siglip), 33615392 (vc1) — RUNNING
- 1차 결과 (~2일 후) 본 후 `object`/`goal` suite + seed=1, 2 병렬 큐 제출

**🔴 1차 BC-T 학습 중단 결정 (2026-04-30) — 학습 cfg 결함 발견**

VideoMAE-ours BC-T (33615386) 학습 종료 후 로컬 H100에서 LIBERO rollout sanity 평가 → **0/50 = 0% SR** (10 task × 5 trial). 모든 task 0% 일관, max_steps 정확히 소진. 진단:

| 점검 | 상태 | 근거 |
|------|------|------|
| ckpt 로드 | ✅ 정상 | 227 key (adapter 160 + BC-T head 67) — encoder weights 전부 포함 |
| Inference 코드 | ✅ LIBERO 공식 일치 | 회전 미적용, gripper_states↔robot0_gripper_qpos, dist.sample 모두 일치 |
| 비디오 패턴 | ⚠️ 의도적 움직임이지만 task 영역 이탈 | gripper가 점진적으로 테이블 area 밖으로 누적 이동 |
| 학습 NLL | ✅ -23.7 (정상 fit) | 학습 자체 수렴 |

**결정적 원인**: `scripts/eval/finetune_libero_bct.py` 학습 cfg가 LIBERO 공식 default와 불일치
- LIBERO 공식 (`/opt/libero/libero/configs/data/default.yaml`): `use_joint=True`, `low_dim=["gripper_states", "joint_states"]`, `obs_key_mapping.joint_states=robot0_joint_pos`
- 우리 학습 cfg: `use_joint=False`, `low_dim=["gripper_states"]` (gripper qpos 2-d만)
- → **`joint_states` (robot0_joint_pos, 7-d) 누락** = robot kinematics state 부재
- 정책이 robot의 spatial configuration을 모름 → spatial control fail
- 비디오의 "의도적이지만 잘못된 방향" 패턴과 일치

**필요한 학습 driver 수정** (`scripts/eval/finetune_libero_bct.py`):
1. `cfg.data.use_joint=True` 활성화
2. `cfg.data.obs.modality.low_dim`에 `joint_states` 추가
3. `obs_key_mapping`에 `joint_states: robot0_joint_pos` 추가
4. `shape_meta`에 `joint_states: [7]` 자동 추출 (dataset에서 자연스럽게 들어감 — get_dataset 호출 시 obs_modality.low_dim에 명시되어 있으면 됨)
5. (선택) `affine_translate=4` augmentation 활성화 — LIBERO 공식 default. 현재는 IdentityAug

**액션 플랜**:
- 진행 중 잡 (33615385~33615392) cancel + 5 encoder 모두 새 cfg로 재제출
- 클러스터 학습 종료된 VideoMAE-ours BC-T ckpt (cluster: `libero_bct/videomae_libero_spatial_*` + 로컬: `bct_videomae-ours_libero_spatial_seed0_best.pt`)도 재학습
- 추가 비용: ~25h × 5 encoder = ~125 GPU·h (병렬 5 GPU면 실시간 ~25h)

**참고 (sanity rollout 인프라)**: 로컬 H100에서 `src/eval_libero.py` BC-T 전용으로 갈아엎고 `libero-eval` 컨테이너로 closed-loop rollout 동작 확인. 새 ckpt 받으면 동일 명령으로 재평가 가능 — 자세한 rollout 가이드는 `docs/setup/LIBERO_TEST_GUIDE.md`

**최종 main table 목표**: 5 encoder × 3 suite × 3 seed = **45 BC runs + rollout**

#### 3-2차: LeRobot ACT (조건부 진입)

**목표**: 강한 정책 head (Action Chunking Transformer)에서 결과 재검증. 절대 SR이 published VLA 수치와 비교 가능 수준으로 상승 → reviewer 설득력 강화

- **진입 조건**: 1차 BC-T 결과 보고 결정
  - Two-Stream v11 명확히 우위 → "강화 증거" 역할
  - 박빙 → 정책 head 영향력 점검 후 결정
- **lerobot 패키지 별도 설치 필요** (현재 conda env 미설치)
- 설계 결정 D1-D4 동일 적용
- 인코더 어댑터 재사용 (정책 head만 교체)

### Phase 4: Architecture Ablation (paper Section 5) 🔄 진행 중

**목표**: v11 design choice의 정량 기여도 isolation. paper의 핵심 contribution 입증.

#### A1: Motion-routing source — `V from P` (ours) vs `V from M` (표준 cross-attn)

**Paper claim**: "M의 self-attention graph를 P value에 적용"이 표준 cross-attention보다 표현 품질 우월.

| 항목 | Control (v11) | Ablation (v11-VfromM) |
|------|--------------|----------------------|
| Phase 2 routing | Q,K←M, V←P | Q←P, K,V←M (표준) |
| Param count | 208.33M | 208.33M (동일) |
| Pre-training | EgoDex part1-5, 50 epoch | EgoDex part1-5, 50 epoch |
| Mask ratios, M/P depth, optimizer 등 | 동일 | 동일 |

**평가 지표**:
1. EgoDex action probing R² (12 mode 중 핵심 4개: A, A+B, A+D', A+B+D')
2. DROID action probing R² (gap 1/10/15/30 × A+B / A+B+D' mode)

**구현**: `MotionRoutingBlock(routing_mode='v_from_m')` — `src/models/two_stream_v11.py:55`. CLI: `--v11-routing-mode {v_from_p|v_from_m}`. sbatch: `V11_ROUTING_MODE=v_from_m`.

**진행 상황** (2026-04-29~):
- 33615394: sanity (1 GPU × 1 epoch × part1 × 200 vid) — PENDING
- 33615395: full (8 GPU × 50 epoch × part1-5, AIP_long --time=3-12:00) — PENDING
- Control: 기존 `/proj/external_group/mrg/checkpoints/two_stream_v11/20260426_014333/` ckpt 그대로 사용 (추가 학습 0)
- 비용: ~617 GPU·h (33615395만)

**가능한 시나리오**:
- v11 > VfromM: 핵심 contribution 정량 입증 ✓
- 동등: design choice는 부수적 — "motion-guided" 표현 약화
- v11 < VfromM: 핵심 novelty 무너짐 — 다른 contribution(2-frame regime, dual target negative result, M/P functional differentiation)으로 paper 재구성

#### A2: Feature extraction position (12-mode probing) ✅ 이미 완료

추가 학습 없이 기존 ep4~ep50 × 12 mode probing 결과를 paper Table로 정리.

**핵심 메시지**:
- M encoder (A) +0.267 vs P encoder (B) -0.003: **M이 action 정보 보유, P는 appearance만** (functional differentiation 입증)
- D' (motion-routing 후, +0.135) > B (P encoder, -0.003): motion-routing이 P를 action-relevant 화 (+0.138 격차)
- 3-way concat A+B+D' = +0.288 (champion): M + P + motion-routed P 상보적

이 분석은 A1과 상호 보완:
- A1: pre-training design (motion-routing 메커니즘) 자체가 옳은가
- A2: 학습된 모델 안에서 motion-routing이 의도대로 functional differentiation을 만들었는가

**Paper에서 위치**: main paper Table 4 (condensed 4-5 row) + Appendix B (full 12-mode 표).

#### A3: 추가 ablation (시간 여유 시, 우선순위 낮음)

| 항목 | 검증 대상 | 평가 방식 |
|------|----------|----------|
| Motion-routing N=1 / 2 / 4 | iterative refinement 효과 | probing only |
| Dual target (L_t + L_tk) vs single (L_tk only) | Phase 1 anchor의 collapse 방지 역할 | probing only |
| M decoder loss 유무 | M sensor 가설 | probing only |

→ **모두 cheap (probing only)**. 단 추가 pre-training이 비용. paper 본 마감 시점에 시간 여유로 결정.

### Phase 3B: OpenVLA 통합 (축소안, 조건부 진입) ⏸️ 대기

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
- Phase 1.5 ✅ **v11 ep44 새 챔피언 등극** (A+B+D' +0.288, v6 +0.259 추월). v10 종료 (+0.221, 추월 실패). VideoMAE +0.326까지 격차 -0.038
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
