# Factorization Crossover — 이중분리 게이트 계획 (2026-07-02)

> **상태**: **Phase A 실행됨(2026-07-02, directional 지지)** · Phase B 강등 · Phase C(STEP 1 인과) 규율 확정. 이 문서는 **계획·판정 규율**(구현·config·집계는 dev 세션, 결과 로그는 `cluster_sessions.md`).
> **진행 요약**:
> - **Phase A (✅ 실행)**: libero_object 2×2. de-confound(rot+trans) 후 identity **P_t 0.851 vs M 0.278**(chance 0.10) = directional 이중분리. BUT arena가 위치↔identity↔motion 얽혀 **단일 arena clean 2×2 미달**(문제셀 P-motion 0.547). 상세 `cluster_sessions.md`(2026-07-02 Factorization crossover Phase A).
> - **🚫 EgoDex clean arena 조사 = 불가**: repo의 EgoDex는 hand-pose HDF5(transforms/confidences)만 추출 → object/scene identity label 부재, task=폴더명이나 motion-confounded, video-id는 k-NN proxy뿐. **EgoDex 단일 arena clean 2×2는 신규 annotation 필요**(값싼 경로 아님). 재조사 불필요.
> - **Phase B(readout-free) 강등**: 남은 gap = arena(데이터 얽힘)지 readout 아님 → k-NN도 이 confound 못 고침(별개 문제).
> - **다음**: ① LIBERO 통계적 de-confound(위치 partial-out / same-position-diff-object 서브셋) **또는** ② triangulation 수용 + **STEP 1 인과**(§4 Phase C 규율 준수).
> **결정 출처**: Obsidian Vault `Projects/Action-Agnostic Paper/2. Experiments.md §4 남은 게이트 A` / `README.md §다음 수` / `History.md`(2026-07-02).
> **관련 dev docs**: [`restart_plan.md`](restart_plan.md) §3.3(cross-leakage TODO), [`comp_mae_plan.md`](comp_mae_plan.md) §6(dissociation probe·§6.1 M 배포 무효), [`eval_protocols.md`](eval_protocols.md), [`PROBING_GUIDE.md`](PROBING_GUIDE.md).

---

## 1. 배경 — 왜 이 작업인가

- STEP 0.5 de-confound 후 **3a(slope/dissociation) 폐기**(regression-to-ceiling), **3b(efficiency) 확정**. 논문 무게중심을 **주장 1(factorization)**으로 재확정 — 3b·2a는 그 아래 증거.
- factorization = "M stream은 dynamics(움직임), P stream은 form(외형)을 구조적으로 분리해 담는다"는 **본 논문의 핵심 미입증 가설**.
- 이를 실험 명제로 번역하면 **2×2 이중분리(double dissociation / crossover)**로 수렴:

| readout \ stream | **M-stream** | **P-stream** |
|---|---|---|
| **motion**(pose Δ 회귀 R²) | 높음 (기대) | 낮음 (기대) |
| **identity**(object/scene 분류 acc) | chance (기대) | 높음 (기대) |

- **대각선 crossover**(M-motion 高 ↔ P-identity 高, off-diagonal 低)가 서야 factorization 성립.

## 2. 왜 이건 3a(slope)처럼 안 죽나 — 설계 규율 (🔴 최우선 가드)

- 3a는 **main effect**(in-domain 0.47 천장 vs OOD 0.85, 타깃 난이도 비대칭)에 지배당해 폐기됐다.
- crossover는 **interaction**이라 stream별 전체 정보량(capacity) 차이와 target 난이도 main-effect를 상쇄한다:
  - M이 그냥 정보가 적어 identity를 못 맞히는 것이면 → P도 motion을 비슷하게 못 맞혀야 함.
  - 그런데 P는 **motion만 특히** 못 맞히면(P_t 단독 이미 R²=−0.009) capacity가 아니라 *내용의 분리*.
- 🔴 **판정 조건 = 진짜 interaction**: "M이 두 readout 다 낮고 P가 두 readout 다 높음"인 **독립 main effect 2개**면 slope를 죽인 confound의 재판이다. **대각선(off-diagonal 열위)** 이 확인돼야 crossover로 인정. interaction 유의성(예: 2-way ANOVA interaction term, 또는 stream×readout 상호작용 부호)까지 리포팅.

## 3. 손에 있는 것 / 없는 것

- **있음 (motion 열)**: CoMP-MAE-S에서 이미 측정 — P_t 단독 **−0.009**(P는 motion 거의 0) / M **+0.094(mean)→+0.239(attentive)** / 3b efficiency 표 전체(`paper_artifacts/tables/step0_ood_efficiency/`). → crossover의 motion 열은 채워졌고, **P가 motion을 못 담는다**는 절반이 이미 확인됨.
- **없음 (identity/appearance 열)**: 🔴 **유일한 실질 블로커.** M·P 각각을 identity로 재는 label·프로토콜이 필요. EgoDex엔 깨끗한 object identity label이 애매.

## 4. 실행 순서 (저비용 → 깔끔, 게이트 구조)

### Phase A — 저비용 probe-based crossover (지금, 학습 0)
identity 축을 값싸게 붙여 crossover **신호 유무**부터 확인. 통과 시에만 Phase B 투자.

- **identity 축 후보** (dev 세션에서 택1 또는 병행):
  1. **LIBERO object·task 분류 probe**: frozen feature(stream별) → linear classifier로 "이 프레임이 어느 object/task인가". LIBERO suite는 object 구성이 명확 → label 확보 쉬움. appearance 모델(DINOv2/SigLIP)이 여기선 이겨야 정상(sanity).
  2. **CALVIN scene/task 분류**: 동일 발상, CALVIN 환경 label 사용.
- **motion 축**: 기존 `probe_action.py` per-dim R²(이미 산출) 재사용 — 재측정 불필요, Phase A는 identity 축만 신규.
- 두 축을 **동일 frozen feature·동일 readout(attentive uniform)·동일 split**에서 뽑아 2×2 채움.

### Phase B — readout-free crossover (Phase A 통과 시, 배선 필요)
학습된 probe 없이 crossover를 재현 = concat-probe artifact 면역(논문용 bulletproof 버전).
- **identity 축 (readout-free)**: scene/instance **k-NN retrieval** — stream feature 최근접이 same-object/same-scene을 얼마나 회수하나. P 높음 / M chance 기대.
- **motion 축 (readout-free)**: **correspondence label-propagation**(DAVIS/JHMDB, restart_plan §3.3) 또는 temporal-order/direction 분류. M 높음 / P 낮음 기대.
- 두 축 다 학습 probe 없음 → 3a·concat artifact 취약성 원천 회피.

### Phase C — 인과 (STEP 1, 비쌈 ~110 GPU·h)
Phase A/B는 상관("표현이 factored 되어 있다")까지. 인과("M-recon 설계 때문에")는 재학습 필요. ablation 학습본(no-M · routing on/off · V-source `V_M` vs `V_P`, comp_mae_plan §6)에서 factorization이 유지/소멸하는지 본다. STEP 1 arm과 공유.

#### 🔴 STEP 1 인과 판정 규율 (반드시 준수 — raw 오독 방지)

> 배경: **no-M은 mechanism만 제거하는 게 아니라 task 난이도까지 바꾼다.** CoMP의 P-recon은 M-full 라우팅으로 "미래에 무엇이 어디서 움직이는지"(실현된 ΔL)를 받아 예측이 쉬워진다(leakage 가드, comp_mae_plan §CoMP·"ΔL raw→copy trivial"). M을 빼면 P는 그 정보 없이 masked 프레임만으로 예측 → **VideoMAE급 이상으로 가혹.** ∴ no-M의 raw 하락은 *factorization 손실*인지 *task 난이도*인지 **안 갈린다.**

1. **raw 성능이 아니라 crossover *signature*로 읽어라.** ablation 모델의 recon/BC/probe **절대 R²·SR 하락을 factorization 증거로 쓰지 말 것**(난이도 confound 오염). 판정 = "**ablation 후 이중분리(crossover) 패턴이 유지되나 무너지나**". 유지 → M-recon이 원인 아님 / 무너져 entangle → M-recon이 factorization의 원인.
2. **왜 난이도-robust한가**: crossover는 *interaction*(칸 사이 상대 패턴)이라 난이도가 전체를 같이 올리/내려도 상대 패턴은 유지. = **slope(main effect, 폐기) vs crossover(interaction, 생존)** 와 동일 논리(§2).
3. **🔴 same-probe 일관성 (하드 가드 — 이거 깨지면 위 상쇄 효과 소멸)**: 모든 STEP 1 ablation 모델(CoMP · no-M · routing · V-source)을 **Phase A와 완전히 동일한** arena(libero_object)·de-confound aug(rot+trans)·readout(attentive)·probe protocol로 측정. **동일한 (dirty) probe를 공유하기 때문에 arena confound가 CoMP↔ablation 상대 비교에서 상쇄**된다. 인과 증거 = 절대값 아니라 **모델 간 signature 차이(Δ)**. arena·aug·readout을 조금이라도 바꾸면 비교 불가.
4. **ablation별 signature 읽는 법**:
   - **V-source (M-recon `V_M` vs `V_P`)** = **주력 인과 arm**(P task 난이도 덜 건드림). "M이 진짜 motion 배우나 vs no-op 통과냐". comp_mae_plan §6 게이트.
   - **routing on/off** = P 난이도 건드림 → **raw 금지, signature로만**.
   - **no-M (single-stream)** = "M vs P 2×2" 불가 → "**한 stream 안에서 motion·identity가 분리되나 뒤섞이나**(separability)"로 읽고, **난이도-매칭 참조(VideoMAE·Image MAE matched)와 삼각측량.** raw headline 절대 금지.

## 5. Critical guards (구현 시 실수 방지)

- 🔴 **진짜 interaction 확인** (§2) — 독립 main effect 2개 금지. 대각선 crossover + interaction 항 리포팅.
- 🔴 **baseline 공정성**: 모든 인코더(DINOv2/SigLIP/VC-1/VideoMAE/CoMP-MAE-S) **동일 프로토콜·frozen·동일 readout**. identity probe도 동일 classifier capacity.
- 🔴 **chance level 명시**: identity 분류의 chance(=1/n_class)를 표에 병기 — "M이 chance 근처"의 기준.
- 🟠 **stream 분리 정확성**: M-stream readout이 실제 M encoder 출력인지(P leakage 없음), `encode_batch_tokens` stream 인덱싱 확인(comp_mae 조합 probing과 동일 경로).
- 🟠 **identity label 누수 방지**: object 분류가 배경·carpet 같은 spurious cue로 풀리지 않게(특히 LIBERO 배경 고정) — 가능하면 object-centric crop 또는 여러 suite 교차.
- 🟠 **M 배포 무효와 무관**(comp_mae_plan §6.1): 이건 frozen feature **측정**이지 rollout 입력 아님. P+M causal-confusion 붕괴는 여기 적용 안 됨.
- 🟢 **readout = attentive uniform**: mean이 M motion under-read(수확 ①) → 모든 stream attentive pooling 일관 적용.
- 🔴 **STEP 1 same-probe 일관성** (§4 Phase C 규율): ablation 인과 판정은 **Phase A와 동일 arena·aug·readout 공유**가 전제. 프로토콜 바꾸면 arena-confound 상쇄가 깨져 raw 오독으로 회귀. ablation은 **raw 아닌 crossover signature 차이(Δ)**로만.

## 6. Pseudocode 골격 (probe 집계 — 신규 학습 없음)

```
# 입력: CoMP-MAE-S frozen encoder + (LIBERO/CALVIN) frame + object/task label
# motion 열: 기존 probe_action.py per-dim R² 재사용 (M, P_t 각각)
# identity 열 (신규): stream feature → linear classifier → top-1 acc

for stream in [M, P]:                      # stream별 frozen feature (attentive)
    feat = encode_stream(encoder, frames, stream)
    motion_r2[stream]  = probe_regression(feat, pose_delta)      # 기존
    identity_acc[stream] = probe_classify(feat, object_label)    # 신규
                                                                 # chance = 1/n_class

# crossover 판정 (진짜 interaction):
assert motion_r2[M]  >  motion_r2[P]        # off-diagonal: P motion 열위
assert identity_acc[P] > identity_acc[M]    # off-diagonal: M identity 열위
assert identity_acc[M] ~ chance             # M은 identity chance 근처
report_interaction(stream x readout)         # main effect 2개 아님을 명시
```

## 7. 검증 체크리스트 (hand-off 전)

- [ ] 2×2 네 셀 모두 **동일 frozen feature·readout(attentive)·split**에서 산출.
- [ ] 대각선 crossover 성립(M-motion > P-motion **및** P-identity > M-identity) + M-identity ≈ chance.
- [ ] interaction이 두 독립 main effect가 아님을 수치로 확인(§2).
- [ ] baseline 인코더 동일 프로토콜 sanity(appearance 모델이 identity에서 이기고 motion에서 지는지).
- [ ] identity label spurious cue 누수 점검(배경 고정 등).
- [ ] Phase A 통과 시에만 Phase B(readout-free) 배선 착수.
- [ ] **STEP 1 인과 (§4 Phase C)**: ablation 모델 전부 Phase A와 **동일 arena·de-confound aug·readout·probe protocol**로 측정했는지(same-probe 일관성).
- [ ] **STEP 1 판정**: raw recon/BC/R² 하락이 아니라 **crossover signature 차이(Δ)**로 인과 판정했는지. no-M raw를 factorization headline으로 쓰지 않았는지.
- [ ] no-M signature는 single-stream separability + 난이도-매칭 참조(VideoMAE·Image MAE) 삼각측량으로 읽었는지.

## 8. Cross-refs

- **Vault 결정 출처**: Obsidian `Projects/Action-Agnostic Paper/2. Experiments.md §4`(남은 게이트 A/B) · `README.md §다음 수` · `History.md`(2026-07-02).
- **dev**: [`restart_plan.md`](restart_plan.md) §3.3(cross-leakage·correspondence TODO), [`comp_mae_plan.md`](comp_mae_plan.md) §6(dissociation probe·ablation)·§6.1(M 배포 무효), [`eval_protocols.md`](eval_protocols.md)(parity 가드), [`PROBING_GUIDE.md`](PROBING_GUIDE.md).
- **코드 진입점(참고)**: `scripts/eval/probe_action.py`(per-dim R²·stream 조합·attentive), `paper_artifacts/tables/step0_ood_efficiency/`(3b 표·provenance).
