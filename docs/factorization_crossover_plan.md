# Factorization Crossover — 이중분리 게이트 계획 (2026-07-02)

> **상태**: **Phase A 실행됨(2026-07-02, directional 지지)** · Phase B 강등 · Phase C(STEP 1 인과) 규율 확정. 이 문서는 **계획·판정 규율**(구현·config·집계는 dev 세션, 결과 로그는 `cluster_sessions.md`).
> **진행 요약**:
> - **Phase A (✅ 실행)**: libero_object 2×2. de-confound(rot+trans) 후 identity **P_t 0.851 vs M 0.278**(chance 0.10) = directional 이중분리. BUT arena가 위치↔identity↔motion 얽혀 **단일 arena clean 2×2 미달**(문제셀 P-motion 0.547). 상세 `cluster_sessions.md`(2026-07-02 Factorization crossover Phase A).
> - **🚫 EgoDex clean arena 조사 = 불가**: repo의 EgoDex는 hand-pose HDF5(transforms/confidences)만 추출 → object/scene identity label 부재, task=폴더명이나 motion-confounded, video-id는 k-NN proxy뿐. **EgoDex 단일 arena clean 2×2는 신규 annotation 필요**(값싼 경로 아님). 재조사 불필요.
> - **Phase B(readout-free) 강등**: 남은 gap = arena(데이터 얽힘)지 readout 아님 → k-NN도 이 confound 못 고침(별개 문제).
> - **① 통계 de-confound = ✅ 완료** (2026-07-02, 위치 partial-out beyond-position 2×2, git 5bfcf9c): aug 경로(P 0.851/M 0.278)와 통계 경로가 독립 수렴 → **directional 이중분리 확정**. → **다음 = ② STEP 1 인과**. 실행 우선순위·저비용 선결·구현 TODO = **§4.1**(2026-07-04 결정).
> - **② STEP 1 인과 = ✅ 실행·판정 완료** (2026-07-08, 2런 학습 36652563/564 + same-probe 16잡 36785365~380): **M-recon 존재 = M grounding의 인과**(plain에서 M motion 0.835→0.107 붕괴) · **V 소유는 인과 아님**(V_P 스칼펠서 M signature 생존 0.829/Δ+0.331) · **대신 V_P는 P를 오염**(P_t identity 0.999→0.224) = **V_M 설계의 인과적 정당화**. 판정 상세 = **§4.1 말미** · 수치 로그 = `cluster_sessions.md`(2026-07-08). **다음 = plain을 value 지표로 연장**(3b 효율 표·LIBERO BC-T에 plain 추가 = headline control 완결).
> **결정 출처**: Obsidian Vault `Projects/Action-Agnostic Visual Representation (AAAI)/2. Experiments.md §4 남은 게이트 A` / `README.md §다음 수` / `History.md`(2026-07-02).
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

- **있음 (motion 열)**: CoMP-MAE-S에서 이미 측정 — P_t 단독 **−0.009**(P는 motion 거의 0) / M **+0.094(mean)→+0.239(attentive)** / 3b efficiency 표 전체(`paper_artifacts/ood_efficiency/`). → crossover의 motion 열은 채워졌고, **P가 motion을 못 담는다**는 절반이 이미 확인됨.
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

### 4.1 STEP 1 실행 우선순위 — part1 최소 런 (2026-07-04 결정)

> Phase A directional 이중분리 확정 후. part1 학습 기회 제한 → **최소 런으로 M-recon 인과 판별**. 코드 구현은 **dev 세션**(이 저장소 별도 세션); 아래는 spec·순서·판정만.

**중심 질문 먼저**: 이 프로젝트 #1 리스크 = **M no-op**(v11/v15 재발). M이 no-op이면 라우팅 스타일은 애초 무의미 → **"M이 grounded인가"를 먼저** 확정.

**우선순위** (전부 part1 · same-probe[Phase C §3] · **signature Δ로만** 판정):

1. **[최우선] M-recon `V_M → V_P` 스칼펠** — M-recon의 **value 소유만 M→P로 뒤집어** M grounding만 외과적으로 off. P-recon(=P 난이도) 불변 → **난이도 매칭**(no-M보다 깨끗). V_P에서 M motion-probing/crossover가 무너지면 → **V_M(M-recon grounding)이 factorization 인과 성분** 입증. no-op 판정의 정본. 구현 = `blocks.py:210` M-recon 라우팅 인스턴스 forward의 `v_owner_state`를 M→P로 스위치하는 플래그(예 `--m-recon-v-source {m,p}`).
2. **[차선] plain baseline** — P-recon `v_from_m`(표준 cross-attn) + M-recon off (두 노브 동시 off). "우리 mechanism이 plain cross-modal MAE(≈temporal MultiMAE)를 이기나" = **외부 headline control**. 골격만으로 factored되는지 분리(#1의 인과 몫과 상보).
3. **[생략]** (a) P-recon `v_from_p→v_from_m` 단독(M-recon on) = 라우팅 스타일 곁가지, reviewer 요구 시만. (b) M-recon **통째 제거** 단독(v_from_p 유지) = #1 스칼펠에 지배(난이도까지 바뀌어 덜 깨끗) → 불필요.

⇒ 3 런 아니라 **2 런(#1·#2)로 충분**. #1(no-op 판정) 먼저.

**저비용 선결** (본학습 전, 코드=dev 세션):

- **Case A 효율** (zero-encode 낭비): ΔL(t,t)=0을 M-encoder에 통과시키는 forward가 입력-무관 낭비([two_stream_v15.py:944](../src/models/two_stream_v15.py)). **`caseA_prob` 낮추기(0.25~0.5)** 우선 — Case A는 calibration이고 Case B floor 가중이 정지 신호를 이미 일부 보유. 매 step 유지 시 zero 입력 batch=1 1회 인코딩 후 broadcast. ⚠️ encoder를 learned token으로 **완전 대체 금지**(Case A calibration 소멸).
- **rotation** (motion 신호 오염 방지): 사전학습 **`independent_rotation_prob=0`(joint rotation만)**. 서로 다르게 회전한 pair는 `ΔL(rot_k0, rot_k1)`이 90° 재배치 아티팩트로 지배 → motion 구조 소멸(cross-rotation 매칭 아니라 노이즈). joint는 `ΔL(rot,rot)=rot(ΔL)`로 유효. 교차회전 equivariance가 목표면 pixel-ΔL 아닌 별도 타깃 필요(범위 밖). ⚠️ probe-side de-confound aug(Phase C §3 same-probe의 rot+trans)와 무관 — 그건 유지.
- **(선택) 직렬 encoder pass batching**: `_forward_pair_comp`의 masked_t/masked_tk, Case A/B masked-M을 `[2B]`로 묶어 occupancy↑(작은 S에서 kernel-launch 이득). **프로파일 before/after 후 적용** — batch 크면 이득 작음. 정합성 무관 최적화(결함 아님).

**판정**: Phase C §3 규율 그대로 — raw 아닌 **crossover signature Δ**, Phase A와 동일 arena·probe protocol.

### 4.2 STEP 1 판정 결과 (2026-07-08, ✅ 완료)

2런 학습(36652563 스칼펠 / 36652564 plain, 각 50ep 완주) + same-probe 16잡(36785365~380, libero_object·attentive·gap20, raw 2×2 + beyond-position Δ 2×2). 결과 = `paper_artifacts/libero_action_probing/*_s1{vp,px}_*` · 수치 로그 = `cluster_sessions.md`(2026-07-08).

| 셀 (raw / Δ=⊕pos−ctrl) | CoMP-MAE-S (기준) | #1 V_P 스칼펠 | #2 plain |
|---|---|---|---|
| M motion | 0.835 / +0.338 | **0.829 / +0.331** (생존) | 🚨 **0.107 / +0.016** (붕괴) |
| M identity | 0.526 / +0.307 | 0.497 / +0.274 | 0.134 / +0.029 |
| P_t identity | 0.999 (ceiling) | 🚨 **0.224** (붕괴) | 0.800 / +0.508 |
| P_t motion | 0.547 | 0.055 | 0.014 |

**판정 3항**:

1. **M-recon 존재 = M grounding의 인과 (확정)** — plain에서 M motion 완전 붕괴. plain에서도 M은 `v_from_m` cross-attn으로 **사용되며 gradient를 받는데도** 자기 recon 목적 없이는 grounding 안 생김("학습 신호 부족" 반론 차단). 골격만으로 factored 안 됨 = **CoMP mechanism > plain cross-modal MAE** (외부 headline control, 표현 signature 레벨).
2. **V 소유(V_M vs V_P)는 M grounding의 인과 아님** — 스칼펠에서 M signature가 기준과 동일 수준으로 생존. §4.1 #1의 조건문("V_P에서 무너지면 V_M 인과")은 불성립.
3. **V_P는 M을 끄는 게 아니라 P를 오염** — P_t identity 0.999→0.224 (chance 0.10 인접). "P-recon 불변 = 난이도 매칭"이라는 스칼펠 설계 전제가 깨짐: M-recon grad가 P 12-layer를 관통(학습 중 throughput 저하 관찰과 정합)해 P의 appearance 선형가독성을 붕괴. ⟹ 뒤집으면 **V_M이 유일하게 "M grounding + P 무손상"을 동시 달성** = CoMP 대칭 설계의 인과적 정당화.

**Caveats**: ① 판정 1의 귀속은 plain이 2노브 동시 off(M-recon+라우팅)라 M-recon 단독 몫 미분리(#3a 생략분) — "사용돼도 붕괴" 논리로 실질 방어, reviewer 요구 시만 #3a. ② same-probe는 통계 경로(beyond-position Δ) 기준 — aug(rot+trans) 경로 미실행이나 효과 크기(0.835→0.107)가 경로 선택에 강건. ③ 스칼펠 P 붕괴는 linear-probe 가독성 기준(P pixel recon L_t 0.0071 건강) — 스토리 주력으로 쓸 경우 mean readout 교차확인 권장.

**다음 → §4.3** (STEP 2 value-level headline control).

### 4.3 STEP 2 — value-level headline control (task spec, 2026-07-09)

STEP 1이 표현 signature 레벨에서 "CoMP mechanism > plain"을 확정 → **value 레벨(probing R²·BC SR)로 완결**하는 단계. 비교 대상 = CoMP-MAE-S(ckpt `two_stream_v15b_step1_comp_mae_s_vp`의 기준런 = `…_comp_mae_s`, deployed **P-only**) vs plain(ckpt `two_stream_v15b_step1_plain_xmae_s`). **둘 다 학습 완료 → 신규 pretrain 없음.**

⚠️ **CoMP 자신의 reportable rollout도 미완** — 지금까지 task0·aug-off 탐색(P-only 68.7 vs P+M 2.0, cluster_sessions 2026-07-01)만. 따라서 (B)는 plain 추가가 아니라 **CoMP-S + plain 동시** reportable 매트릭스.

**(A) OOD 효율 표에 plain 행 추가 — 저비용 (frozen probing, ~0.7 GPU·h)**
- 프로토콜 = CoMP-S와 동일: CALVIN xfold(gap30) + LIBERO spatial/object/goal(gap20), **mean+attn**, position R²(dims 0–2). readout = `P_t⊕M`(plain도 M stream 구조 보유, M-recon만 off). probe forward = `_encode_p/m_unmasked`(step1 판정 16잡과 동일 경로).
- 산출 = `paper_artifacts/ood_efficiency/efficiency.csv`에 plain 행 → `scripts/eval/build_step0_efficiency_table.py` 재생성. 판정 = CoMP-S signature 우위가 probing value로도 이어지는지(같은 param·data).
- 위치 = 클러스터 or 로컬(둘 다 가능, frozen).
- **✅ 완료 (2026-07-09, 잡 36785986–993, ~1.1 GPU·h)**: plain `P_t⊕M`(attn) = CALVIN **0.030** / spatial **0.127** / object **0.109** / goal **0.059** — CoMP-S(0.487/0.814/0.851/0.751) 대비 4벤치 전부 붕괴 수준. **게이트 (A) PASS**(same 32.3M·same data에서 efficiency = CoMP mechanism의 산물). 내부 정합: object attn 0.109 ≈ STEP 1 M motion raw 0.107 + P_t motion 0.014(동일 arena). parity 앵커 전부 일치.

**(B) LIBERO BC-T rollout — reportable, full-suite·aug-on·P-only (load-bearing)**
- CoMP-S와 plain **둘 다**, 3 suite(spatial/object/goal), **P-only**(P+M은 causal confusion 배포 제외 = comp_mae_plan §6), **aug-on**, seed 0/1/2, 500 ep/seed — 기존 `libero_rollout/summary.csv` baseline(dinov2/siglip/vc1/videomae) 매칭.
- 위치 = **로컬**(finetune+rollout 모두, ckpt 로컬 전송 선결 — 역할 분담: 클러스터는 탐색만, cluster_sessions 2026-07-01 §다음).
- 집계 = `scripts/eval/aggregate_libero_rollouts.py` → summary.csv에 CoMP-S·plain 두 행 추가.
- ⚠️ **probing R² ≠ SR** (memory `feedback_evaluation_metric`): value 판정의 본 측정은 (B). (A)는 효율 표 완성용.

**선택(옵션)**: CoMP-MAE-B 효율 행 — size-scaling 참조. deployed-P B는 feature-geometry 병리(추격 실익 낮음, cluster_sessions 2026-06)라 필수 아님. 넣는다면 M-stream만.

**성공 기준(게이트)**:
- (A) plain probing < CoMP-S(`P_t⊕M`) at matched param/data → efficiency 주장 강화.
- (B) CoMP-S P-only SR ≥ plain P-only SR(3 suite 평균 + suite별) → headline control이 value로 성립.
- 둘 다 만족 시 **Paper 2 factorization claim이 value 레벨까지 완결**.

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

- **Vault 결정 출처**: Obsidian `Projects/Action-Agnostic Visual Representation (AAAI)/2. Experiments.md §4`(남은 게이트 A/B) · `README.md §다음 수` · `History.md`(2026-07-02).
- **dev**: [`restart_plan.md`](restart_plan.md) §3.3(cross-leakage·correspondence TODO), [`comp_mae_plan.md`](comp_mae_plan.md) §6(dissociation probe·ablation)·§6.1(M 배포 무효), [`eval_protocols.md`](eval_protocols.md)(parity 가드), [`PROBING_GUIDE.md`](PROBING_GUIDE.md).
- **코드 진입점(참고)**: `scripts/eval/probe_action.py`(per-dim R²·stream 조합·attentive), `paper_artifacts/ood_efficiency/`(3b 표·provenance).
