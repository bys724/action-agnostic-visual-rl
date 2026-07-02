# Restart Plan — subset-matched small 재시작 (2026-06-23)

> **상태**: 계획 확정, 실행 전. **문서 전용**(계획·주의·체크리스트·pseudocode). 실제 코드(slope 집계 스크립트·analog routing 분기·run config)는 이 저장소 dev 세션에서.
> **결정 출처**: Obsidian Vault `Projects/Action-Agnostic Paper/2. Experiments.md` §"결정: subset-matched small 재시작".
> **명명**: 코드 식별자 우선(rename deferred). 기능명↔코드 매핑 = `REFACTOR_PLAN.md` §1 / `CLAUDE.md` "명명 · 2논문 구조".

---

## 1. 배경 — 왜 재시작인가

1. **데이터 예산 사고**: `MS-JEPA`(=Parvo, code `v15b`)와 `Image MAE`(`--v15-no-motion`)는 실측 로그상 **EgoDex part1 서브셋(46,234 vid)만** 학습됨 — full part1-5(~314k) 아님(`cluster_sessions.md` L96, "기존 doc part1-5는 오류"). `VideoMAE-ours`는 full part1-5. → 기존 in-house 비교는 **cross-budget confound**.
2. **붕괴(recipe) 사고**: 서브셋 `MS-JEPA`는 clean checkpoint도 없음 — Run A ep5 cancel(collapse), Run B-2 CLS collapse(patch R²=0.2884 < VideoMAE 0.4705). LIBERO BC 78.5는 붕괴 잔재 위 값 → trend·development 참고용, 절대·앵커 주장 불가.
3. **비용 구조**: 지배 레버는 **데이터**(part1→part1-5 ~6.8×) ≫ 모델 크기(ViT-S↔B ~2×). 실측: subset single-stream 50ep ≈ 107 GPU·h(no-M JobID 36126892+36128321), full ViT-B 50ep = 343~420 GPU·h(v15 343.3 / VideoMAE ~400 / vfromm 420.4). → **"small+full"은 "small+subset"보다 ~6× 비쌈.**

→ 죽일 confound은 *cross-budget*이지 subset 자체가 아님. **subset에서 matched로 맞추면 같은 목적을 1/6 비용에 달성.** 현 narrative는 절대 LIBERO BC SR을 떠나 OOD/dissociation으로 이동했으므로 full 불필요.

---

## 2. 결정 (묶였던 4개를 분리)

| # | 결정 | 근거 |
|---|---|---|
| ① | matched 재시작 = **subset(part1)에서** | confound = cross-budget. subset matched면 1/6 비용 해소. 서브셋 MS-JEPA는 데이터+collapse 이중 오염이라 어차피 재시작 필요 |
| ② | ours 축 `MS-JEPA` → **`MCP-MAE`(`--v15-pixel-pred`)** 교체, **gate** | JEPA self-ref constant collapse 원천 제거. 단 미학습·미검증 → 첫 런이 곧 건강성 첫 증명 |
| ③ | **`VideoMAE-ours` 유지 (드롭 X, 재학습 X)** | dissociation 앵커(in-domain probe +0.47 ↔ control 0.293). 유지 비용 0. ViT-S 변종 없음 → 재학습 시 full-data monolithic 역할 파괴(zero gain) |
| ④ | 대조군 = **SiamMAE-analog** (claim-bearing) | routing Q/K를 M(ΔL) 대신 P, V는 P 유지 → "ΔL-where vs RGB-where" 단일변수 격리. vanilla literature SiamMAE는 objective·decoder·masking·routing 동시 변동이라 mechanism 입증 불가 → **외부 reference로만** |

---

## 3. 실행 순서 (비용 오름차순)

- **STEP 0 — value 게이트 (거의 무료, ~1 GPU·h)**: **🔼 2026-07-01 정련으로 승격 → §3.1 참조.** CoMP-MAE-S가 학습 완료라 이제 ours를 STEP 1 전에 probe 가능 → same-corpus slope(ours vs VideoMAE) + OOD-motion 절대값으로 **value 현상 유무 판정 → 규모 결정**. (옛 버전: ours probing 부재로 baseline 특성화만 = STEP 1 이후로 미뤘음. CoMP-MAE-S 완료로 무효.)
- **STEP 1 — subset·small·matched 3런 → probe → OOD slope(ours 포함)** (~100 GPU·h each): `MCP-MAE`-S + `SiamMAE-analog`-S + `no-M`(Image MAE)-S, **part1 동일·epoch 동일·size 동일**. 학습+probing까지 = thesis slope **첫 진짜 신호**. **게이트 = `MCP-MAE`-S 첫 run health(collapse 없이 patch+CLS R² 정상) + slope가 `VideoMAE` 대비 우위** → 아니면 매트릭스 확장 보류.
- **STEP 2 — 핵심 비교만 2-scale**: `MCP-MAE`-S vs `SiamMAE-analog`-S 를 part1의 10%/30%에서 재실행 → rank 안정성(scale-interaction 방어).
- **STEP 3 — (보류·조건부) 승자만 ViT-B full 1회**: 2-size 확인 + dissociation de-confound(full `VideoMAE`-B vs ours-B)를 **한 런으로 동시 충족**. 절대 경쟁력 주장 재추구 시에만.
- **KEEP**: `VideoMAE-ours`-B(full) + frozen baselines(VC-1/DINOv2/SigLIP) = 그대로. cross-size external reference로 표기.

---

## 3.1 🔼 STEP 0 정련 + value-first (2026-07-01)

> Vault 대화로 재정렬. **STEP 0가 "baseline 특성화"에서 "value 게이트"로 승격** — CoMP-MAE-S가 학습 완료(2026-06-30)라 이제 ours를 STEP 1 전에 probe 가능. 큰 학습(≈3M원+) 지르기 전 value 현상 유무를 공짜로 판정. 출처: Vault `2. Experiments.md §4` / `History.md`(2026-07-01).

**value 축 (무엇을 입증하나 — SOTA 아님)**:
- **(A) same-corpus readout↔control dissociation**: 동일 EgoDex에서 VideoMAE는 probing 최강인데 control 파탄 → "probing≠control". 현 데이터 방어 가능.
- **(B) OOD-motion 효율**: 좁은 라벨없는 데이터가 도메인 shift·motion 축서 internet-scale과 경합. 미입증 → STEP 0에서 확인.
- mechanism(analog·photometric)은 value 확인 **후** 착수(보조).

**클러스터에서 할 일 (STEP 0, 학습 0, ~1 GPU·h)**:
1. **CoMP-MAE-S를 CALVIN(ABC→D)·LIBERO motion-dim probe** — `probe_action.py`로 per-domain R² CSV 산출(BC rollout만 있고 probing 없던 gap 메움). in-domain=EgoDex 이미 있음(P_t⊕P_tk +0.236 / M +0.239). ⚠️ CALVIN 정규 소스 = `<enc>_training_*_gapsweep` gap30(§5 parity 가드), `_validation_*_seg`는 in-dist라 OOD 아님.
2. **same-corpus slope 집계** = `aggregate_dissociation_slope.py`에 CoMP-MAE-S append → **가족 내부만**(CoMP-S · VideoMAE · no-M). 신호 = `slope_ours − slope_VideoMAE`(음수=factorization 이득). 현 baseline VideoMAE slope −0.083.
3. **OOD-motion 절대값** = internet-scale 포함 전원, motion-dim 분해(gripper 섞지 말 것). "CoMP-S가 DINOv2와 경합?"(옛 CALVIN: +0.263 vs +0.223).

**게이트 판정 → 규모 결정**: (A)/(B) 신호 보이면 STEP 1(작은+matched)만. 안 보이면 (A) 단독 성립·scope·venue 재고(큰 학습 전).

**변경점 (옛 STEP 0 대비)**:
- ✅ **DROID 폐기**: action-probing R²≈0 floor = 변별 0 → OOD 축 제외(uninformative 명시). OOD = **CALVIN per-dim + LIBERO motion-dim(semantic-shift) + photometric corrupt-in-place**.
- ✅ **slope는 same-corpus 가족만**(전부 EgoDex home). internet-scale은 home 도메인 달라 slope 무의미(평평=robust 아니라 "떨어질 home 없음") → **난이도 calibration + OOD 절대참조로만**. (doc §5 "frozen slope 불가"와 정합.)
- ⚠️ **ours 축 = CoMP-MAE(v16)**, 옛 "MCP-MAE" 표기 대체(comp_mae_plan.md 정본). STEP 1 arm = CoMP-MAE-S.
- ⚠️ **SiamMAE-analog 재정의 필요 — dev 세션 확인 요망**: 이번 대화 결정 = **faithful rig-matched**(ΔL→**RGB helper** 교체 + **helper-recon 생략**, rig 고정). 현 구현 `--v15-routing-source p`(Q/K만 P로, V=P, M encoder frozen)는 *single-variable routing swap*이라 **다른 대조군**. 역할 분담: faithful analog=외부 헤드라인 대조 / routing-swap·no-M·V-source=내부 mechanism attribution. **어느 것을 STEP 1 외부 대조로 쓸지 dev 세션에서 확정**(구현 변경은 dev 세션).

**photometric-shift 축 (신규, ②)**:
- clean-probing 셋(EgoDex/LIBERO)에 **corrupt-in-place**. 🔴 **가드**: 전역 상수 밝기 offset은 ΔL서 상쇄 + per-image z-score 정규화로 지워짐 → **프레임 간 비대칭·공간 국소**(이동 그림자·국소 노출) 섭동이라야 ΔL 취약성 드러남. 취약성은 **M-stream readout 별도 측정**(P appearance가 가림).
- **예측 = ΔL 약세**(취약함이 곧 selectivity 증거, pre-registered). semantic-shift robust + photometric fragile **교차**가 mechanism 증거.

---

## 3.1.1 STEP 0.5 — de-confound (STEP 1 전 선결, 2026-07-01)

> STEP 0 결과(cluster_sessions 2026-07-01)가 **잠정 green이나 confound**됨 → STEP 1(3M원) 올리기 전 반드시 선결. **핵심 규율: 절대 성능 낮음 = size/data efficiency reframe로 방어 가능. 하지만 slope robust 신호는 측정 confound라 size/data와 무관 — de-confound로만 풀림.**

**STEP 0 결과 요약 (slope = in−OOD, mean, CALVIN pos)**:
| readout | comp in(EgoDex 18d) | comp OOD(CALVIN pos) | comp slope | slope-diff(vs vmae) |
|---|---|---|---|---|
| comp p_t_m (motion) | **0.099** ← 최저 | 0.405 | −0.306 | **−0.247** |
| comp p_t_p_tk (appearance) | 0.236 | 0.257 | −0.021 | +0.038 |
| VideoMAE | 0.470 | 0.529 | −0.059 | (기준) |

**🔴 confound**: comp p_t_m의 robust해 보이는 slope(−0.247)는 **in-domain이 0.099로 바닥 출발**한 탓일 수 있음. p_t_m(−0.247) vs p_t_p_tk(+0.038) 대조는 content(motion/appearance)뿐 아니라 **in-domain baseline(0.099 vs 0.236)도 다름** → "M robust"와 "M baseline 낮음"이 안 갈림. diff-in-diff는 target-space offset만 상쇄, baseline 효과는 못 잡음.

**작업 (비용 오름차순 — 클러스터)**:
- [x] **① [공짜] dim-matched in-domain 재계산**: EgoDex in-domain을 **rightHand(3D, end-effector 대응)**로 재계산 → 18d 집계와 거의 동일(손 joint 균일 이동): comp p_t_m 0.095≈0.099, p_t_p_tk 0.244≈0.236, vmae 0.466≈0.470. slope-diff **p_t_m −0.248≈−0.247** 유지 → **18d 아티팩트 아님**. rightHand가 여전히 낮음 = 저 baseline은 genuine.
- [x] **② [공짜] LIBERO slope-diff**: p_t_m CALVIN −0.247 / LIBERO −0.327 (둘 다 −, replication) **BUT** p_t_p_tk CALVIN +0.05(null) / LIBERO −0.11(불일치) → **confound 노출**. LIBERO는 in-suite(OOD saturate 0.74~0.85)라 baseline 낮은 모델이 기계적으로 −slope.
- [x] **🔴 결정타 (로그 스캔, 2026-07-02)**: **EgoDex in-domain은 300개 probe 전부 ~0.47 천장**(최고 VideoMAE 0.470). in-domain(어려움 0.47) vs OOD(쉬움 0.85)는 **target 난이도 비대칭** → 전 인코더가 OOD서 상승. slope는 이 난이도차가 지배 = **regression-to-ceiling, motion robustness 아님**. → **3a(slope) 폐기 확정**.
- [ ] **③ [배선] controlled-shift corrupt-in-place** (유일한 클린 mechanism 테스트): 공통 셋에 통제 섭동 → **각 모델 자기 clean baseline 대비 ΔR²**. dim·baseline·도메인 confound 원천 회피. semantic(ours robust 예상) + photometric(프레임 간 비대칭, ours fragile 예상) **교차 = selectivity**. ⚠️ 약한 feature는 잃을 게 적어 trivially robust — baseline R² 충분한 셋(CALVIN comp p_t_m 0.41) + **교차**로 판정.
- [ ] ④ (유보) attentive slope-diff — 폐기(3a와 함께). VideoMAE in-domain EgoDex attentive 545GB OOM.

**결론 (STEP 0.5 종료)**: **3a(slope/dissociation) 폐기** — dataset 난이도 천장 confound. **살아남은 주장 = 3b(efficiency 절대값)**: CoMP-MAE-S(~32M P+M, 좁은 unlabeled home-video)가 internet-scale ViT-B(DINOv2/SigLIP) 이기고 same-data VideoMAE(86M)에 근접. **검증된 표·방법론·provenance = [`paper_artifacts/tables/step0_ood_efficiency/`](../paper_artifacts/tables/step0_ood_efficiency/README.md)** (parity: CALVIN n_eval=32183 / LIBERO_spatial 9690, 전 인코더 일치). **다음 = ③ mechanism(selectivity)** → 그 후 STEP 1(no-M·SiamMAE-analog 인과).

**STEP 1이 보이려는 것 (참고)**: STEP 0=상관("M 경유 신호"), STEP 1=인과. no-M(M 빼면 robustness 사라지나) + SiamMAE-analog(ΔL-where가 RGB-where 이기나). 단 confound된 STEP 0 위에 올리면 그대로 상속 → 위 de-confound가 선결.

---

## 3.2 평가 축 probing-primary + BC 재배치 (2026-07-01)

> Vault 대화로 재정렬. **BC를 인코더 품질 주 지표에서 강등.** probing(정보 존재)과 control(closed-loop 안전 사용)은 다른 질문 — P+M NLL↔SR 역전이 증명. 출처: Vault `README.md §평가 축`, `2. Experiments.md §5`, `History.md`(2026-07-01).

**평가 축 재편 (클러스터/로컬 작업 함의)**:
- **주 축 = probing** (클러스터, 학습 후 frozen probe):
  - action probing(motion 디코딩) — 기존 `probe_action.py`.
  - **correspondence/tracking** (DAVIS/JHMDB류) — SiamMAE·MotionMAE 홈그라운드. **신규 eval 배선 필요**(현재 없음).
  - OOD 선택성(§3.1), equivariance probe(입력 known transform → 표현 구조 변화 측정 — 신규), spatial task(seg/depth — "holistic RGB는 공간정보 날림" 직접 시험, 신규·조건부).
  - ⚠️ raw R² 부족(concat-probe artifact 2회 철회) → 선택성·ablation 민감도·readout 불변성·probe OOD 전이로 방어(타협 불가).
- **BC = dissociation의 control 반쪽** (로컬 rollout, 주 지표 아님): "probing↑ but control↓(VideoMAE 78 vs 29)"를 *드러내는* 축. NLL↔SR 역전 = causal-confusion 사례(기여). **M→0 ablation 리포팅 권고**(rollout서 M feature 상수 치환 → SR이 P-only 회복하면 copycat 확정 = NLL 이득 spurious 증명). comp_mae_plan §6.1 결정적 후속 테스트와 동일.

**서술 규율 (P-only 격차)**:
- efficiency reframe: "작아도 근접"(S 40M < baseline 86M) ✅ / "키우면 이긴다" ❌(B가 오히려 나빴음). 깨끗한 주장은 **matched(VideoMAE)**, internet-scale은 unmatched 참조. bitter-lesson: 저데이터 특화라 격차 안 닫혀도 thesis 정합.
- P+M 붕괴(causal confusion, M-특정)로 **P-only 절대 격차를 덮지 말 것**(별도 정직 인정).

**tiering 재확인**: **STEP 0 = 주 value(A/B), SiamMAE 불필요·무료** → **STEP 1 = SiamMAE-analog(보조 mechanism), 조건부.** SiamMAE 비교는 LIBERO 단일 아니라 OOD-slope 전체 + correspondence.

---

## 3.3 평가 구현 TODO — 3-claim spine (2026-07-01, 무게중심 갱신 2026-07-02)

> 논문 = 주장 3개(① Factorization ② 표현 품질≠벤치 dissociation ③ 도메인-robust 효율). 각 방법을 in-domain·OOD 두 모드로. Vault `README §논문 spine` / `2. Experiments §4`. **핵심: readout-free(②correspondence·k-NN) + cross-leakage(③)가 probing artifact·BC 교란 두 약점의 면역 축.**
>
> **🔼 2026-07-02 무게중심 재확정**: 3a(slope) 폐기 후 논문 핵심 = **주장 1(factorization)**, 2a(dissociation-방법론)·3b(efficiency)는 그 아래 증거. 다음 최우선 = **factorization 이중분리(2×2 crossover) 게이트** — 아래 cross-leakage TODO를 저비용 probe→readout-free 순으로 operationalize. **정본 계획 = [`factorization_crossover_plan.md`](factorization_crossover_plan.md).**

**즉시 (STEP 0, 학습 0, 클러스터) — 주장 2·3 첫 신호**:
- [ ] **action probing OOD** — CoMP-MAE-S를 CALVIN(gap30)·LIBERO motion-dim probe(`probe_action.py`), **per-dim 분해**(선택성). 기존 코드, config만.
- [ ] **same-corpus slope 집계** — `aggregate_dissociation_slope.py`에 CoMP-MAE-S append(§3.1).

**신규 배선 (claim별, 우선순위 순)**:
- [ ] **correspondence label-propagation** (주장 2·3, 면역 축) — 🔴 신규. frozen patch feature nearest-neighbor affinity로 첫 프레임 라벨 전파. DAVIS-2017(J&F)·JHMDB(PCK). 프로토콜 = Jabri 2020(Contrastive Random Walk) 표준. **학습된 probe 없음 = concat-probe artifact 면역.** EgoDex→DAVIS = OOD 전이 겸함. baseline(DINOv2·SigLIP·VideoMAE) 동일 프로토콜.
- [ ] **cross-leakage disentanglement** (주장 1, factorization *측정*) — 🔴 **최우선(무게중심)**, 정본 = [`factorization_crossover_plan.md`](factorization_crossover_plan.md). 2×2 crossover(M→motion↑/identity chance, P는 역). **motion 열 이미 有**(P_t −0.009 / M +0.239) → **블로커=identity 열**. Phase A 저비용 probe(LIBERO/CALVIN object·task 분류, 학습 0)로 crossover 신호 먼저 → 서면 Phase B readout-free(k-NN). 🔴 진짜 interaction(대각선)이어야 3a 재confound 회피. = comp_mae §6 dissociation probe에 cross-leakage 정량 부여.
- [ ] **k-NN retrieval** (주장 2, 면역 축) — 🟢 신규·저비용. frozen feature 최근접. readout-free 보강.
- [ ] **manipulation-centricity** (주장 2, 로봇 지표) — 🟡 신규. 2410.22325 지표 구현(rollout 없이 조작-관련성 정량).
- [ ] **action probing 변형** (주장 2) — 🟢 기존 확장: velocity/acceleration(2-frame primitive 시험), gap-sweep(sampling-time), cross-embodiment.

**기존·재배치**:
- [x] action probing in-domain (EgoDex) — 완료.
- [ ] **BC = dissociation foil**(강등, 로컬 rollout) — 주 지표 아님. **M→0 ablation 추가**(rollout서 M feature 상수 치환 → SR이 P-only 회복하면 copycat 확정, comp_mae §6.1). P-only full-suite aug-on = credibility floor 확인용(≈78 예상).
- [ ] **equivariance probe** (주장 1 보조, Tier 3) — custom(입력 known transform → 표현 변화). off-the-shelf 지표 없음, 후순위.

**주의**: 신규 배선은 전부 dev 세션(runnable code). 본 문서는 계획만. baseline 공정성 = 모든 인코더 동일 프로토콜·frozen.

---

## 4. Critical guards (구현 시 실수 방지)

- 🔴 **`MCP-MAE` ~207M FROZEN param 삭제**: §9 경로가 teacher EMA + interpreter(미사용)를 들고 있어 메모리 낭비 → **본 학습 전 `del`** (sanity서 확인됨). 안 지우면 "small"이 메모리상 small 아님.
- 🔴 **analog ≠ vanilla SiamMAE**: `MotionRoutingBlock`(`src/models/common/blocks.py`, v11/v15 공유)에 routing-source 옵션 추가(M→Q/K vs P→Q/K). **V는 항상 P 유지**, param-symmetric. 별도 Siamese ViT 만들지 말 것.
- 🔴 **3런 matched 보장**: 동일 part1(MAX_VIDEOS 동일), 동일 epoch, 동일 ViT-S. subset fraction을 로그에 명시(향후 calibration).
- 🟠 **`VideoMAE` 재학습 금지**: cross-size external reference로만. cross-size mechanism attribution 금지(작은 ours를 큰 VideoMAE와 직접 메커니즘 비교 X).
- 🟠 **small-only 종착 금지**: ΔL inductive-bias 이점은 데이터·크기↑서 약화 가능(bitter-lesson) → 핵심 비교는 STEP 2(2-scale) 또는 STEP 3(ViT-B)로 재확인해야 paper claim.
- 🟢 **gate 위치 (정정)**: STEP 0는 게이트 아님(baseline 특성화만). 실제 게이트 = **STEP 1의 MCP-MAE-S 첫 run** — collapse 없이 healthy patch+CLS R² + OOD slope가 VideoMAE 대비 우위인지 확인 후에야 매트릭스(2-scale·ViT-B) 확장. ⚠️ thesis slope는 ours 학습 전엔 산출 불가(ours probing 부재, v15 no-op 프록시 부적합).

---

## 5. 구현 TODO 체크리스트 (dev 세션)

- [x] **STEP 0 집계 스크립트** = [`scripts/eval/aggregate_dissociation_slope.py`](../scripts/eval/aggregate_dissociation_slope.py) (학습 없음, 즉시 실행). 결과·실측 정정 ↓.
  - **in-domain(EgoDex) 데이터 현실**: canonical은 **VideoMAE-ours(+0.4705)·MS-JEPA(=Parvo v15b, +0.2884) 뿐**(eval_protocols §1). frozen(VC-1/SigLIP=0개, DINOv2=stale)은 EgoDex 학습 자체가 없어 in-domain 앵커 부재 → **slope 불가, OOD floor 참조만**. ⟹ slope는 EgoDex-trained만(VideoMAE; ours-S는 STEP 1 후 `ENCODERS`에 append).
  - **CALVIN OOD 정규 소스**(parity 가드가 오source 적발): `<enc>_training_20260526_213639_gapsweep`(§4 cross-folder, gap30) — 4인코더 §4값 정확 일치. ⚠️ `_validation_*_seg`는 within-validation(in-dist)이라 **OOD 아님**. LIBERO는 within-suite라 보조 표기만(slope 미사용).
  - **slope 절대값 해석 불가**(EgoDex 18-dim hand vs CALVIN 3-dim pos = target-space confound). **신호 = `slope_ours − slope_VideoMAE`**(동일 target → confound 상쇄). 현 baseline: VideoMAE slope **−0.083**, frozen OOD-pos floor = VC-1 +0.536 / DINOv2 +0.223 / SigLIP −0.314.
- [x] **SiamMAE-analog routing 분기**: `MotionRoutingBlock`(blocks.py)에 `routing_source ∈ {m,p}`(default m). `v_from_p`에서 `p`면 Q/K 입력을 `m_completed`→`p_state`로 교체(V는 항상 P, qk_m·norm_m·v_p 재사용). `RoutingInterpreterStep`→`TwoStreamV15Model`(`routing_source`)→CLI `--v15-routing-source` threading. analog = `--v15-pixel-pred --v15-routing-source p`. **smoke PASS**: param/state_dict 키 동일(ckpt 무손상)·flag 동작변경 확인(loss 2.060≠2.069).
- [x] **MCP-MAE frozen-param del**: pixel_pred에서 미사용 모듈 **freeze→del** (teacher_p/teacher_m=full P/M encoder copy + interpreter_1 + M decoder/pos/mask). smoke: 삭제 attr·state_dict 키 제거·param 감소·forward+backward·grad 정상. **no_motion(Image MAE)도 동일 teacher del 확장**(no-M-S = STEP 1 런). 비-pixel/비-no_motion(MS-JEPA)은 teacher 유지(scope 격리).
  - **🔴 점검 중 발견·수정한 버그**: SiamMAE-analog(routing_source="p")는 routing이 M을 무시(Q/K=P)하는데 M encoder가 **trainable+forward**돼 ① DDP `find_unused_parameters=False` hang ② 불필요 M forward ×2/step. → analog일 때 M encoder **freeze + forward skip**(param-symmetric 유지). smoke: analog trainable-no-grad=0(DDP-safe), MCP-MAE는 M 여전히 학습.
  - ⏳ 후속(미적용): no_motion의 M encoder/decoder·p_motion_decoder도 forward 미사용 → freeze 상태(DDP-safe). 완전 P-only로 추가 del 가능(고위험 회피로 이번 보류).
- [x] **run config** = [`scripts/cluster/run_step1_matched.sh`](../scripts/cluster/run_step1_matched.sh) + `pretrain.sbatch`에 `V15_ROUTING_SOURCE` env 추가. matched 공통(part1·ViT-S `EMBED_DIM=384 NUM_HEADS=6`·`V11_M_DEPTH=6`·50ep·`NO_SOBEL=1`·`PAIR_MODE=1`·동일 batch/LR) + per-run flag만 차이: mcp_mae_s(`V15_PIXEL_PRED=1`)·analog_s(`+V15_ROUTING_SOURCE=p`)·no_m_s(`V15_NO_MOTION=1`). 3런 모두 `two-stream-v15b`(vanilla siammae 모델 아님). no-arg=usage(대량제출 가드), gate=mcp_mae_s 먼저. `bash -n` 통과. **제출은 별도**(cluster_sessions 로그 필수).
- [ ] **STEP 2 subset**: part1의 10%/30% 2런(핵심 비교만).

### STEP 0 pseudocode (집계만 — 신규 학습 없음)

```
# 입력: probe_action 출력의 per-domain R² CSV (이미 존재)
# STEP 0 (무료): probing CSV 보유 인코더만 = baseline + VideoMAE
#   (MS-JEPA·no-M = probing 없음 / v15·v11 = 논문 제외·no-op → 제외)
for enc in [VideoMAE, VC-1, DINOv2, SigLIP]:
    r_in  = mean(R2[enc, EgoDex in-domain dims])
    r_ood = mean(R2[enc, CALVIN/LIBERO OOD dims])
    slope[enc] = r_in - r_ood          # 작을수록 도메인-robust
# STEP 1 이후: MCP-MAE-S / SiamMAE-analog-S / no-M-S 를 probe 후 같은 식으로 append
rank(enc by slope)                      # 지표 = 절대 R²가 아니라 slope
# 해석: ours(STEP 1)의 slope가 monolithic(VideoMAE)보다 작으면 factorization 이득 신호
```

---

## 6. 검증 체크리스트 (hand-off 전)

- [ ] 3런이 동일 part1·epoch·ViT-S로 학습됐는지 로그 확인.
- [ ] `MCP-MAE` frozen-param del 반영 + 메모리 감소 확인.
- [ ] analog가 V=P 유지(M은 Q/K만 생성)인지 weight shape/forward로 확인.
- [ ] STEP 0 slope 부호·순위를 예측 시그니처(semantic-shift서 ours robust, photometric서 ours 약세)와 대조.
- [ ] STEP 1 결과를 paper claim화하기 전 STEP 2(2-scale) rank 유지 확인.

---

## 7. Cross-refs

- **Vault 결정 출처**: Obsidian `Projects/Action-Agnostic Paper/2. Experiments.md` §"결정: subset-matched small 재시작" / `README.md` "현재 상태".
- **관련 dev docs**: [`v15b_retraining_status.md`](v15b_retraining_status.md) §9(MCP-MAE 채택)·§10(BC 78.5)·§11(no-M), [`siammae_baseline_plan.md`](siammae_baseline_plan.md), [`cluster_sessions.md`](cluster_sessions.md)(part1 확정·GPU·h), [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md), [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) §1(명명표).
