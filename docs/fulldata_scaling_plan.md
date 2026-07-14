# Full-Data Scaling — CoMP-B × EgoDex part1-5 검증 계획 (2026-07-10)

> **목적**: 마지막 살아있는 제약 "모델만 키우면 좁은 데이터서 악화(B deployed-P 발산)"의 원인 판정 — **데이터 기아 vs 구조적 병리**.
> 가설 출처 = cluster_sessions 2026-07 §가설: "B의 P-appearance overfit-proneness = part1 대비 모델 규모 과대. EgoDex 전체 학습이 appearance 일반화로 완화 기대 (M은 이미 part1로도 clean·scale)."
> Vault 결정 = `Projects/Action-Agnostic Visual Representation (AAAI)/2. Experiments.md` §규모 결정 (2026-07-10 항목). 논문 반영 = **attach-only** (아래 §4).
> 본 문서 = 계획·주의사항. 잡 제출·구현은 dev/클러스터 세션에서.

## 1. 설계 — compute-matched 10ep 우선 (단계적)

**핵심**: part1 50ep ↔ full(part1-5, ≈5×) 10ep는 **본 샘플 수가 같음** → 같은 비용으로 "**다양성만 바꾼**" 깨끗한 대조. 병리 완화 확인 후에만 epoch 연장 결정.

| 항목 | 값 |
|---|---|
| 모델 | CoMP-B (기준런 `36186569` = step1_comp_mae_b와 config 완전 동일) |
| config 고정 | embed768/h12/m_depth4 · no-Sobel · pair · mask_m0.6 · caseA_prob0.25 · floor0.02 · batch128/GPU(eff1024) · LR 2.8e-4 |
| 변경 2개만 | **DATA = part1~5 full** · **EPOCHS = 10** |
| 데이터 | `/proj/external_group/mrg/datasets/egodex/frames/` — part1~5 + test 추출 완료 (07 확인) |
| 비용 추정 | ≈152 GPU·h ≈ 19h wall (2×4 H100, throughput 앵커 418.9 samp/s/GPU) — **48h 캡 내 단일 잡** |
| SUFFIX 제안 | `fulldata_comp_mae_b_10ep` |

**후속 옵션 (10ep 판정 후)**: ① epoch 연장(체인 잡, 50ep full ≈ 760 GPU·h — 완화 확인 시에만) ② S-full 10ep 추가로 2×2(scale × 다양성) 완성 — scale-interaction 축(2. Experiments §규모 결정 ⚠️: 확인 축은 part1→part1-5 **다양성**) ③ WD probe 보조 — 병리 지속 시 readout-overfit vs data-부족 disambiguate.

## 2. Critical guards

- 🔴 **LR 스케줄 step-매칭**: 스케줄러/warmup이 epoch 기반이면 10ep 기준으로 재설정할 것 — full 10ep의 총 step은 part1 50ep와 동일하므로 **step 수 기준으로 기준런과 매칭** (epoch 수 naive 적용 금지).
- 🔴 **part1~5 실제 비디오 수 확인** 후 "≈5×" 추정 보정 (part1=46,234 vid) → 비용·EPOCHS 재산정.
- 🟠 데이터 glob이 part1~5를 정확히 포괄하는지 + test 파티션 누출 없는지 확인.
- 🟠 checkpoint 저장 cadence = **매 epoch** (10ep라 촘촘히 필요, B-part1은 ep28 overfit 관찰 이력).
- 🟢 sanity(1ep MV200 급) 선행 — 신규는 데이터 경로뿐이나 throughput·loss 스케일 앵커 확인용.

## 3. 판정 (기존 파이프라인 재사용, 신규 배선 없음)

측정 = frozen probing (~1 GPU·h 수준), **same-probe 규율** (B-part1 병리 측정과 동일 arena·readout·protocol):

1. **deployed-P 기하 병리 재측정** — B-part1의 발산·overfit 시그니처가 B-full서 완화되는가 (주 판정).
2. **OOD 효율 4벤치** (CALVIN xfold + LIBERO 3 suite, `P_t⊕M` mean+attn) — efficiency.csv 참조 행.
3. **M-stream 규모 단조성** — S 0.293 → B-part1 0.352 → B-full ? (유지/상승 기대, M은 part1로도 clean했음).

**해석**: 완화 → **데이터 기아 확정** (B+part1 = 규모-데이터 미스매치, 구조 무죄). 지속 → 구조적 병리 (WD probe로 보조 판정, negative도 기록).

**사전 등록 판정 기준 (2026-07-12, probe 결과 확인 전 고정)** — 참조값: B-part1 deployed-P best −0.026→ep40 −0.49 발산 / S 0.329 healthy / M: S 0.293→B-part1 0.352:

- **완화** = ① 발산 시그니처 소멸 (40ep 동안 eval R² 단조 붕괴 없음, 최종 ≥ best−0.1) **AND** ② best R² ≥ +0.15 (S 0.329의 절반).
- **지속** = ① 실패 (B-part1과 같은 음수 발산).
- ① 충족·② 미달 = **부분 완화** → WD probe 보조 판정으로 이월.
- 보조 기대치: M 단조성 B-full ≳ 0.35 (B-part1 수준 유지 — M은 part1로도 clean했으므로 하락 시 별도 flag), P_t⊕M이 M 단독을 하회하는 red-flag 잔존 여부 병기.
- caveat 유지: 지속이어도 "CoMP 구조 결함" 즉시 확정 아님 — attentive readout 아티팩트 가능성(WD probe 2026-06 결론) 병기.

### ✅ 판정 결과 (2026-07-12, 게이트 13잡 `36828291~301`+`338/339` ≈2.5 GPU·h)

**완화 확정 = 데이터 기아, 구조 무죄.** same-probe (attentive·test·gap10·1500vids·40ep):

| Probe 축 | S(part1) | B(part1) | **B(full 7ep)** |
|---|---:|---:|---:|
| deployed-P (P_t⊕P_tk)★ | +0.329 | −0.03→**−0.49 발산** | **+0.375** (ep40 +0.323, 발산 소멸 — 기준 ①② 충족) |
| M | +0.293 | +0.352 | **+0.401** (단조 지속) |
| P_t⊕M | +0.286 (M−0.007) | +0.236 (M−0.116 red flag) | +0.361 (M−0.040, 대폭 완화·P 기여 여전히 ≤0) |

- **B-part1 병리 = 규모-데이터 미스매치**로 판정. deployed-P가 S도 상회(+0.375>+0.329) → "size가 P를 망친다" 서사 해소. scale은 다양성이 받쳐줄 때 양 스트림 모두에 이로움.
- **OOD 효율**(pos R², `efficiency.csv` 정식 행 추가·parity PASS): B-full attn CALVIN **0.571**/spatial **0.850**/object **0.881**/goal **0.703** (mean 0.532/0.813/0.857/0.753). S 대비 +0.03~0.08에 그침 = **3.5× params+6.8× data의 한계효용 작음 → S 효율 headline 강화**. same-data VideoMAE(attn 0.610/0.879/0.903/0.830)는 전 벤치 하회 — "small but close" 프레이밍 유지.
- ⚠️ 관찰: goal만 attn(0.703)<mean(0.753)·S(0.751) — 유일한 attn<mean 역전(단일 관찰, 해석 보류). B attentive probe는 host RAM 63GB 초과(OOM 2잡 → 16cpu 재제출) — B급 attentive 잡은 `--cpus-per-task=16` 필수.
- **후속 결정**: 50ep 연장 **불채택**(목적 달성·한계효용 낮음·I/O 병목 시 ~1,750 GPU·h). §4 attach-only대로 limitations 완화 재서술 + 효율 표 참조 행까지만. ~~S-full(옵션②)은 7/28 이후 판단~~ → **번복(07-12)**: SSv2 경로 1-c가 S-full을 요구, 겸용 조기 착수 (§4-b).

## 4. 논문 반영 정책 — attach-only (사전 등록)

- **7/28 전 spine 재편 금지.** 결과가 인상적이어도 3-claim 골격·헤드라인 불변.
- 완화 시: §4.4 효율 표 **참조 행** + limitations "part1 서브셋"·"B 악화" 두 항목 완화 재서술만.
- 지속 시: limitations 유지, negative 결과 본 문서·cluster_sessions에 기록.

## 4-b. S-full 7ep (옵션② · SSv2 1-c 겸용) — 착수 기록 + probing 관찰 기준 사전 등록

> **착수 경위 (07-12)**: §3 "S-full은 7/28 이후" 결정을 번복 — SSv2 경로 1-c(data-matched 확증, [`correspondence_eval_plan.md`](correspondence_eval_plan.md) §7)가 S-full을 요구해 옵션② 겸용 조기 착수. 설계 = §1과 동일하되 모델만 CoMP-S, 7ep compute-matched(part1-5 실측 6.8× 반영).
> 잡: sanity `36829370` PASS → 본학습 `36829403` 제출(07-12) → 노드 장애 requeue → 07-13 20:32 재시작 → **✅ COMPLETED (07-14 12:23, 15h54m = 127.3 GPU·h)**. 후속 15잡(EgoDex 3 + OOD 8 + SSv2 1-c 4)은 orchestrator `36831319`가 자동 제출 완료(`36832609~623`).

**⚠️ 측정 0 (sanity) 판독 — ep6 loss spike (07-14, probe 결과 확인 전 고정)**:
- train loss ep1→7 = 0.114/0.077/0.077/0.052/**0.032**/0.035/**0.051** — B-full(단조 감소)과 달리 **ep6 batch ~26,200에서 전 성분 동시 spike**(0.026→0.155, L_t/L_tk/L_pred/L_mB 모두). NaN 없음, 이후 회복 추세였으나 LR≤1.88e-5(cosine 말단)라 **부분 회복 상태로 종료**(ep7末 배치 ~0.045 vs spike 전 ~0.027).
- ckpt 상태: **latest.pt(ep7) = post-spike 부분회복** / **best_model.pt(ep5) = spike 이전**(유일 eval 0.0288).
- **판독 규율**: 측정 1·2 판독은 사전 등록대로 **latest.pt 기준 유지**(B-full·S-part1과 same-probe 정합). 기준 미충족 시 spike를 원인으로 귀속하려면 **ep5 best_model 재probe를 별도 관찰로 등록 후 실행** — 사후 기준 변경·구제 금지 원칙 동일. 기준 충족 시 spike는 관찰 기록만.

**🔴 측정 1 판독 — latest.pt(ep7) 기준 ①②③ 전부 미충족, 붕괴 수준 (2026-07-14, 잡 36832609~619)**:

| | S-full ep7 | S-part1 참조 | VideoMAE-ours |
|---|---|---|---|
| EgoDex deployed-P (best) | **0.019** | +0.329 | — |
| EgoDex M (best) | **0.095** | +0.293 (B-full 0.401) | — |
| EgoDex P_t⊕M (best) | **0.082** | +0.286 | — |
| OOD attn pos R² (CALVIN/spatial/object/goal) | **~0.19 / 0.306 / 0.531 / 0.297** | 0.487 / 0.814 / 0.851 / 0.751 | 0.610 / 0.879 / 0.903 / 0.830 |

- ① 격차 전 벤치 확대 → data-matched 행 보강 없음 (기존 S-part1 표 유지) ② M 다양성 재현 실패 ③ deployed-P 대폭 악화. 단 **하락 폭이 "이득 없음" 수준이 아니라 in-domain까지 붕괴** — ep6 spike 손상 가설이 유력 (다양성/scale 효과 판정 불가능 상태).
- **🟠 ep5 best_model 재probe 등록 (2026-07-14, 실행 전 고정)**: 동일 11잡(EgoDex 3 + OOD 8, same-probe)을 `best_model.pt`(ep5, spike 이전, eval 0.0288)로 재측정. 해석 기준: **ep5가 S-part1 수준 이상으로 회복되면 = spike 손상 확정**(7ep compute-matched 런은 무효, ep5는 5/7 compute라 §4-b 기준 ①② 판정엔 참고 관찰로만) / **ep5도 낮으면 = S-full 학습 자체 문제**(스케일링 negative, spike와 무관). SSv2 1-c 게이트 판정은 ckpt 대표성 문제(latest 손상)로 **보류** — 대기 중 4잡(latest.pt)은 손상 정량화 관찰로만 사용, 게이트 적용 여부는 ep5 재probe 후 결정.

**🔴 ep5 재probe 판독 (2026-07-14, 잡 36832860~870)** — 결과는 두 등록 분기의 **중간**이나 spike 손상은 확정:

| 측정 (attn) | ep7 post-spike | ep5 pre-spike | S-part1 |
|---|---|---|---|
| OOD pos (CALVIN/spat/obj/goal) | ~0.19/0.31/0.53/0.30 | **~0.39/0.66/0.74/0.50** | 0.49/0.81/0.85/0.75 |
| EgoDex M / P_t⊕M (best) | 0.095 / 0.082 | **0.195 / 0.158** | 0.293 / 0.286 |
| EgoDex deployed-P (best) | 0.019 | **0.006** | 0.329 |

1. **spike 손상 확정** — ep5→ep7이 학습 진행인데도 OOD·M이 반토막(직접 증거). latest.pt는 §4-b 기준 ①② 판정 부적격 → **7ep 런 측정 무효**.
2. ep5 < S-part1 은 **under-training confound**(5/7ep, LR 미annealed)로 스케일링 negative 결론 불가 — 청정 7ep 완주본이 있어야 판정 가능.
3. ⚠️ **deployed-P(P_t⊕P_tk) ep5에서도 ~0** — spike 무관 신호 후보. B-part1 병리(발산)의 S-full 재현이면 scale-interaction 역전(§3 B는 full-data로 완화, S는 full-data로 발병) 관찰이 되나, 역시 under-training 유보. 청정 완주본에서 재판정.
4. spike 원인: stderr 무결(데이터/worker 에러 없음), grad clip 1.0 존재 — 수치/optimizer 이벤트 추정, 데이터 기인 배제 불가. ⚠️ DistributedSampler seed 고정(0) → **ep6 재실행 시 동일 배치 순서 재현**(데이터 기인이면 같은 지점 재발; 재발 자체가 원인 진단이 됨).
- **수리 결정 (07-14, 사용자 확정)**: **A안 — ep5 재개 ep6-7 재실행, seed 동일**(재발 = 데이터 기인 진단). 잡 `36833097`(재개 확인: ep6부터, LR 7e-5 ✓) + orchestrator 재장전 `36833098`(afterok, 청정 완주 시 15잡 재제출). 손상 ep7 산출물은 `_sfullspike_`로 rename 격리, 손상-ckpt SSv2 4잡은 시작 전 취소. (B) ep7 연장·15ep 확장은 compute-matched 훼손으로 비채택 — 청정 7ep 판정 **후** 필요 시 별도 등록해 이어붙이기 가능(순차 실행이 정보 손실 없이 동등).

**🔴 수리 결과 (07-14 저녁): spike 재발 → 단순 재실행 전략 무효, 시스템적 불안정 확정**:
- 원 spike 지점(ep6 b26,200)은 무사 통과(0.033) 후 **ep6 b~28,400에서 재발**(0.029→0.086~0.098) — 동일 signature(전 성분 동시 + **L_mA(정지 calibration, 평시 정확 0) 발화**), ep7 평균 ~0.05로 종료 = 원 런과 동일 손상.
- **진단**: 고정 데이터 지점 아님(같은 seed·같은 순서인데 다른 batch에서 발생) · 노드 아님(g[001,003]→g[003-004] 모두 발생) · S-part1 50ep·B-full 7ep에는 없음 → **S(32M) × part1-5(다양) × LR 말단(≤2e-5) 조합의 시스템적 불안정, 2/2 재현**. 기전 추정: 후기 sharp minimum에서 드문 대형 grad 이벤트가 가중치를 걷어참 → LR이 작아 재하강 불가(fp16 autocast 사용 중, grad clip 1.0으로는 미방어).
- orchestrator `36833098` 발동 전 취소(손상 ckpt 15잡 차단). **유일 청정 ckpt = ep5** (양 런 공통). 누적 비용: 원 런 132 + 수리 44 ≈ 176 GPU·h.
- **다음 옵션 (미결)**: (i) **3차 시도 + 방어 코드** — spike-guard(배치 loss가 이동평균 대비 k배 초과 시 step skip) 또는 bf16 전환 후 ep5 재개(~35 GPU·h, 코드 수정 필요·S-part1 config와의 엄밀 동일성 각주) / (ii) **S-full 서랍** — §4-b는 attach-only 관찰이라 spine 무피해, SSv2 1-c는 "S-full 확보 실패" 기록, B-full 결과(§3)만 유지 / (iii) 스케줄 변경 재학습(LR floor 등, compute-matched 훼손 각주).

**측정 순서 (학습 완료 후)**:
0. sanity — loss curve·collapse 여부·recon 품질 (분 단위).
1. **action probing 매트릭스** (same-probe 규율, §3 B-full 판정과 동일 프로토콜): in-domain deployed-P/M/P_t⊕M + OOD 4벤치(CALVIN xfold + LIBERO 3 suite, mean+attn). 아래 사전 등록 기준으로 판독.
2. **SSv2 경로 1-c** (gate는 `correspondence_eval_plan.md` §7에 기등록, 07-12 학습 시작 전 고정): 표준+compositional 양 split. probing과 독립 — 병렬 제출 가능.

**🟠 probing 관찰 기준 사전 등록 (2026-07-14, probe 결과 확인 전 고정)** — 전부 **attach-only 관찰 기준, gate 아님** (spine 불변, §4 규율 동일). 참조값: S-part1 deployed-P +0.329 / M +0.293 / P_t⊕M +0.286 · B-full M +0.401 · OOD attn VideoMAE-ours 0.610/0.879/0.903/0.830 (S-part1 행 = `efficiency.csv`):

1. **3b 효율 data-matched 재서술 조건 (주)**: OOD 4벤치에서 격차 (VideoMAE-ours − S-full)가 S-part1 대비 **과반 벤치에서 확대되지 않으면** → 논문 3b 효율 표에 data-matched 행 보강 ("small but close"가 same-data 32M vs 86M으로 승격). 확대되면 → 관찰 기록만, 기존 S-part1 표 유지.
2. **다양성 이득 재현 (부)**: S-full M ≥ S-part1 +0.293 (B 패턴 0.352→0.401의 S 재현 기대). 미상승 = "다양성 이득이 B 전용"이라는 scale-interaction 관찰로 기록.
3. **무병리 확인 (부)**: deployed-P가 S-part1 +0.329에서 악화되지 않는지 (S는 무병리 셀, B-full +0.375 선례상 상승 기대).
4. **해석 규율**: 기준 1 미충족이어도 efficiency claim 훼손 아님 — 기존 표(S-part1)는 이미 parity PASS 상태. 이 측정은 **강화 기회**지 방어 의무가 아님. 사후 기준 변경 금지.

**❌ 금지 (gate 사후 구제 방지)**: JHMDB correspondence 재측정(경로 2는 FAIL 서랍 — B-full exploratory로 scale 견고성 기확인) · LIBERO BC 재실행(dissociation foil 강등·고비용·7/28 전 spine 재편 금지).

## 5. 데이터 로딩 I/O 병목 — 학습 종료 후 점검·개선 (2026-07-11 기록)

본학습 `36822727`(7ep, part1-5 실측 6.8× 반영)에서 **per-sample throughput 2.3× 저하** 관측 — 단순 배수 스케일링 아님:

| | part1 기준런 `36186569` | full-data `36822727` |
|---|---|---|
| 샘플/epoch | 4.62M | 31.48M (6.8×) |
| epoch 시간 | ~1,360s | ~21,400s (**15.7×**) |
| throughput | ~3,400 samp/s 유지 | 2,710(ep1) → **~1,470**(ep3+ 안정) |

- 하드웨어(2×4 H100)·global batch(1024)·모델(207M) 동일 → 연산 아닌 **데이터 로딩 병목 추정**: 315k 비디오·31.5M 샘플 셔플 랜덤 액세스가 페이지 캐시 워킹셋 초과 → GPFS 실제 리드 직행. §1 비용 앵커(418.9 samp/s/GPU)는 part1 캐시 우호 조건 값 — **full-data 추정에 재사용 금지** (실측 ~184 samp/s/GPU).
- **학습 종료 후 점검·개선 후보** (50ep 연장 등 후속 full-data 잡 전 필수 — 현 속도면 연장 비용 2.3× 증가):
  1. **node-local scratch(NVMe) stage-in** — 잡 시작 시 데이터 노드 로컬 복사 (용량 확인 필요)
  2. **샤드 순차 읽기** (webdataset/tar 샤드 + 샤드 단위 셔플) — 랜덤 액세스 자체를 제거
  3. num_workers·prefetch 재점검 (16 CPU/task 기준 포화 여부 프로파일)
- 이번 런은 개입 불필요 (~6h/ep 안정, ~37.5h 완료 전망 < 48h 캡).

## 6. Cross-refs

- 가설·데이터 준비: [`cluster_sessions.md`](cluster_sessions.md) 2026-07 §가설 · B 본학습 `36186569` (config·비용 앵커).
- Vault 결정: `Projects/Action-Agnostic Visual Representation (AAAI)/2. Experiments.md` §규모 결정 · §STEP 2 (FAIL 스코핑 맥락).
- 모델 정의: [`comp_mae_plan.md`](comp_mae_plan.md).
