# Full-Data Scaling — CoMP-B × EgoDex part1-5 검증 계획 (2026-07-10)

> **목적**: 마지막 살아있는 제약 "모델만 키우면 좁은 데이터서 악화(B deployed-P 발산)"의 원인 판정 — **데이터 기아 vs 구조적 병리**.
> 가설 출처 = cluster_sessions 2026-07 §가설: "B의 P-appearance overfit-proneness = part1 대비 모델 규모 과대. EgoDex 전체 학습이 appearance 일반화로 완화 기대 (M은 이미 part1로도 clean·scale)."
> Vault 결정 = `Projects/Action-Agnostic Paper/2. Experiments.md` §규모 결정 (2026-07-10 항목). 논문 반영 = **attach-only** (아래 §4).
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

## 4. 논문 반영 정책 — attach-only (사전 등록)

- **7/28 전 spine 재편 금지.** 결과가 인상적이어도 3-claim 골격·헤드라인 불변.
- 완화 시: §4.4 효율 표 **참조 행** + limitations "part1 서브셋"·"B 악화" 두 항목 완화 재서술만.
- 지속 시: limitations 유지, negative 결과 본 문서·cluster_sessions에 기록.

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
- Vault 결정: `Projects/Action-Agnostic Paper/2. Experiments.md` §규모 결정 · §STEP 2 (FAIL 스코핑 맥락).
- 모델 정의: [`comp_mae_plan.md`](comp_mae_plan.md).
