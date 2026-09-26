# STATUS — action-agnostic-visual-rl

> 정본형 문서: 본문은 **현재 상태만**. 무엇이 일어났는지는 `docs/cluster_sessions.md`, 왜 그렇게 정했는지는 하단 결정 이력.
> 갱신: 2026-09-25 dev 세션 — refinement-floor 1–3단계(C1 학습 중·바닥선 표·교란 시험 파일럿) 반영. 이전: 2026-09-23 Vault 세션에서 `docs/forecast_sufficient_plan.md`(09-15/16)·`CLAUDE.md` 현재 상태(07-21)·Vault 메모리(08-30·09-20 확인분)를 근거로 쓴 **초안**. dev 세션이 첫 실행 때 검증·수정할 것. 이후 실험 결과를 보고한 턴과 세션 종료 시 갱신 · 본문 80줄 이내

## 지금 어디인가

CoMP(대칭 cross-reconstruction Magno-Parvo MAE, 코드 v16) 논문은 **AAAI-27 Reject (2026-09-25)**. 리뷰 수렴 지적 = raw ΔL 바닥선 부재·거울 ablation 미학습·ΔL≠물리 motion → 후속 = `refinement_floor_plan.md`. (아래 문단은 제출 시점 서술) 이 저장소의 논문 작업은 대기 상태다. 지금의 활성 축은 후속 **Forecast-Sufficient Representation**(v17) — 동결 교사 + 학생 구조로 "예측에 충분한 표현"을 목적함수로 만드는 연구인데, 조기 게이트 구현 계획(09-15/16)만 있고 **구현은 0**이며, 게이트 스펙의 미결 5건이 사용자 판단을 기다린다. 클러스터 잡은 7/12 이후 돌지 않았다.

## 확정된 것

| 주장 | 신뢰도 | 근거 (무엇을 어떻게 재서) | 빠진 것 |
|---|---|---|---|
| M-recon 존재가 M grounding의 인과 (STEP 1, 07-08) | [확정] | V_P 스칼펠·plain 2런 same-probe: plain에서 M motion 0.835→0.107. V_P는 P 오염(identity 0.999→0.224) | — |
| 효율은 CoMP 구조의 산물 (STEP 2-A, 07-09) | [확정] | same param(32.3M)·same data plain의 OOD probing이 4벤치 붕괴(CALVIN 0.030 등 vs CoMP-S 0.487) | — |
| 폐루프 BC에서는 CoMP ≈ plain (STEP 2-B, 07-10) | [확정 · 사전 등록대로 FAIL] | LIBERO 3suite × seed 0/1/2, 500ep/seed, pooled Δ−0.9pt (Wilcoxon p=0.763) | control-level value 이득은 미입증 — 논문에선 dissociation 근거로 흡수 |
| B 모델의 deployed-P 발산은 데이터 기아 (07-12) | [확정 · attach-only] | CoMP-B × part1-5 compute-matched 7ep: deployed-P −0.49 → +0.375, M 0.352→0.401 | 논문 spine 밖 |
| qk-norm 버그 재측정 후 EgoDex 기준② FAIL→PASS, OOD 확대 4/4→2/4 반전 (07-20) | [잠정] | probe 로더 qk-norm silent drop 수정 후 15잡 재측정: EgoDex P 0.328 / M 0.333 | **서랍 재판정 여부 미결** (attach-only라 spine 무피해) |

## 열린 것 · 다음 결정

- **🔴 최우선 (2026-09-25 Vault 세션 지시)**: AAAI-27 **Reject**(리뷰어 2명 5/3). 후속 실험 계획 = [`docs/refinement_floor_plan.md`](refinement_floor_plan.md) — **자족적 문서(09-25 재작성본). §1 논리 → §2 행동 규칙(변경 금지·허용 범위·멈추고 물을 것) → §7 순서로 집행.** 새 base C1(밝기 증강 CoMP-S) 학습 → 바닥선 팔·시험(§5) → C2·C3·C4(§4). 판정 기준 §6 사전 등록(사후 수정 금지). 문서와 충돌하는 판단이 서면 코드를 고치지 말고 여기 "열린 것"에 적고 멈춘다.

- **사용자가 정할 것 (refinement-floor)**: DINOv2+ΔL 팔(F4)의 입력 규약 — 부호 있는 ΔL([-1,1])을 DINOv2 [0,1] 이미지로 넣는 방식이 계획서에 없음. 추천 = (ΔL+1)/2 3채널 복제 후 ImageNet 정규화(0=중간 회색).
- **사용자가 정할 것 (먼저)**: Forecast-Sufficient **조기 게이트 스펙 5건** (09-20 제기) — ① 기준선 0.52~0.70은 32-d 헤더 출력값인데 게이트는 M_student 인코더 출력을 잼 ② 상대선(> M_teacher)과 절대선(≥ 0.52) 공존 ③ M_teacher 정본값 = Table I `ours` 0.576 ④ seed 수·마진 미명시 ⑤ 최종 판정 RAW-MOVE 1,536-d vs FSR 배포 2,304-d 차원 비대조. 이게 정해져야 v17 구현 착수.
- **사용자가 정할 것 (다음)**: qk-norm 재측정 결과로 **서랍(supplement 부록) 재판정**을 할지.
- **진행 중 검증**: refinement-floor 1단계 — 밝기 증강 구현(`--bright-aug`, 기본 off) [확정] 증강 off 시 수정 전 코드와 전 loss 항 소수 8자리 일치(C0 ckpt, CPU smoke) · sanity 1ep [확정 · 게이트 PASS] L_mB C0 동 step 대비 +20%·L_mA 0 붕괴 없음·증강 오프셋 0.051 vs motion |ΔL| 0.080 같은 자릿수 → C1 본학습 대기 중. 2단계 최소 칸(CALVIN, gap30, attentive, seed 1개) [잠정]: M 단독 위치 R² = raw ΔL 0.225 vs 제출본 M 0.444 — 같은 분포 참조 시험이라 판정축 아님, raw probe가 20ep 마지막 epoch에 best(미수렴 가능), 같은 설정 P_t⊕M이 기존 0.487→0.535로 흔들려 변동 폭 미확인. seed 변동 [확정 · V100 결정론·seed 3]: raw ΔL 0.218±0.024 / P_t⊕M 0.488±0.043 (0.535는 유리한 seed, 기존 0.487과 정합). M 단독 바닥선 표(seed 3, 같은 분포 참조 시험) [잠정]: 제출본 M 0.470±0.049 vs random-init M 0.246 · 투영 raw 0.239 · raw 0.218 · 정규화 0.130 · 증강 raw 0.052 (`paper_artifacts/tables/refinement_floor/calvin_m_alone_floor.csv`). 60ep 수렴 진단 [확정 · seed 3]: raw 0.223±0.005(평탄) / 제출본 M 0.576±0.033(아직 상승) → 20ep는 raw가 아니라 학습된 M에 불리, 격차 유지·확대. 전 팔 best epoch = 마지막(20) → probe 미수렴 공통 이슈. **교란 파일럿 (판정축 ⓢ, seed 1)**: 구현 [확정](교란 0 = clean 정확 재현). 제출본 M [잠정 · seed 1·CALVIN만]은 모든 교란에서 raw ΔL보다 크게 붕괴 — 노이즈 σ0.01(8bit 2.5단계)에 0.496→−2.0(raw 0.225 불변), 그림자 0.6에 −2.95(raw −0.31), 밝기 배율 0.9에 −9.9(raw −0.80). C1 판정의 조기 경보(중지 판단은 C1 측정 때 1회). CSV `calvin_perturb_pilot.csv`. 교란 시험 절차 [확정 · 2중 검증] = eval_protocols §4-b. part1 학습이라 아래 full-data I/O 병목과 무관.
- **🔴 방법론 문제 (사용자 판단 필요, 09-26)**: 판정 기준 (C)의 기준점 = "증강 raw(F1-aug)의 R²가 절반이 되는 그림자 강도 s*"인데, F1-aug는 **깨끗한 조건부터 0.05**(probe가 증강 데이터에서 거의 못 배움)라 그림자 0.6에서도 0.04 → **s*가 정의되지 않음**. 계획서 §6 사후 수정 금지 → 코드·기준 건드리지 않고 대기.
- **기준 (A) 전이 범위 = 계획서대로 6방향 평균 (사용자 결정 09-26 05:41, 결과 보기 전)**: object(Floor 장면·카메라 높이 0.65)와 spatial·goal(Tabletop·1.61)의 시점 차이 [확정]를 알고도 유지 — "기대 밖 수확 가능". spatial↔goal 2방향·object 관련 4방향 분해는 참고 열로 함께 보고(판정 = 6방향).
- **🔴 §6 판정 (09-26, C1 측정 완료 · M 단독 · seed 3) — 중지 신호 발동, 사용자 결정 대기 (C2–C4 미제출)**
  - (A) LIBERO suite 간 전이 6방향 [잠정 · seed 3]: C1 −0.33 CI[−0.49,−0.17] vs raw −0.32 · 증강 raw −0.22 · 정규화 −0.31 · random-init −0.73 → **불성립**(raw 계열과 겹치거나 낮음). 같은 suite(ⓘ)에선 C1 0.71 vs raw 0.16.
  - (B) CALVIN 라벨 효율(step 매칭) [잠정 · seed 3]: 5% C1 0.39 CI[0.30,0.49] vs raw 0.22 CI상한 0.34 → CI 겹침 **불성립**(평균은 C1 우위), 2% C1 0.18 CI 넓음. 고정 epoch 참고 열: 학습된 M이 저라벨에서 붕괴(5% −0.07) — step 매칭은 raw가 아니라 학습된 M에 유리했음.
  - (C) 계산 불가(기준점 부재). C1 교란(seed 2개): 그림자 0.6 −2.1/−2.7, 노이즈 0.01 −2.2/−1.3 vs raw −0.32/+0.22 → 밝기 증강으로도 C0와 같은 붕괴. 경사(ramp)만 개선(0.3에서 +0.05/0.15 vs C0 −0.40/−0.58).
  - 중지 신호(§6): ⓢ 전부(그림자·노이즈·전이)에서 C1 M ≤ raw → **해당**. 단 계획서 1차 기준인 P_t⊕C1 M 열은 ⓢ에서 미측정.
- **교란 비교군 seed 3 [잠정 · CALVIN만]**: 학습된 제출본 M은 noise σ0.01에 0.49→−2.57인데 **같은 구조 random-init M은 +0.24(불변)** → 노이즈 취약성은 구조가 아니라 학습이 만든 것. 그림자 0.6: C0 −2.59 vs raw −0.32 vs random-init −4.93.
- **재개 시 할 일**: ① C1(40275371) 완료 확인 → C1 M 단독으로 CALVIN 바닥선(20ep)·교란 시험 seed 3 ③ LIBERO suite 간 무재학습 전이 구현(판정 (A), 구현 0) ④ 결정 대기: DINOv2+ΔL 입력 규약.
- 실행 전 필수 점검(full-data 잡 한정) = **full-data 데이터 로딩 2.3× 병목(GPFS 랜덤 액세스) 미해결** — 후속 full-data 잡을 내기 전에 먼저 봐야 함.
- **형제 프로젝트 Cross-View**(09-18 개시, 제목 잠정): head/wrist 뷰 충분성 + action 조건화. DROID 3뷰 페어링 로더 신규 필요. 구현 0.

## 돌아가는 잡

| 잡 ID | 무엇을 왜 | 시작 | 결과 확인 방법 |
|---|---|---|---|
| 40285696~40285765 | C1 측정 + 판정 (A)·(B) 비교군 64잡 (CALVIN 교란·라벨 효율, LIBERO 전이; 7팔 × seed 3) | 09-26 13:4x | `paper_artifacts/{calvin,libero}_action_probing/*refine_{pert,label,labelfix,xfer}_*` |

## 이 문서의 용어

- **CoMP** — 이 저장소의 모델 (구 CoMP-MAE, 코드 v16). P(form)·M(dynamics) 두 스트림을 서로 재구성시켜 분리시킴. 코드·ckpt 키는 옛 이름 유지
- **deployed-P / P-only** — 배포 시 쓰는 P 스트림 출력. P+M 이어붙임은 causal confusion으로 유해(LIBERO 68.7 vs 2.0)
- **STEP 0/1/2** — 논문 게이트: 0 = 효율 headline, 1 = factorization 인과, 2 = control-level value (A 효율 / B BC)
- **서랍** — 본문에 넣지 않고 supplement에 두는 결과(attach-only). "서랍 재판정" = 그 결과를 다시 판정할지
- **조기 게이트** — 후속 연구를 하루 안에 싸게 죽일 수 있는 선행 조건 (v17: M_student 릿지 프로브 R² > M_teacher)
- **v15 / v16 / v17** — `src/models/two_stream_v15.py`의 config 계보. v16 = comp_mae 플래그, v17 = forecast-sufficient (새 파일 아님)
- 전체 사전: `docs/GLOSSARY.md` (없으면 `docs/FILE_INDEX.md`·`RESEARCH_PLAN.md`)

---

## 결정 이력

- 2026-09-26 · refinement-floor 기준 (A) 전이 = 계획서대로 6방향 평균 유지 (object 시점 차이 확인 후, 전이 결과 보기 전 사용자 결정)
- 2026-09-16 · 조기 게이트 계획에 배포 대상(`p_teacher + m_teacher + m_student`)·항 1 포함·항 2 변위 타깃 확정 반영 (Vault 세션 결정)
- 2026-07-21 · Paper 1(input-prior)은 전용 repo로 분리, 이 저장소 문서 동결
- 2026-07-10 · STEP 2-B 게이트 FAIL → 사전 등록 스코핑 발동, "signature 인과 확정·control-level value 미입증"으로 논문 반영
