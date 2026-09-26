# STATUS — action-agnostic-visual-rl

> 정본형 문서: 본문은 **현재 상태만**. 무엇이 일어났는지는 `docs/cluster_sessions.md`, 왜 그렇게 정했는지는 하단 결정 이력.
> 갱신: 2026-09-26 dev 세션 — refinement-floor C1 학습·바닥선·판정 측정 완료, 결과 정리. 이후 실험 결과를 보고한 턴과 세션 종료 시 갱신 · 본문 80줄 이내

## 지금 어디인가

CoMP(대칭 cross-reconstruction Magno-Parvo MAE, 코드 v16) 논문은 **AAAI-27 Reject (2026-09-25)**. 리뷰 수렴 지적(raw ΔL 바닥선 부재 등)에 답하는 `refinement_floor_plan.md`의 측정을 09-26 마쳤다: **같은 분포에선 학습된 M이 raw의 2–4배, 분포 이동(다른 suite·그림자·노이즈)에선 raw 단독이 가장 강건** → §6 판정상 효율(정제) 주장 불성립·중지 신호. C2–C4 학습과 P_t⊕X 비교군 확장은 보류, 논문 방향 판단(Vault) 대기. 결과 요약 = `paper_artifacts/tables/refinement_floor/README.md`. 별도 활성 축 Forecast-Sufficient Representation(v17)은 구현 0, 게이트 스펙 5건 사용자 판단 대기.

## 확정된 것

| 주장 | 신뢰도 | 근거 (무엇을 어떻게 재서) | 빠진 것 |
|---|---|---|---|
| M-recon 존재가 M grounding의 인과 (STEP 1, 07-08) | [확정] | V_P 스칼펠·plain 2런 same-probe: plain에서 M motion 0.835→0.107. V_P는 P 오염(identity 0.999→0.224) | — |
| 효율은 CoMP 구조의 산물 (STEP 2-A, 07-09) | [확정] | same param(32.3M)·same data plain의 OOD probing이 4벤치 붕괴(CALVIN 0.030 등 vs CoMP-S 0.487) | — |
| 폐루프 BC에서는 CoMP ≈ plain (STEP 2-B, 07-10) | [확정 · 사전 등록대로 FAIL] | LIBERO 3suite × seed 0/1/2, 500ep/seed, pooled Δ−0.9pt (Wilcoxon p=0.763) | control-level value 이득은 미입증 — 논문에선 dissociation 근거로 흡수 |
| B 모델의 deployed-P 발산은 데이터 기아 (07-12) | [확정 · attach-only] | CoMP-B × part1-5 compute-matched 7ep: deployed-P −0.49 → +0.375, M 0.352→0.401 | 논문 spine 밖 |
| qk-norm 버그 재측정 후 EgoDex 기준② FAIL→PASS, OOD 확대 4/4→2/4 반전 (07-20) | [잠정] | probe 로더 qk-norm silent drop 수정 후 15잡 재측정: EgoDex P 0.328 / M 0.333 | **서랍 재판정 여부 미결** (attach-only라 spine 무피해) |

## 열린 것 · 다음 결정

- **🔴 라운드 2 지시 (2026-09-26 Vault 세션, 사용자 확정)** — [`refinement_floor_plan.md` §9](refinement_floor_plan.md). **학습 없음.** ① P_t⊕P_tk 팔을 ⓢ 시험(전이·perturbation)에 추가 ② P_t⊕raw의 zero-pad를 계획대로 학습 선형 투영으로 고쳐 재측정(옛 행 보존) ③ probe 학습·시험 양쪽에 σ0.01/0.02 노이즈를 넣은 "현실적 sim" 조건에서 같은 분포 표 재측정 ④ 진단: CALVIN·LIBERO·EgoDex 정확히-0 ΔL 비율 ⑤(선택) probe weight decay. 답할 질문 = 정확히-0 의존이 원인인가 / P가 어디까지 짐인가 / P 두 장은 버티나. Case A 입력 노이즈 수정 **학습은 결과 후 사용자 결정**.
- **🔴 refinement-floor 결론 (09-26, M 단독·P_t⊕X 열, probe seed 3, 표현 학습 1회) [잠정]** — 표·파일 = `paper_artifacts/tables/refinement_floor/README.md`
  - 같은 분포(ⓘ): C1 M 0.46(CALVIN)·0.71(LIBERO suite 내) vs raw 0.22·0.16. random-init M ≈ raw → 차이는 학습된 가중치.
  - §6 (A) 전이 6방향: C1 −0.33 ≈ raw −0.32 → 불성립. (B) 라벨 5%: 0.39 vs 0.22, CI 겹침 → 불성립. (C) 계산 불가(증강 raw가 절반 강도 없음). ⇒ 중지 신호(M 단독 열).
  - P_t⊕X 열: P_t⊕C1 M이 P_t⊕raw를 전이(−1.29 vs −10.15)·그림자(−1.99 vs −3.75)에서 CI로 이기나, **raw 단독(−0.32/−0.32)이 둘 다보다 강건** — 우위는 P를 붙인 raw의 악화(P 외형 과적합 추정) 탓. 노이즈는 raw 우위. object suite만 카메라 다름(Floor)이 전이 음수의 주 원인.
  - 밝기 증강(C1 vs C0): 경사(ramp)만 개선, 배율·그림자·노이즈 불변성 없음. 노이즈 취약성은 학습이 만든 것(random-init M은 불변; CALVIN 정지 ΔL 73%가 정확히 0).
  - 보류: P_t⊕X 비교군 4개 확장(코드 `git stash` 'WIP parvo-randm', 미검증) · C2–C4 학습 · DINOv2+ΔL 팔(입력 규약 미정).
- **라운드 2 진행 (09-26 23시)**: R2-4 진단 완료 · R2-2/R2-3 최소 3칸 완료 → 전체 102잡 제출(09-27, 사용자 승인) · R2-1 1/18 완료, 17 대기. 요약 = `paper_artifacts/tables/refinement_floor/agg_round2.py`.
  - R2-3 최소 [잠정·seed 1·노이즈 조건 raw 대조 없음]: probe 학습·시험 양쪽 노이즈 σ0.01에서 C1 M CALVIN 0.469(깨끗 0.466)·LIBERO 같은 suite 0.666(0.716). → 라운드 1 노이즈 붕괴(−2.21, s42)는 probe 학습·시험 불일치. 같은 분포 우위(raw 깨끗 0.22·0.16)는 정확히-0 산물이 아닐 가능성↑.
  - R2-2 최소 [잠정·seed 1·전이 미측정]: 학습 투영으로 P_t⊕raw 그림자0.6 −3.49→−1.71, 깨끗 0.208→0.237, best_ep 3 불변. raw 단독(−0.31)보다 여전히 나쁨 → zero-pad는 악화의 일부, P 결합 자체가 판독기를 흔듦.
  - R2-1 첫 칸 [잠정·seed 1/3]: P_t⊕P_tk(C1) 깨끗 0.369 · 노이즈 σ0.04까지 불변(+0.37) · 그림자0.6 −0.24(raw −0.31과 비슷). → 노이즈 취약성은 M 국한, 배포 P는 무사(배율 1.3은 −5.15로 취약).
  - R2-4 [확정·500쌍]: 1초 간격 정확히-0 픽셀 CALVIN 77% · LIBERO 80–85% vs **EgoDex 2.3%**(전부-0 패치 0%). 1프레임 간격 EgoDex 16%/1.3%.
  - ⚠️ 해석 제약 [확정·코드]: C1의 Case A 입력은 독립 밝기 배율 두 벌(`two_stream_v15.py` `_photometric_aug` ×2) → 정확히 0이 아님. 즉 C1은 사전학습에서 정확히-0 패치를 거의 못 봤는데도 노이즈에 무너짐(−1.43; C0 −2.57) → §9.2 "Case A 정확히-0이 원인" 사슬은 C1에 성립 안 함. 남는 후보 = probe가 깨끗한 sim의 정확히-0에 기댐(→ R2-3가 판별).
- ~~사용자가 정할 것 (먼저)~~ → **09-26 답: 라운드 2(위) 먼저, 논문 방향은 그 결과 후.** (원문: refinement-floor 결과를 재투고 주장으로 쓸지·논문 방향.) 판정 기준 (C) 처리는 "계산 불가"로 기록(사후 수정 금지 준수).
  - Vault 논의용 해석 (09-26 dev 세션, 측정 아님): ① 노이즈 σ0.01(≈2.5/255)은 실제 센서 수준 → sim→real 제한점으로 무게 높임, 그림자 0.6(1초 내 화면 1/5 ×0.4 급출현)은 극한 스트레스로 무게 낮춤 — 판정 자체는 불변. ② 인코더 주장은 "작은 고정 헤드에서의 접근성"까지만 성립, 큰 헤드(BC-T)에선 CoMP≈plain → 논문을 결과물(인코더)보다 방법·메커니즘(M-recon 인과) 프레임으로. 인코더 주장을 살리려면 raw ΔL + 큰 헤드 대조가 필요 [미검증].
- **사용자가 정할 것**: Forecast-Sufficient **조기 게이트 스펙 5건** (09-20 제기) — ① 기준선 0.52~0.70은 32-d 헤더 출력값인데 게이트는 M_student 인코더 출력을 잼 ② 상대선(> M_teacher)과 절대선(≥ 0.52) 공존 ③ M_teacher 정본값 = Table I `ours` 0.576 ④ seed 수·마진 미명시 ⑤ 최종 판정 RAW-MOVE 1,536-d vs FSR 배포 2,304-d 차원 비대조. 이게 정해져야 v17 구현 착수.
- **사용자가 정할 것 (다음)**: qk-norm 재측정 결과로 **서랍(supplement 부록) 재판정**을 할지.
- 운영 메모: C1 학습이 C0 대비 34% 느렸음(원인 미확인, 노드 전용) — C2–C4 재개 시 비용 +60 GPU·h/셀 가능. full-data 잡은 GPFS 랜덤 액세스 2.3× 병목 미해결.
- **형제 프로젝트 Cross-View**(09-18 개시, 제목 잠정): head/wrist 뷰 충분성 + action 조건화. DROID 3뷰 페어링 로더 신규 필요. 구현 0.

## 돌아가는 잡

| 잡 ID | 무엇을 왜 | 시작 | 결과 확인 방법 |
|---|---|---|---|
| 40310547~563 (17) | R2-1 P_t⊕P_tk × {C1,C0} × {교란·라벨·전이} × seed 3 — 배포 P가 분포 이동에서 버티나 | 09-26 21:33 제출, 대기 | `python paper_artifacts/tables/refinement_floor/agg_round2.py` (env aavrl-train) |
| 40310996~311097 (102) | R2-2 나머지 8 + R2-3 전체 94 (노이즈 양쪽 조건 × 팔 8 × σ2 × 벤치 2 × seed 3) — Q1 판정에 필요한 raw 노이즈 조건값 포함 | 09-27 00시 제출 | 같은 스크립트 |

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

- 2026-09-26 · refinement-floor P_t⊕X 비교군 확장 보류 — raw 단독이 C1 계열 최선보다 분포 이동에 강건해 확장해도 결론 불변 (사용자 결정, 결과 정리 후 방향 판단)
- 2026-09-26 · 판정 기준 (C) = "계산 불가"로 기록 (증강 raw 기준점 부재, 사후 수정 금지)
- 2026-09-26 · refinement-floor 기준 (A) 전이 = 계획서대로 6방향 평균 유지 (object 시점 차이 확인 후, 전이 결과 보기 전 사용자 결정)
- 2026-09-16 · 조기 게이트 계획에 배포 대상(`p_teacher + m_teacher + m_student`)·항 1 포함·항 2 변위 타깃 확정 반영 (Vault 세션 결정)
- 2026-07-21 · Paper 1(input-prior)은 전용 repo로 분리, 이 저장소 문서 동결
- 2026-07-10 · STEP 2-B 게이트 FAIL → 사전 등록 스코핑 발동, "signature 인과 확정·control-level value 미입증"으로 논문 반영
