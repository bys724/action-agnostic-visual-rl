# STATUS — action-agnostic-visual-rl

> 정본형 문서: 본문은 **현재 상태만**. 무엇이 일어났는지는 `docs/cluster_sessions.md`, 왜 그렇게 정했는지는 하단 결정 이력.
> 갱신: 2026-10-02 Vault 세션 (주장 구조 v2 = `docs/claim_spine_v2.md`) · 2026-10-02 dev 세션 (C2 판정) · 이전 2026-09-28 dev 세션 — refinement-floor 라운드 1·2 + §10 denoising 파일럿 완료, 주장 모델 확정, eval hang 수정, C1-DN factorization 재측정. 이후 실험 결과를 보고한 턴과 세션 종료 시 갱신 · 본문 80줄 이내

## 지금 어디인가

CoMP(대칭 cross-reconstruction Magno-Parvo MAE, 코드 v16) 논문 **AAAI-27 Reject (09-25)** 후, 리뷰 지적(raw ΔL 바닥선 부재)에 답하는 측정을 마쳤다. 결과: 같은 분포에선 학습된 M이 raw의 2–4배(판독기를 노이즈 조건으로 학습해도 유지), 분포 이동에선 raw가 가장 강건 → 사전 등록상 효율(정제) 주장 불성립. **논문 뼈대(09-27 합의) = 메커니즘(M-recon → factorization 인과) 중심 + 같은 분포 접근성 보조 + 분포 이동·제어 성능은 한계.** **주장 모델(09-28 사용자 결정) = C1-DN**(C1 + 10ep 센서 노이즈 제거·밝기 보존 증강; 잡음 붕괴 −1.4 → raw 동급). C1-DN에서도 M/P 방향성 이중분리 유지 확인(09-28, 아래 표). 결과 표 = `paper_artifacts/tables/refinement_floor/README.md`. C2 거울 ablation 판정 완료(10-02): M-recon 단독이 M grounding의 인과. **→ 그 결과를 받아 주장 구조를 재배치했다(10-02 Vault): 정본 = `docs/claim_spine_v2.md`. 다음 작업은 전부 그 문서의 사전 등록을 따른다.**

## 확정된 것

| 주장 | 신뢰도 | 근거 (무엇을 어떻게 재서) | 빠진 것 |
|---|---|---|---|
| M-recon 존재가 M grounding의 인과 (STEP 1, 07-08) — **M-recon 단독 몫 분리 완료 (C2, 10-02)** | [확정] | V_P 스칼펠·plain 2런 same-probe: plain에서 M motion 0.835→0.107. V_P는 P 오염(identity 0.999→0.224). C2(CoMP − M-recon, routing 유지): M motion Δ +0.017 ≤ 사전 등록 +0.05 → routing 형태 무관, C2 ≈ plain | M-recon 없는 런은 2/2 학습 불안정 — ep4(학습 건강) 비교로 방어(아래 행) |
| 효율은 CoMP 구조의 산물 (STEP 2-A, 07-09) | [확정] | same param(32.3M)·same data plain의 OOD probing이 4벤치 붕괴(CALVIN 0.030 등 vs CoMP-S 0.487) | — |
| 폐루프 BC에서는 CoMP ≈ plain (STEP 2-B, 07-10) | [확정 · 사전 등록대로 FAIL] | LIBERO 3suite × seed 0/1/2, 500ep/seed, pooled Δ−0.9pt (Wilcoxon p=0.763) | control-level value 이득은 미입증 — 논문에선 dissociation 근거로 흡수 |
| B 모델의 deployed-P 발산은 데이터 기아 (07-12) | [확정 · attach-only] | CoMP-B × part1-5 compute-matched 7ep: deployed-P −0.49 → +0.375, M 0.352→0.401 | 논문 spine 밖 |
| C1-DN(주장 모델)도 M/P 방향성 이중분리 유지 (09-28) | [잠정] | STEP 1 same-probe 그대로(LIBERO-object·attentive·gap20), C0 재측정 정확 재현(drift 0): 위치 너머 motion Δ M +0.320 vs P +0.209 · P identity 1.000 · M identity 잔여 +0.241 (C0 +0.307) | probe 1회 · **M−P motion 격차 축소**(C0 2.7× → 1.5×, P_t motion 0.547→0.621) · C1 단계 vs denoise 단계 기여 미분리 · plain 대조(인과)는 C0 기반만 |
| M-recon 없으면 학습이 건강한 시점(ep4)에도 M에 motion이 안 생기고, P 보조만 하는 M은 미학습보다 낮아짐 (10-01) | [잠정] | LIBERO-object same-probe M motion, ep4: C0 0.795 / plain 0.183 / C2 0.094 / 미학습 M 0.400 · ep4 P identity 셋 다 1.000. C2는 ep6 이후 P 표현 붕괴(ep12 P identity 0.186) | probe 1회·미학습 init seed 1 · ep4는 사전 등록 판정 시점(ep50) 아님 · M-recon 없는 학습은 2/2 불안정 → ep50 판정은 붕괴와 섞일 수 있음 |
| qk-norm 버그 재측정 후 EgoDex 기준② FAIL→PASS, OOD 확대 4/4→2/4 반전 (07-20) | [잠정] | probe 로더 qk-norm silent drop 수정 후 15잡 재측정: EgoDex P 0.328 / M 0.333 | **서랍 재판정 여부 미결** (attach-only라 spine 무피해) |

## 열린 것 · 다음 결정

- **🔴 다음 (지시 확정 10-02 · 정본 `docs/claim_spine_v2.md`)**: **E0 → E3 최소 셀** 순서로 최단 경로. E0 = 외부 인코더(DINOv2·SigLIP·VC-1)에 ΔL 입력 = 리뷰어 UnGc W1 미측정분, probe만. E3 = BC 가치 실험(같은 프레임 예산 5팔 · 간격 교정 · copycat 대책 전 팔 동일 · 데모 수 스윕), 인코더는 C0 고정. E3 최소 셀 중지 신호 = 최저 데모에서 CoMP 모션이 RGB 스택을 못 넘으면 주장 C 폐기. E1(증강 scratch ≈165 GPU·h)은 E3 양성 확인 후 발주. C3는 보류.
- **✅ 해결 (10-02)**: 재투고 논문 주장 확정 = `docs/claim_spine_v2.md` §1 (주장 A 메커니즘 / B 불변성=설계 / C 정제된 모션의 제어 가치 + 봉합 E0). **남은 사용자 확정 2건** = ① 주장 모델 C0 유지 vs C1-DN(09-28 결정) ② venue(E0·E3 결과 후). 쓸 수 있는 서술 경계: 노이즈는 판독기 노이즈 조건 학습으로 해결(R2-3, 제한점) · denoising 증강은 붕괴를 raw 수준까지 없앰(§10, 관찰) · "장면 교란은 데이터 다양성으로 해결"은 내부 증거 없음(파일럿 지지 없음 → 문헌 Fang et al. 2022 가설로만) · 시점 이동 전이 실패는 판독기 기하 문제(추정).
- 요약 [잠정·probe seed 3·표현 학습 1회]:
  - 라운드 1 (09-26): §6 (A) 전이 C1 −0.33 ≈ raw −0.32 · (B) 라벨 5% CI 겹침 · (C) 계산 불가 → 중지 신호.
  - 라운드 2 (09-27): 노이즈 붕괴 = probe 학습·시험 불일치(R2-3) · P를 붙이면 판독기 붕괴는 실재(R2-2) · 배포 P는 분포 이동에서 raw 수준, 취약성은 M 국한(R2-1) · 정확히-0 ΔL: sim 77–85% vs EgoDex 2.3%(R2-4).
  - §10 파일럿 (09-28, 10ep): 사전 등록 ① 불통과(학습에 없던 잡음 0.01에서 raw 동급, 0.005에선 초과) → 50ep 본학습 없음. ② CALVIN 0.43·LIBERO 0.68.
- **사용자가 정할 것**: Forecast-Sufficient **조기 게이트 스펙 5건** (09-20 제기) — ① 기준선 0.52~0.70은 32-d 헤더 출력값인데 게이트는 M_student 인코더 출력을 잼 ② 상대선(> M_teacher)과 절대선(≥ 0.52) 공존 ③ M_teacher 정본값 = Table I `ours` 0.576 ④ seed 수·마진 미명시 ⑤ 최종 판정 RAW-MOVE 1,536-d vs FSR 배포 2,304-d 차원 비대조. 이게 정해져야 v17 구현 착수.
- **사용자가 정할 것 (다음)**: qk-norm 재측정 결과로 **서랍(supplement 부록) 재판정**을 할지.
- 운영 메모: H100 1노드×3 GPU 학습 = 배치 341(유효 1023), ~81분/ep. eval hang(rank 0 단독 DDP forward)은 09-28 수정·검증. 대기 잡 `scontrol hold/release` 금지(자원 요청이 노드 전체로 부풂). GPFS 랜덤 읽기 병목: 노드 로컬 복사(`USE_SCRATCH=1`, part1 72분)로 1 GPU 처리량 2.0× (09-30 sanity) — `/scratch/tmp` 없는 노드는 `/tmp` fallback.
- **형제 프로젝트 Cross-View**(09-18 개시, 제목 잠정): head/wrist 뷰 충분성 + action 조건화. DROID 3뷰 페어링 로더 신규 필요. 구현 0.

## 돌아가는 잡

| 잡 ID | 무엇을 왜 | 시작 | 결과 확인 방법 |
|---|---|---|---|
| (없음 — C2 학습·판정 10-02 완료) | | | |

## 이 문서의 용어

- **CoMP** — 이 저장소의 모델 (구 CoMP-MAE, 코드 v16). P(form)·M(dynamics) 두 스트림을 서로 재구성시켜 분리시킴. 코드·ckpt 키는 옛 이름 유지
- **deployed-P / P-only** — 배포 시 쓰는 P 스트림 출력. P+M 이어붙임은 causal confusion으로 유해(LIBERO 68.7 vs 2.0)
- **STEP 0/1/2** — 논문 게이트: 0 = 효율 headline, 1 = factorization 인과, 2 = control-level value (A 효율 / B BC)
- **서랍** — 본문에 넣지 않고 supplement에 두는 결과(attach-only). "서랍 재판정" = 그 결과를 다시 판정할지
- **조기 게이트** — 후속 연구를 하루 안에 싸게 죽일 수 있는 선행 조건 (v17: M_student 릿지 프로브 R² > M_teacher)
- **C1-DN** — 주장 모델(09-28): C1 + 10ep, M 입력에 RGB 센서 노이즈(σ~U[0,0.01]) + 쌍 공유 장면 밝기(±1 stop), M-recon 타깃 = 밝기 보존·노이즈 제거 ΔL(`--bright-target aug`). 평가 팔 이름 `PilotM10`
- **v15 / v16 / v17** — `src/models/two_stream_v15.py`의 config 계보. v16 = comp_mae 플래그, v17 = forecast-sufficient (새 파일 아님)
- 전체 사전: `docs/GLOSSARY.md` (없으면 `docs/FILE_INDEX.md`·`RESEARCH_PLAN.md`)

---

## 결정 이력

- 2026-10-02 · **주장 구조 v2 확정 (Vault 세션)** — 신규성 자리를 "라우팅이 모션을 만든다"(C2가 반증)에서 "갈래별 복구 = 접지 / 라우팅 = P 보호" + "정제된 모션의 제어 가치"로 재배치. 증강은 리뷰 회피가 아니라 설계 원칙으로 승격(scratch 본 레시피), 주장 B 판정 자는 raw 초과 → 선택성으로 교체. 실행 = E0 → E3. 상세·사전 등록 = `docs/claim_spine_v2.md`
- 2026-10-02 · C2 사전 등록 판정: M motion 위치 너머 Δ +0.017 ≤ +0.05 → "M-recon 단독이 M grounding의 인과" 확정 (STEP 1 caveat ① 해소)
- 2026-10-01 · C2 학습은 P 붕괴에도 50ep까지 계속, 판정은 사전 등록대로 ep50 (ep4 비교는 보조 증거로만) (사용자 결정)
- 2026-09-30 · C2 2노드 사본 취소, 1노드 4 GPU 사본만 유지 (사용자 결정 — 2노드가 우선순위상 1노드 시작을 막을 수 있고 1노드가 GPU·h·속도 우위; plain과 GPU 수·데이터 순서 차이는 각주)
- 2026-09-30 · 대량 학습 시 손목 카메라는 별도 인코더·시점 토큰 없이 **함께 학습, 손목 쌍만 프레임 간격을 짧게**(초 단위 간격, 카메라 종류별 최대 간격) — 긴 간격 손목 쌍은 ΔL이 움직임이 아닌 두 장면 겹침이 됨(교차 회전 ΔL 아티팩트와 같은 종류) (사용자 결정)
- 2026-09-29 · C2 본학습 파티션 AIP_long → AIP (설정 동일; AIP_long은 AIP보다 우선순위 등급이 낮아 AIP 대기열이 있는 한 시작 불가) (사용자 승인)
- 2026-09-28 · C2 본학습은 2노드 8 GPU 대기 유지, 1노드 3 GPU 전환 안 함 (사용자 결정 — plain과 배치·구성 동일 유지)
- 2026-09-28 · C1-DN의 P motion 증가 원인(C1 단계 vs denoise 단계) 분리 안 함 (사용자 결정)
- 2026-09-28 · **논문 주장 모델 C1-DN = §10 파일럿(C1 + 10ep, 센서 노이즈 제거·밝기 보존 타깃, ckpt `two_stream_v15b_refine_comp_s_denoise_augtgt/20260927_235906/checkpoint_epoch0010.pt`)** — C1 대비 잡음 붕괴 해소(−1.4 → raw 동급), 같은 분포 유지 (사용자 결정. 사전 등록 ① 불통과와 별개로 C1 대비 개선 근거)
- 2026-09-28 · §10 파일럿 사전 등록 ① 불통과 → 50ep scratch 본학습 없음 (계획서 §10.3 규칙)
- 2026-09-27 · §10 파일럿 LR = 2.8e-5 (B) — 적응 부족은 1ep loss로 바로 보이고 표현 손상은 평가 후에야 보임 (사용자 선택; 실제 1ep에 적응 확인)
- 2026-09-27 · 밝기 증강 타깃 = 밝기 보존(aug), 노이즈만 제거 — 사용자 원래 의도 (원래 조명 복원 파일럿 1ep에서 취소)
- 2026-09-27 · 논문 뼈대 = 메커니즘 중심·같은 분포 접근성 보조·분포 이동/제어는 한계 (사용자 합의)
- 2026-09-26 · refinement-floor P_t⊕X 비교군 확장 보류 — raw 단독이 C1 계열 최선보다 분포 이동에 강건해 확장해도 결론 불변 (사용자 결정, 결과 정리 후 방향 판단)
- 2026-09-26 · 판정 기준 (C) = "계산 불가"로 기록 (증강 raw 기준점 부재, 사후 수정 금지)
- 2026-09-26 · refinement-floor 기준 (A) 전이 = 계획서대로 6방향 평균 유지 (object 시점 차이 확인 후, 전이 결과 보기 전 사용자 결정)
- 2026-09-16 · 조기 게이트 계획에 배포 대상(`p_teacher + m_teacher + m_student`)·항 1 포함·항 2 변위 타깃 확정 반영 (Vault 세션 결정)
- 2026-07-21 · Paper 1(input-prior)은 전용 repo로 분리, 이 저장소 문서 동결
- 2026-07-10 · STEP 2-B 게이트 FAIL → 사전 등록 스코핑 발동, "signature 인과 확정·control-level value 미입증"으로 논문 반영
