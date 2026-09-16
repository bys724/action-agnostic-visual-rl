# Forecast-Sufficient Representation — 조기 게이트 구현 계획 (code v17) (2026-09-15)

> **이름**: 교사(P_teacher, 동결) + 학생(P_student, M_student)의 예측 충분성으로 maintainability를
> *동기*가 아니라 **목적함수**로 만드는 연구. 코드 버전 = **v17** (v15 → v16(comp_mae) 계보의 다음
> config). 새 파일·새 클래스 아님 — `TwoStreamV15Model`(`src/models/two_stream_v15.py`)의 config
> variant로 추가한다 (v16이 `comp_mae: bool` 플래그로 추가된 것과 동일한 패턴).

> 설계 출처: Vault `Projects/Forecast-Sufficient Representation/1. 설계.md` (2026-09-16 대화
> 세션에서 배포 대상·항 1 포함 여부·항 2 타깃 확정, 이 문서도 그 결정 반영).
> 본 문서는 **조기 게이트 범위로 한정한 구현 참고**(계획·주의·pseudocode)다. 전체 항1+항2
> 이중목적 학습(λ 최종 값 튜닝 포함)은 게이트 통과 후 범위. 실제 코드는 dev 세션에서 작성.

> **배포 대상 (확정)**: `p_teacher + m_teacher + m_student` 이어붙임. `p_student`는 학습 중
> M_student에 gradient를 나르는 라우팅 통로일 뿐 배포에는 안 쓴다. 조기 게이트가 재는 것도
> 정확히 이 `m_student`.

> ⚠️ **README 경로 정정**: Vault README와 구 메모가 "`src/models/comp_mae.py`의 `CoMPMAE`"를
> 언급하나, 그 파일은 `release/aaai27_supplement/model/comp_mae.py`(AAAI 논문 supplement용 동결
> 스냅샷)이고 활성 dev 코드가 아니다. 실제 확장 대상은 **`src/models/two_stream_v15.py`의
> `TwoStreamV15Model`**이며, v16(`comp_mae=True`)이 이미 그 파일 안의 config로 존재한다.

---

## 1. 목표 (조기 게이트만)

> 전체 연구의 성공 기준(최종 판정: RAW-MOVE 14.2° 돌파)은 1.설계.md 소관. 이 문서는 그 앞의
> **조기 게이트 하나**만 구현한다 — 하루 안에 싸게 죽일 수 있는 조건.

**성공 기준(한 줄)**: `M_student`(항 2 학습 중 gradient를 받는 M encoder)의 말단 변위 릿지 프로브
R²가 `M_teacher`(현행 v15/v16의 EMA M encoder)보다 **높다**. SFA 기준선(차분 스트림을 읽는 팔들,
Table I) = **0.52~0.70**. 이 밑이면 다음 단계(전체 이중목적 학습)를 볼 필요 없다 — 1.설계.md
§게이트가 이미 이렇게 처분했다.

---

## 2. 현재(v15/v16) vs 추가(v17 조기 게이트 스코프)

| 구성요소 | 현재 (`TwoStreamV15Model`) | 조기 게이트에 필요한 추가 |
|---|---|---|
| P teacher | `TeacherPv15` — **EMA** copy of student P encoder (`update()` 메서드, momentum) | 다른 것: **동결**(freeze, EMA 아님) `P_teacher` — 세션 1(현행 CoMP 학습 완료본)에서 가져와 파라미터 고정. `TeacherPv15`를 그대로 쓰지 말 것 (EMA는 학생을 따라가므로 "닿기에 충분"의 고정 타깃이 못 됨) |
| M teacher | `TeacherMv15` — EMA copy, `forward_unmasked_encoder_only()` | 조기 게이트에는 **불필요** — 비교 기준은 `M_teacher`(=현재 학습된 v15/v16 체크포인트의 M encoder, 그대로 동결 로드)면 충분. 새 EMA 클래스 필요 없음 |
| M encoder 학습 신호 | routing helper로만 gradient 수령 (`v_from_p`+`src='m'`, comp_mae_plan.md guard 1) → v15에서 no-op 이력 | **항 2**(`routing(P_s(a), M_s(a→b)) → P_teacher(a+W) − P_teacher(b)`, 변위)만 켜고 **항 1은 λ1=0**으로 배선은 두되 끔 — M_student의 forecast 능력 상한선부터 본다 |
| 릿지 프로브 연동 | 없음 (v15/v16은 SFA 쪽 `ENCODERS` 화이트리스트에 없음) | `M_student` 체크포인트를 SFA `repr_score.py` 파이프라인이 읽을 수 있는 형태로 export (§5) |

---

## 3. 조기 게이트에 한정한 아키텍처 spec

전체 항1+항2 설계(라우팅 대수, λ 밸런스, 붕괴 방지 논리)는 1.설계.md 그대로 따른다. 여기서는
**게이트에 필요한 최소 조각**만 명시한다.

- **입력 경로**: `P_student`, `M_student` 둘 다 raw 관측에서 시작 (1.설계.md "입력이 학생이라 두
  인코더 모두 gradient를 받는다"). `_encode_p_unmasked`/`_encode_m_unmasked`류 기존 인코더
  forward를 재사용하되 **student 파라미터**로 통과시킨다 (teacher forward가 아님).
- **라우팅**: 기존 `MotionRoutingBlock`(`common/blocks.py`) + `RoutingInterpreterStep`
  (`decode_first=True`) 재사용. comp_mae_plan.md guard 1의 "V=owner, Q/K=helper" 규약을 그대로
  따른다 — 새 routing_mode 불필요.
- **항 2 타깃 (확정 = 변위)**: `P_teacher(a+W) − P_teacher(b)`, 둘 다 **동결** 교사로 인코딩한
  실제 프레임(a+W, b). 상태가 아니라 변위인 이유: 설계 문서의 SFA 가산성(`h_S+h_M=h_S'`) 유비와
  정합적 + 라우팅이 "고르고 옮길 뿐 변형 못한다"는 한계에 덜 물림(§근거는 1.설계.md §확정 3).
- **M-recon은 그대로 유지** (건드리지 않음 — 1.설계.md "건드리지 말 것", STEP 1 인과 확정 사항).
  조기 게이트 학습에서도 M-recon loss는 켜둔다. 껐을 때와 비교하는 것은 **M 제거 게이트**(별도,
  이 문서 범위 밖)의 몫이다.
- **항 1은 배선은 하되 λ1=0 (조기 게이트 한정)**: 항 1(짧은 지평·교사 공간 고정)은 최종 설계에
  포함되는 게 확정됐다(안전장치가 아니라 세션1→세션2 커리큘럼의 일부). 하지만 조기 게이트는
  M_student의 forecast 능력 **상한선**부터 싸게 보는 게 목적이라, 항 1의 코드 경로는 만들어두고
  가중치만 `λ1=0`으로 꺼서 돌린다. 여기서 기준(R² > M_teacher)을 못 넘으면 λ1을 얼마로 줘도
  no-op 쪽으로만 가므로 그대로 킬 — 넘으면 λ1 값 자체는 본 학습 튜닝 과제로 이월.
- **타이밍 (확정)**: `a→b ∈ [0.5, 1.0]초`(세션 2 한정 — 세션 1의 M 자체 학습은 기존 다양한 gap
  pair 그대로 불변), `a+W`는 a 기준 고정 1.5초. 데이터는 pair가 아니라 (a, b, a+W) triple
  샘플링 필요.

---

## 4. Critical guards (구현 시 실수 방지)

1. **P_teacher를 EMA로 두지 말 것** — `TeacherPv15`를 그대로 쓰면 학생을 따라가는 타깃이 되어
   "닿기에 충분"이 무의미해진다(움직이는 표적). 별도 로드: 세션 1 체크포인트에서 P encoder
   가중치만 읽어 `requires_grad_(False)`.
2. **M no-op 재발 감시 (필수, 1.설계.md 게이트 그대로)** — 조기 게이트 자체가 이 감시다. R²가
   `M_teacher` 대비 안 오르면 M_student가 항 2의 학습 신호를 못 받고 있다는 뜻 — λ·라우팅 배선을
   먼저 의심.
3. **M-recon 목적함수는 손대지 않는다** — ΔL 픽셀 복구 그대로. 인과가 잡힌 유일한 조각
   (comp_mae_plan.md, CoMP STEP 1: M-recon 제거 시 M motion 0.835→0.107).
4. **λ 밸런스 최종 값은 조기 게이트 판정에 영향 없음** — 릿지 프로브는 `M_student` 인코더 출력
   자체를 재므로(λ1=0 고정 실행), 본 학습에서 λ1을 얼마로 줄지와 무관하게 먼저 답이 나온다.
   순서: **조기 게이트(λ1=0) → 통과 시 λ 스윕(본 학습) → 최종 판정.**
5. **항 1을 코드에서 아예 빼지 말 것** — λ1=0으로 끄더라도 배선(routing → `P_teacher(b)` 비교)은
   구현해둔다. 게이트 통과 후 λ 스윕이 바로 이 경로를 켜는 것이라, 나중에 다시 짤 필요 없게.

---

## 5. SFA 측 연동 (측정 — 새 파일 만들지 않음)

측정 인프라는 `source-field-alternation` repo가 canonical (Vault README에 이미 명시). 이 문서는
**연동 지점만** 적어둔다 — 실제 등록은 M_student 체크포인트가 나온 뒤, 그 repo의 dev 세션에서.

- 물리 자 릿지 프로브 본체 = `src/sfa/eval/repr_score.py`의 `_disp_panel`/`_physics`/`_probe`
  (말단 변위 R², Table I/II를 만드는 그 경로).
- 동결 인코더 교체 팔의 선례 = `src/sfa/registry.py`의 `ENCODERS`
  (`vipenc`/`vipstate`/`r3mstate`/`r3mchange`) — `M_student`도 같은 패턴으로 **다섯 번째** 진입점이
  될 후보(README의 "여섯 번째"는 현재 4개 기준 착오 — 실제로는 5번째. Vault README도 같이 정정).
  단, 조기 게이트 자체는 이 화이트리스트 등록 없이 **단발 스크립트로 R² 하나만 뽑아도 충분** —
  화이트리스트 등록은 본 학습 통과 후 정식 비교에 필요.
- 캐시 규약은 `_repr_cache_of`/`_state_path`(같은 파일)가 기존 인코더에 쓰는 형식을 따른다.

---

## 6. 조기 게이트 실행 체크리스트 (TODO — 미구현)

- [ ] 세션 1 체크포인트에서 P·M encoder 가중치 로드 → 동결 `P_teacher`/`M_teacher` 모듈 작성
      (`TeacherPv15`/`TeacherMv15` 상속하지 않고 별도, guard 1 — EMA 아님)
- [ ] `TwoStreamV15Model`에 v17 config 분기 추가 (`forecast_sufficient: bool` 플래그, v16의
      `comp_mae: bool` 패턴을 그대로 따름)
- [ ] 데이터 샘플러: pair → (a, b, a+W) triple로 확장. `a→b`는 [0.5, 1.0]초 범위 샘플, `a+W`는
      a 기준 고정 1.5초 오프셋 (세션 1의 기존 pair 샘플링은 건드리지 않음 — 이 triple 샘플러는
      v17 전용 신규 경로)
- [ ] 항 1 loss 배선 (구현은 하되 λ1=0으로 실행): `routing(P_s(a), M_s(a→b))` → 동결
      `P_teacher(b)`와 비교
- [ ] 항 2 loss 배선: `routing(P_s(a), M_s(a→b))` → 동결 `P_teacher(a+W) − P_teacher(b)`(변위)와
      비교 (거리 함수는 기존 `loss_pred`류 재사용 가능한지 확인 — 실현 ΔL을 안 주는 버전인지
      검증 필수, §3)
  - stub 형태 예 (실제 구현은 dev 세션):
    ```python
    def forecast_sufficient_loss(
        self, p_visible_a, m_completed_ab, p_teacher_b, p_teacher_target_bw,
        lambda_1: float = 0.0,  # 조기 게이트 = 0.0 고정, 본 학습에서 스윕
    ) -> torch.Tensor:
        """항 1: routing(P_s(a), M_s(a→b)) → P_teacher(b) (동결, EMA 아님).
        항 2: 같은 routing → P_teacher(a+W) − P_teacher(b) (변위). raw ΔL(b→a+W)을
        타깃에 절대 섞지 말 것 (guard, 1.설계.md '진짜 예측이다' 문단)."""
        raise NotImplementedError("TODO: forecast-sufficient v17 항1+항2 loss — dev session")
    ```
- [ ] 짧은 학습 실행(하루 예산, `lambda_1=0.0` 고정 — 정확한 step/epoch 수는 dev 세션에서 기존
      sanity-run 관행 참고해 결정, 이 문서에서 임의로 못 박지 않음)
- [ ] `M_student` 체크포인트 → SFA repo로 export
- [ ] SFA `repr_score.py` 경로로 말단 변위 릿지 R² 측정 (§5, 화이트리스트 등록 없이 단발 실행)
- [ ] `M_student` R² vs `M_teacher` R² vs SFA 기준선(0.52~0.70) 3자 비교 → 게이트 판정 기록

---

## 7. 관련

- Vault `Projects/Forecast-Sufficient Representation/README.md`, `1. 설계.md` — 설계 원본, 이
  문서의 상위 문서
- `docs/comp_mae_plan.md` — v16 설계 문서, 이 문서가 따르는 형식·guard 스타일의 선례
- `src/models/two_stream_v15.py` — 확장 대상 파일
