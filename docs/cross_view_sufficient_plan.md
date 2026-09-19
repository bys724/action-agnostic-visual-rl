# Cross-View Sufficient Representation — 항 A(head↔action) 구현 계획 (2026-09-18)

> **이름**: CoMP 세션1(정적/동적 상호예측)을 뷰를 가로질러 확장하는 연구. 이 문서는 그 첫
> 실험(항 A, head↔action)만 다룬다. head+action→wrist(본 주제)는 항 A 통과 후 범위.

> 설계 출처: Vault `Projects/Cross-View Sufficient Representation/1. 설계.md` §항 A 구현 스펙.
> 본 문서는 **구현 참고**(계획·주의·pseudocode)다. 실제 코드는 dev 세션에서 작성.

---

## 1. 목표 (항 A만)

**과제**: `P_head(t)`를 마스킹하고, `M_action(t→t+k)`로 라우팅해서 `P_head(t+k)`를 복구한다.
CoMP의 기존 P-recon과 구조가 완전히 동일하다 — helper만 head 자신의 M에서 action으로 바뀐다.

**목적**: head+action→wrist라는 최종 아키텍처가 성립하려면 "action이 vision과 CoMP식으로
결합되는가"가 필요조건이다. wrist(뷰가 매번 새로워 예측이 원리적으로 어려움)로 바로 가기 전에,
훨씬 안정적인 head로 이 필요조건부터 가장 싸게 검증한다.

**성공 기준**: (구체 지표·임계값 미확정 — 우선은 loss가 head 자신의 M을 helper로 쓸 때 대비
얼마나 근접하는지를 정성적으로 본다. 정량 게이트는 초기 결과를 보고 정한다.)

---

## 2. 현재(CoMP) vs 추가(항 A)

| 구성요소 | 현재 (`TwoStreamV15Model`) | 항 A에 필요한 추가 |
|---|---|---|
| owner(P) 파이프라인 | 마스킹 → mask token 주입 → `_build_full_seq_p` → interpreter
self-attn → `RoutingInterpreterStep` → `recon_head` | **변경 없음.** head-cam 프레임에 그대로
적용 |
| 라우팅 블록(`MotionRoutingBlock`, cross-attention) | `v_from_p`+`src='m'` 컨벤션 | **변경
없음.** helper 텐서만 바뀔 뿐 블록 내부는 그대로 |
| helper(M) 소스 | `_encode_m_unmasked` — head 자신의 ΔL을 ViT로 인코딩 | **신규**:
`action_proj(action_delta)` — 학습된 인코더 없음, linear projection 하나 |
| 데이터 로더 | `egodex.py` — RGB만 반환 | **신규**: 같은 영상의 HDF5(`transforms`/
`confidences`)를 join해서 action delta도 반환하도록 확장 |

---

## 3. 아키텍처 spec

- **owner = head-cam P.** 기존 P encoder(`patch_embed_p`, `blocks_p`)와 기존 P-recon
  파이프라인을 100% 재사용한다. 새 코드 없음.
- **helper = action.** `M_action(t→t+k)` = action_dim 벡터(EgoDex는 18차원 hand-pose delta,
  §4 참고) → 신규 `action_proj: nn.Linear(action_dim, embed_dim)` → cross-attention의 K/V
  소스로 들어간다. **여기가 이 실험의 유일한 신규 학습 파라미터.**
- **차원 정합 지점**: cross-attention 직전에만. self-attention(interpreter)이나 라우팅
  블록 내부에는 절대 넣지 않는다(guard 1).
- **복구 대상은 여전히 P_head(t+k)뿐** — action을 복구하는 방향(action 자체를 vision으로
  맞히는 것)은 이 실험 범위 밖(1.설계.md의 "항 B", 별도 실험으로 미룸).

---

## 4. Critical guards (구현 시 실수 방지)

1. **차원 정합은 action 쪽에만, cross-attention 직전에만** — 기존 검증된 라우팅 블록
   (`MotionRoutingBlock`)이나 interpreter self-attention 코드는 한 줄도 건드리지 않는다.
   action에만 `action_proj` 레이어를 새로 추가해서 embed_dim으로 맞춘 뒤 helper 자리에
   그대로 꽂는다.
2. **action이 단일 토큰이면 cross-attention이 퇴화한다** — action_delta를 벡터 하나(토큰
   길이 1)로 projection하면, cross-attention의 key가 1개뿐이라 `softmax(1개 키) = 1.0`이 되어
   **P_head의 모든 마스킹 위치가 무조건 같은 action 임베딩을 받는다** (Q가 사실상 무의미,
   attention이 아니라 균일 broadcast와 수학적으로 동치). 이게 의도한 동작인지 구현 직전에
   확인할 것:
   - 의도한 동작이면(action은 원래 공간 구조가 없으니 균일 조건화가 맞다) 그대로 진행.
   - 위치별로 다르게 반응해야 한다고 판단되면, action을 여러 토큰(예: 윈도우 안의 서브스텝
     시퀀스)으로 쪼개야 cross-attention이 실질적인 일을 한다 — §5 시간 스케일 정합과 연동.
3. **owner 쪽 기존 코드는 절대 수정하지 않는다** — 마스킹, mask token, `_build_full_seq_p`,
   `recon_head` 전부 재사용. 이 실험이 실패해도 기존 CoMP(STEP 1에서 인과 확정된 부분)에
   영향이 없어야 한다.
4. **M_action은 실제 action 이력에서 계산** — raw 픽셀에서 유도하지 않는다(guard, 리키지
   방지 — action은 정의상 이미 독립적인 채널이라 유출 문제 자체가 없지만, 혹시 파생값을
   쓸 경우 head 프레임에서 역산하지 않았는지 확인).

---

## 5. 데이터 (EgoDex, DROID 아님)

- **로더**: `src/datasets/egodex.py` (현재 RGB만 반환) — 같은 `video_name`의 HDF5
  (`transforms/{joint}`, `confidences/{joint}`)를 join하도록 확장. 기존 `probe_action.py`
  (line ~134, ~141-184)가 이미 이 HDF5를 읽는 패턴을 참고 — 새 파싱 로직을 만들 필요 없이
  그 방식을 재사용.
- **action delta 계산**: `probe_action.py`의 기존 관행(신뢰도 필터링된
  `pose[t+k]-pose[t]`, 18차원, gripper 없음)을 우선 그대로 쓴다 — 시작-끝 델타 방식.
  누적합·서브스텝 시퀀스는 guard 2가 필요하다고 판명되면 그때 바꾼다.
- **`k` (gap)**: 미확정. 기존 M 채널이 이미 검증한 [0.5, 1.0]초 폭에서 시작하는 게 자연스럽다
  (Cross-View Sufficient Representation의 다른 실험들과 같은 타이밍 관행).

---

## 6. 실행 체크리스트 (TODO — 미구현)

- [ ] `egodex.py`: 영상별 HDF5 join 추가 — 프레임 인덱스 → `pose[t]`, `pose[t+k]` 매핑,
      confidence 필터링 재사용
- [ ] `action_proj: nn.Linear(action_dim, embed_dim)` 신규 모듈 추가 (v15/v16과 별개
      config, 예: `cross_view_action_gate: bool` 플래그)
  - stub 형태 예 (실제 구현은 dev 세션):
    ```python
    def encode_action_helper(self, action_delta: torch.Tensor) -> torch.Tensor:
        """action_delta: [B, action_dim] (EgoDex = 18차원 hand-pose delta, 시작-끝).
        -> [B, 1, embed_dim] (guard 2: 토큰 길이 1 = cross-attention이 균일 broadcast로
        퇴화함을 인지한 상태에서의 설계 — 의도 확인 후 진행)."""
        raise NotImplementedError("TODO: cross-view-sufficient 항A action helper — dev session")
    ```
- [ ] P-recon forward에서 helper 소스를 `_encode_m_unmasked(head_M)` 대신
      `encode_action_helper(action_delta)`로 교체하는 분기 추가 — 나머지 forward는 전부
      기존 코드 재사용
- [ ] 짧은 학습 실행 — head 자신의 M을 helper로 쓸 때(기존 CoMP) 대비 loss 비교
- [ ] guard 2 판단: attention weight가 실제로 균일한지 확인, 의도와 다르면 서브스텝
      토큰화로 재설계

---

## 7. 관련

- Vault `Projects/Cross-View Sufficient Representation/README.md`, `1. 설계.md` — 설계
  원본, 이 문서의 상위 문서
- `docs/comp_mae_plan.md`, `docs/forecast_sufficient_plan.md` — 같은 형식·guard 스타일의
  선례
- `src/models/two_stream_v15.py` — 재사용 대상 파일 (owner 파이프라인·라우팅 블록 전부)
- `scripts/eval/probe_action.py` — action delta 계산 관행의 출처
