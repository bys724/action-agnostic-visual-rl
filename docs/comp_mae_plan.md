# CoMP-MAE — 대칭 Cross-Reconstruction Plan (M-recon 분기, code v16) (2026-06-26)

> **이름**: CoMP-MAE = **Co**-reconstructive **M**agno-**P**arvo **MAE**. Co=상호 cross-reconstruction(이번 본질), MP=magno(ΔL)/parvo(RGB) 두 stream(불변 골격), MAE=pixel objective. 코드 버전 = **v16** (개념 이름과 분리). 선행: MS-JEPA(v15) → MCP-MAE → **CoMP-MAE(v16)**.

> 설계 출처: Vault `Projects/Action-Agnostic Paper/1. Core Idea.md` §대칭 Cross-Reconstruction 설계 (MotionMAE 읽기 + 대화로 정련).
> 본 문서는 **구현 참고(계획·주의·pseudocode)** 다. 실제 코드는 dev 세션에서 작성.
> 기존 MCP-MAE 노선(JEPA 제거·pixel 통일)의 연장 — `docs/v15b_retraining_status.md §9` 위에 쌓는다.

> **버전 결정 (2026-06-26)**: 이 설계(**CoMP-MAE**, code v16)가 **MCP-MAE를 대체하는 현재 ours 축** (replace). ⚠️ MCP-MAE도 미학습이라 "미학습 위에 미학습" stack → plain MCP-MAE를 별도 baseline으로 학습하지 **않음**. **attribution은 본 문서 §6 내부 ablation**(routing on/off · no-M · V-source `V_M` vs `V_P`)으로 보존 — 별도 plain baseline 없이도 "M-recon 기여"를 격리. restart_plan STEP 1의 ours arm = dual-cross-recon-S (MCP-MAE-S 대체), 나머지 STEP·대조군·게이트 불변.

---

## 1. 목표

현재 두 stream은 **P 복구로만** 학습된다(M은 routing helper로만 gradient 수령 → v15에서 no-op underdeliver, §9·Vault 핵심 리스크). M에 **자기 ΔL을 복구하는 grounded pixel 목적**을 직접 주어, M-encoder가 실제 motion을 표상하도록 강제한다. 구조는 P-recon의 **대칭 미러**.

성공 기준(한 줄): M-recon 추가 후 **routing ablation에서 recon/transfer가 유의하게 하락**(= M이 실제로 P를 돕는다) + **dissociation probe 통과**(M=motion, ≠identity). 둘 중 하나라도 실패면 no-op 재발로 판정.

## 2. 현재 vs 추가 (구현 현황)

| 구성요소 | 현재 (v15) | 추가 (M-recon) |
|---|---|---|
| P 복구 루트 | ✅ `RoutingInterpreterStep` × N (two_stream_v15.py:54-82) | — |
| M 복구 루트 | ❌ (M decoder = V-JEPA predictor / MCP-MAE서 M-side 축소) | ✅ **신규**: masked ΔL pixel 복구 |
| cross-attn | `v_from_p`+`routing_source='m'` (V_P / Q_M·K_M) | **신규 variant**: V_M / Q_P·K_P (§4 주의) |
| 복구 ordering | complete-first (`decode_first=masked_anchor`, :314) | 미러로 동일 |
| collapse 대책 | pixel-recon, no JEPA/EMA | 동일 (student only) |

## 3. 아키텍처 spec

- **각 stream: 자기 마스킹 / 상대 full.** P-recon은 M-full로, M-recon은 P-full로 cross-condition.
- **cross-attn 원칙 — "V는 복구 대상, Q·K는 helper"**:
  - P-recon (기존): `V_P / Q_M·K_M` — attention M→M(correspondence), gather P.
  - M-recon (신규): `V_M / Q_P·K_P` — attention P→P(grouping), gather M.
  - 출력은 V 공간에 산다 → M-recon은 출력=ΔL이므로 V_M 필연. helper(P)는 패턴(Q·K)만.
- **복구 대상: static(gap=0) + future(gap=k)** 둘 다 (multi-gap, MCP-MAE 정신).
- **디코더 step**: mask token 주입 → **self-attn(완성) → routing**(complete-first) interleave. M-recon은 M-self-attn 완성 + P-grouping routing 미러.
- **deliverable**: 전이 인코더는 P-encoder(기존). M-recon은 M-encoder를 grounding + (mutual coupling으로) P-encoder도 간접 개선.

## 4. Critical guards (구현 시 실수 방지)

1. **grouping = `v_from_p` 입력 swap (새 mode 아님) ≠ `v_from_m`** ⚠️ (가장 흔할 실수)
   - 핵심: `MotionRoutingBlock`의 `v_from_p`는 이미 **역할 대칭** — 메커니즘은 `softmax(QK←둘째 인자) @ V(←첫째 인자)`. V는 항상 첫 인자(`p_state`)에서, QK는 `src='m'`일 때 둘째 인자에서 (blocks.py:216-225, residual도 owner=첫 인자에 붙음).
   - 기존 `MotionRoutingBlock` (common/blocks.py:141-248):
     - `v_from_p`+`src='m'`, 인자=(P, M): V=P / Q,K=M → **M→M attention, gather P** (P-recon)
     - `v_from_m`: Q=P / K,V=M → **P→M attention, gather M** ← 우리가 원하는 게 **아님**
   - M-recon이 원하는 **Q,K=P / V=M (P→P grouping, gather M)** = 같은 `v_from_p`+`src='m'`을 **인자만 swap**(`v_owner=M, qk_helper=P`)하면 그대로 나옴 → **새 routing_mode 추가 불필요.** 필요한 것: ① forward 인자명 generic화(`p_state,m_completed` → `v_owner_state,qk_helper_state`), ② M-recon용 **별도 인스턴스**(자기 qk/v projection = param-symmetry, P-recon과 weight 공유 금지).
2. **leakage 비대칭 (trivial 방지)**: ΔL = |F(t+1)−F(t)| 는 P 프레임의 결정론적 함수. P-full이 타깃 ΔL의 **양쪽 프레임을 bracket하면 빼기로 trivial** → 특히 static(gap=0). M-recon 타깃 ΔL을 **P visible이 bracket하지 않게** 시간 offset 구성. M bottleneck 타이트 유지, M target=ΔL 고정.
3. **complete-first ↔ routing no-op trade-off**: interpreter(self-attn)가 너무 강하면 routing 없이도 복구돼 M 기여 redundant. interpreter depth 과다 금지 + **routing ablation 필수**(§6).
4. **mutual coupling용 gradient**: M-recon의 `Q_P·K_P`는 **P-encoder full-pass에서 grad 흐르게** (raw patch 아님). 그래야 L_M이 P-encoder도 빚음(=진짜 mutual). 단 guard 2와 동시 만족.
5. **M recon head**: ΔL은 1채널(`[ΔL]`, Sobel-free 확정) → P recon head(RGB 3ch) 재사용 금지, **1채널 head 별도**.
6. **helper full-pass 비용**: P-recon은 M-full, M-recon은 P-full 필요 → 각 encoder가 masked 1회 + full 1회. M을 full 인코딩하면 M-recon 타깃 leak → **M-recon의 M은 반드시 masked 별도 pass** (full pass 재사용 금지).

## 5. 구현 TODO 체크리스트 (dev 세션)

- [ ] `MotionRoutingBlock` forward 인자명 generic화(`v_owner_state, qk_helper_state`) + M-recon용 **별도 인스턴스**(`v_owner=M, qk_helper=P`로 호출 → P→P grouping, gather M). **새 routing_mode 아님** — guard 1 참고. param-symmetric(별도 projection) 유지.
- [ ] M-recon용 `RoutingInterpreterStep` 미러: M-self-attn `interp` + P-grouping `routing`, `decode_first=True`.
- [ ] M decoder를 pixel-recon 분기로: mask_token_m 주입(이미 존재) → APE → interleave step × N → **1채널 ΔL recon head**.
- [ ] P-encoder full-pass 경로 추가 (M-recon helper용 Q_P,K_P, grad on) + guard 2/6 만족하는 마스킹/시간 구성.
- [ ] Loss에 `L_M_recon` 추가: static(gap=0)+future(gap=k), λ_M 가중. moving-region 가중 옵션(sparse-ΔL zero collapse 방지).
- [ ] flag/config: `--dual-cross-recon` (기존 `--v15-pixel-pred` 계열과 정합), λ_M·interpreter depth·routing on/off 노출.
- [ ] DDP-safe: 미사용 모듈 freeze/skip 패턴 기존(v15:373-388) 따름.

### pseudocode skeleton (구조만 — 실제 구현은 dev 세션)

```
# grouping = 기존 v_from_p 인자 swap (새 mode 아님; forward 인자명만 generic화)
#   routing_M = MotionRoutingBlock(routing_mode="v_from_p", routing_source="m")  # 별도 인스턴스
#   m_state   = routing_M(v_owner=m_state, qk_helper=p_full)
#   → softmax(Q_P @ K_P^T) @ V_M          # P→P grouping, gather motion (residual on M)
#   (v_from_m 과 혼동 금지 — 그건 P→M, gather M 으로 우리가 원하는 게 아님)

# M-recon step (RoutingInterpreterStep 미러)
#   if decode_first:  m_state = interp_M(m_state)         # self-attn 완성 먼저
#                     m_state = routing_grouping(m_state, p_full)  # P-grouping
#   m_recon = head_dL(m_state[:, 1:])                     # 1채널 ΔL

# loss
#   L_M = sum_gap MSE(m_recon[gap], dL_target[gap])[masked]   # gap∈{0,k}
#   L_total = L_P(기존) + lambda_M * L_M
#   # guard 2: dL_target[gap=0] 의 양쪽 프레임이 P visible에 동시 존재하지 않게
```

## 6. 검증 체크리스트 (hand-off 전)

- [ ] **routing ablation**: M-recon의 P-grouping routing 제거 → ΔL recon 악화 확인 (안 악화면 grouping no-op).
- [ ] **P-side routing ablation**: 동일하게 M→P routing 제거 → P recon/transfer 하락 확인 (complete-first no-op 점검).
- [ ] **dissociation probe**: M-encoder 출력에서 motion(where/Δ크기)은 디코딩 ↑, object identity는 디코딩 ↓ (chance 근처).
- [ ] **V-source ablation**: M-recon `V_M` vs `V_P` 비교 → V_P일 때 M-encoder 표상 빈약(probe 하락) 확인.
- [ ] **trivial 체크**: gap=0 M-recon loss가 비정상적으로 0에 근접하지 않는지(=빼기 leak) 모니터.
- [ ] **transfer**: held-out domain probing/BC로 M-recon on/off 비교 (factorization OOD 이득 — Vault §일반화 프레임).

## 7. Cross-refs

- 설계 source: Vault `1. Core Idea.md` §대칭 Cross-Reconstruction 설계, §Motion-Routing 설계 근거
- 선행 노선: `docs/v15b_retraining_status.md §9` (MCP-MAE: JEPA→pixel 통일), `docs/restart_plan.md` (subset-matched 검증)
- 관련 코드: `src/models/common/blocks.py:141-248` (MotionRoutingBlock), `src/models/two_stream_v15.py:54-82` (RoutingInterpreterStep), `:457-461` (mask 주입)
- 선례 대조: MotionMAE(motion-as-target), MultiMAE(cross-modal masked recon) — 본 설계 ≈ temporal MultiMAE + motion routing
