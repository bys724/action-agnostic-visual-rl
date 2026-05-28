# CoRL 2026 Paper — v15 Baseline & 전면 수정 계획

> **이 branch (`paper-corl2026`)의 정체**: 논문(CoRL 2026)용 **v15 baseline 코드 고정 + 논문 작업 가이드**.
> 분기점 `8210f0d` = 논문 모든 결과를 낸 v15 (**teacher anchor, predictor-only**).
> `main`(`0fb74c8`+, **student anchor = 새 구조**)과 **영구 분리**. 클러스터 논문 작업은 이 branch에서.

**작성**: 2026-05-28 (main.pdf 정독 + 코드 검증 중 scaffolding 인과 미검증 발견)
**상세 결정 기록 (Vault)**: `Obsidian Vault/Projects/Action-Agnostic Paper/우려사항 및 대응방안.md` §0 / §0.1 / §0.2
**논문 문구 review (paper repo)**: `action-agnostic-paper/notes/main_pdf_review.md` (최상단 CRITICAL 플래그)

---

## 0. 왜 이 branch가 필요한가

`main`의 `0fb74c8`이 `src/models/two_stream_v15.py`를 **student anchor**(V-JEPA gradient ON)로 수정 = **새 구조(catalyst 검증용)**. 논문 v15는 `8210f0d`의 **teacher anchor**(predictor-only). `main`을 pull하면 논문 코드가 새 구조로 바뀌므로, **논문 작업은 반드시 이 branch(`8210f0d` 기반)에서** 한다.

| | 논문 v15 (이 branch) | 새 구조 (`main`) |
|---|---|---|
| 기준 commit | `8210f0d` | `0fb74c8`+ |
| V-JEPA P anchor | `teacher_p(frame_t).detach()` | student P encoder (grad ON) |
| P encoder 학습 | reconstruction(MAE) only | + motion routing gradient |
| 용도 | **논문 (클러스터)** | redesign 재학습 (워크스테이션) |

---

## 1. 상황 요약 — scaffolding 인과 미검증

- **student P encoder는 reconstruction(3-frame MAE)만으로 학습** — motion track gradient 차단 (teacher = `copy.deepcopy` + EMA, V는 `.detach()`). 코드 확정.
- **probe(+0.390) · BC(86.4/83.7) 모두 P encoder를 두 frame에 raw forward한 `P_t ⊕ P_tk` concat** — motion routing / M encoder / `predicted_repr` **미사용**.
  - 근거: `scripts/eval/probe_action_libero.py:152` (`p_t_p_tk` → `_p_encoder_forward` ×2), `src/encoders/adapters/two_stream_v15_pt_ptk.py` (P encoder ×2, "Motion-routing/M encoder 미사용").
- → 논문의 **"scaffolding effect(motion routing이 P를 motion-aware하게)" 인과는 코드와 불일치**.
- **D1 확정 (코드 diff)**: +0.380 gap(v11 `+0.010` → v15 `+0.390`) = **reconstruction path asymmetry → symmetry**.
  - v11(`two_stream_v11.py` docstring l.25-41): frame_t는 interpreter_1로 direct, frame_{t+k}는 motion routing ×2 → interpreter_2로 **routed** → raw P_{t+k} 약함.
  - v15(`two_stream_v15.py` `_mae_one_frame` ×3): 모든 frame **direct MAE** → raw P_t·P_{t+k} 둘 다 강함.
  - **함의**: motion routing을 P encoder 학습에서 **빼니까(v15)** raw P가 frame-discriminative — scaffolding의 정반대.

> ⚠️ **정정**: `0fb74c8` 진단의 *"M routing +0.31 기여 (M=0 0.66 vs M 0.97)"* 는 **motion track `predicted_repr`** 품질 측정. 논문 probe/BC는 **raw P encoder**(`predicted_repr` 안 씀)라 **무관**.

---

## 2. 자산 A/B/C/D — 평가 결과는 전부 진실, 인과 해석만 교체

**A. 그대로 사용 (raw evidence)**: LIBERO BC(v15 63.0/86.4/83.7), EgoDex probe(+0.390), DROID cross-domain, CALVIN cross-folder, CortexBench(Meta-World 89.9), view-sensitivity(Δ +0.211).

**B. 정당한 주장 (mechanism 주장 제외, 현상+비교만)**:
- ~100M frames로 LIBERO object/goal baseline 매칭 (spatial 약점 정직 보고)
- v15 `P_t⊕P_tk` frame-discriminative (+0.390) vs single-frame(−0.05) / v11(+0.010)
- v11(+0.010) → v15(+0.390) architecture 변경이 raw P encoder 개선

**C. 폐기 (코드와 불일치 — 못 씀)**:
- scaffolding effect (motion routing → student P motion-aware)
- "deploy 시 motion machinery 제거" (probe/BC가 애초에 미사용)
- motion routing이 load-bearing
- "+0.384 = scaffolding 증거" (실제 v11 vs v15 차이)
- "reconstruction-only Two-Stream(+0.010)" 라벨 (실제 **v11**, motion routing 있는 모델)

**D. 규명 후 확정 (클러스터 ablation)**:
- **D1 ✅** — reconstruction symmetry (위 §1).
- **D2 ✅** — VideoMAE-ours(RGB pixel MAE, **Sobel 없음**) `P_t⊕P_tk` EgoDex gap=10 = **+0.4705** > v15(Sobel+RGB) **+0.3898**. → **Sobel-channel은 +0.390의 핵심 아님** (오히려 RGB-only가 더 높음). 핵심은 **symmetric multi-frame MAE** (D1과 정합). 데이터: [`paper_artifacts/fig2_catalyst/d2_sobel_isolation.csv`](../paper_artifacts/fig2_catalyst/d2_sobel_isolation.csv).
  - 보강 비교 (동일 probe): v11(asymmetric, frame_t+k routed) **+0.0097**, dinov2(single-frame) **+0.0062**. → symmetric MAE(videomae 0.47 / v15 0.39) ≫ asymmetric(v11 0.01) / single-frame(dinov2 0.01).
  - ⚠️ VideoMAE vs v15는 Sobel 외에도 frame수(2 vs 3)·architecture 차이 있음 → 순수 Sobel 격리는 **D3** (v15에서 Sobel만 제거)에서 확정.
- **D3**: v15에서 요소(multi-frame / sampling-time / Sobel / EMA) 하나씩 제거 → +0.390 어디서 무너지나.

---

## 3. 논문 전면 수정 방향

- scaffolding / motion-routing 서사 **전체 폐기**.
- **새 framing 후보** (D2/D3 확정 후 finalize):
  > "복잡한 motion routing 없이, **Sobel-channel 대칭 multi-frame MAE 단일 P encoder**가 two-frame concat으로 frame-discriminative representation을 만들어 robot control에 경쟁력. v11→v15는 motion routing을 P encoder 학습에서 **분리**한 것이 핵심."
- §3 Method, §4 mechanism, C1-C4 contribution 전면 재작성.
- 진행 순서: **D2/D3로 진짜 원인 확정 → 그에 맞춰 재서술** (추측으로 다시 쓰지 말 것).

---

## 4. 클러스터 작업 규칙 (혼란 방지)

- ✅ 논문 작업(D2/D3, LaTeX 연동 실험)은 **이 branch(`paper-corl2026`)에서만**.
- ✅ D3 ablation **variant**(v15에서 요소 뺀 실험)는 이 branch에 추가 OK.
- ❌ **새 구조(student anchor, `main`)를 이 branch에 merge 절대 금지.** 양방향 영구 분리.
- ℹ️ `8210f0d`는 **코드 commit** — 학습된 v15 weight(`.pt`)는 별도 로드 (클러스터 저장소 / `paper_artifacts`).
