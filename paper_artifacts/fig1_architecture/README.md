# Fig 1 — Architecture figures (MCP-MAE · MS-JEPA)

**Paper position**: Paper 2 (AAAI) §3 Method (main figure) + method-history.
**생성 소스·코드**: [`scripts/viz/arch_figs/`](../../scripts/viz/arch_figs/) (source/artifact 분리 — 이 폴더는 렌더 결과만).
명명·맥락 → [`CLAUDE.md`](../../CLAUDE.md) "명명 · 2논문 구조" · [`docs/REFACTOR_PLAN.md`](../../docs/REFACTOR_PLAN.md) §1.

## 모델 한 줄 정의

- **MCP-MAE** (현행 ours, code `--v15-pixel-pred`): appearance(P)·motion(M) 분리, M이 `where`(Q,K)로 P의 `what`(V)을 routing해 **미래 픽셀** 예측. JEPA/EMA 제거 → **collapse 불가**, scaffold 유지.
- **MS-JEPA** (predecessor, Parvo code v15b): 동일 routing이나 **EMA-teacher LATENT** 예측(V-JEPA). self-referential target → collapse 위험(target-LN·var-reg 필요) → MCP-MAE가 픽셀 target으로 대체.

## 산출물 (현행)

| File | 모델 | 용도 | 도구 |
|------|------|------|------|
| **`mcp_mae_fig.png` / `.pdf`** | MCP-MAE | **메인 Fig 1** — SiamMAE 스타일 직관 도식 (실제 EgoDex 프레임) | matplotlib |
| `mcp_mae_architecture.png` | MCP-MAE | 상세 dataflow — 통일 predict() ×3 | Mermaid |
| **`ms_jepa_fig.png` / `.pdf`** | MS-JEPA | 동일 스타일 도식 (predecessor 비교) | matplotlib |
| `ms_jepa_architecture.png` | MS-JEPA | 상세 dataflow — two heads + EMA teacher | Mermaid |

- **`*_fig`** = "한눈에 이해" SiamMAE-스타일 (발표·논문 main 1순위). 두 모델이 **나란히 비교**되도록 동일 레이아웃.
- **`*_architecture` (mermaid)** = 정확한 학습 dataflow (supplementary / 내부 검토).
- 발표용 컨셉/hero 이미지(Nano Banana) → [`../presentation/`](../presentation/).

## 핵심 시각 메시지 (공통)

1. **what / where factorization**: P=appearance(V), M=ΔL motion(Q,K). ΔL은 attention 가중치로만 — 출력은 P appearance remix (additive 아님; `MotionRoutingBlock`).
2. **MCP-MAE vs MS-JEPA**: 유일 차이 = prediction **target**. MS-JEPA=EMA-teacher latent(붕괴 위험), MCP-MAE=실제 픽셀(붕괴 불가). 두 figure를 나란히 두면 이 한 점이 드러남.
3. **scaffold**: 양쪽 다 L_pred만 M-조건 → M이 gradient로 P를 shaping. recon/grounding은 M-비조건.
4. **배포**: downstream은 **P encoder만** 사용(patch-mean, P_t⊕P_tk). M 기여는 사전학습 gradient로 P에 frozen-in.

## STEP 1 matched 3런과의 관계 (restart_plan §3)

같은 `two_stream_v15.py` 코드, 플래그만 차이 — `mcp_mae_fig`는 **MCP-MAE(thesis)** 를 그림:

| 런 | 플래그 | routing | 그림 대응 |
|----|--------|---------|-----------|
| **MCP-MAE** (ours) | `--v15-pixel-pred` | Q,K ← **M (ΔL)** | `mcp_mae_fig` |
| SiamMAE-analog (대조) | `+ --v15-routing-source p` | Q,K ← **P (RGB)** | "where"를 ΔL 대신 RGB로 |
| Image MAE / no-M | `--v15-no-motion` | routing off | two-frame image MAE (Paper 1 baseline) |

## 재생성

[`scripts/viz/arch_figs/README.md`](../../scripts/viz/arch_figs/README.md) 참조 (repo 루트에서 실행).

## Source

- 모델: [`src/models/two_stream_v15.py`](../../src/models/two_stream_v15.py) — MCP-MAE `_forward_pair_pixel`/`_predict_pixels`, MS-JEPA `_forward_pair`/`_vjepa_p_masked`/`_mae_one_frame`
- routing: [`src/models/common/blocks.py`](../../src/models/common/blocks.py) `MotionRoutingBlock`
- 채널: [`src/models/common/preprocessing.py`](../../src/models/common/preprocessing.py) (no-Sobel: P=RGB 3ch, M=ΔL 1ch)
- 설계 근거: [`docs/v15b_retraining_status.md`](../../docs/v15b_retraining_status.md) §9, [`docs/restart_plan.md`](../../docs/restart_plan.md)
