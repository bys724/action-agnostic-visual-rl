# Fig 1 — v15 Architecture Overview

**Paper position**: §3 Method (main figure)
**Status**: 🟡 partial — v11 method-history만 보유, v15 main 미작성

## Spec (Vault [[7. Outline § Fig 1]])

- Top half: reconstruction track (student P encoder space) — `P_enc → interpreter_1 → recon_head`
- Bottom half: motion track (teacher P encoder space) — `Teacher_P → motion_routing (Q,K←M / V←teacher_P) → p_motion_decoder`
- EMA loop이 두 half 연결 (momentum 0.999→0.9999)
- L_compose path: `composition_head(M_short, M_long) ≈ Teacher_M(target)`
- 색 컨벤션: M=blue / P=red / motion routing=purple / loss=green / mask=gray hatched
- 도구: TikZ 권장 (Mermaid는 draft)

## Current artifacts

| File | 용도 | Paper position |
|------|------|----------------|
| `v11_method_history.mmd` | v11 Mermaid source | §method history paragraph (1-2 문장) |
| `v11_method_history.png` | rendered v11 diagram | 동일, supplementary 가능 |

## TODO

- [ ] **v15 architecture diagram** — `.mmd` draft + `.tikz` 최종본
  - 4 layer (P_enc / p_motion_decoder / M_enc / M_decoder) × 본인 paradigm 명시
  - 5 loss 표시: `L_t`, `L_tk_recon`, `λ·L_pred`, `λ·L_m_jepa`, `λ·L_compose`
  - 가장 강조해야 할 시각 메시지: **space-level paradigm separation** (두 paradigm이 서로 다른 representation space에서 작동)
- [ ] **Channel composition mini-diagram** (§3.1 inset 또는 Fig 1 측면): M=3ch(ΔL+2Sobel), P=5ch(2Sobel+RGB)

## Source

- 모델 코드: [`src/models/two_stream_v15.py`](../../src/models/two_stream_v15.py)
- v11 (참고): [`src/models/two_stream_v11.py`](../../src/models/two_stream_v11.py)
- Mermaid 재렌더: `mmdc -i v11_method_history.mmd -o v11_method_history.png -t neutral -b transparent`
