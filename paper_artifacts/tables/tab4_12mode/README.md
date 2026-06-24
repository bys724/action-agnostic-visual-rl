# Tab 4 — 12-Mode Probing Breakdown (v15 ep32)

**Paper**: 2 (AAAI, Action-Agnostic)

**Paper position**: §5.4 Analysis (catalyst mechanism quantitative breakdown)
**Status**: 🔴 source CSV 별도 없음 — [`../tab2_probing/v11_egodex_summary.csv`](../tab2_probing/v11_egodex_summary.csv) 패턴으로 v15 export 필요

## Spec (Vault [[7. Outline § Tab 4]])

v15 ep32 EgoDex within-domain (gap=10):

| Mode | R² |
|---|---|
| P_t only | -0.053 |
| P_tk only | -0.014 |
| M only | -0.083 |
| **P_t + P_tk** | **+0.390** ★ |
| M + P_t | -0.081 |
| P_tk + M | -0.103 |
| A+B+D' (all) | -0.129 |

→ Caption: "두 frame P feature가 서로 다른 정보 인코딩, linear probe가 implicit difference 학습. Single-position mode 모두 음수 — catalyst의 mechanism은 *frame discrimination*에 있음."

## Source data

[`../tab2_probing/`](../tab2_probing/) 에서 v15 12-mode export 후 ep32 row 필터링.

## TODO

- [ ] tab2_probing의 v15 ep32 EgoDex 12-mode CSV가 도착하면 → 본 표는 그 subset
- [ ] **참고 비교 추가** (선택): v11 ep44 동일 12-mode (P_t+P_tk +0.010 vs v15 +0.390 격차 강조)
- [ ] **LaTeX 표 스크립트** — 단일 column 강조 표

## Notes

본 표는 tab3 (architecture ablation) 의 sub-evidence — catalyst가 specific mode (P_t+P_tk)에서만 발현되는 것을 정량 보여줌.
