# Tab 6 — Catalyst Mechanism Evidence (Appendix or §5 supplementary)

**Paper position**: §5 Analysis (signature 보강) 또는 Appendix
**Status**: 🔴 not started — VideoMAE-ours C7 결과 + v15 ep32 P_t+P_tk 추출 대기

## Spec (Vault [[7. Outline § Tab 6]])

| Encoder | Pretrain data | P_t+P_tk R² (EgoDex) | LIBERO BC goal | 격차 |
|---|---|---|---|---|
| v15 | EgoDex 100M | **+0.390** | **83.7%** | reference |
| VideoMAE-ours | EgoDex 100M (same) | +X.XX (C7) | (대기) | controlled, same data |
| DINOv2 | LVD-142M images | +0.006 | 83.8% | single-frame SSL, more data |
| v11 | EgoDex 100M | +0.010 | 26.4% | no catalyst path |

**Caption 핵심**: "Same data (VideoMAE), more data (DINOv2), no catalyst path (v11) 모두 P_t+P_tk emerge X. v15 architectural choice가 catalyst의 인과."

## Source data 의존성

같은 4-encoder × 2-metric 데이터가 [`../../fig2_catalyst/`](../../fig2_catalyst/) Panel B (bar chart) 와 공유.

| Cell | Source |
|------|--------|
| v15 P_t+P_tk | [`../tab2_probing/`](../tab2_probing/) — TODO |
| v15 BC goal | [`../../fig3_bc_main/summary.csv`](../../fig3_bc_main/summary.csv) — TODO (round2) |
| VideoMAE-ours P_t+P_tk | C7 probing — 미실시 |
| DINOv2 P_t+P_tk | DINOv2 controlled — 결과 위치 확인 필요 |
| DINOv2 BC goal | fig3_bc_main round1 — ✅ 보유 (83.8%) |
| v11 P_t+P_tk | tab2_probing/v11_egodex_summary.csv — ✅ 보유 (+0.010) |
| v11 BC goal | fig3_bc_main round1 — ✅ 보유 (26.4%) |

## TODO

본 표는 fig2_catalyst와 같은 데이터 source — fig2 TODO 충족 시 자동 반영.

- [ ] **집계 스크립트** — 4 encoder × 2 metric (R², BC SR) 1-table
- [ ] **격차 column 자동 계산** — v15 vs others
- [ ] **LaTeX 표 + 강조** — v15 row highlight

## Cross-link

- [`../../fig2_catalyst/`](../../fig2_catalyst/) — 같은 데이터의 bar chart 시각화
- [`../tab3_ablation/`](../tab3_ablation/) — architecture ablation은 V-from-P vs V-from-M에 집중, 본 표는 같은 architecture 다른 encoder family 비교
