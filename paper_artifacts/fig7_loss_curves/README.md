# Fig 7 — Training Loss Curves (Appendix C)

**Paper position**: Appendix C (training dynamics)
**Status**: 🔴 not started — wandb 추출 필요

## Spec (Vault [[7. Outline § Fig 7]])

- X = epoch (1-50)
- Y = loss value (log scale)
- 5 lines: `L_t`, `L_tk_recon`, `L_pred`, `L_m_jepa`, `L_compose`
- **L_m_jepa 폭증 별표** (ep7 0.0009 → ep28 0.086) — Appendix F 결함 분석과 연계

## Argument

v15 학습 결함 (P encoder CLS collapse + L_m_jepa 폭증)을 정직 보고. 그럼에도 **catalyst representation은 emerge** — 결함의 본질 = "P CLS collapse는 patch 학습에 무관" (사용자 framing, 2026-05-13).

## Current artifacts

없음.

## TODO

- [ ] **wandb run에서 5 loss curve CSV 추출** — v15 본 학습 (JobID 34288968, 2026-05-12 완주)
  - 또는 ckpt directory의 학습 로그 (`train_log.txt` 등)에서 파싱
- [ ] **matplotlib line plot 스크립트** — log scale Y, 5 line, 색 통일 (loss type별)
- [ ] **L_m_jepa annotation** — ep7 / ep28 marker + 텍스트
- [ ] **secondary plot** (선택): cos_intra_p / patch_cos diagnostics — Appendix F 동반 figure

## Source

- v15 학습 ckpt: `/proj/external_group/mrg/checkpoints/two_stream_v15/<JobID>/`
- diagnose 스크립트 (가능시 재활용): [`scripts/eval/diagnose_v15_collapse.py`](../../scripts/eval/diagnose_v15_collapse.py)
- wandb project: 확인 필요 (CLAUDE.md / cluster_sessions에 표기 X)
