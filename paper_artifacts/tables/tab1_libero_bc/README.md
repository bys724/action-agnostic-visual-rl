# Tab 1 — LIBERO BC Main Results

**Paper**: 2 (AAAI, Action-Agnostic)

**Paper position**: §4 Experiments (★ main table)
**Status**: source = [`../../libero_rollout/summary.csv`](../../libero_rollout/summary.csv) — parvo + v15 variants 포함 최신 (구 fig3 round1 CSV는 통합·제거)

## Spec (Vault [[7. Outline § Tab 1]])

| Row | Column |
|---|---|
| 5 encoder (v15-ptptk highlighted leftmost ★) | spatial / object / goal × {mean ± std} = 6 cols + average |

- Best per column: **bold**
- v15-ptptk row: highlight
- v15-mp 별도 §5 ablation (Tab 3)
- V-JEPA 2.1 footnote: "excluded, 24× input cost"
- v11 row supplementary only (paper main 제외 5/12 결정)

## Source

[`../../libero_rollout/summary.csv`](../../libero_rollout/summary.csv) — single source of truth (LIBERO BC rollout 집계).

## TODO

[`../../fig3_bc_main/README.md`](../../fig3_bc_main/README.md) TODO 와 동일 (data = `libero_rollout/`).

추가:
- [ ] **LaTeX 표 생성 스크립트** — `summary.csv` → booktabs 형식
- [ ] **bold/highlight 로직** — column별 best 자동 검출
