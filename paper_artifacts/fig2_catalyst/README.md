# Fig 2 — Catalyst Mechanism Evidence (★ signature)

**Paper**: 2 (AAAI, Action-Agnostic)

**Paper position**: §5 Analysis (★ key finding figure)
**Status**: 🔴 not started — schematic + bar chart 모두 생성 필요

## Spec (Vault [[7. Outline § Fig 2]])

**2-panel layout**:

- **Panel A — Schematic** (좌 50%): 3 encoder side-by-side
  - v15: `P_enc → MAE` ⊕ `Teacher_P → motion routing` → EMA loop **(catalyst path highlighted)**
  - DINOv2: single image → augmentation → self-distillation (no motion path)
  - VideoMAE-ours: 2-frame MAE only (no motion routing)
- **Panel B — Empirical** (우 50%, bar chart):
  - X: encoder type (v15 / VideoMAE-ours / DINOv2 / v11)
  - Y: P_t+P_tk linear probe R² (EgoDex, gap=10)
  - v15 별표 + annotation "+0.384 gap = catalyst signature"

## Source data

| Encoder | P_t+P_tk R² | Source file |
|---------|-------------|-------------|
| **v15 ep32** | **+0.390** ★ | C4 결과 필요 (probe_action.py with `--cls-mode patch_mean_concat_p_t_p_tk`) |
| VideoMAE-ours | TBD (C7) | 미실시 — v15와 같은 EgoDex 100M 데이터, controlled |
| DINOv2 controlled | +0.006 | 기존 single-frame probe (controlled comparison 실험) |
| v11 ep44 | +0.010 | [`tables/tab2_probing/v11_egodex_summary.csv`](../tables/tab2_probing/v11_egodex_summary.csv) (mode `patch_mean_concat_p_t_p_tk` row) |

## TODO

- [ ] **v15 ep32 P_t+P_tk probing CSV 추출** — `data/probing_results/` 의 v15 결과를 export
- [ ] **VideoMAE-ours P_t+P_tk probing** (C7) — same-data controlled — `scripts/eval/probe_action.py --encoder videomae-ours --cls-mode patch_mean_concat_p_t_p_tk`
- [ ] **bar chart 스크립트** — matplotlib 4-encoder × 1-mode 바 (error bar = k-fold variance)
- [ ] **schematic 스크립트** — Inkscape/Illustrator 또는 TikZ (Panel A)

## 활용 cross-link

같은 데이터가 [`tables/tab6_catalyst_evidence/`](../tables/tab6_catalyst_evidence/)에서도 사용됨.
