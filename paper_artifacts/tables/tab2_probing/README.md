# Tab 2 — Probing Results (EgoDex + LIBERO)

**Paper position**: §4 Experiments (probing table)
**Status**: 🟡 partial — v11 / baseline 보유, **v15 EgoDex / LIBERO 미통합**

## Spec (Vault [[7. Outline § Tab 2]])

| Section | Content |
|---|---|
| **A. EgoDex within-domain** (gap=10) | 5 encoder × 다양한 mode (P_t / P_tk / M / P_t+P_tk / A+B+D' / patch_mean_concat_all) |
| **B. LIBERO action probing** (4 gap × 3 suite) | 5 encoder × 12 cell (best mode per cell highlighted) |

## Current artifacts

| File | Section | 내용 |
|------|---------|------|
| `v11_egodex_summary.csv` | A | v11 11 epoch × 12 mode (probing JSON 집계) |
| `libero_all_gaps_summary.csv` | B | baseline + v11 LIBERO 12-cell (gap × suite) |
| `libero_per_cell/` | B raw | per-cell JSON / summary, _archive/ 포함 |

## CSV schema (`v11_egodex_summary.csv`)

| Column | 설명 |
|--------|------|
| `epoch` | v11 ckpt epoch (50=latest) |
| `mode` | feature 추출 모드 (12종) |
| `r2` / `mse` / `cosine_sim` | linear probe 결과 |
| `train_samples` / `eval_samples` | 180921 / 40914 (EgoDex test) |
| `embed_dim` | 768 / 1536 / 2304 |
| `source_file` | 원본 probe JSON |

### v11 mode 사전

| mode | position | pooling |
|------|----------|---------|
| `patch_mean_m_enc` | A (M encoder) | patch_mean |
| `patch_mean_p_enc` | B (P encoder) | patch_mean |
| `patch_mean_p_state_after_routing` | D' (motion-routing 후 P state) | patch_mean |
| `patch_mean_p_features_tk` | D (Phase 3 final) | patch_mean |
| `patch_mean_concat_enc_only` | A + B | concat |
| `patch_mean_concat_enc_phase3` | A + D | concat |
| `patch_mean_concat_enc_d_prime` | A + D' | concat |
| `patch_mean_concat_p_enc_d_prime` | B + D' | concat |
| `patch_mean_concat_all` | A + B + D' | concat — **v11 champion ep44 R²=+0.288** |
| `patch_mean_concat_p_t_p_tk` | B(t) + B(tk) | concat — **v15 champion mode, v11에서는 +0.010** |
| `cls_m_enc` / `cls_p_enc` / `cls_concat_enc` | A / B / A+B | CLS |

## TODO

### Section A (EgoDex)

- [ ] **v15 ep32 EgoDex 12-mode** export — `data/probing_results/` v15 JSON → CSV
- [ ] **v15 ep50 EgoDex** 동일 (ep32 vs ep50 trajectory)
- [ ] **VideoMAE-ours EgoDex P_t+P_tk** (C7 — controlled comparison)
- [ ] **DINOv2 / SigLIP / VC-1 EgoDex P_t+P_tk** — single-frame SSL controlled (DINOv2 +0.006 결과 위치 확인)

### Section B (LIBERO)

- [ ] **v15 ep32/ep50 LIBERO 12-cell probing** — `scripts/eval/probe_action_libero.py`
- [ ] `libero_all_gaps_summary.csv`에 v15 row 추가 (export 스크립트 재실행)

### 표 생성

- [ ] **LaTeX 표 스크립트** — Section A + B 합치는 multi-row table
- [ ] **best mode highlight** — 5 encoder별 column-wise best
- [ ] **v15 P_t+P_tk +0.390** 행 별표

## Cross-link

- v11 P_t+P_tk +0.010, v15 P_t+P_tk +0.390 → [`../../fig2_catalyst/`](../../fig2_catalyst/) bar chart 동일 data
- v11 mode `patch_mean_concat_all` ep44 +0.288 → [`../tab3_ablation/`](../tab3_ablation/) v11 reference row

## Source

- Probing script (EgoDex): [`scripts/eval/probe_action.py`](../../../scripts/eval/probe_action.py)
- Probing script (LIBERO): [`scripts/eval/probe_action_libero.py`](../../../scripts/eval/probe_action_libero.py)
- Export: [`scripts/analysis/export_probing_summary.py`](../../../scripts/analysis/export_probing_summary.py)
- 원본 JSON: `data/probing_results/{egodex,libero}/` (gitignored)
