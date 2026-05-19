# Tab 2 — Probing Results (EgoDex + LIBERO)

**Paper position**: §4 Experiments (probing table)
**Status**: 🟡 partial — v11 / baseline + **v15 EgoDex 8-mode export 완료 (2026-05-19)**. LIBERO 12-cell v15 통합 진행 중.

## Spec (Vault [[7. Outline § Tab 2]])

| Section | Content |
|---|---|
| **A. EgoDex within-domain** (gap=10) | 5 encoder × 다양한 mode (P_t / P_tk / M / P_t+P_tk / A+B+D' / patch_mean_concat_all) |
| **B. LIBERO action probing** (4 gap × 3 suite) | 5 encoder × 12 cell (best mode per cell highlighted) |

## Current artifacts

| File | Section | 내용 |
|------|---------|------|
| `v11_egodex_summary.csv` | A | v11 11 epoch × 12 mode (probing JSON 집계) |
| `../../probing/v15_egodex_summary.csv` | A | **v15 main ep32/ep50 + v15-vfromm ep28~44** × {patch_mean_m_enc, p_enc, p_enc_tk, concat_p_t_p_tk, concat_enc_only, concat_p_tk_m, concat_all, concat_p_t_p_tk_m} (26 rows, target_mode=same/future 모두 포함) |
| `libero_all_gaps_summary.csv` | B | baseline + v11 LIBERO 12-cell (gap × suite) |
| `libero_per_cell/` | B raw | per-cell JSON / summary, _archive/ 포함 |

## v15 EgoDex 핵심 결과 (`../../probing/v15_egodex_summary.csv` 발췌, target_mode=same, gap=10)

| mode | composition | v15 main ep32 | v15 main ep50 (paper main) | v15-vfromm ep28~44 |
|------|-------------|---------------|-----------------------|---------------------|
| `patch_mean_m_enc` | M only | −0.083 | −0.058 | — |
| `patch_mean_p_enc` | P_t only | −0.053 | −0.043 | — |
| `patch_mean_p_enc_tk` | P_tk only | −0.014 | **−0.006** | — |
| `patch_mean_concat_enc_only` | M + P_t (= v11 A+B) | −0.081 | −0.138 | — |
| `patch_mean_concat_p_tk_m` | P_tk + M | −0.103 | −0.087 | — |
| `patch_mean_concat_all` | M + P_t + (P_tk routed) | −0.129 | −0.116 | — |
| **`patch_mean_concat_p_t_p_tk`** ★ | **P_t + P_tk** | **+0.390** | **+0.405** | +0.380~+0.415 |

**핵심 관찰**:
- v15 champion mode `patch_mean_concat_p_t_p_tk` (P_t + P_tk concat) = **+0.405 @ ep50** > 이전 v11 champion (`patch_mean_concat_all` ep44 = +0.288). +0.117 개선
- v11 동일 mode (`patch_mean_concat_p_t_p_tk`) = +0.010 ≈ 0 → P_t+P_tk pattern은 v15 specific
- 단독 mode (M only, P_t only, P_tk only)는 모두 음수 → P_t와 P_tk가 별도 정보 인코딩, concat이 implicit difference 학습
- DINOv2 controlled (single-frame SSL) = +0.006 → motion-routing이 P encoder에 motion-friendly pressure transfer 인과 확정 (catalyst evidence, fig2)

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

- [x] **v15 ep32 EgoDex 8-mode** export → `../../probing/v15_egodex_summary.csv` (2026-05-19 완료)
- [x] **v15 ep50 EgoDex** 동일 (`latest.pt`) — 8 mode 모두 포함
- [x] **v15-vfromm ep28~44** P_t+P_tk sweep — 동일 CSV에 통합
- [ ] **VideoMAE-ours EgoDex P_t+P_tk** (C7 — controlled comparison) — `data/probing_results/probe_videomae-ours_*.json` export
- [ ] **DINOv2 / SigLIP / VC-1 EgoDex P_t+P_tk** — single-frame SSL controlled (DINOv2 +0.006 결과 export 위치 확인)

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
