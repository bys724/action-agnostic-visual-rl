# Fig 5 — Cross-Domain DROID Probing

**Paper**: 공유 (Paper 1 + Paper 2) — DROID cross-domain은 양 논문 공통 평가. 공유 데이터(`*_droid_summary.csv`)를 두고 각 논문이 자기 narrative로 figure를 별도 렌더 (Paper 1=input-prior 일반화 / Paper 2=action-agnostic 일반화)

**Paper position**: §4 Experiments (cross-domain probing supplementary)
**Status**: 🟡 partial — v11 / VideoMAE-ours 보유, **v15 미실시 (C5)**

## Spec (Vault [[7. Outline § Fig 5]])

- Line plot: X = gap (1, 10, 15, 30) — DROID 15 Hz
- Y = R² (probing best mode per encoder)
- 5 lines, v15 굵게 highlight
- gap=15 vertical dashed: "EgoDex training distribution alignment"
- Y range 신중히 (절대값 작음, 격차 시각화 우선)
- **정직 보고**: DROID 절대값 작음 (~0.005), all encoders 한계 — image-only feature setting

## Current artifacts

| File | Encoder × dataset | Rows | 비고 |
|------|-------------------|------|------|
| `v11_droid_summary.csv` | two-stream-v11 × DROID | 13 rows (ep12, partial) | gap × 12 mode |
| `videomae_droid_summary.csv` | videomae-ours × DROID | 4 (gap × patch_mean only) | partial baseline |

DROID coverage 부족 (v11 ep12 only): 원래 supplementary 결정으로 acceptable.

## TODO

- [ ] **v15 ep32/ep50 DROID 4-gap probing** (C5) — `scripts/eval/probe_action_droid.py --encoder two-stream --checkpoint <v15_ckpt> --cls-mode patch_mean_concat_p_t_p_tk`
- [ ] **DINOv2 / SigLIP / VC-1 DROID probing** — baseline 비교 위해 4-gap 추가 (or 기존 cluster_sessions §C5에 있는지 확인)
- [ ] **line plot 스크립트** — matplotlib 5-line (encoder) × 4-x (gap)
- [ ] **v15 P_t+P_tk DROID 결과** 별도 highlight — EgoDex within과 동일 mode 비교 위해

## Source code

- DROID probing: [`scripts/eval/probe_action_droid.py`](../../scripts/eval/probe_action_droid.py) / `probe_action_droid_v11.py`
- 원본 JSON: `data/probing_results/droid/` (재집계 source)
- Export script: `scripts/analysis/export_probing_summary.py`

## Notes

DROID는 cross-domain (이전 = EgoDex). 절대값 작아 paper에서 "gap-wise consistency"로 framing, 단독 SOTA 주장 X.
