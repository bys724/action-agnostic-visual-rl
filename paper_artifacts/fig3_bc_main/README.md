# Fig 3 — LIBERO BC Main Result

**Paper**: 2 (AAAI, Action-Agnostic)

**Paper position**: §4 Experiments (★ main result figure + Tab 1 데이터 source)
**Status**: 🟡 데이터는 [`../libero_rollout/`](../libero_rollout/)로 통합 (parvo + v15 variants 포함, 최신). 이 폴더는 **figure 렌더 spec** — bar chart 미생성.

> 데이터 단일 출처 = `libero_rollout/{summary,per_task,episodes}.csv`. fig3 전용 stale round1 CSV는 제거(중복). figure/표는 거기서 추출.

## Spec (Vault [[7. Outline § Fig 3]])

- Bar chart: X = encoder (v15-ptptk leftmost ★ / DINOv2 / SigLIP / VC-1 / VideoMAE-ours)
- Y = success rate 0~100%
- 3 color groups: spatial (blue) / object (orange) / goal (green)
- Error bars: ±1 std over 3 seed
- v15-ptptk highlight, footnote: "v15 pretrained on 100M frames; DINOv2 142M / SigLIP 10B"
- v11 row 제외 (5/12 결정), v15-mp는 §5 ablation table로 이동

## Current artifacts

데이터는 [`../libero_rollout/`](../libero_rollout/)에 통합 (`{episodes,per_task,summary}.csv`):
- `summary.csv` (encoder, suite) — **paper Tab 1 직접 source**, mean ± std
- `per_task.csv` — per-task SR (failure 분석), `episodes.csv` — episode 단위 (재집계 가능)

**현재 포함**: dinov2 / siglip / vc1 / videomae-ours / two-stream-v11 / v15-mp / v15-ptptk / **parvo-ptptk** (+vfromm variants) × 3 suite × 3 seed.

## TODO

- [ ] aggregate output-dir = `paper_artifacts/libero_rollout`: `python scripts/eval/aggregate_libero_rollouts.py --input-dir data/libero/results --output-dir paper_artifacts/libero_rollout` (raw rollout 추가 시 재실행, 로컬 머신)
- [ ] **bar chart 스크립트** — matplotlib 5-encoder × 3-suite grouped bar (parvo highlight). 데이터 = `libero_rollout/summary.csv`
- [ ] v11 row를 paper figure에서는 **제외** (data CSV엔 유지)

## Source code

- Rollout driver: [`src/eval_libero.py`](../../src/eval_libero.py)
- Aggregator: [`scripts/eval/aggregate_libero_rollouts.py`](../../scripts/eval/aggregate_libero_rollouts.py)
- BC ckpt 위치: cluster `/proj/external_group/mrg/checkpoints/libero_bct/`, 로컬 `/mnt/data/checkpoints/libero_bct/`

## Notes for paper repo

- SR은 % 표기, 소수 1자리 반올림 (예: 83.7%)
- 3 seed mean ± std 보고 (per-task uncertainty는 episodes.csv binomial CI)
- v15 paper main 결과는 `summary.csv` 에 v15 row가 들어오는 시점부터 confirmed
