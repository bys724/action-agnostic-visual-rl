# Fig 3 — LIBERO BC Main Result

**Paper**: 2 (AAAI, Action-Agnostic)

**Paper position**: §4 Experiments (★ main result figure + Tab 1 데이터 source)
**Status**: 🟡 partial — round1 (baseline + v11) 도착, **v15 round2 미통합**

## Spec (Vault [[7. Outline § Fig 3]])

- Bar chart: X = encoder (v15-ptptk leftmost ★ / DINOv2 / SigLIP / VC-1 / VideoMAE-ours)
- Y = success rate 0~100%
- 3 color groups: spatial (blue) / object (orange) / goal (green)
- Error bars: ±1 std over 3 seed
- v15-ptptk highlight, footnote: "v15 pretrained on 100M frames; DINOv2 142M / SigLIP 10B"
- v11 row 제외 (5/12 결정), v15-mp는 §5 ablation table로 이동

## Current artifacts

| File | Granularity | 비고 |
|------|-------------|------|
| `episodes.csv` | episode 단위 | 가장 세밀, 재집계 가능 |
| `per_task.csv` | (encoder, suite, seed, task) | per-task SR — failure 분석 |
| `summary.csv` | (encoder, suite) | **paper Tab 1 직접 source** — mean ± std |

**Round 1 (현재 포함)**: dinov2 / siglip / vc1 / videomae-ours / two-stream-v11 × 3 suite × 3 seed

## TODO

- [ ] **v15-ptptk round2 (9 ckpt)** 로컬 rollout → `data/libero/results/`에 추가
  - 클러스터 tar (`libero_bct_v15_9jobs_round2_20260514.tar` 7.6 GB) 전송 + 실행
  - `bash scripts/local/run_libero_rollouts.sh two-stream-v15-ptptk 50`
- [ ] **v15-mp round2 (9 ckpt)** 동일 패턴 (C-variant 비교용, Tab 3 ablation)
- [ ] aggregate 재실행: `python scripts/eval/aggregate_libero_rollouts.py --input-dir data/libero/results --output-dir paper_artifacts/fig3_bc_main`
- [ ] **bar chart 스크립트** — matplotlib 5-encoder × 3-suite grouped bar (v15-ptptk highlight)
- [ ] v11 row를 paper figure에서는 **제외**, `_archive/v11_bc_round1.csv` 같이 별도 보관 검토

## Source code

- Rollout driver: [`src/eval_libero.py`](../../src/eval_libero.py)
- Aggregator: [`scripts/eval/aggregate_libero_rollouts.py`](../../scripts/eval/aggregate_libero_rollouts.py)
- BC ckpt 위치: cluster `/proj/external_group/mrg/checkpoints/libero_bct/`, 로컬 `/mnt/data/checkpoints/libero_bct/`

## Notes for paper repo

- SR은 % 표기, 소수 1자리 반올림 (예: 83.7%)
- 3 seed mean ± std 보고 (per-task uncertainty는 episodes.csv binomial CI)
- v15 paper main 결과는 `summary.csv` 에 v15 row가 들어오는 시점부터 confirmed
