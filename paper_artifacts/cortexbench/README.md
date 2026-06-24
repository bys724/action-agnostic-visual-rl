# CortexBench — Frozen-Encoder Adaptation (Adroit + Meta-World)

**Paper**: 1 (ICRA, Input-Prior) — **핵심 증거**

Paper 1 주장("단일프레임 image MAE(Sobel+RGB) = `v15_p_only` **> VideoMAE-ours**")의 정량 근거.
계획·재사용 자산 → [`docs/paper1_input_prior_plan.md`](../../docs/paper1_input_prior_plan.md),
실행·환경 → [`docs/setup/CORTEXBENCH_GUIDE.md`](../../docs/setup/CORTEXBENCH_GUIDE.md).

## 구성

| Encoder | 역할 |
|---------|------|
| `v15_p_only` | **ours** (Parvo P-stream 단독 = image MAE). Paper 1 모델 |
| `videomae_ours` | 직접 대비 baseline ("same corpus, no input-prior") |
| `siglip_base` / `dinov2_base` / `vc1_vitb` | 외부 비교 baseline (양 논문 공유) |
| `_logs/` | 매트릭스 runner 로그 |

- 각 encoder 하위 = task 디렉토리 (Adroit `pen-v0`/`relocate-v0`/…, Meta-World `*-v2-goal-observable`) × seed.
- 집계 CSV (`per_run.csv`/`per_task.csv`/`summary.csv`)는 **gitignored** — raw `_DONE`/log에서 재생성:
  `python scripts/eval/aggregate_cortexbench.py --root paper_artifacts/cortexbench`.

## 주의

- "v15_p_only"의 v15는 obsolete 버전명이나, **여기 데이터는 Paper 1 image-MAE 증거로 유효**. ckpt 원본은 클러스터 보존.
- 정규화 사고(2026-05-25 ImageNet Normalize → OOD) 재실행 이력 → [`docs/eval_protocols.md`](../../docs/eval_protocols.md) §정규화.
