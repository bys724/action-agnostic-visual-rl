# Tab 5 — Hyperparameters (Appendix B)

**Paper position**: Appendix B (reproducibility)
**Status**: 🔴 not started — config 추출 필요

## Spec (Vault [[7. Outline § Tab 5]])

3 section 표:
1. **Pretraining**: 50ep, batch global 256, EMA momentum 0.999→0.9999, λ values (pred / m_jepa / compose), mask_p=0.75 / mask_m_jepa=0.5, max_gap=30 sample_center=15, optimizer + lr schedule
2. **BC-T**: frozen encoder + LIBERO BCTransformerPolicy, use_joint=True, augmentation (ImgColorJitter + TranslationAug), 50ep, lr / batch size
3. **Probing**: linear head, 20ep, lr 1e-3, batch 256, gap별

## Source

- Pretraining cfg: [`scripts/cluster/pretrain.sbatch`](../../../scripts/cluster/pretrain.sbatch) + [`scripts/pretrain.py`](../../../scripts/pretrain.py) defaults
- BC-T cfg: [`scripts/eval/finetune_libero_bct.py`](../../../scripts/eval/finetune_libero_bct.py) (V3 cfg)
- Probing cfg: [`scripts/eval/probe_action.py`](../../../scripts/eval/probe_action.py) defaults
- 마스터 설명: [`docs/RESEARCH_PLAN.md`](../../../docs/RESEARCH_PLAN.md), [`CLAUDE.md`](../../../CLAUDE.md)

## TODO

- [ ] **3 section LaTeX 표 작성** — 위 source에서 값 추출
- [ ] **λ warmup schedule** 명시: 10ep 0.01→1.0 (cluster_sessions §2026-05-15 cfg)
- [ ] **학습 비용 footnote**: v15 = 343 GPU·h (8 H100 × 42.9h)
- [ ] **데이터 footnote**: EgoDex part1-5 ≈ 100M frames (gap triple 샘플링)
