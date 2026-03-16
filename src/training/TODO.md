# Training Module

## 구현 완료

- `pretrain.py`: Pre-training 학습 루프
  - Future frame prediction (Two-Stream, Single-Stream)
  - Masked autoencoding (VideoMAE)
  - Multi-gap sampling, Gap-weighted loss
  - Checkpoint management, Multi-GPU support

## 미구현

- Fine-tuning 학습 루프 (encoder freeze → action head 학습)
- Action probing 스크립트 (DROID primary, Bridge V2 secondary)

실험 로드맵 및 우선순위는 `docs/RESEARCH_PLAN.md` 참고.
