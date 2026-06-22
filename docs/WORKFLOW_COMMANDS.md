# 워크플로우 명령어 (Quick Reference)

> 자주 쓰는 실행 명령 모음. 매 세션 필수가 아니라 [CLAUDE.md](../CLAUDE.md)에서 분리. 각 워크플로우의 상세 가이드는 아래 cross-link 참조. 활성 모델·명명은 CLAUDE.md "명명 · 2논문 구조".

## 1. EgoDex Pre-training

**Parvo 본학습 명령**: [v15b_retraining_status.md](v15b_retraining_status.md) §6 (재제출 명령 + 환경 + warmup 설정)이 단일 출처. 직전 v15(teacher-anchor)는 `paper-corl2026` 동결 → git history.

- Sanity 권장:
  ```bash
  sbatch --export=ALL,MODEL=two-stream-v15b,MAX_VIDEOS=50,EPOCHS=3,BATCH_SIZE=8,NUM_WORKERS=4 scripts/cluster/sanity_v15.sbatch
  ```
- 데이터 다운로드 (EgoDex CDN 직접):
  ```bash
  bash scripts/local/download_egodex.sh   part2 part3 part5  # 로컬
  bash scripts/cluster/download_egodex.sh part2 part3 part5  # 클러스터
  ```

## 2. Action Probing (사전학습 완료 후)

학습된 표현이 행동 정보를 인코딩하는지 검증. 상세·결과 → [PROBING_GUIDE.md](PROBING_GUIDE.md).

```bash
# EgoDex (within-domain)
python scripts/eval/probe_action.py \
    --encoder two-stream \
    --checkpoint /mnt/data/checkpoints/two_stream/.../best_model.pt \
    --egodex-root /mnt/data/egodex --frames-root /mnt/data/egodex_frames \
    --egodex-split part4 --gap 10 --cls-mode patch_mean_concat

# DROID (cross-domain, primary)
python scripts/eval/probe_action_droid.py \
    --encoder two-stream --checkpoint <ckpt> \
    --droid-root /mnt/data/droid_frames/ext1 --gap 10
```

## 3. LIBERO BC-Transformer Fine-tuning & Rollout

표준 평가 = **BC-T policy head** (frozen encoder + LIBERO 공식 `BCTransformerPolicy`). 상세 → [setup/LIBERO_TEST_GUIDE.md](setup/LIBERO_TEST_GUIDE.md) · [../scripts/eval/README.md](../scripts/eval/README.md).

```bash
# Cluster (sbatch, V3 cfg = use_joint=True + augmentation)
ENCODER=two-stream-v15-ptptk SUITE=libero_spatial SEED=0 \
    sbatch scripts/cluster/finetune_libero_bct.sbatch

# Local rollout (closed-loop SR)
bash scripts/local/run_libero_rollouts.sh two-stream-v15-ptptk 50
```
