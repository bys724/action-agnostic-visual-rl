#!/bin/bash
# RoPE 인코더 실험 (docs/rope_encoder_plan.md) 제출 — C1-DN 이어 학습 레시피 env 를 한 곳에 고정하고 판만 바꾼다.
#   ARM=R → 위치 전부 2D RoPE (--rope) · ARM=A → APE 대조 판 (레시피 동일)
#   레시피 = C1-DN(40313842) Model args 그대로 · EgoDex part1 · INIT_FROM = C1-DN ep10 · LR 2.8e-5 · 10 ep
#   (C1-DN 자체가 C1 + 10 ep 이어 학습. 에폭 1 은 워밍업 = LR 1e-6 고정 — 스케줄러가 에폭 단위)
# 사용: ARM=R SUFFIX=sanity_R [EPOCHS=2 MAX_VIDEOS=200 BATCH_SIZE_PER_GPU=16 NUM_WORKERS=4 ...] \
#       bash scripts/cluster/rope/submit.sh <sbatch 자원 인자...>
# 공학 단계 공통 수정(RoPE 고유 아님)은 env 로 넘기고 A 에도 같이 넣는다 (계획서 §2 단일 변수 규칙).
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${ARM:?ARM=R|A}" "${SUFFIX:?SUFFIX}"
export MODEL=two-stream-v15b NO_SOBEL=1 PAIR_MODE=1 V15_COMP_MAE=1 V15_M_RECON_FLOOR=0.02 \
       BRIGHT_AUG=1 M_NOISE_MAX=0.01 BRIGHT_TARGET=aug BRIGHT_SCENE_GAIN="0.73 1.37" \
       EMBED_DIM=384 NUM_HEADS=6 SPLITS=part1 LR="${LR:-2.8e-5}" EPOCHS="${EPOCHS:-10}" \
       INIT_FROM=/proj/external_group/mrg/checkpoints/two_stream_v15b_refine_comp_s_denoise_augtgt/20260927_235906/checkpoint_epoch0010.pt \
       CHECKPOINT_SUFFIX="rope_${SUFFIX}"
case "$ARM" in R) export ROPE=1 ;; A) export ROPE=0 ;; *) echo "ARM=R|A"; exit 1 ;; esac
sbatch --job-name="rope_${SUFFIX}" --export=ALL "$@" scripts/cluster/pretrain.sbatch
