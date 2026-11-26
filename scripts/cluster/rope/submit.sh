#!/bin/bash
# RoPE 인코더 실험 (docs/rope_encoder_plan.md) 제출 — C1-DN 이어 학습 레시피 env 를 한 곳에 고정하고 판만 바꾼다.
#   ARM=R → 위치 전부 2D RoPE (--rope) · ARM=A → APE 대조 판 (레시피 동일)
#   레시피 = C1-DN(40313842) Model args 그대로 · EgoDex part1 · INIT_FROM = C1-DN ep10 · LR 2.8e-5 · 10 ep
#   (C1-DN 자체가 C1 + 10 ep 이어 학습. 에폭 1 은 워밍업 = LR 1e-6 고정 — 스케줄러가 에폭 단위)
# 사용: ARM=R SUFFIX=sanity_R [EPOCHS=2 MAX_VIDEOS=200 BATCH_SIZE_PER_GPU=16 NUM_WORKERS=4 ...] \
#       bash scripts/cluster/rope/submit.sh <sbatch 자원 인자...>
# 공학 단계 공통 수정(RoPE 고유 아님)은 env 로 넘기고 A 에도 같이 넣는다 (계획서 §2 단일 변수 규칙).
#   RECIPE=c1 → 처음부터 C1 레시피(40275371 Model args 그대로 · LR 2.8e-4 · 50 ep · 워밍업 5 ep · init 없음 · 노이즈 제거 단계 인자 없음)
#   = §7 R-full · T1 θ 점검(ROPE_THETA=100 등). 기본 RECIPE=dn 은 위 이어 학습 레시피(변경 없음).
#   ROPE_SCOPE=m → M 전용 판(§8 안 1 · --rope-scope m · ARM=R 과 함께). 미지정 = 전부 RoPE.
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${ARM:?ARM=R|A}" "${SUFFIX:?SUFFIX}"
export MODEL=two-stream-v15b NO_SOBEL=1 PAIR_MODE=1 V15_COMP_MAE=1 V15_M_RECON_FLOOR=0.02 \
       BRIGHT_AUG=1 EMBED_DIM=384 NUM_HEADS=6 SPLITS=part1 CHECKPOINT_SUFFIX="rope_${SUFFIX}"
case "${RECIPE:-dn}" in
  dn) export M_NOISE_MAX=0.01 BRIGHT_TARGET=aug BRIGHT_SCENE_GAIN="0.73 1.37" LR="${LR:-2.8e-5}" EPOCHS="${EPOCHS:-10}" \
             INIT_FROM=/proj/external_group/mrg/checkpoints/two_stream_v15b_refine_comp_s_denoise_augtgt/20260927_235906/checkpoint_epoch0010.pt ;;
  c1) unset M_NOISE_MAX BRIGHT_TARGET BRIGHT_SCENE_GAIN INIT_FROM; export LR="${LR:-2.8e-4}" EPOCHS="${EPOCHS:-50}" ;;
  *) echo "RECIPE=dn|c1"; exit 1 ;;
esac
case "$ARM" in R) export ROPE=1 ;; A) export ROPE=0 ;; *) echo "ARM=R|A"; exit 1 ;; esac
sbatch --job-name="rope_${SUFFIX}" --export=ALL "$@" scripts/cluster/pretrain.sbatch
