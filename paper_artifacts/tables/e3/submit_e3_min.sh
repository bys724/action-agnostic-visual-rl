#!/bin/bash
# E3 1단계 최소 셀 (docs/claim_spine_v2.md §4.4): libero_object · 데모 10 · g=10 · dropout 0.1 · attentive · 50ep.
# 팔 ② rgb_prev (seed 0 = 0단계 재사용) vs ④ comp_m. 학습 → afterok 롤아웃(osmesa, 500 ep) 체인.
# 인자 = 팔:seed 목록, 예) "comp_m:0" 또는 "comp_m:1 comp_m:2 rgb_prev:1 rgb_prev:2". 출력 "arm seed train_id roll_id".
set -euo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
C0=/proj/external_group/mrg/checkpoints/two_stream_v15b_step1_comp_mae_s/20260629_101634/latest.pt
for spec in "$@"; do
  arm=${spec%:*}; seed=${spec#*:}
  tag=$(case $arm in comp_m) echo p1m;; none) echo p1;; *) echo p2;; esac)
  suf=e3c1_${tag}_d10_g10_dp01_s${seed}
  tid=$(sbatch --parsable --partition=${PART:-AIP} --cpus-per-task=8 --job-name=e3_${tag}_s${seed} \
    --export=ALL,ENCODER=parvo-ptptk,CHECKPOINT=$C0,TASK_SUITE=libero_object,SEED=$seed,POOLING=attentive,MOTION_GAP=10,MOTION_DROPOUT_P=0.1,MAX_DEMOS=10,MOTION_SOURCE=$arm,SUFFIX=$suf \
    scripts/cluster/finetune_libero_bct.sbatch)
  rid=$(sbatch --parsable --partition=${ROLL_PART:-AIP} --gres=gpu:1 --cpus-per-task=8 --time=12:00:00 --dependency=afterok:$tid --kill-on-invalid-dep=yes \
    --job-name=roll_${tag}_s${seed} --export=ALL,CKPT_SUFFIX=$suf,TASK_SUITE=libero_object,NUM_TRIALS=50,RENDER=osmesa \
    scripts/cluster/rollout_libero_bct.sbatch)
  echo "$arm $seed $tid $rid"
done
