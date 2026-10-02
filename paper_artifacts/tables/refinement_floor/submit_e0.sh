#!/bin/bash
# E0 제출 (docs/claim_spine_v2.md §3): 외부 RGB 인코더 + ΔL 한 장 → 라운드 1과 같은 probe 프로토콜.
# 인자 = 단계(min | full). 잡 목록을 stdout에 "test arm seed id"로. 대조 행(C0/C1 M·F1·F3)은 라운드 1 재사용.
set -euo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
CAL="SPLIT=training,CROSS_FOLDER=1,MAX_EPISODES=200,GAPS=30,READOUT=attentive"
LIB="TASK_SUITE=libero_spatial,TRANSFER=1,READOUT=attentive"
EVAL_PERTURB=""; LABEL_FRACS=""; export EVAL_PERTURB LABEL_FRACS
sub() {  # test arm seed exports script
  local id; id=$(sbatch --parsable --partition=${PART:-mig-1g.10gb} --gres=gpu:1 --time=02:30:00 --job-name=e0_$1_$2_s$3 \
     --export=ALL,$4,PROBE_SEED=$3,SUFFIX=e0_$1_$2_s$3 scripts/cluster/$5)
  echo "$1 $2 $3 $id"
}
arm() { echo "ENCODER=${1%_*},INPUT_SOURCE=dl_${1#*_}"; }   # dinov2_signed → ENCODER=dinov2,INPUT_SOURCE=dl_signed
case "$1" in
min)  # 최소 칸: DINOv2 · 부호 유지 · CALVIN · seed 42
  sub cal dinov2_signed 42 "$CAL,$(arm dinov2_signed)" probe_action_calvin.sbatch ;;
full) # 나머지: 3 인코더 × 2 입력 변형 × seed {42,1,2} × {CALVIN, LIBERO 전이 행렬}. 최소 칸은 건너뜀
  for s in 42 1 2; do for e in dinov2 siglip vc1; do for v in signed abs; do
    [[ $e == dinov2 && $v == signed && $s == 42 ]] || sub cal ${e}_$v $s "$CAL,$(arm ${e}_$v)" probe_action_calvin.sbatch
    sub xfer ${e}_$v $s "$LIB,$(arm ${e}_$v)" probe_action_libero.sbatch
  done; done; done ;;
esac
