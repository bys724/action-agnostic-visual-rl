#!/bin/bash
# 라운드 2 제출 (refinement_floor_plan §9). 인자 = 단계 이름(r21 | min). 잡 목록을 stdout에 "test arm seed id"로.
set -euo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
C1=/proj/external_group/mrg/checkpoints/two_stream_v15b_refine_comp_s_bright/20260925_163504/latest.pt
C0=/proj/external_group/mrg/checkpoints/two_stream_v15b_step1_comp_mae_s/20260629_101634/latest.pt
PERT="gain:0.9,1.1,1.2,1.3 ramp:0.1,0.2,0.3 shadow:0.2,0.4,0.6 noise:0.01,0.02,0.04"   # eval_protocols §4-b 고정 문자열
CAL="SPLIT=training,CROSS_FOLDER=1,MAX_EPISODES=200,GAPS=30,READOUT=attentive"
LIB="TASK_SUITE=libero_spatial,TRANSFER=1,READOUT=attentive"
sub() {  # test arm seed exports script
  local id; id=$(sbatch --parsable --partition=normal --gres=gpu:1 --time=02:30:00 --job-name=r2_$1_$2_s$3 \
     --export=ALL,$4,PROBE_SEED=$3,SUFFIX=refine2_$1_$2_s$3 scripts/cluster/$5)
  echo "$1 $2 $3 $id"
}
# 잡별 옵션(EVAL_PERTURB·LABEL_FRACS)은 export한 셸 변수로 전달 (--export=ALL, 라운드 1과 같은 방식)
pert()  { EVAL_PERTURB="$PERT"; LABEL_FRACS="";                      sub pert  "$@" probe_action_calvin.sbatch; }
label() { EVAL_PERTURB="";       LABEL_FRACS="1.0 0.2 0.05 0.02";    sub label "$@" probe_action_calvin.sbatch; }
xfer()  { EVAL_PERTURB="";       LABEL_FRACS="";                      sub xfer  "$@" probe_action_libero.sbatch; }
plain() { EVAL_PERTURB="";       LABEL_FRACS="";                      sub "$@"; }
EVAL_PERTURB=""; LABEL_FRACS=""; export EVAL_PERTURB LABEL_FRACS
case "$1" in
r21)  # R2-1: P_t⊕P_tk (사다리 0단) — 기존 경로, 새 코드 없음
  for s in 42 1 2; do
    for A in C1 C0; do CK=$([ $A = C1 ] && echo $C1 || echo $C0)
      pert PtPtk$A $s "$CAL,ENCODER=parvo,CHECKPOINT=$CK,PARVO_MODE=p_t_p_tk"
      label PtPtk$A $s "$CAL,ENCODER=parvo,CHECKPOINT=$CK,PARVO_MODE=p_t_p_tk"
      xfer PtPtk$A $s "$LIB,ENCODER=parvo,CHECKPOINT=$CK,PARVO_MODE=p_t_p_tk"
    done
  done ;;
min)  # R2-2·R2-3 최소 칸 (새 코드)
  pert PtRawLin 42 "$CAL,ENCODER=parvo-raw,CHECKPOINT=$C1,RAW_DL_VARIANT=raw,RAW_PAD=linear"
  plain noise01 C1 42 "$CAL,ENCODER=parvo,CHECKPOINT=$C1,PARVO_MODE=m_only,PROBE_NOISE_SIGMA=0.01" probe_action_calvin.sbatch
  plain xnoise01 C1 42 "$LIB,ENCODER=parvo,CHECKPOINT=$C1,PARVO_MODE=m_only,PROBE_NOISE_SIGMA=0.01" probe_action_libero.sbatch ;;
esac
