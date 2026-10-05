#!/bin/bash
# 분리 축 1단계 제출 (docs/factor_shift_plan.md §2): CALVIN 판독기 학습 환경 ABC(이동) vs D(기준), 시험 = validation(D) 공통.
# 프로토콜 = 라운드 1 승계(cross-folder · attentive · cap 200 · gap 10/20/30/45 · probe seed {42,1,2}).
# 인자 = 단계(min | full). 잡 목록을 stdout에 "arm scenes seed id"로.
set -euo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
CK=/proj/external_group/mrg/checkpoints
C0=$CK/two_stream_v15b_step1_comp_mae_s/20260629_101634/latest.pt
PLAIN=$CK/two_stream_v15b_step1_plain_xmae_s/20260708_012539/latest.pt
CAL="SPLIT=training,CROSS_FOLDER=1,MAX_EPISODES=200,GAPS=10 20 30 45,READOUT=attentive"
EVAL_PERTURB=""; LABEL_FRACS=""; export EVAL_PERTURB LABEL_FRACS
declare -A ARM=(
  [c0_m]="ENCODER=parvo,CHECKPOINT=$C0,PARVO_MODE=m_only"
  [c0_ptptk]="ENCODER=parvo,CHECKPOINT=$C0,PARVO_MODE=p_t_p_tk"
  [plain_ptptk]="ENCODER=parvo,CHECKPOINT=$PLAIN,PARVO_MODE=p_t_p_tk"
  [c0_ptm]="ENCODER=parvo,CHECKPOINT=$C0,PARVO_MODE=p_t_m"
  [raw]="ENCODER=raw-dl,RAW_DL_VARIANT=raw"
  [vmae]="ENCODER=videomae-ours,CHECKPOINT=$CK/videomae/20260415_012017/best_model.pt,VIDEOMAE_ENCODER=vla"  # 10-04 추가: 섞인 표현 3 (plain 바닥 대체)
)
sub() {  # arm scenes seed
  local id; id=$(sbatch --parsable --partition=${PART:-mig-1g.10gb} --gres=gpu:${NGPU:-1} ${MEM:+--mem=$MEM} --time=06:00:00 --job-name=fs1_$1_$2_s$3 \
     --export="ALL,$CAL,${ARM[$1]},TRAIN_SCENES=$2,PROBE_SEED=$3,SUFFIX=fs1_$1_$2_s$3" scripts/cluster/probe_action_calvin.sbatch)
  echo "$1 $2 $3 $id"
}
case "$1" in
min)  sub c0_m ABC 42; sub c0_m D 42 ;;
full) for s in 42 1 2; do for a in c0_m c0_ptptk plain_ptptk c0_ptm raw; do for sc in ABC D; do
        [[ $a == c0_m && $s == 42 ]] || sub $a $sc $s
      done; done; done ;;
vmae) for s in 42 1 2; do for sc in ABC D; do sub vmae $sc $s; done; done ;;
esac
