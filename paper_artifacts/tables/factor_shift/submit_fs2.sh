#!/bin/bash
# 분리 축 2단계 제출 (docs/factor_shift_plan.md §3): P 외형 판독 = 프레임 t 블록 3개 xyz(9-d) · D→D(같은 분포).
# 판정 = C0 P_t ≥ plain P_t (CI 비열위). gap 스윕 10/20/30/45 = §3-b 관찰. 인자 = min | full.
set -euo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
CK=/proj/external_group/mrg/checkpoints
C0=$CK/two_stream_v15b_step1_comp_mae_s/20260629_101634/latest.pt
PLAIN=$CK/two_stream_v15b_step1_plain_xmae_s/20260708_012539/latest.pt
NOM=$CK/two_stream_v15b_noM_cont/20260622_172841/checkpoint_epoch0030.pt   # ViT-B(768) — 크기 불일치, 참고 팔(판정 외)
CAL="SPLIT=training,CROSS_FOLDER=1,MAX_EPISODES=200,GAPS=10 20 30 45,READOUT=attentive,TARGET=scene_pos,TRAIN_SCENES=D"
EVAL_PERTURB=""; LABEL_FRACS=""; export EVAL_PERTURB LABEL_FRACS
declare -A ARM=(
  [c0_pt]="ENCODER=parvo,CHECKPOINT=$C0,PARVO_MODE=p_t_only"
  [plain_pt]="ENCODER=parvo,CHECKPOINT=$PLAIN,PARVO_MODE=p_t_only"
  [c0_m]="ENCODER=parvo,CHECKPOINT=$C0,PARVO_MODE=m_only"
  [c0_ptptk]="ENCODER=parvo,CHECKPOINT=$C0,PARVO_MODE=p_t_p_tk"
  [nom_pt]="ENCODER=parvo,CHECKPOINT=$NOM,PARVO_MODE=p_t_only"
)
sub() {  # arm seed
  local id; id=$(sbatch --parsable --partition=${PART:-normal} --gres=gpu:${NGPU:-1} --time=06:00:00 --job-name=fs2_$1_s$2 \
     --export="ALL,$CAL,${ARM[$1]},PROBE_SEED=$2,SUFFIX=fs2_$1_s$2" scripts/cluster/probe_action_calvin.sbatch)
  echo "$1 $2 $id"
}
case "$1" in
min)  sub c0_pt 42 ;;
full) for s in 42 1 2; do for a in c0_pt plain_pt c0_m c0_ptptk nom_pt; do
        [[ $a == c0_pt && $s == 42 ]] || sub $a $s
      done; done ;;
esac
