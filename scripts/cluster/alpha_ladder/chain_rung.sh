#!/bin/bash
# alpha_plan §4-c 단 체인: 학습 잡 완료 → ep10 ckpt 확인 → 판독 1 칸(최소 잡) → 성공 시 나머지 14 칸.
#   bash chain_rung.sh <train_jobid> <ckpt_suffix> <rung_tag>   (예: 42619451 alpha_ladder_m192x2 m192x2)
set -uo pipefail
J=$1; SUF=$2; RUNG=$3
R=/proj/home/mrg/bys724/action-agnostic-visual-rl
OUT=$R/paper_artifacts/alpha_value/r4/$RUNG
wait_job() { while [ -n "$(squeue -j "$1" -h 2>/dev/null)" ]; do sleep 120; done
             sacct -j "$1" -X -n -o State | head -1 | tr -d ' '; }
st=$(wait_job $J); echo "$(date) train $J state=$st"
CK=$(ls -d /proj/external_group/mrg/checkpoints/two_stream_v15b_${SUF}/*/ 2>/dev/null | tail -1)checkpoint_epoch0010.pt
[ "$st" = "COMPLETED" ] && [ -f "$CK" ] || { echo "STOP: train not complete or no ep10 ckpt ($CK)"; exit 1; }
mkdir -p "$OUT"
sub() { sbatch --parsable --job-name=alpha_value_$RUNG \
        --export=ALL,ARM=comp,AGE=1,FOLD=$1,SEED=$2,EXTRA="--checkpoint $CK --out-root $OUT" $R/scripts/cluster/alpha_value.sbatch; }
j0=$(sub 0 1); echo "$(date) readout f0 s1 = $j0"
st=$(wait_job $j0); echo "$(date) readout $j0 state=$st"
[ "$st" = "COMPLETED" ] && [ -f "$OUT/value_comp_f0_s1.json" ] || { echo "STOP: first readout failed"; exit 1; }
for s in 1 2 3; do for f in 0 1 2 3 4; do [ "$f$s" = "01" ] && continue
  echo "$(date) readout f$f s$s = $(sub $f $s)"; done; done
echo "$(date) all 15 readout cells submitted"
