#!/bin/bash
# alpha_plan §4-f 비선형 판독기 105 칸 체인: P+M f0 s1 한 칸(최소) → 성공 시 나머지 104 칸.  bash chain_tf1.sh
set -uo pipefail
R=/proj/home/mrg/bys724/action-agnostic-visual-rl; A=$R/paper_artifacts/alpha_value/r6
SMALLM=/proj/external_group/mrg/checkpoints/two_stream_v15b_alpha_ladder_m192x2/20261024_014642/checkpoint_epoch0010.pt
wait_job() { while [ -n "$(squeue -j "$1" -h 2>/dev/null)" ]; do sleep 120; done
             sacct -j "$1" -X -n -o State | head -1 | tr -d ' '; }
# 칸 = "이름|env|out"  (env 는 sbatch --export 추가분 · out 은 r6 하위)
CELLS=("star|ARM=star,AGE=1|base" "comp|ARM=comp,AGE=1|base" "none|ARM=none,AGE=1|base" "raw|ARM=raw,AGE=1|base"
       "p2|ARM=none,AGE=1,P2=1|p2" "p9|ARM=none,AGE=1,PK=9|p9" "m192x2|ARM=comp,AGE=1|m192x2")
sub() {  # $1 = 칸 · $2 fold · $3 seed
  IFS='|' read -r nm env out <<< "$1"; ex="--readout tf1 --out-root $A/$out"; [ "$nm" = m192x2 ] && ex="--checkpoint $SMALLM $ex"
  sbatch --parsable --job-name=alpha_tf1_$nm --time=04:00:00 "--export=ALL,$env,FOLD=$2,SEED=$3,EXTRA=$ex" $R/scripts/cluster/alpha_value.sbatch; }
j0=$(sub "${CELLS[1]}" 0 1); echo "$(date) first cell comp f0 s1 = $j0"
st=$(wait_job $j0); echo "$(date) $j0 state=$st"
[ "$st" = "COMPLETED" ] && [ -f "$A/base/value_comp_f0_s1.json" ] || { echo "STOP: first cell failed"; exit 1; }
n=0; for c in "${CELLS[@]}"; do for s in 1 2 3; do for f in 0 1 2 3 4; do
  [ "${c%%|*}" = comp ] && [ "$f$s" = 01 ] && continue
  echo "$(date) ${c%%|*} f$f s$s = $(sub "$c" $f $s)"; n=$((n+1)); done; done; done
echo "$(date) submitted $n more cells (total 105)"
