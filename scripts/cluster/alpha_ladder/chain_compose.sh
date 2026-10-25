#!/bin/bash
# alpha_plan §4-g 합성 복원 시험 30 칸 체인: 선형 f0 s1 한 칸(최소) → 성공 시 나머지 29 칸 (선형·tf1 × seed 3 × 폴드 5).  bash chain_compose.sh
set -uo pipefail
R=/proj/home/mrg/bys724/action-agnostic-visual-rl; A=$R/paper_artifacts/alpha_value/r7
wait_job() { while [ -n "$(squeue -j "$1" -h 2>/dev/null)" ]; do sleep 120; done
             sacct -j "$1" -X -n -o State | head -1 | tr -d ' '; }
sub() {  # $1 = linear|tf1 · $2 fold · $3 seed
  sbatch --parsable --job-name=alpha_compose_$1 "--export=ALL,ARM=star,AGE=1,COMPOSE=$1,FOLD=$2,SEED=$3" $R/scripts/cluster/alpha_value.sbatch; }
j0=$(sub linear 0 1); echo "$(date) first cell linear f0 s1 = $j0"
st=$(wait_job $j0); echo "$(date) $j0 state=$st"
[ "$st" = "COMPLETED" ] && [ -f "$A/linear/value_compose_f0_s1_preds.npz" ] || { echo "STOP: first cell failed"; exit 1; }
n=0; for r in linear tf1; do for s in 1 2 3; do for f in 0 1 2 3 4; do
  [ "$r$f$s" = linear01 ] && continue
  echo "$(date) $r f$f s$s = $(sub $r $f $s)"; n=$((n+1)); done; done; done
echo "$(date) submitted $n more cells (total 30)"
