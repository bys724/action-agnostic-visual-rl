#!/bin/bash
# alpha_plan §4-c 경쟁자 (10-25 Vault): 주기-k P 전용 판독기 15 칸 · 1 칸 먼저 → 성공 시 14 칸.  bash chain_competitor.sh 9
set -uo pipefail
PK=$1; R=/proj/home/mrg/bys724/action-agnostic-visual-rl; OUT=$R/paper_artifacts/alpha_value/r4/competitor_p$PK
wait_job() { while [ -n "$(squeue -j "$1" -h 2>/dev/null)" ]; do sleep 120; done
             sacct -j "$1" -X -n -o State | head -1 | tr -d ' '; }
sub() { sbatch --parsable --job-name=alpha_value_p$PK --export=ALL,ARM=none,AGE=1,PK=$PK,FOLD=$1,SEED=$2 $R/scripts/cluster/alpha_value.sbatch; }
j0=$(sub 0 1); echo "$(date) competitor p$PK f0 s1 = $j0"
st=$(wait_job $j0); echo "$(date) $j0 state=$st"
[ "$st" = "COMPLETED" ] && [ -f "$OUT/value_p${PK}_f0_s1.json" ] || { echo "STOP: first cell failed"; exit 1; }
for s in 1 2 3; do for f in 0 1 2 3 4; do [ "$f$s" = "01" ] && continue
  echo "$(date) competitor f$f s$s = $(sub $f $s)"; done; done
echo "$(date) all 15 competitor cells submitted"
