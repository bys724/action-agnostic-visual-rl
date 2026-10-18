#!/bin/bash
#SBATCH --job-name=e1_smoke
#SBATCH --partition=mig-1g.10gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/proj/external_group/mrg/logs/e1_smoke_%j.out
# E1 스모크: 기본값(off·C1-DN) 손실이 수정 전 코드(HEAD worktree)와 같아야 통과 (≤1e-6 상대).
set -euo pipefail
NEW="${NEW_ROOT:?}"; OLD="${OLD_ROOT:?}"
PY=/proj/external_group/mrg/conda_envs/aavrl-train/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for md in off dn; do
  # (tee /dev/stderr 금지 — 로그 파일을 다시 열며 앞부분을 잘라냄)
  outn=$($PY "$NEW/scripts/cluster/e1/smoke_e1.py" "$NEW" $md); echo "$outn"
  outo=$($PY "$NEW/scripts/cluster/e1/smoke_e1.py" "$OLD" $md); echo "$outo"
  n=$(awk '/ loss /{print $3}' <<<"$outn"); o=$(awk '/ loss /{print $3}' <<<"$outo")
  echo "PARITY $md new=$n old=$o"
  $PY -c "import sys; n,o=float('$n'),float('$o'); sys.exit(0 if abs(n-o)<=1e-6*max(1,abs(o)) else 1)" || { echo "PARITY FAIL $md"; exit 1; }
done
$PY "$NEW/scripts/cluster/e1/smoke_e1.py" "$NEW" e1
echo SMOKE_PASS
