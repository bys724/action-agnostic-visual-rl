#!/bin/bash
# α v2 남은 스텝 판독 (docs/alpha_plan.md §3 · scripts/eval/alpha_readout.py --target remaining) — 한 잡 = 팔 × 폴드 × seed.
# 학습 없음(판독기만) · train 특징은 CPU 에 상주 (4 stream 칸 ≈ 50GB) → --mem 넉넉히.
#
#   sbatch --export=ALL,ARM=comp,FOLD=0,SEED=1 scripts/cluster/alpha_value.sbatch
#   # ARM ∈ {star, comp, raw, none} · NOQ=1 → q 제외 · SMOKE=1 (에피소드 4 · 나이 1 10 20 · probe 2 ep)
#
#SBATCH --job-name=alpha_value
#SBATCH --partition=mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/proj/external_group/mrg/logs/alpha_value_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/alpha_value_%j.err

set -euo pipefail
PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
ARM="${ARM:?ARM 필요 (star|comp|raw|none)}"
FOLD="${FOLD:-0}"
SEED="${SEED:-1}"
EXTRA="${EXTRA:-}"
TAG="value_${ARM}"; [ "${NOQ:-0}" = "1" ] && { TAG="${TAG}_noq"; EXTRA="--no-q $EXTRA"; }
TAG="${TAG}_f${FOLD}_s${SEED}"
if [ "${SMOKE:-0}" = "1" ]; then
  TAG="${TAG}_smoke"; EXTRA="--max-episodes 4 --ages 1 10 20 --probe-epochs 2 $EXTRA"
fi

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /proj/external_group/mrg/conda_envs/aavrl-train

cd "$PROJECT_ROOT"
"$CONDA_PREFIX/bin/python" -u scripts/eval/alpha_readout.py --target remaining --arm "$ARM" --fold "$FOLD" \
  --seed "$SEED" --tag "$TAG" $EXTRA
