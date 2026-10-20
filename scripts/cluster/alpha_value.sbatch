#!/bin/bash
# α v2 남은 스텝 판독 (docs/alpha_plan.md §3 · scripts/eval/alpha_readout.py --target remaining) — 한 잡 = 팔 × 폴드 × seed.
# 학습 없음(판독기만) · train 특징은 CPU 에 상주 (4 stream 칸 ≈ 50GB) → --mem 넉넉히.
#
#   sbatch --export=ALL,ARM=comp,FOLD=0,SEED=1 scripts/cluster/alpha_value.sbatch
#   # ARM ∈ {star, comp, raw, none} · NOQ=1 → q 제외 · DINO=1 → 앞 카메라 P = DINOv2 (J4) · AGE=1 → 앵커 나이 판독기 입력 (2차) · P2=1 → 주기-2 P (§4-b) · SMOKE=1 (에피소드 4 · 나이 1 10 20 · probe 2 ep)
#
#SBATCH --job-name=alpha_value
#SBATCH --partition=mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/proj/external_group/mrg/logs/alpha_value_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/alpha_value_%j.err

set -euo pipefail
PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
ARM="${ARM:?ARM 필요 (star|comp|raw|none)}"
FOLD="${FOLD:-0}"
SEED="${SEED:-1}"
EXTRA="${EXTRA:-}"
TAG="value_${ARM}"; [ "${DINO:-0}" = "1" ] && { TAG="value_dino_${ARM}"; EXTRA="--p-source dino $EXTRA"; }
[ "${AGE:-0}" = "1" ] && EXTRA="--age-input $EXTRA"  # 2차 (10-19 Vault) · 출력 alpha_value/r2
# 주기-2 P (alpha_plan §4-b · 10-20): 앵커 P만 · 나이 {1,2} 로 학습·평가 · 출력 alpha_value/r3 (ARM=none AGE=1 P2=1)
[ "${P2:-0}" = "1" ] && { TAG="value_p2"; EXTRA="--ages 1 2 --train-ages 1 2 --out-root $PROJECT_ROOT/paper_artifacts/alpha_value/r3 $EXTRA"; }
[ "${NOQ:-0}" = "1" ] && { TAG="${TAG}_noq"; EXTRA="--no-q $EXTRA"; }
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
