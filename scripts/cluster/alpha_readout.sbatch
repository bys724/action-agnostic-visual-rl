#!/bin/bash
# α 선행 판독 4-a·4-b (docs/alpha_plan.md §4 · scripts/eval/alpha_readout.py) — 한 잡 = 한 칸 (P 출처 × 변화 채널 × seed).
# 학습 없음(판독기만) · frames + 격자 P 캐시 + 나이별 특징이 GPU 에 상주 (DINO 칸 ≈ 20GB) → mig-3g.40gb.
#
# 사용 (6칸 × seed 42/1/2, suite 기본 libero_object; CKPT 기본 = C1-DN):
#   sbatch --export=ALL,P_SOURCE=ckpt,CHANGE=comp,SEED=42 scripts/cluster/alpha_readout.sbatch
#   # P_SOURCE ∈ {ckpt, dino} · CHANGE ∈ {comp, raw, none} · SUITE=libero_goal 로 goal
#   # 스모크: SMOKE=1 (데모 8 · 나이 1 10 20 · probe 5 ep · TAG 에 _smoke)
#   # 집계·판정 ⓐ: python scripts/eval/alpha_readout.py --summarize --task-suite libero_object
#
#SBATCH --job-name=alpha_readout
#SBATCH --partition=mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/proj/external_group/mrg/logs/alpha_readout_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/alpha_readout_%j.err

set -euo pipefail
PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
P_SOURCE="${P_SOURCE:?P_SOURCE 필요 (ckpt|dino)}"
CHANGE="${CHANGE:?CHANGE 필요 (comp|raw|none)}"
SEED="${SEED:-42}"
SUITE="${SUITE:-libero_object}"
CKPT="${CKPT:-}"   # 비우면 스크립트 기본 = C1-DN
VIEW="${VIEW:-agentview_rgb}"   # 손목 = eye_in_hand_rgb
TAG="${SUITE}_${P_SOURCE}_${CHANGE}_s${SEED}"; [ "$VIEW" = "eye_in_hand_rgb" ] && TAG="${TAG}_wrist"
EXTRA="${EXTRA:-}"
if [ "${SMOKE:-0}" = "1" ]; then
  TAG="${TAG}_smoke"; EXTRA="--max-demos 8 --ages 1 10 20 --probe-epochs 5 $EXTRA"
fi

module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /proj/external_group/mrg/conda_envs/aavrl-train

cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$CONDA_PREFIX/bin/python" -u scripts/eval/alpha_readout.py --p-source "$P_SOURCE" --change "$CHANGE" \
  --seed "$SEED" --task-suite "$SUITE" --view "$VIEW" ${CKPT:+--checkpoint "$CKPT"} --tag "$TAG" $EXTRA
