#!/bin/bash
# Two-Stream v15 visualization: nomask reconstruction × 3 frame + motion routing × 3 segment + L_compose
# 사용:
#   sbatch --export=ALL,CKPT=/proj/external_group/mrg/checkpoints/two_stream_v15/<ts>/checkpoint_epochXXXX.pt scripts/cluster/viz_v15.sbatch
#SBATCH --job-name=viz_v15
# 가시화=추론 only → 풀 H100 불필요. V100(normal)이 최소사양+최저단가(8천원/일)+최다가용.
# bf16 autocast는 V100에서 비가속이나 viz(이미지 몇 장)엔 무관(검증: 35763305 1m37s).
# H100 MIG가 비면 --partition=mig-1g.10gb로 override 가능(아래는 module 없이 env python 직접 호출 → 어느 파티션이든 동작).
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/proj/external_group/mrg/logs/viz_v15_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/viz_v15_%j.err

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
# GPFS conda env python 직접 호출 — module load 불필요(V100 노드엔 miniforge3 모듈 없음).
PY="/proj/external_group/mrg/conda_envs/aavrl-train/bin/python"
CKPT="${CKPT:?CKPT required}"
# Parvo(no-Sobel) 모델이면 NO_SOBEL=1, OUT_DIR로 산출물 경로 지정
NO_SOBEL="${NO_SOBEL:-0}"
OUT_DIR="${OUT_DIR:-paper_artifacts/v15_main_train_samples}"

cd "$PROJECT_ROOT"

EXTRA_ARGS=()
[ "$NO_SOBEL" = "1" ] && EXTRA_ARGS+=(--no-sobel)
[ "${MASKED_ANCHOR:-0}" = "1" ] && EXTRA_ARGS+=(--masked-anchor)  # Run B: decode_first + masked 라우팅 컬럼

"$PY" scripts/eval/visualize_v15_no_mask.py \
    --ckpt "$CKPT" --num-egodex-train 3 --num-droid 3 --seed 42 \
    --out-dir "$OUT_DIR" "${EXTRA_ARGS[@]}"
