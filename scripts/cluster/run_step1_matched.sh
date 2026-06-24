#!/bin/bash
# STEP 1 — subset-matched small 3런 (restart_plan.md §3 STEP 1).
#
# part1·ViT-S(384/6)·50ep·no-Sobel matched 3런. 세 런의 차이는 **loss/routing 구조뿐** —
# 데이터(part1)·epoch·size·전처리는 동일 (single-variable 비교):
#   - mcp_mae_s  (ours, gate) : §9 motion-cond pixel MAE, routing=M(ΔL).        [--v15-pixel-pred]
#   - analog_s   (대조)        : 동일하되 routing=P(RGB). M encoder freeze.       [+ --v15-routing-source p]
#   - no_m_s     (Image MAE)   : P two-frame MAE 단독 (routing 없음).            [--v15-no-motion]
#
# 🔴 게이트(restart_plan §3): **mcp_mae_s 첫 run health**(collapse 없이 patch+CLS R² 정상 +
#    slope가 VideoMAE 대비 우위) 확인 후에야 매트릭스(STEP 2·ViT-B) 확장. → mcp_mae_s 먼저 제출 권장.
# 🔴 제출과 동시에 docs/cluster_sessions.md 표 업데이트 필수 (비용 청구 대조).
#
# 사용: scripts/cluster/run_step1_matched.sh mcp_mae_s          # 게이트 먼저
#       scripts/cluster/run_step1_matched.sh analog_s no_m_s    # 게이트 통과 후 나머지
#       scripts/cluster/run_step1_matched.sh all                # 3개 한번에 (게이트 생략 주의)
set -euo pipefail

# ── matched 공통 (3런 동일, env로 override 가능) ──────────────────────────
SPLITS="${SPLITS:-part1}"          # subset = part1 (matched 데이터 예산 ~46k vid)
EPOCHS="${EPOCHS:-50}"
EMBED_DIM="${EMBED_DIM:-384}"      # ViT-S
NUM_HEADS="${NUM_HEADS:-6}"
V11_M_DEPTH="${V11_M_DEPTH:-6}"    # M encoder = P depth(12)의 절반
# 🔵 batch 튜닝 (sanity 36137148 관찰, 2026-06-23): ViT-S(40M)·batch32에서 throughput
#   ~330 samp/s/GPU·COMPUTE-bound 99%인데 — 작은 모델/batch라 H100(80GB) GPU 저활용 추정
#   (메모리는 batch32에서 극히 일부만 사용). full part1 50ep ≈ ~200 GPU·h(no-M 대비 ~2× = §9 3×predict).
#   → **batch 128~256로 키워 GPU util↑·throughput↑·비용↓ 권장** (메모리 여유 충분).
#   LR은 batch에 비례 스케일(예: batch128 → LR~4e-4 + warmup). 본학습 전 batch sweep 1회로 최적값 확정 권장.
BATCH="${BATCH:-32}"
LR="${LR:-2e-4}"
MAX_GAP="${MAX_GAP:-30}"
SAMPLE_CENTER="${SAMPLE_CENTER:-15}"
PARTITION="${PARTITION:-AIP_long}"
NODES="${NODES:-1}"
GPUS="${GPUS:-4}"
TIME="${TIME:-1-12:00:00}"         # ViT-S subset 50ep 추정 <36h (timeout guard 여유; 첫 run 후 재조정)

COMMON="MODEL=two-stream-v15b,EPOCHS=$EPOCHS,BATCH_SIZE_PER_GPU=$BATCH,LR=$LR,\
SPLITS=$SPLITS,EMBED_DIM=$EMBED_DIM,NUM_HEADS=$NUM_HEADS,V11_M_DEPTH=$V11_M_DEPTH,\
NO_SOBEL=1,PAIR_MODE=1,MAX_GAP=$MAX_GAP,SAMPLE_CENTER=$SAMPLE_CENTER,V11_ROUTING_MODE=v_from_p"

submit() {  # $1=tag  $2=per-run env
  echo "→ submitting step1_$1  (per-run: $2)"
  sbatch --job-name="step1_$1" --partition="$PARTITION" \
    --nodes="$NODES" --ntasks-per-node="$GPUS" --gres="gpu:$GPUS" --time="$TIME" \
    --export="ALL,$COMMON,$2,CHECKPOINT_SUFFIX=step1_$1" \
    scripts/cluster/pretrain.sbatch
}

run_one() {
  case "$1" in
    mcp_mae_s) submit mcp_mae_s "V15_PIXEL_PRED=1" ;;
    analog_s)  submit analog_s  "V15_PIXEL_PRED=1,V15_ROUTING_SOURCE=p" ;;
    no_m_s)    submit no_m_s     "V15_NO_MOTION=1" ;;
    *) echo "unknown run: $1 (mcp_mae_s|analog_s|no_m_s)"; exit 1 ;;
  esac
}

if [ $# -eq 0 ]; then
  sed -n '2,16p' "$0"  # usage(헤더 주석) 출력 후 종료 — 실수 대량 제출 방지
  echo; echo "인자 없음 → 제출 안 함. 위 사용법 참고."
  exit 0
fi
[ "$1" = "all" ] && set -- mcp_mae_s analog_s no_m_s
for r in "$@"; do run_one "$r"; done
echo "제출 완료 ($*). 🔴 docs/cluster_sessions.md 업데이트 필수."
