#!/bin/bash
# C10 CALVIN action probing 매트릭스 launcher (5 enc × 4 ABCD splits = 20잡)
#
# 전제: CALVIN 데이터 압축 해제 + loader sanity 검증 완료.
#
# 사용:
#   bash scripts/cluster/run_calvin_probing_matrix.sh           # 본 20잡
#   bash scripts/cluster/run_calvin_probing_matrix.sh sanity    # 1잡 sanity (validation, 10 ep)
#
# 종료 후 paper §4 ¶2 (iii) CALVIN row → paper_artifacts/calvin_action_probing/*

set -euo pipefail

REPO=/proj/home/mrg/bys724/action-agnostic-visual-rl
MODE="${1:-full}"

# 5 encoder
ENCODERS=(two-stream-v11 videomae-ours dinov2 siglip vc1)
V15_CKPT=/proj/external_group/mrg/checkpoints/two_stream_v15/20260511_045319/latest.pt
VIDEOMAE_CKPT=/proj/external_group/mrg/checkpoints/videomae/20260415_012017/best_model.pt

# 4 splits
# CALVIN task_ABCD_D는 training (A,B,C,D 모두) + validation (D split, OOD test)
# paper §C10 plan: "5 enc × 4 ABCD splits × static view" → 보통 training/<split> 형태 분리됨
# 실제 directory 구조 확인 후 split 명 조정 필요
SPLITS=(training validation)  # 다운로드 후 실제 구조 확인하여 수정

if [[ "$MODE" == "sanity" ]]; then
  echo "=== Sanity mode: v15 × validation × max_episodes=10 ==="
  sbatch --job-name=calvin_sanity \
    --export=ALL,ENCODER=two-stream-v11,CHECKPOINT=$V15_CKPT,SPLIT=validation,MAX_EPISODES=10,SUFFIX=sanity \
    $REPO/scripts/cluster/probe_action_calvin.sbatch
  exit 0
fi

# Full matrix
echo "=== C10 CALVIN probing matrix: 5 enc × ${#SPLITS[@]} splits = $((5 * ${#SPLITS[@]}))잡 ==="
for ENC in "${ENCODERS[@]}"; do
  for SPLIT in "${SPLITS[@]}"; do
    CKPT_ARG=""
    case "$ENC" in
      two-stream-v11) CKPT_ARG="CHECKPOINT=$V15_CKPT," ;;
      videomae-ours)  CKPT_ARG="CHECKPOINT=$VIDEOMAE_CKPT," ;;
      # dinov2/siglip/vc1은 ckpt 불필요 (HF/vc_models 자동)
    esac

    SHORT_ENC=$(echo $ENC | tr '/-' '_')
    # training (178 ep)은 ~4-5h 필요 → 8h. validation (4 ep)은 sbatch default 2h OK
    if [[ "$SPLIT" == "training" ]]; then
      TIME_LIMIT="08:00:00"
    else
      TIME_LIMIT="02:00:00"
    fi
    sbatch --job-name=calvin_${SHORT_ENC}_${SPLIT} \
      --time=$TIME_LIMIT \
      --export=ALL,ENCODER=$ENC,${CKPT_ARG}SPLIT=$SPLIT,MAX_EPISODES=200 \
      $REPO/scripts/cluster/probe_action_calvin.sbatch
  done
done

echo
echo "=== 제출 완료. squeue로 진행 확인 ==="
squeue -u $USER --format="%.10i %.30j %.8T" | grep calvin | head
