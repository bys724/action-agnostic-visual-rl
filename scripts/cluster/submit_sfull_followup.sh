#!/bin/bash
# S-full 7ep 본학습(36829403) 완료 시 후속 측정 15잡 자동 제출 orchestrator.
#   ① EgoDex same-probe 3잡  — fulldata_scaling_plan §4-b 측정 1 (36197899 프로토콜)
#   ② OOD probing 8잡        — STEP 2(A) 36785986~993 프로토콜 (CALVIN xfold + LIBERO 3suite, mean+attn)
#   ③ SSv2 경로 1-c 4잡      — correspondence_eval_plan §7 (표준+compositional × p_t_m/p_t_p_tk, mean·fp16)
#
# 제출 (dependency — 학습 성공 시에만 실행):
#   sbatch --dependency=afterok:36829403 --kill-on-invalid-dep=yes scripts/cluster/submit_sfull_followup.sh
# 검증 (로그인 노드, 제출 없이 명령만 출력):
#   DRY_RUN=1 bash scripts/cluster/submit_sfull_followup.sh
#
# ckpt는 런타임 glob — requeue 시 새 timestamp dir이 생기므로 (36829403 선례:
# 20260712_172523 사장 → 20260713_203245) 제출 시점 하드코딩 금지.

#SBATCH --job-name=sfull_followup_submit
#SBATCH --partition=core_s
#SBATCH --qos=core_s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/proj/external_group/mrg/logs/sfull_followup_submit_%j.out

set -euo pipefail

PROJECT_ROOT="/proj/home/mrg/bys724/action-agnostic-visual-rl"
CKPT_BASE="/proj/external_group/mrg/checkpoints/two_stream_v15b_fulldata_comp_mae_s_7ep"
SSV2_COMP_ANN="/proj/external_group/mrg/datasets/ssv2/splits_something_else/compositional"
DRY_RUN="${DRY_RUN:-0}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [followup] $*"; }

CKPT=$(ls -t "$CKPT_BASE"/*/latest.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] || { log "ERROR: latest.pt not found under $CKPT_BASE"; exit 1; }
RUN_DIR=$(dirname "$CKPT")

# 완주 확인: afterok가 exit 0을 보장해도 ep7 ckpt 실재로 이중 확인 (부분완료 방어)
if [ "$DRY_RUN" = 0 ] && [ ! -f "$RUN_DIR/checkpoint_epoch0007.pt" ]; then
  log "ERROR: checkpoint_epoch0007.pt missing in $RUN_DIR — 7ep 미완주, 제출 중단"; exit 1
fi
log "CKPT=$CKPT"

cd "$PROJECT_ROOT"
SUBMITTED=()
submit() {  # submit <설명> <sbatch args...>
  local desc="$1"; shift
  if [ "$DRY_RUN" != 0 ]; then log "DRY: sbatch $*"; return; fi
  local out; out=$(sbatch "$@") || { log "SUBMIT FAIL: $desc"; exit 1; }
  SUBMITTED+=("${out##* }  $desc")
  log "submitted ${out##* }  $desc"
}

# ── ① EgoDex same-probe 3잡 (H100 AIP, test·gap10·MAX_VIDEOS 1500·40ep) ──
for CM in attentive_concat_p_t_p_tk attentive_m attentive_concat_p_m; do
  submit "egodex_$CM" -J "sfull_egodex_$CM" \
    --export=ALL,ENCODER=parvo,CHECKPOINT=$CKPT,SPLIT=test,GAP=10,EPOCHS=40,MAX_VIDEOS=1500,CLS_MODE=$CM \
    scripts/cluster/probe_action.sbatch
done

# ── ② OOD 8잡 (H100 AIP, p_t_m — CALVIN xfold MAX_EPISODES=200 · LIBERO spatial/object/goal) ──
for RO in mean attentive; do
  SFX=sfull_mean_ptm; [ "$RO" = attentive ] && SFX=sfull_attn_ptm
  submit "calvin_$RO" -J "sfull_calvin_$RO" \
    --export=ALL,ENCODER=parvo,CHECKPOINT=$CKPT,CROSS_FOLDER=1,PARVO_MODE=p_t_m,READOUT=$RO,SUFFIX=$SFX \
    scripts/cluster/probe_action_calvin.sbatch
  for TS in libero_spatial libero_object libero_goal; do
    submit "${TS}_$RO" -J "sfull_${TS#libero_}_$RO" \
      --export=ALL,ENCODER=parvo,CHECKPOINT=$CKPT,TASK_SUITE=$TS,PARVO_MODE=p_t_m,READOUT=$RO,SUFFIX=$SFX \
      scripts/cluster/probe_action_libero.sbatch
  done
done

# ── ③ SSv2 경로 1-c 4잡 (V100 normal, mean·fp16 — gate = §7 사전 등록) ──
for PM in p_t_m p_t_p_tk; do
  submit "ssv2_std_$PM" -J "sfull_ssv2_std_$PM" --partition=normal \
    --export=ALL,ENCODER=parvo,PARVO_MODE=$PM,CHECKPOINT=$CKPT,SUFFIX=sfull \
    scripts/cluster/ssv2_linear_probe.sbatch
  submit "ssv2_comp_$PM" -J "sfull_ssv2_comp_$PM" --partition=normal \
    --export=ALL,ENCODER=parvo,PARVO_MODE=$PM,CHECKPOINT=$CKPT,ANN_DIR=$SSV2_COMP_ANN,SUFFIX=sfull_comp \
    scripts/cluster/ssv2_linear_probe.sbatch
done

# 제출 목록 기록 → cluster_sessions.md 기입용
if [ "$DRY_RUN" = 0 ]; then
  REC=/proj/external_group/mrg/logs/sfull_followup_jobids.txt
  { echo "# S-full followup — $(date '+%Y-%m-%d %H:%M:%S'), CKPT=$CKPT"
    printf '%s\n' "${SUBMITTED[@]}"; } > "$REC"
  log "done — ${#SUBMITTED[@]}잡 제출, 목록 = $REC"
fi
