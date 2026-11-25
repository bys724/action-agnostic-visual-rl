#!/bin/bash
# RoPE §7-2 사전 점검 ①② 나머지 칸 (docs/rope_encoder_plan.md §7-2 + "dev 조작화 11-25") — 최소 칸 3 개 통과 뒤 실행.
#   ① 해상도 보간: R0 × seed 1·2 (seed 0 = 최소 칸) · §3 과 같은 칸 + EVAL_PERTURB "res:288,320 resi:288,320"
#      (APE 판은 보간 = §3 res 값 그대로라 다시 재지 않음)
#   ② 판독기 측 위치 임베딩(PROBE_PE=1): 세 모델 × seed 3 × (건강 관문 8 칸 + OOD 4 벤치 attentive)
#      − 최소 칸 2 개(R0 goal s0 · R0 calvin s0)
#   끝나면 precheck_summary.py. 동시 실행 상한 MAXQ (잡 이름 접두사 pc_).
# 사용(로그인 노드 · 세션 분리 · 고정 사본):
#   cp scripts/cluster/rope/precheck_chain.sh /proj/external_group/mrg/logs/rope_precheck_chain.sh
#   (setsid nohup bash /proj/external_group/mrg/logs/rope_precheck_chain.sh > /proj/external_group/mrg/logs/rope_precheck_chain.log 2>&1 < /dev/null &)
set -uo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
L=/proj/external_group/mrg/logs; PY=/proj/external_group/mrg/conda_envs/aavrl-train/bin/python
PART=${PART:-mig-1g.10gb}; MAXQ=${MAXQ:-8}
declare -A CK=(
  [fdn]=/proj/external_group/mrg/checkpoints/two_stream_v15b_refine_comp_s_denoise_augtgt/20260927_235906/checkpoint_epoch0010.pt
  [R0]=/proj/external_group/mrg/checkpoints/two_stream_v15b_rope_R0/20261124_104046/checkpoint_epoch0010.pt
  [A0]=/proj/external_group/mrg/checkpoints/two_stream_v15b_rope_A0/20261124_173501/checkpoint_epoch0010.pt
)
LB=scripts/cluster/probe_action_libero.sbatch; CB=scripts/cluster/probe_action_calvin.sbatch
IDS=()

throttle() { while [ "$(squeue -h -u "$USER" -o %j | grep -c '^pc_')" -ge "$MAXQ" ]; do sleep 60; done; }
sub() {   # sub <name> <sbatch script> [env ...] — 콤마 값은 env 로만(--export= 안 금지)
  local name=$1 script=$2; shift 2; throttle
  local id=$(env "$@" sbatch --parsable -p "$PART" --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=03:00:00 \
             --job-name="$name" --output="$L/${name}_%j.out" --export=ALL "$script")
  [ -n "$id" ] || { echo "sbatch failed for $name"; exit 1; }   # 빈 ID → squeue 전체 큐 → 무한 대기
  echo "$name $id"; IDS+=("$id")
}

echo "[1/2] ① 해상도 보간 R0 seed 1·2 $(date)"
for s in 1 2; do
  n=pc_ri_R0_m_act_pos_s$s
  sub $n $LB ENCODER=parvo CHECKPOINT=${CK[R0]} PROBE_SEED=$s READOUT=attentive TASK_SUITE=libero_object GAPS=20 \
      PARVO_MODE=m_only TARGET=action POSCTRL=concat SUFFIX=$n "EVAL_PERTURB=res:288,320 resi:288,320"
done

echo "[2/2] ② 판독기 측 위치 임베딩 $(date)"
for m in fdn R0 A0; do for s in 0 1 2; do
  C="ENCODER=parvo CHECKPOINT=${CK[$m]} PROBE_SEED=$s READOUT=attentive PROBE_PE=1"
  for st in m pt; do for t in act id; do for p in "" _pos; do
    mode=$([[ $st == m ]] && echo m_only || echo p_t_only); tgt=$([[ $t == act ]] && echo action || echo identity)
    pc=$([[ -n $p ]] && echo concat || echo none); n=pc_pe_${m}_${st}_${t}${p}_s$s
    sub $n $LB $C TASK_SUITE=libero_object GAPS=20 PARVO_MODE=$mode TARGET=$tgt POSCTRL=$pc SUFFIX=$n
  done; done; done
  n=pc_pe_${m}_ood_calvin_attn_s$s
  [[ $m$s == R00 ]] || sub $n $CB $C PARVO_MODE=p_t_m CROSS_FOLDER=1 MAX_EPISODES=200 GAPS=30 SUFFIX=$n
  for su in spatial object goal; do
    n=pc_pe_${m}_ood_${su}_attn_s$s
    [[ $m$s$su == R00goal ]] || sub $n $LB $C PARVO_MODE=p_t_m TASK_SUITE=libero_$su GAPS=20 TARGET=action POSCTRL=none SUFFIX=$n
  done
done; done

echo "완료 대기 $(date)"
J=$(IFS=,; echo "${IDS[*]}")
while squeue -h -j "$J" 2>/dev/null | grep -q .; do sleep 120; done
sacct -j "$J" -X --format=JobID,JobName%34,State,Elapsed | grep -v COMPLETED
$PY scripts/cluster/rope/precheck_summary.py
