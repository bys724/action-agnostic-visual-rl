#!/bin/bash
# RoPE 판정 단계 격자 (docs/rope_encoder_plan.md §3 + "dev 조작화") — 세 모델 × 판독기 seed 3, 동시 실행 상한 MAXQ.
#   칸(모델 · seed 마다): 건강 관문 8(m_act_pos 칸에서 같은 판독기로 평행이동 shift:8,16,32 · 해상도 res:288,320 교란을 같이 잰다)
#     · 분포 밖 4 벤치 × {attentive, mean}(p_t_m · CALVIN gap30 교차 폴더 · LIBERO gap20) = 16
#   모델마다 한 번: rope_position.py(등변성 + 위치 판독 · 내부 seed 3) · 4-a(alpha_readout · seed 42/1/2)
#   C1-DN · R0 는 바로, A0 는 ep10 체크포인트가 생기면 제출. 끝나면 judge_summary.py.
# 사용(로그인 노드 · 세션 분리 · 고정 사본):
#   cp scripts/cluster/rope/judge_chain.sh /proj/external_group/mrg/logs/rope_judge_chain.sh
#   (setsid nohup bash /proj/external_group/mrg/logs/rope_judge_chain.sh > /proj/external_group/mrg/logs/rope_judge_chain.log 2>&1 < /dev/null &)
set -uo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
L=/proj/external_group/mrg/logs; PY=/proj/external_group/mrg/conda_envs/aavrl-train/bin/python
PART=${PART:-mig-1g.10gb}; MAXQ=${MAXQ:-8}; A0_JOB=${A0_JOB:-44502065}; SEEDS=${SEEDS:-"0 1 2"}
CK_fdn=/proj/external_group/mrg/checkpoints/two_stream_v15b_refine_comp_s_denoise_augtgt/20260927_235906/checkpoint_epoch0010.pt
CK_R0=/proj/external_group/mrg/checkpoints/two_stream_v15b_rope_R0/20261124_104046/checkpoint_epoch0010.pt
A0_GLOB="/proj/external_group/mrg/checkpoints/two_stream_v15b_rope_A0/*/checkpoint_epoch0010.pt"
IDS=()

throttle() { while [ "$(squeue -h -u "$USER" -o %j | grep -c '^rj_')" -ge "$MAXQ" ]; do sleep 60; done; }
sub() {   # sub <job name> <sbatch script | --wrap cmd> [env ...] — 콤마 값(EVAL_PERTURB)은 env 로만 넘긴다(--export= 안 금지)
  local name=$1 script=$2; shift 2; throttle
  local id
  if [[ $script == --wrap ]]; then
    id=$(env "${@:2}" sbatch --parsable -p "$PART" --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=03:00:00 \
         --job-name="$name" --output="$L/${name}_%j.out" --export=ALL --wrap "$1")
  else
    id=$(env "$@" sbatch --parsable -p "$PART" --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=03:00:00 \
         --job-name="$name" --output="$L/${name}_%j.out" --export=ALL "$script")
  fi
  [ -n "$id" ] || { echo "sbatch failed for $name"; exit 1; }   # 빈 ID → squeue 전체 큐 → 무한 대기
  echo "$name $id"; IDS+=("$id")
}

model_cells() {   # model_cells <tag> <ckpt>
  local m=$1 ck=$2 s
  for s in $SEEDS; do
    local LB="scripts/cluster/probe_action_libero.sbatch" C="ENCODER=parvo CHECKPOINT=$ck PROBE_SEED=$s READOUT=attentive"
    for st in m pt; do for t in act id; do for p in "" _pos; do
      local mode=$([[ $st == m ]] && echo m_only || echo p_t_only) tgt=$([[ $t == act ]] && echo action || echo identity)
      local pc=$([[ -n $p ]] && echo concat || echo none) ex=()
      [[ $st$t$p == mact_pos ]] && ex=("EVAL_PERTURB=shift:8,16,32 res:288,320")   # 공백 포함 → 배열로 한 인자
      sub rj_${m}_${st}_${t}${p}_s$s $LB $C TASK_SUITE=libero_object GAPS=20 PARVO_MODE=$mode TARGET=$tgt POSCTRL=$pc \
          SUFFIX=rj_${m}_${st}_${t}${p}_s$s "${ex[@]}"
    done; done; done
    for ro in attentive mean; do
      local rn=$([[ $ro == attentive ]] && echo attn || echo mean)
      sub rj_${m}_ood_calvin_${rn}_s$s scripts/cluster/probe_action_calvin.sbatch ENCODER=parvo CHECKPOINT=$ck PROBE_SEED=$s \
          READOUT=$ro PARVO_MODE=p_t_m CROSS_FOLDER=1 MAX_EPISODES=200 GAPS=30 SUFFIX=rj_${m}_ood_calvin_${rn}_s$s
      for su in spatial object goal; do
        sub rj_${m}_ood_${su}_${rn}_s$s $LB ENCODER=parvo CHECKPOINT=$ck PROBE_SEED=$s READOUT=$ro PARVO_MODE=p_t_m \
            TASK_SUITE=libero_$su GAPS=20 TARGET=action POSCTRL=none SUFFIX=rj_${m}_ood_${su}_${rn}_s$s
      done
    done
  done
  sub rj_${m}_position --wrap "cd $PWD && $PY -u scripts/eval/rope_position.py --checkpoint $ck --out paper_artifacts/rope_judge/position_${m}.json"
  for s in 42 1 2; do   # 4-a = alpha_readout.sbatch 자체 자원(mig-3g · 4 h) · 출력은 별도 폴더(기존 C1-DN 4-a 파일 덮어쓰기 방지)
    throttle
    local id=$(env P_SOURCE=ckpt CHANGE=comp SEED=$s CKPT=$ck EXTRA="--out-root paper_artifacts/rope_judge/alpha4a_$m" \
               sbatch --parsable --job-name=rj_${m}_alpha4a_s$s --output="$L/rj_${m}_alpha4a_s${s}_%j.out" \
               --export=ALL scripts/cluster/alpha_readout.sbatch)
    [ -n "$id" ] || { echo "sbatch failed for rj_${m}_alpha4a_s$s"; exit 1; }
    echo "rj_${m}_alpha4a_s$s $id"; IDS+=("$id")
  done
}

echo "[1/3] C1-DN · R0 $(date)"
model_cells fdn "$CK_fdn"
model_cells R0 "$CK_R0"
echo "[2/3] A0 ep10 대기 $(date)"
while ! ls $A0_GLOB >/dev/null 2>&1; do
  if ! squeue -h -j "$A0_JOB" 2>/dev/null | grep -q .; then echo "A0 $A0_JOB ended without $A0_GLOB"; exit 1; fi
  sleep 300
done
sleep 120   # 저장 완료 여유
model_cells A0 "$(ls $A0_GLOB | tail -1)"
echo "[3/3] 완료 대기 $(date)"
J=$(IFS=,; echo "${IDS[*]}")
while squeue -h -j "$J" 2>/dev/null | grep -q .; do sleep 120; done
sacct -j "$J" -X --format=JobID,JobName%34,State,Elapsed | grep -v COMPLETED
$PY scripts/cluster/rope/judge_summary.py
