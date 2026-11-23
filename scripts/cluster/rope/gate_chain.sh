#!/bin/bash
# RoPE 공학 단계 예비 건강 관문: 학습 잡 $1 의 epoch $3 체크포인트가 생기면 STEP 1 same-probe 8 칸 제출 → 집계.
#   설정 = C1-DN fdn_* 와 같음: libero_object · attentive · gap20 · {m_only,p_t_only} × {action,identity} × POSCTRL {none,concat}
#   학습은 계속 돌린다(의존성 없음). 결과는 "공학 단계 수치" — 판정(§3-1)에 쓰지 않는다 (rope_encoder_plan.md §2).
# 사용: bash scripts/cluster/rope/gate_chain.sh <train_jobid> <run 이름(예 R0)> <epoch> [PART=mig-1g.10gb]
set -uo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
TRAIN=$1; RUN=$2; EP=$(printf "%04d" "$3"); PART=${PART:-mig-1g.10gb}
GLOB="/proj/external_group/mrg/checkpoints/two_stream_v15b_rope_${RUN}/*/checkpoint_epoch${EP}.pt"
while ! ls $GLOB >/dev/null 2>&1; do
  if ! squeue -h -j "$TRAIN" 2>/dev/null | grep -q .; then echo "train $TRAIN ended without $GLOB"; exit 1; fi
  sleep 300
done
sleep 120   # 저장 완료 여유
CK=$(ls $GLOB | tail -1); echo "ckpt $CK"
TAG=rope_${RUN}_ep$3; IDS=()
for m in m pt; do for t in act id; do for p in "" _pos; do
  mode=$([[ $m == m ]] && echo m_only || echo p_t_only); tgt=$([[ $t == act ]] && echo action || echo identity)
  pc=$([[ -n $p ]] && echo concat || echo none)
  id=$(sbatch --parsable -p "$PART" --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=02:00:00 \
    --job-name=${TAG}_${m}_${t}${p} --output=/proj/external_group/mrg/logs/probe_libero_${TAG}_${m}_${t}${p}_%j.out \
    --export=ALL,ENCODER=parvo,TASK_SUITE=libero_object,GAPS=20,READOUT=attentive,PARVO_MODE=$mode,TARGET=$tgt,POSCTRL=$pc,SUFFIX=${TAG}_${m}_${t}${p},CHECKPOINT=$CK \
    scripts/cluster/probe_action_libero.sbatch)
  echo "${TAG}_${m}_${t}${p} $id"; IDS+=("$id")
done; done; done
J=$(IFS=,; echo "${IDS[*]}")
while squeue -h -j "$J" 2>/dev/null | grep -q .; do sleep 120; done
sacct -j "$J" --format=JobID,JobName%34,State,Elapsed -X
/proj/external_group/mrg/conda_envs/aavrl-train/bin/python scripts/cluster/rope/gate_summary.py "$TAG"
