#!/bin/bash
# T1 θ 점검 (docs/rope_encoder_plan.md §7-2 ③ 재설계 · 사용자 11-25 승인) — 처음부터 학습 잡 하나를 감시한다.
#   ① 학습 인자 확인: Model args 에서 --rope · --rope-theta 를 뺀 것 = C1(40275371) Model args · 헤더 = epochs 50 · lr 2.8e-4 · part1
#      다르면 scancel 후 종료(레시피 어긋난 판은 비교 무효)
#   ② checkpoint_epoch0004.pt 가 생기면 120 s 뒤 학습 scancel (이긴 판은 Vault 확인 뒤 같은 SUFFIX 로 다시 제출 → latest.pt 자동 이어 학습)
#   ③ ep4 체크포인트로 건강 관문 8 칸 × 판독기 seed 0·1·2 제출(C1-DN fdn_* 와 같은 설정) → 끝나면 theta_summary.py
# 사용: setsid nohup bash <고정 사본> <train_jobid> <run 이름 = SUFFIX 예 T1th100> <theta> > <log> 2>&1 < /dev/null &
set -uo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
TRAIN=$1; RUN=$2; THETA=$3; L=/proj/external_group/mrg/logs; PART=${PART:-mig-1g.10gb}
LOG=$L/pretrain_rope_${RUN}_${TRAIN}.out
alive() { squeue -h -j "$TRAIN" 2>/dev/null | grep -q .; }

C1=$(grep -a -m1 "Model args:" $L/pretrain_refine_comp_s_bright_40275371.out | sed 's/.*Model args: //')
while ! grep -aq "Model args:" "$LOG" 2>/dev/null; do alive || { echo "train $TRAIN ended before Model args"; exit 1; }; sleep 120; done
RAW=$(grep -a -m1 "Model args:" "$LOG" | sed 's/.*Model args: //')
GOT=$(echo "$RAW" | sed -E 's/ --rope-theta [0-9.e+-]+//; s/ --rope( |$)/\1/')
HDR=$(grep -a -m1 "Model: " "$LOG")
ok=1
[ "$GOT" == "$C1" ] || { echo "args mismatch vs C1:"; diff <(echo "$C1" | tr ' ' '\n') <(echo "$GOT" | tr ' ' '\n'); ok=0; }
echo "$RAW" | grep -q -- "--rope" || { echo "--rope missing"; ok=0; }
if [ "$THETA" != "10000" ]; then echo "$RAW" | grep -q -- "--rope-theta $THETA" || { echo "--rope-theta $THETA missing"; ok=0; }; fi
echo "$HDR" | grep -q "epochs: 50, per-GPU batch: 256, lr: 2.8e-4" || { echo "header mismatch: $HDR"; ok=0; }
[ $ok == 1 ] || { scancel "$TRAIN"; echo "cancelled $TRAIN (recipe mismatch)"; exit 1; }
echo "args OK $(date) · $HDR"

GLOB="/proj/external_group/mrg/checkpoints/two_stream_v15b_rope_${RUN}/*/checkpoint_epoch0004.pt"
while ! ls $GLOB >/dev/null 2>&1; do alive || { echo "train $TRAIN ended without ep4"; exit 1; }; sleep 300; done
sleep 120; scancel "$TRAIN"; CK=$(ls $GLOB | tail -1); echo "ep4 $CK · scancel $TRAIN $(date)"

IDS=()
for s in 0 1 2; do for m in m pt; do for t in act id; do for p in "" _pos; do
  mode=$([[ $m == m ]] && echo m_only || echo p_t_only); tgt=$([[ $t == act ]] && echo action || echo identity)
  pc=$([[ -n $p ]] && echo concat || echo none); n=t1_${RUN}_${m}_${t}${p}_s$s
  id=$(env ENCODER=parvo CHECKPOINT=$CK PROBE_SEED=$s READOUT=attentive TASK_SUITE=libero_object GAPS=20 PARVO_MODE=$mode \
       TARGET=$tgt POSCTRL=$pc SUFFIX=$n sbatch --parsable -p "$PART" --gres=gpu:1 --cpus-per-task=4 --mem=48G \
       --time=02:00:00 --job-name=$n --output=$L/${n}_%j.out --export=ALL scripts/cluster/probe_action_libero.sbatch)
  [ -n "$id" ] || { echo "sbatch failed for $n"; exit 1; }   # 빈 ID → squeue 전체 큐 → 무한 대기
  echo "$n $id"; IDS+=("$id")
done; done; done; done
J=$(IFS=,; echo "${IDS[*]}")
while squeue -h -j "$J" 2>/dev/null | grep -q .; do sleep 120; done
sacct -j "$J" -X --format=JobID,JobName%34,State,Elapsed | grep -v COMPLETED
/proj/external_group/mrg/conda_envs/aavrl-train/bin/python scripts/cluster/rope/theta_summary.py
