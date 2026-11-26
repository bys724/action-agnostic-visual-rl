#!/bin/bash
# §8 1 단계 (docs/rope_encoder_plan.md §8-5 · Vault 11-26 (2)) — M 전용 RoPE 판(--rope-scope m) 처음부터 학습 잡 하나를 감시한다.
#   ① 학습 인자 확인: Model args 에서 --rope · --rope-scope m · --rope-theta 를 뺀 것 = C1(40275371) Model args · 헤더 = epochs 50 · 256/GPU · lr 2.8e-4
#      다르면 scancel 후 종료 (T1 과 같은 레시피여야 같은 θ 의 전부-RoPE 판과 짝이 됨)
#   ② checkpoint_epoch0004.pt 가 생기면 120 s 뒤 학습 scancel (고른 판은 Vault 등록 뒤 같은 SUFFIX 로 재제출 → latest.pt 이어 학습)
#   ③ ep4 로 §8-3 자 측정: 복원 평가(recon_eval · 판정 4 칸 + 평행이동 · 확대 · 독립 창 보고 행) + 토큰 등변성(rope_position --only equiv)
#   ④ 두 판(M1th100 · M1th10000) 산출물이 다 있으면 2 단계 읽기 표: recon_summary --out recon_stage1 (판정 = 못 본 장면 4 칸)
# 사용: setsid nohup bash <고정 사본> <train_jobid> <run 이름 = SUFFIX 예 M1th100> <theta> > <log> 2>&1 < /dev/null &
set -uo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
TRAIN=$1; RUN=$2; THETA=$3; L=/proj/external_group/mrg/logs; PART=${PART:-mig-1g.10gb}
PY=/proj/external_group/mrg/conda_envs/aavrl-train/bin/python
LOG=$L/pretrain_rope_${RUN}_${TRAIN}.out
alive() { squeue -h -j "$TRAIN" 2>/dev/null | grep -q .; }

C1=$(grep -a -m1 "Model args:" $L/pretrain_refine_comp_s_bright_40275371.out | sed 's/.*Model args: //')
while ! grep -aq "Model args:" "$LOG" 2>/dev/null; do alive || { echo "train $TRAIN ended before Model args"; exit 1; }; sleep 120; done
RAW=$(grep -a -m1 "Model args:" "$LOG" | sed 's/.*Model args: //')
GOT=$(echo "$RAW" | sed -E 's/ --rope-theta [0-9.e+-]+//; s/ --rope-scope m//; s/ --rope( |$)/\1/')
HDR=$(grep -a -m1 "Model: " "$LOG")
ok=1
[ "$GOT" == "$C1" ] || { echo "args mismatch vs C1:"; diff <(echo "$C1" | tr ' ' '\n') <(echo "$GOT" | tr ' ' '\n'); ok=0; }
echo "$RAW" | grep -q -- "--rope --rope-scope m" || { echo "--rope --rope-scope m missing"; ok=0; }
if [ "$THETA" != "10000" ]; then echo "$RAW" | grep -q -- "--rope-theta $THETA" || { echo "--rope-theta $THETA missing"; ok=0; }
else echo "$RAW" | grep -q -- "--rope-theta" && { echo "unexpected --rope-theta"; ok=0; }; fi
echo "$HDR" | grep -q "epochs: 50, per-GPU batch: 256, lr: 2.8e-4" || { echo "header mismatch: $HDR"; ok=0; }
[ $ok == 1 ] || { scancel "$TRAIN"; echo "cancelled $TRAIN (recipe mismatch)"; exit 1; }
echo "args OK $(date) · $HDR"

GLOB="/proj/external_group/mrg/checkpoints/two_stream_v15b_rope_${RUN}/*/checkpoint_epoch0004.pt"
while ! ls $GLOB >/dev/null 2>&1; do alive || { echo "train $TRAIN ended without ep4"; exit 1; }; sleep 300; done
sleep 120; scancel "$TRAIN"; CK=$(ls $GLOB | tail -1); echo "ep4 $CK · scancel $TRAIN $(date)"

sub() { local n=$1; shift; local id
  id=$(sbatch --parsable -p "$PART" --gres=gpu:1 --cpus-per-task=16 --mem=32G --time=01:30:00 --job-name=$n --output=$L/${n}_%j.out \
       --export=ALL --wrap "cd $PWD && $PY -u $*")
  [ -n "$id" ] || { echo "sbatch failed for $n"; exit 1; }   # 빈 ID → squeue 전체 큐 → 무한 대기
  echo "$n $id" >&2; echo "$id"; }
unset ROPE_THETA   # θ 는 ckpt 기록에서 읽는다(11-26 로더 수정 뒤 학습 = 기록 있음 · 없으면 로더가 중단)
I1=$(sub rc_$RUN scripts/eval/recon_eval.py --checkpoint $CK --out paper_artifacts/rope_judge/recon/$RUN)
I2=$(sub eq_$RUN scripts/eval/rope_position.py --checkpoint $CK --out paper_artifacts/rope_judge/position_$RUN.json --only equiv)
while squeue -h -j "$I1,$I2" 2>/dev/null | grep -q .; do sleep 60; done
sacct -j "$I1,$I2" -X --format=JobID,JobName%20,State,Elapsed
grep -a "rope_theta\|rope_scope" $L/rc_${RUN}_$I1.out $L/eq_${RUN}_$I2.out   # 평가가 읽은 θ · scope 확인 (기대: θ = $THETA · scope = m)
grep -aq "rope_theta=$THETA " $L/rc_${RUN}_$I1.out && grep -aq "rope_scope=m" $L/rc_${RUN}_$I1.out || { echo "EVAL THETA/SCOPE MISMATCH — 결과 무효"; exit 1; }

R=paper_artifacts/rope_judge
if [ -f $R/recon/M1th100.npz ] && [ -f $R/recon/M1th10000.npz ] && [ -f $R/position_M1th100.json ] && [ -f $R/position_M1th10000.json ]; then
  $PY scripts/cluster/rope/recon_summary.py --arms M1th100 M1th10000 T1th100 T1th10000 C1ep4 \
      --pair M1th100 T1th100 --pair M1th10000 T1th10000 --pair M1th100 M1th10000 --out recon_stage1 > $L/rope_scope_stage1_summary.log 2>&1
  echo "summary exit $? → $R/recon/recon_stage1.md"
else echo "other arm not finished — summary skipped"; fi
echo "chain done $(date)"
