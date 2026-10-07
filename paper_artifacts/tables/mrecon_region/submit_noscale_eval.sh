#!/bin/bash
# M-recon 비례항 제거 모델(학습 잡 $1) 완료 후 자동 평가 체인 (afterok).
# ① 영역별 M-recon 오차(고정 표본, C0와 동일 기본값) ② STEP 1 same-probe 8칸(건강 확인; C1-DN fdn_* 설정 복원:
#    libero_object · attentive · gap20 · {m_only,p_t_only} × {action,identity} × POSCTRL {none,concat}).
# ckpt 경로는 학습 디렉터리 타임스탬프를 실행 시점에 해석.
set -euo pipefail
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
TRAIN=$1
RES='CKPT_PATH=$(ls -d /proj/external_group/mrg/checkpoints/two_stream_v15b_refine_comp_s_mrecon_noscale/*/checkpoint_epoch0010.pt | tail -1)'
id=$(sbatch --parsable --dependency=afterok:$TRAIN --kill-on-invalid-dep=yes -p mig-1g.10gb --gres=gpu:1 --cpus-per-task=4 --mem=24G \
  --time=02:00:00 --job-name=mrecon_region_noscale --output=/proj/external_group/mrg/logs/mrecon_region_%j.out \
  --wrap "$RES; export CKPT=\$CKPT_PATH; bash scripts/cluster/mrecon_region_error.sbatch")
echo "region $id"
for m in m pt; do for t in act id; do for p in "" _pos; do
  mode=$([[ $m == m ]] && echo m_only || echo p_t_only); tgt=$([[ $t == act ]] && echo action || echo identity)
  pc=$([[ -n $p ]] && echo concat || echo none)
  id=$(sbatch --parsable --dependency=afterok:$TRAIN --kill-on-invalid-dep=yes -p ${PART:-mig-3g.40gb} --gres=gpu:1 --cpus-per-task=8 \
    --time=00:40:00 --job-name=fns_${m}_${t}${p} --output=/proj/external_group/mrg/logs/probe_libero_fns_${m}_${t}${p}_%j.out \
    --export=ALL,ENCODER=parvo,TASK_SUITE=libero_object,GAPS=20,READOUT=attentive,PARVO_MODE=$mode,TARGET=$tgt,POSCTRL=$pc,SUFFIX=fns_${m}_${t}${p} \
    --wrap "$RES; export CHECKPOINT=\$CKPT_PATH; bash scripts/cluster/probe_action_libero.sbatch")
  echo "fns_${m}_${t}${p} $id"
done; done; done
