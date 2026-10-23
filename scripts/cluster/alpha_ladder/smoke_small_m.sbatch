#!/bin/bash
#SBATCH --job-name=alpha_ladder_smoke
#SBATCH --partition=mig-3g.40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/proj/external_group/mrg/logs/alpha_ladder_smoke_%j.out
# alpha_plan §4-c 작은 M 스모크: ① 옵션 off 손실 = 수정 전 코드(OLD_ROOT)와 동일 ② 작은 M + P 동결 grad 경로 ③ 평가 로더 로드
set -euo pipefail
NEW="${NEW_ROOT:?}"; OLD="${OLD_ROOT:?}"; OUT="${OUT_PT:?}"
PY=/proj/external_group/mrg/conda_envs/aavrl-train/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
$PY "$NEW/scripts/cluster/alpha_ladder/smoke_small_m.py" "$OLD" off
$PY "$NEW/scripts/cluster/alpha_ladder/smoke_small_m.py" "$NEW" off
$PY "$NEW/scripts/cluster/alpha_ladder/smoke_small_m.py" "$NEW" small "$OUT"
cd "$NEW/scripts/eval" && $PY - "$OUT" <<'PYEOF'
import sys, torch
from probe_action_libero import build_parvo_encoder
enc = build_parvo_encoder(sys.argv[1], torch.device("cuda"))
x = torch.rand(2, 3, 224, 224, device="cuda")
t = enc._encode_m_unmasked(enc.preprocessing.compute_m_channel(x, x))
print("loader ok · M tokens", tuple(t.shape), "· m_depth", len(enc.blocks_m), "· m_up", enc.m_up is not None)
PYEOF
