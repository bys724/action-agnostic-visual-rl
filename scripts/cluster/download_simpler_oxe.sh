#!/bin/bash
# SIMPLER 학습 데이터 (OXE 서브셋) 다운로드 — 로그인 노드 전용, 순차, resume 지원.
#   Bridge (WidowX/Bridge setup, 416GB) → Fractal (RT-1/Google Robot setup, 119GB)
# 계획: docs/simpler_eval_plan.md §A. WidowX/Bridge 먼저(widowx250s asset 준비됨) → Google Robot.
#
# 사용 (로그인 노드, nohup 백그라운드):
#   nohup bash scripts/cluster/download_simpler_oxe.sh > /proj/external_group/mrg/logs/download_simpler.log 2>&1 &
#
# 주의 (docs/dataset_todo.md):
#   - 로그인 노드에서만 (compute 노드 외부 네트워크 제한)
#   - 동시 2개 이상 다운로드 금지 → 이 스크립트는 순차 실행 (rsync 1개씩)
#   - gsutil rsync = resume 지원: 끊기면 재실행하면 이어받음 (기존 파일 skip)
set -u

GSUTIL=/proj/external_group/mrg/conda_envs/aavrl-extract/bin/gsutil
DEST=/proj/external_group/mrg/datasets/oxe
# gsutil 병렬 throttle (사용자 process limit 초과 방지, dataset_todo.md 교훈)
OPTS=(-o "GSUtil:parallel_thread_count=8" -o "GSUtil:parallel_process_count=4")

download_one() {
    local name="$1" src="$2" dst="$3"
    mkdir -p "$dst"
    echo "[$(date '+%F %T')] START $name → $dst"
    "$GSUTIL" "${OPTS[@]}" -m rsync -r "$src" "$dst"
    echo "[$(date '+%F %T')] DONE  $name (rc=$?)"
}

# Bridge 먼저 (계획 우선), 이후 Fractal — 순차
download_one "bridge (416GB)"  "gs://gresearch/robotics/bridge/0.1.0/"                 "$DEST/bridge/0.1.0/"
download_one "fractal (119GB)" "gs://gresearch/robotics/fractal20220817_data/0.1.0/"  "$DEST/fractal20220817_data/0.1.0/"

echo "[$(date '+%F %T')] ALL SIMPLER OXE downloads complete → $DEST"
