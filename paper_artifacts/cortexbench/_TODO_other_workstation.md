# CortexBench videomae_ours — 다른 워크스테이션 작업 분배

**작성**: 2026-05-26 KST 19:05 / **상태**: 진행 중

## 배경

CortexBench §C11 videomae_ours 21잡을 wrapper normalization fix 후 재실행 중. 로컬 워크스테이션(H100 × 2, 64코어) 한 곳에서 다 돌리면 잡당 8h (CPU oversubscribed 진단). 분배로 가속.

자세한 fix 경위 — commit `b80d56c` "fix(cortexbench): wrapper에 추가된 ImageNet Normalize 제거".

## 진행 상태 (이 시점)

| 상태 | 잡 |
|---|---|
| ✅ DONE (10잡) | pen-v0 × 3, relocate-v0 × 3, assembly × 3, bin-picking seed_100 |
| 🏠 로컬 워크스테이션 담당 (5잡, 진행 중) | **bin-picking seed_200/300, button-press × 3 seed** |
| 🌐 다른 워크스테이션 담당 (6잡, 미시작) | **drawer × 3 seed + hammer × 3 seed** |

## 🌐 다른 워크스테이션에서 할 일

### 사전 준비

```bash
git pull                            # commit b80d56c 이후 wrapper fix + launcher TASKS_OVERRIDE 지원 반영
docker ps                           # cortexbench-eval 컨테이너 떠있는지 확인 (없으면 docker compose up -d)
ls /mnt/data/cortexbench/checkpoints/videomae_ours_best.pt  # encoder weights 존재 확인
```

### 실행

```bash
# 6잡 (drawer × 3 + hammer × 3) background 실행
ENCODERS="videomae_ours" \
TASKS_OVERRIDE="drawer-open-v2-goal-observable hammer-v2-goal-observable" \
OMP_NUM_THREADS_VAL=<코어수×0.4정도> \
NUM_GPUS=<해당 머신 GPU수> \
nohup bash scripts/local/run_cortexbench_matrix.sh \
    > paper_artifacts/cortexbench/_logs/_matrix_videomae_other_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### OMP/NUM_GPUS 권장값 가이드

`NUM_GPUS × OMP_NUM_THREADS_VAL ≤ 호스트 코어 수 × 0.75` 유지 (oversubscription 회피).

| 호스트 코어 | NUM_GPUS | OMP_NUM_THREADS_VAL |
|---|---|---|
| 16 | 1 | 12 |
| 16 | 2 | 6 |
| 32 | 2 | 12 |
| 64 | 2 | 24 (로컬과 동일) |

### 결과 동기화

잡 6개 모두 끝나면 (`_DONE` 마커 6개 생성):

```bash
# (옵션 1) git add + commit + push — 결과 파일이 작아서 권장
git add paper_artifacts/cortexbench/videomae_ours/drawer-open-v2-goal-observable/ \
        paper_artifacts/cortexbench/videomae_ours/hammer-v2-goal-observable/ \
        paper_artifacts/cortexbench/_logs/videomae_ours_drawer-* \
        paper_artifacts/cortexbench/_logs/videomae_ours_hammer-* \
        paper_artifacts/cortexbench/_logs/_matrix_videomae_other_*.log
git commit -m "feat(cortexbench): videomae_ours drawer+hammer 6잡 결과 (other workstation)"
git push origin main

# (옵션 2) 로컬 워크스테이션에서 직접 rsync
# scp/rsync paper_artifacts/cortexbench/videomae_ours/{drawer-open,hammer}-* etri@local:...
```

로컬 워크스테이션에서 `git pull` 후 aggregate 실행:

```bash
python3 scripts/eval/aggregate_cortexbench.py --root paper_artifacts/cortexbench
```

## 이 파일 정리 시점

videomae_ours 21잡 모두 완료 + aggregate 확정되면 이 파일 삭제 (작업 분배 완료).