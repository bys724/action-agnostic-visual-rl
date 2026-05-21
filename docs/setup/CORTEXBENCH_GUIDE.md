# CortexBench §C11 Guide

CortexBench manipulation subset (Adroit + Meta-World) BC training & evaluation.
Paper §5.1 추가 ablation (§C11 appendix).

## 매트릭스 구성

- **Encoders (5)**: `v15_p_only`, `videomae_ours`, `siglip_base`, `dinov2_base`, `vc1_vitb`
- **Tasks (7)**: Adroit (pen, relocate) + Meta-World (assembly, bin-picking, button-press-topdown, drawer-open, hammer)
- **Seeds (3)**: 100, 200, 300
- **Total**: 5 × 7 × 3 = **105잡** BC training (epoch=99, eval 25 episodes / 5 epoch)

각 잡 = encoder 별 frozen embedding precompute → 작은 BC head 학습 → 25-episode rollout. 잡당 1~9h (env별 편차 큼). 2 GPU round-robin.

## 환경

- Docker image: `cortexbench-eval` (Miniforge + conda-forge, defaults channel 미사용 — 라이센스 free)
- Dockerfile: `docker/cortexbench/Dockerfile`
- 호스트 요구: NVIDIA driver ≥ 520 (CUDA 11.8), `nvidia-container-toolkit`
- Datasets:
  - `/mnt/data/cortexbench/datasets/adroit-expert-v1.0/` (5.9 GB)
  - `/mnt/data/cortexbench/datasets/metaworld-expert-v1.0/` (12 GB)
- Checkpoints (v15 / videomae_ours만 별도 ckpt 필요):
  - `/mnt/data/cortexbench/checkpoints/v15_ep50_latest.pt`
  - `/mnt/data/cortexbench/checkpoints/videomae_ours_best.pt`
  - SigLIP / DINOv2 / VC-1은 vc_models가 HF에서 자동 다운로드 (인터넷 필요)

## 단일 머신 실행

```bash
# 전체 105잡
nohup bash scripts/local/run_cortexbench_matrix.sh > paper_artifacts/cortexbench/_logs/_matrix.log 2>&1 &
echo $! > paper_artifacts/cortexbench/_logs/_matrix.pid

# 집계만
bash scripts/local/run_cortexbench_matrix.sh --aggregate
```

ETA: H100 × 2 + 64 CPU 기준 **~7일** wallclock (잡당 평균 ~3h, 2 GPU 병렬).

## 분산 운영 (2대 워크스테이션)

CortexBench는 **CPU bound** (mujoco rollout). GPU는 8% 수준이라 GPU 종류 영향 적음. CPU 코어 수가 ETA 결정 요인.

### 분할 plan

| 머신 | 담당 encoder | 잡 수 | 비고 |
|---|---|---|---|
| **머신 A (현재 로컬)** | `v15_p_only` + `videomae_ours` | 42 | ours + 직접 비교 baseline. v15/videomae ckpt 보유 |
| **머신 B (새 WS)** | `siglip_base` + `dinov2_base` + `vc1_vitb` | 63 | HF 자동 다운로드. ckpt 전송 불필요 |

이 분할은 **ckpt 전송 0**으로 새 머신 setup을 최소화한다. 양쪽 ETA가 비슷하지 않으므로 (A=42잡, B=63잡), GPU/CPU 균등하다면 B가 ~1.5배 더 오래 걸리지만 baseline 3종 single ViT-B라 forward는 v15 dual encoder보다 가벼움 → 실제 ETA 비슷할 가능성.

### 실행

머신 A:
```bash
ENCODERS="v15_p_only videomae_ours" bash scripts/local/run_cortexbench_matrix.sh
```

머신 B:
```bash
ENCODERS="siglip_base dinov2_base vc1_vitb" bash scripts/local/run_cortexbench_matrix.sh
```

각 머신 `paper_artifacts/cortexbench/<enc>/<task>/seed_*/_DONE`이 idempotent 마커이므로 재시작 안전.

### 새 머신 setup 단계

1. **Docker image 전송** (재빌드 비추천 — setup gotchas 多)

   ```bash
   # 머신 A
   docker save cortexbench-eval | gzip > /tmp/cortexbench-eval.tar.gz   # ~20GB
   rsync -avP /tmp/cortexbench-eval.tar.gz USER@B:/tmp/

   # 머신 B
   gunzip -c /tmp/cortexbench-eval.tar.gz | docker load
   ```

2. **데이터셋 sync** (~18 GB, 1Gbps LAN에서 ~3분)

   ```bash
   # 머신 B에서
   rsync -avP USER@A:/mnt/data/cortexbench/datasets/ /mnt/data/cortexbench/datasets/
   ```

3. **Repo clone + 컨테이너 기동**

   ```bash
   git clone <repo> /home/USER/action-agnostic-visual-rl
   cd /home/USER/action-agnostic-visual-rl
   docker run -d --name cortexbench-eval --gpus all \
       -v $PWD:/workspace -v /mnt/data:/mnt/data \
       cortexbench-eval tail -f /dev/null
   docker exec cortexbench-eval bash /usr/local/bin/install_eai_vc.sh
   ```

4. **매트릭스 실행** (위 "실행" 참고)

### 결과 통합 (git 기반)

산출물 분리 원칙:

| 항목 | 위치 | git track? | 비고 |
|---|---|---|---|
| `<enc>/<task>/seed_*/_DONE` | repo | ✅ | 빈 마커, 잡 완료 신호 |
| `<enc>/<task>/seed_*/.hydra/*.yaml` | repo | ✅ | hydra config (~5 KB) |
| `<enc>/<task>/seed_*/{adroit,metaworld}_cortex_vil.log` | repo | ✅ | wandb summary 추출 가능 (~25 KB/잡) |
| `<enc>/<task>/seed_*/job_config.json` | repo | ✅ | 작음 |
| `_logs/<enc>_<task>_seed*.log` | repo | ✅ | stdout 학습 로그 (aggregator 입력) |
| `per_run.csv` / `per_task.csv` / `summary.csv` | repo | ❌ `.gitignore` | derived data, 충돌 방지 |
| `<...>/wandb/`, `<...>/{adroit,metaworld}_cortex_vil/` | repo | ❌ `.gitignore` | wandb 폴더 + ckpt, 무거움 |

**충돌 0**: 각 머신이 자기 encoder 디렉토리만 commit → 두 머신 변경 행이 완전히 분리됨.

#### 운영 절차

머신 B (잡 완료 후):
```bash
git add paper_artifacts/cortexbench/{siglip_base,dinov2_base,vc1_vitb}/ \
        paper_artifacts/cortexbench/_logs/
git commit -m "feat(cortexbench): §C11 baseline 3종 63잡 결과"
git push
```

머신 A (또는 어떤 머신에서든):
```bash
git pull                                                                     # 양쪽 raw 결과 모임
python3 scripts/eval/aggregate_cortexbench.py --root paper_artifacts/cortexbench  # 통합 csv 생성
```

csv는 git에 안 들어가므로 누구든 pull 후 1회 aggregate로 최신 통합 결과 확인 가능.

#### Launcher 자동 집계

`run_cortexbench_matrix.sh`는 매트릭스 끝나면 자동으로 `aggregate_cortexbench.py`를 호출함. 머신별 자기 encoder만 채워진 partial csv가 로컬에 생성되며 (git track X), 진행 중에도 임의 시점에 `--aggregate`로 수동 갱신 가능.

## 트러블슈팅

- **Cython<3 / mujoco<3 호환성**: `install_eai_vc.sh`가 pin. 직접 setup 시 주의.
- **mujoco offscreen rendering**: `MUJOCO_GL=egl` + `nvidia-container-toolkit` 필수.
- **CPU 경합**: 같은 머신에 LIBERO rollout 등 다른 mujoco 작업 병행 시 둘 다 1.5~2배 느려짐.
- **wandb offline**: ckpt 디렉토리 안 `wandb/offline-run-*/` → 필요 시 `wandb sync` (paper 결과는 stdout `eval/highest_*`로 추출).
