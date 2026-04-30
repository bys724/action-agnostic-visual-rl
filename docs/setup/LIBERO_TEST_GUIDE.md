# LIBERO BC-T Rollout 가이드

학습된 BC-Transformer 정책(`scripts/eval/finetune_libero_bct.py` 산출물)을
LIBERO 시뮬레이터에서 closed-loop 평가하기 위한 가이드.

> 학습은 **클러스터** (1 GPU H100), rollout은 **로컬 워크스테이션** (mujoco
> 시뮬레이터). BC-T `best.pt`만 클러스터→로컬로 전송하면 되고, LIBERO
> demonstration HDF5는 rollout에 불필요.

## 사전 요구사항

- Docker + NVIDIA Container Toolkit
- GPU (CUDA 12.1+, H100 OK)
- 로컬에 BC-T `best.pt` 1개 (340 MB 정도)
- Docker 이미지 `libero-env:latest` (이미 빌드됨, 31.7 GB)

## 1. 컨테이너 기동

```bash
docker compose up -d libero
docker exec libero-eval python -c "import libero; print('libero ok'); import mujoco; print('mujoco', mujoco.__version__)"
```

마운트:
- 호스트 `.` → 컨테이너 `/workspace`
- `MUJOCO_GL=egl` (헤드리스 NVIDIA 렌더링)

## 2. ckpt 전송 (클러스터 → 로컬)

```bash
# 옵션 1 (rsync, 권장)
mkdir -p /mnt/data/checkpoints/libero_bct
rsync -avzP bys724@<cluster>:/proj/external_group/mrg/checkpoints/libero_bct/<run_dir>/best.pt \
  /mnt/data/checkpoints/libero_bct/bct_<encoder>_<suite>_seed<N>_best.pt
```

ckpt 안의 `cfg.encoder.checkpoint`는 클러스터 path (`/proj/...`)이지만,
`policy_state_dict`이 adapter weights 전부를 포함하므로 rollout 시 자동으로
None override + 덮어쓰기됨 (별도 처리 불필요).

## 3. Sanity rollout (1 task × 1 trial)

```bash
docker exec libero-eval python src/eval_libero.py \
    --checkpoint /mnt/data/checkpoints/libero_bct/bct_videomae-ours_libero_spatial_seed0_best.pt \
    --task-suite libero_spatial \
    --task-ids 0 \
    --num-trials 1
```

- 첫 episode가 success이면 본 평가로
- 실패 시 비디오 (`data/libero/videos/task0_ep0_failure.mp4`)와 stdout으로
  디버깅 (이미지 회전, gripper sign convention, action scale 등)

## 4. 본 평가 (50 trial × 10 task)

```bash
docker exec libero-eval python src/eval_libero.py \
    --checkpoint /mnt/data/checkpoints/libero_bct/bct_<encoder>_<suite>_seed<N>_best.pt \
    --task-suite libero_spatial \
    --num-trials 50 \
    --seed 7
```

- 약 6-12 시간 (10 task × 50 trial × ~max_steps)
- 결과 JSON: `data/libero/results/bct_<...>_libero_spatial_seed7_<ts>.json`
- 비디오: `data/libero/videos/task<i>_ep<j>_{success,failure}.mp4`

## 5. 결과 해석

JSON 구조:
```
{
  "task_suite": "libero_spatial",
  "overall_success_rate": 0.xx,
  "total_successes": N,
  "total_episodes": M,
  "task_results": [{"task_id": ..., "task_description": ..., "success_rate": ...}, ...],
  "metadata": {checkpoint, seed, replan_steps, ...}
}
```

## Task Suite 옵션

| Suite | Tasks | Max Steps |
|-------|-------|-----------|
| `libero_spatial` | 10 | 220 |
| `libero_object`  | 10 | 280 |
| `libero_goal`    | 10 | 300 |
| `libero_10`      | 10 | 520 |
| `libero_90`      | 90 | 400 |

## 트러블슈팅

### MUJOCO_GL EGL 렌더링 실패
```bash
docker exec libero-eval bash -c 'echo $MUJOCO_GL; ls /usr/share/glvnd/egl_vendor.d/'
# MUJOCO_GL=egl, 10_nvidia.json 존재해야 함
```

### action scale / sign 이상
LIBERO env는 7-dim action 기대 (xyz delta 3 + rpy delta 3 + gripper {-1, 1}).
학습 데이터(robomimic HDF5)와 env가 동일 convention인지 첫 비디오로 확인.

### 이미지 회전 (180도)
LIBERO env raw image는 사람 시점에서 거꾸로 보일 수 있음. 학습 데이터(HDF5)도
같은 raw image를 그대로 저장 → 학습/rollout 모두 회전 미적용으로 일관성 유지.
만약 학습 시 별도 회전 처리가 있었다면 `BCTransformerClient._img_to_tensor`도
동일하게 회전 추가 필요.

### Encoder ckpt 경로 문제
ckpt에 박힌 cluster path를 무시하고 `policy_state_dict`로 덮어쓰는 흐름이라
`build_adapter(checkpoint_path=None)` 시 random init이 한 번 일어남.
console에 missing/unexpected keys 0이면 정상.
