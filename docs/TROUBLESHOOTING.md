# 트러블슈팅 로그

> 재발 가능 사고 + 해결 완료 기록. [CLAUDE.md](../CLAUDE.md)에는 must-fire 1-line 가드만 남기고 상세는 여기.
> **Train↔inference preprocessing parity**(가장 비싼 재발 사고)의 체크리스트 단일 출처는 [eval_protocols.md](eval_protocols.md) §0 — 아래엔 요약만.

## Slurm `--time` 결정 가이드 (잡 제출 전 필수 체크)

Slurm timeout으로 학습 손실하는 사고가 반복 발생 (2026-04-29 BC-T 5 encoder: dinov2/siglip/vc1 모두 `--time=24h` 제출 후 epoch당 시간 측정 시 적자 -4.5h~-6.3h → 8h 학습 후 cancel+재제출, ~24 GPU·h 손실). 잡 제출 전 절차:

1. **Epoch당 시간 추정**: 같은 모델/데이터 sanity 또는 과거 잡 elapsed 확인. 추정 어려우면 **`--time`은 partition max** (AIP/AIP_long 모두 max 가능).
2. **계산식**: `필요 시간 = 총 ep × 평균 epoch당 시간 × 1.3 (안전 마진)`. 데이터 로딩/I/O 변동·ckpt 저장 오버헤드 흡수.
3. **multi-encoder 동시 제출 주의**: encoder별 epoch당 시간 크게 다를 수 있음 (실제: BC-T 5 encoder 1082~2731s, **2.5배 차이**). 모든 잡을 가장 느린 encoder 기준 통일된 `--time`으로.
4. **첫 1-2 epoch 후 ETA 재검산**: `잔여 ep × per_ep_time + current wall_time > TIME_LIMIT` 이면 즉시 cancel+재제출. 부분 ckpt보다 paper main table 일관성 유리.
5. **기본값 권장**: 본 학습/fine-tune은 `--time=2-00:00:00` (AIP partition 최대). probing/sanity는 1-3h.

## 재발 가능 — 숙지 필요

- **🔴 Train↔inference preprocessing parity** (2026-05-25 CortexBench v15/videomae 사고): 새 평가/어댑터/wrapper마다 **학습 시 model input range·channel order·정규화** = **inference transform** 정확히 일치 점검. 사고: CortexBench wrapper가 `T.Normalize(ImageNet)` 추가했으나 EgoDex 학습 loader([../src/datasets/base.py:119](../src/datasets/base.py#L119))는 `/255.0`만 → `[0,1]` raw 입력. 두 모델만 OOD inference로 metaworld SR ~53% (vc1/siglip 88%대) 폭락, 30잡(`v15_p_only` 21 + `videomae_ours` 9) 전부 무효. **체크리스트 단일 출처 = [eval_protocols.md](eval_protocols.md) §0** (학습 텐서 range·dtype 확인 / 어댑터 docstring input contract / 첫 batch min/max/dtype 로깅 `finetune_libero_bct.py:_log_first_batch_stats` / 신규 wrapper sanity test).
- **Sanity checkpoint → 본 학습 auto-resume 오염**: 서로 다른 학습 설정 ckpt가 같은 디렉토리에 있으면 scheduler/optimizer 상태 오염 (T_max, lr, batch_size). **sanity 잡은 반드시 `CHECKPOINT_SUFFIX` 사용**.
- **Slurm DDP 3대 함정**: (1) `--gpus-per-task=1` 대신 `--gres=gpu:N` (NCCL PCI 탐색 실패 방지), (2) `MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))` (포트 충돌 방지), (3) srun에서 `$CONDA_PREFIX/bin/python` 절대 경로.
- **로그인 노드 프로세스 제한**: 대용량 다운로드는 순차 실행, 동시 실행 금지. `gsutil -m`은 thread/process count 제한 필수 (`-o parallel_thread_count=8 -o parallel_process_count=4`).
- **Scratch stage-in 비현실적**: 소형 파일 수백만 개는 메타데이터 병목. GPFS 직접 읽기 사용.
- **`num_workers=16` multiprocessing hang** (v15 sanity 사고): 47분 hang → CANCELLED. EgoDex dataset triple은 단독 OK였으나 high worker 수에서 deadlock 추정. **`num_workers=4-8`로 회피**. v15 본 학습은 8 고정.
- **DDP unused-param**: 모델이 forward에 일부 parameter를 안 쓰면 DDP grad reduce가 stuck. **사용 안 하는 mask token 등은 `requires_grad_(False)`** (v14 `mask_token_m` 사고 → 같은 fix). `find_unused_parameters=True`는 임시방편, 정공법은 `requires_grad_(False)`.
- **DINO loss 평형 trap**: K (prototype 수)에 비해 λ가 작으면 `L=log(K)` 근처에 갇혀 prototype space 학습 안 됨 (v13 K=4096+λ=0.005 → L=8.32 plateau). **K=1024 + λ=0.01 권장 (DINOv2 default)**. K↑가 항상 좋은 게 아님.
- **3-frame vs 2-frame 학습 분포 mismatch**: v15는 3-frame triple로 학습. 기존 2-frame pair BC-T 어댑터 그대로 쓰면 학습 분포 불일치. **v15 전용 신규 어댑터(`two-stream-v15-ptptk`, `two-stream-v15-mp`) 사용 필수**. 새 모델 도입 시 BC-T adapter 호환성 사전 검토.

## 해결 완료 — 참고용

- **BF16 autocast**: `use_bf16` 플래그로 scaler와 분리 (GradScaler 불필요). 37.6→62.2 samples/sec.
- **SSIM BF16 NaN**: SSIM 연산 FP32 강제 + sigma clamp(min=0).
- **V-JEPA 발산**: (1) EMA target 기반 학습은 LR warmup 필수, (2) 2-frame에서는 mask ratio 대폭 완화 필요 (16-frame temporal redundancy 없음). 결국 3차 모두 발산 → negative result.
