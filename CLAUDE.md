# Claude Development Guide

**세션 시작 시 필독**: 아래 핵심 문서를 먼저 읽고 작업 맥락을 파악하세요.

## 핵심 문서 (업데이트 우선)

1. **`docs/RESEARCH_PLAN.md`** - 전체 연구 계획 및 현재 phase (마스터 문서)
2. **`docs/PROBING_GUIDE.md`** - Action probing 실험 가이드 + 결과
3. **`docs/setup/LIBERO_TEST_GUIDE.md`** - LIBERO 평가
4. **`docs/cluster_sessions.md`** - IBS 클러스터 sbatch/salloc 세션 로그 (비용 청구 대조)

**문서 작성 원칙**:
- 새 문서를 만들기보다 기존 핵심 문서 **업데이트** 우선
- 핵심 계획은 `RESEARCH_PLAN.md`에 집중 관리
- 일회성 정보는 git commit message에 기록

## 프로젝트 개요

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?

**핵심 아이디어**:
- EgoDex로 행동-독립적(action-agnostic) 시각 표현 사전학습
- LIBERO에서 로봇 조작 태스크로 평가

## 실행 환경

이 프로젝트는 **두 개의 실행 환경**에서 동작합니다. Python 코드 (`scripts/pretrain.py`, `scripts/data/`, `scripts/eval/`, `src/`)는 환경 무관이며, bash launcher만 환경별로 분리되어 있습니다.

| 환경 | 위치 | 컨테이너 | GPU | 데이터 경로 | Launcher |
|------|------|---------|-----|------------|----------|
| **로컬 워크스테이션** | 외부 워크스테이션 | Docker (`dev-env`) | H100 × 2 (DataParallel) | `/mnt/data/...` | `scripts/local/` |
| **IBS 클러스터** (olaf) | Slurm cluster | Apptainer (예정) | H100 × 4 per node, 최대 2 nodes | `/proj/external_group/mrg/datasets/...` (GPFS) | `scripts/cluster/` |

**클러스터 저장소 요약** (관리자 확인, 2026-04-09):
- `/proj` GPFS: ~16 GB/s read/write, 모든 노드, 영구
- `/scratch/tmp/`: 로컬 NVMe (~6.8 GB/s read, ~2 GB/s write), **GPU 노드(olaf-g)에만**, 전체 쓰기 가능 (관리자 설정, 2026-04-10)
- 이 클러스터는 GPFS가 더 빠름 → 일반 작업은 `/proj` 직접 사용. scratch는 small file random I/O에서만 유리
- 자세한 내용 및 stage-in/out 패턴: [scripts/cluster/README.md](scripts/cluster/README.md)

**역할 분담** (2026-04-09):
- **로컬**: probing, 시각화, LIBERO fine-tuning + rollout (인터랙티브, 빠른 turnaround)
- **클러스터**: full pre-training (장시간 백그라운드, 8 GPU DDP)

자세한 클러스터 사용법은 [scripts/cluster/README.md](scripts/cluster/README.md) 참고.

## 워크플로우

### 1. EgoDex Pre-training

2개 모델 사전학습: **Two-Stream v4** (제안), **VideoMAE** (baseline)

**v4 확정 설정**:
```bash
# 로컬 (Docker dev-env)
docker exec -it dev-env bash
python scripts/pretrain.py --model two-stream \
    --depth 12 --num-stages 2 \
    --mask-ratio 0.3 --mask-ratio-p 0.5 \
    --max-gap 60 --sample-dist triangular --sample-center 30 \
    --egodex-splits part1,part2,part3,part4,part5 \
    --epochs 30 --batch-size 64

# 클러스터 (sbatch)
sbatch scripts/cluster/pretrain.sbatch  # TODO: 작성 예정
```

**데이터 다운로드** (EgoDex CDN 직접):
```bash
bash scripts/local/download_egodex.sh   part2 part3 part5  # 로컬
bash scripts/cluster/download_egodex.sh part2 part3 part5  # 클러스터
```

### 2. Action Probing (사전학습 완료 후)

학습된 표현이 행동 정보를 인코딩하는지 검증

```bash
# EgoDex (within-domain)
python scripts/eval/probe_action.py \
    --encoder two-stream \
    --checkpoint /mnt/data/checkpoints/two_stream/.../best_model.pt \
    --egodex-root /mnt/data/egodex --frames-root /mnt/data/egodex_frames \
    --egodex-split part4 --gap 10 --cls-mode patch_mean_concat

# DROID (cross-domain, primary)
python scripts/eval/probe_action_droid.py \
    --encoder two-stream --checkpoint <ckpt> \
    --droid-root /mnt/data/droid_frames/ext1 --gap 10
```

자세한 내용은 `docs/PROBING_GUIDE.md` 참고

### 3. LIBERO Fine-tuning & Evaluation

학습된 인코더로 로봇 조작 태스크 수행

```bash
# Encoder + MLP decoder fine-tuning
python scripts/eval/finetune_libero.py \
    --encoder two-stream \
    --checkpoint <ckpt> \
    --task-suite libero_spatial --epochs 50

# 시뮬레이터 rollout (success rate)
docker compose up -d libero
docker exec libero-eval python src/eval_libero.py ...
```

자세한 내용은 `docs/setup/LIBERO_TEST_GUIDE.md` 참고

## 주요 파일

| 파일 | 용도 |
|------|------|
| `scripts/pretrain.py` | Pre-training 메인 스크립트 (env-agnostic) |
| `scripts/local/download_egodex.sh` | EgoDex CDN 다운로드+추출 (로컬, `/mnt/data`) |
| `scripts/local/pretrain.sh` | 로컬 학습 launcher (Docker dev-env) |
| `scripts/cluster/download_egodex.sh` | EgoDex CDN 다운로드 (클러스터, `/proj/external_group/mrg`) |
| `scripts/cluster/extract_egodex.sbatch` | 프레임 추출 sbatch (144 CPU 병렬) |
| `scripts/cluster/README.md` | 클러스터 사용법 quickstart |
| `scripts/data/extract_droid_frames.py` | DROID 프레임 추출 |
| `scripts/data/extract_droid_actions.py` | DROID action 추출 (cross-domain probing용) |
| `scripts/eval/probe_action.py` | EgoDex action probing |
| `scripts/eval/probe_action_droid.py` | DROID cross-domain probing |
| `scripts/eval/finetune_libero.py` | LIBERO fine-tuning |
| `scripts/eval/visualize_inference.py` | 단일/다중 모델 비교 시각화 |
| `scripts/eval/visualize_sample_detail.py` | 단일 샘플 상세 (M/P channel + attention) |
| `src/models/two_stream.py` | Two-Stream 모델 + 2D RoPE + MAE masking |
| `src/models/videomae.py` | VideoMAE baseline |
| `src/datasets/egodex.py` | EgoDex 데이터셋 |
| `src/datasets/droid.py` | DROID 데이터셋 |
| `src/training/pretrain.py` | Pre-training 루프 |

## 개발 원칙

1. **문서 업데이트 우선**: 새 문서보다 핵심 문서 업데이트
2. **간결한 코드**: 실험 우선, 과도한 추상화 피하기
3. **검증 필수**: 새 스크립트는 sanity test로 검증 후 배포
4. **Best Practice 참고 필수**:
   - 새로운 구현 전에 **반드시** 공식 문서/권장 방법/샘플 코드 조사
   - 특히 데이터 로딩, 학습 파이프라인 등 성능에 영향을 주는 부분
   - 비효율적 구현으로 재작업하지 않도록 사전 조사 우선

## 클러스터 세션 로깅 (필수)

IBS 클러스터에서 sbatch/salloc 잡을 다룰 때마다 [`docs/cluster_sessions.md`](docs/cluster_sessions.md)를 업데이트해야 함. 비용 청구 대조용.

- **공식 단가** (PDF 기준, VAT 별도): H100 = 61,000원/GPU·일, CPU = 7,000원/노드·일
- **청구 방식**: 월 누적 ceil. 한 달의 모든 GPU·초(또는 노드·초) 합산 → 일수 환산 → ceil → 단가 곱셈. **잡 단위 ceil 아님** → 짧은 sanity test 누적해도 부담 적음
- **잡 제출 시 즉시**: "진행 중" 표에 행 추가 (JobID, 시각, 파티션, 자원, 목적)
- **잡 종료 확인 시**: `sacct -j <JobID> --format=JobID,JobName,Partition,AllocTRES,Start,End,Elapsed,State` 로 시각 조회 → "완료" 표로 이동, **자원·시간** (`n_gpus × elapsed_h` 또는 `n_nodes × elapsed_h`) 기입. 비용은 월말 누적 합산 시 계산
- **저장소**: mrg 그룹 10 TB 이미 할당받음 → 추적 불필요. 10 TB 초과 증설 시에만 기록
- **로그인 노드 활동(다운로드, 파일 작업 등)은 미과금**: 기록 대상 아님
- 세션 시작/마무리 시 이 문서 한 번 훑어보기 — 잊은 항목 없는지 확인

## 데이터셋 전처리 워크플로우

새 데이터셋을 학습에 사용하기 전, 아래 프로세스를 따름:

1. **샘플 테스트**: 소수 영상(3~5개)으로 crop/resize 옵션별 결과 비교
2. **결과 기록**: 비교 이미지 + 결정 근거를 `docs/preprocessing/` 하위에 기록
3. **전체 추출**: 결정된 설정으로 전체 데이터셋 프레임 추출
4. **검증**: 샘플 다운로드하여 품질 확인

### 기존 사례
- **EgoDex** (1920x1080): 센터크롭 → 256x256
- **Bridge V2** (480x640, 4:3): 리사이즈 → 256x256 (crop 없음)
- **DROID** (180x320): 리사이즈 → 256x256 (crop 없음)

## 현재 Phase (2026-04-21)

**Phase 1.5 — v7-big/v8 폐기, v9 설계 및 sanity 단계**

**완료 (Two-Stream lineup)**:
- **v4** (RoPE, mask 0.3/0.5): 48ep. Probing R²=0.197
- **v6** (APE + mask 0.5/0.5 + rotation aug): ep23 scancel, ep8 peak R²=0.259
- **VideoMAE-ours**: 50ep, R²=0.326
- **V-JEPA-ours**: 3차 발산, negative result

**폐기 (EMA teacher + L_P feature prediction 라인)**:
- **v7-big**: ep8 cos_st=0.9997 완전 collapse
- **v8 1차**: ep8 R²=-0.22 (L_P scale mismatch)
- **v8 2차** (BYOL form, λ=0.05, grad 0.3, mask 0.5/0.5): **ep12 scancel (2026-04-21)**. Probing 분해:
  - `patch_mean_concat` R²=-0.147, `patch_mean_m` R²=**+0.160**, `patch_mean_p` R²=**-0.468**
  - 진단: Teacher M=zero 설계 → P가 static salience로 수렴. M의 motion signal이 concat에서 P에 오염됨
- **결론**: "EMA teacher + feature prediction"은 2-frame EgoDex 세팅에서 **구조적 failure mode**. 전면 폐기

**진행 중 — Two-Stream v9** (v4 base + MAE P target):
- P decoder target: `frame_t` (standard MAE, mask 0.75) — semantic 학습
- M decoder target: `frame_{t+k}` (mask 0.3, 기존 유지) — motion 학습
- CLS exchange 양방향 유지 — loss 비대칭성이 역할 분담 자동 유도
- EMA teacher 완전 제거
- **설계 반복 과정**:
  1. Residual target 시도 → sanity에서 P 완전 collapse (`cos_intra_p=1.000`, `std_p=0.028`) → decoder가 "0 출력" trivial minimum으로 수렴
  2. MAE 방식(frame_t 복원)으로 전환 → healthy 학습 확인 (`std_p=0.327`, `cos_intra_p=0.857`, ratio p/m=1.25)
- **v8 대비 ~2.1x 빠름** (teacher forward 제거)
- **Full run**: JobID 33492965 (2026-04-21 15:48~), AIP_long 2노드×4 H100, 50 epoch, 예상 ~33-40h

**Encoder lineup** (변경 없음):
1. Two-Stream v9 (ours, 진행 중) / 2. VideoMAE-ours / 3-7. 공개 가중치

**다음 작업**:
1. v9 sanity 분석 (cos_intra_p=1.000 경고 조사) 후 설계 확정
2. v9 full run 제출 (2노드 × 4 H100 DDP)
3. v9 probing (patch_mean_concat + patch_mean_m/p 분리)
4. DROID 프레임 추출 + Phase 2 개시
5. Phase 3: LIBERO

자세한 내용은 `docs/RESEARCH_PLAN.md`, probing 결과는 `docs/PROBING_GUIDE.md` 참고

## 트러블슈팅 로그

### 재발 가능 — 숙지 필요

- **Sanity checkpoint → 본 학습 auto-resume 오염**: 서로 다른 학습 설정의 체크포인트가 같은 디렉토리에 있으면 scheduler/optimizer 상태 오염 (T_max, lr, batch_size). **sanity 잡은 반드시 `CHECKPOINT_SUFFIX` 사용**.
- **Slurm DDP 3대 함정**: (1) `--gpus-per-task=1` 대신 `--gres=gpu:N` 사용 (NCCL PCI 탐색 실패 방지), (2) `MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))` (포트 충돌 방지), (3) srun에서 `$CONDA_PREFIX/bin/python` 절대 경로 사용
- **로그인 노드 프로세스 제한**: 대용량 다운로드는 순차 실행, 여러 다운로드 동시 실행 금지. `gsutil -m`은 thread/process count 제한 필수 (`-o parallel_thread_count=8 -o parallel_process_count=4`)
- **Scratch stage-in 비현실적**: 소형 파일 수백만 개는 메타데이터 병목. GPFS 직접 읽기 사용

### 해결 완료 — 참고용

- **BF16 autocast**: `use_bf16` 플래그로 scaler와 분리 (GradScaler 불필요). 37.6→62.2 samples/sec
- **SSIM BF16 NaN**: SSIM 연산 FP32 강제 + sigma clamp(min=0)
- **V-JEPA 발산**: (1) EMA target 기반 학습은 LR warmup 필수, (2) 2-frame에서는 mask ratio 대폭 완화 필요 (16-frame temporal redundancy 없음). 결국 3차 모두 발산 → negative result
