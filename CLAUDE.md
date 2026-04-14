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

## 현재 Phase (2026-04-14)

**Phase 1.5 — Two-Stream resume + VideoMAE-ours로 encoder lineup 재편**

- **Two-Stream v4**: 50 epoch 중 48 완료 후 TIMEOUT (32712324). Resume 잡 32983533 제출 (AIP, 2노드×4 H100, --time=6h, 남은 2 epoch)
- **V-JEPA-ours**: 3차례 시도 모두 발산 (mask ratio/warmup 완화 무효). **30 epoch까지 기록 후 중단, paper Appendix의 negative result로 기록 (3 attempts loss curve + Two-Stream reference overlay)**
  - 1차(32867433): warmup 없음 → epoch 4 취소
  - 2차(32867645): warmup 추가, mask 0.80/0.85 → epoch 7 취소
  - 3차(32950553): warmup + mask 0.50/0.60 → epoch 20까지 수렴 후 재발산. 30 epoch까지 기록
- **VideoMAE-ours (2-frame)**: V-JEPA-ours 대체 controlled comparison 모델. 구현은 이미 완료 (`src/models/videomae.py` num_frames=2, tubelet_size=2). **mask ratio 0.5** (공식 0.75 대신; 2-frame은 temporal redundancy 부재). V-JEPA 중단 후 제출
- **DROID**: 다운로드 완료 (3.4 TiB), 프레임 추출 대기 중

**Encoder lineup 재편 (2026-04-14)**:
1. **Two-Stream v4** (ours, EgoDex, M/P 구조) — 🔥 학습 (진행 중)
2. **VideoMAE-ours 2-frame** (ours, EgoDex, vanilla MAE, mask 0.5) — 🔥 학습 (대기)
3. VideoMAE-official (16-frame 공식) — 📦 공개 가중치
4. V-JEPA-official (16-frame 공식) — 📦 공개 가중치
5. VC-1-Base (Ego4D+조작, 로봇 SOTA) — 📦 공개 가중치
6. DINOv2-Base (LVD-142M 웹) — 📦 공개 가중치
7. SigLIP-Base (WebLI 웹) — 📦 공개 가중치

**평가 프로토콜**:
- **EgoDex probing**: Two-Stream vs VideoMAE-ours만 (통제 비교, 축 1 — 구조적 bias 기여도)
- **DROID probing (main)**: 7개 모두 (cross-encoder fair comparison, 전부 OOD)
- **LIBERO (Phase 3)**: 7개 모두

**주요 관찰**:
- V-JEPA 실패는 "2-frame 세팅이 feature prediction의 temporal redundancy 전제를 깨뜨림"을 실증 → negative result로 보존
- VideoMAE-ours는 pixel MAE framework에서 encoder 구조만 바꾼 controlled ablation
- 파라미터 불균형(213M vs 101M)은 "Two-Stream M/P 설계상 불가피, 각 스트림 ViT-B per-stream capacity는 동일"로 명시

**다음 작업**:
1. Two-Stream resume 완료 대기 (~6h)
2. V-JEPA-ours 30 epoch 도달 후 중단, 3-attempt + Two-Stream overlay 그래프 생성
3. VideoMAE-ours 2-frame full training (~3일)
4. Phase 2: DROID action probing (7 encoder)
5. Phase 3: LIBERO BC + MLP (7 encoder)
6. Phase 3B: OpenVLA encoder 교체 + LoRA (7 encoder)

자세한 내용은 `docs/RESEARCH_PLAN.md` 참고

## 트러블슈팅 로그

### EgoDex test symlink 문제 (2026-03-23)
- **증상**: eval 시 `FileNotFoundError: .../part2/.../frame_000000.jpg`로 학습 크래시
- **원인**: `scripts/local/pretrain.sh`가 test symlink을 잘못 생성. 추출된 프레임은 `/mnt/data/egodex_frames/test_frames/`
- **수정**: symlink을 `test_frames`로 교체
- **교훈**: dataset 객체는 시작 시 경로를 캐싱하므로, 학습 중 symlink 수정해도 효과 없음 → 재시작 필요

### 학습 프로세스 안정성 (2026-03-23)
- `evaluate()`와 `save_epoch_samples()`를 try/except로 보호
- eval/시각화 실패 시 WARNING 출력 후 학습 계속 (크래시 방지)
- `model.train()` 복구를 finally 블록으로 보장

### SSIM loss BF16 안정성 (2026-03-31)
- BF16 autocast에서 sigma_sq가 음수가 되어 NaN 발생
- 해결: SSIM 연산을 FP32로 강제 + sigma clamp(min=0)
- GradScaler 제거 (BF16에는 불필요)

### AMP BF16 autocast 비활성화 버그 (2026-04-10)
- **증상**: "AMP enabled" 로그가 뜨지만 실제 FP32로 실행. H100 throughput의 절반만 활용.
- **원인**: `scaler = None` → `use_amp = scaler is not None` → False → autocast 꺼짐
- **수정**: `use_bf16` 플래그로 autocast를 scaler와 분리. BF16은 GradScaler 불필요.
- **효과**: 37.6 → 62.2 samples/sec (+65%, batch 4 기준)

### Sanity checkpoint → 본 학습 auto-resume 오염 (2026-04-10)
- **증상**: 본 학습이 sanity test 체크포인트를 auto-resume. CosineAnnealingLR의 T_max=2(sanity)로 덮어써져 lr이 2 epoch 주기로 진동.
- **원인**: sanity와 본 학습이 같은 checkpoint_dir 사용 + pretrain.py auto-resume 로직이 latest.pt 자동 감지
- **수정**: sanity 체크포인트 삭제 후 fresh start. 향후 sanity 잡은 반드시 `CHECKPOINT_SUFFIX` 사용.
- **교훈**: auto-resume은 편리하지만, 서로 다른 학습 설정의 체크포인트가 같은 디렉토리에 있으면 scheduler/optimizer 상태 오염. 특히 T_max, lr, batch_size가 다른 경우 치명적.

### Scratch stage-in 소형 파일 병목 (2026-04-10)
- **증상**: `cp -a`로 1.25 TB JPG 프레임을 GPFS → scratch 복사 시 1시간 37분에 part1의 2개 dir만 복사됨
- **원인**: 수천만 개 소형 파일의 per-file 메타데이터 오버헤드 (stat, open, create, close × 1400만 파일)
- **수정**: scratch stage-in 포기, GPFS 직접 읽기로 전환. tar 파이프 방식은 대안으로 검토 중.
- **교훈**: scratch는 대용량 파일 소수에 유리. 소형 파일 수백만 개는 tar/WebDataset 패키징 없이 stage-in 비현실적.

### 로그인 노드 다운로드로 인한 접속 장애 (2026-04-14)
- **증상**: `download_all_data.sh`를 로그인 노드에서 nohup으로 실행 후 접속 장애 발생 → 관리자 강제 kill
- **원인**: 6개 병렬 `curl`(EgoDex part1~5+test) + `download_droid.sh`의 `gsutil -m`(thread-limit 8×4 적용 상태에서도) 프로세스가 누적되어 로그인 노드의 사용자별 프로세스/스레드 limit 초과. 로그인 노드는 공용 자원이라 한 사용자가 점유하면 다른 사용자도 영향
- **수정**: `download_all_data.sh`, `download_droid.sh`에 로그인 노드 실행 차단 guard 추가 (hostname `olaf[0-9]+` + `SLURM_JOB_ID` 미설정 조건). sbatch로 compute 노드(`normal_cpu` 등)에서 실행 필수. 외부 인터넷은 compute 노드에서도 접근 가능
- **교훈**: "thread count 제한했으니 안전" 가정 금지. 로그인 노드에서는 어떤 병렬 I/O도 누적되면 위험. 네트워크 다운로드도 반드시 sbatch 잡으로 제출

### gsutil -m 로그인 노드 thread limit (2026-04-11)
- **증상**: DROID 다운로드 중 `RuntimeError: can't start new thread` → EOFError → 다운로드 중단
- **원인**: `gsutil -m`이 기본 수백 개 스레드 spawn. 로그인 노드(공용)의 사용자별 스레드 limit 초과
- **수정**: (1) `gsutil -o "GSUtil:parallel_thread_count=8" -o "GSUtil:parallel_process_count=4"`로 병렬도 제한, (2) 로그인 노드 대신 sbatch로 compute 노드에서 실행
- **교훈**: 로그인 노드에서 공격적 병렬 I/O 도구 사용 금지. sbatch 잡으로 독립 리소스 확보

### Slurm DDP 환경 함정 3가지 (2026-04-10)
- **`--gpus-per-task=1`**: CUDA_VISIBLE_DEVICES 제한 → `set_device(local_rank)` 실패 + NCCL `nvmlDeviceGetHandleByPciBusId` 실패. 해결: `--gres=gpu:N`으로 전환
- **MASTER_PORT 충돌**: 동일 노드에 여러 DDP 잡 → EADDRINUSE. 해결: `MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))`
- **srun PATH 미전파**: conda activate 후에도 `srun python` 실패. 해결: `$CONDA_PREFIX/bin/python` 절대 경로

### V-JEPA LR warmup 부재로 loss 발산 (2026-04-12)
- **증상**: V-JEPA 1차 학습(32867433)에서 loss가 epoch 2 이후 단조 상승 (0.17 → 0.25 → 0.35)
- **원인**: CosineAnnealingLR만 사용, warmup 없음. 학습 시작부터 LR=2e-4(max)로 x-encoder가 급변 → EMA y-encoder(momentum 0.998)가 추적 실패 → predictor target drift → loss 상승
- **수정**: `SequentialLR(LinearLR(warmup 5ep, 1e-6→2e-4) + CosineAnnealingLR(45ep))` 적용. 모든 모델 공통 (Two-Stream에도 무해)
- **교훈**: EMA target 기반 학습(V-JEPA, BYOL, DINO 등)은 LR warmup 필수. Pixel reconstruction(Two-Stream)은 target이 고정이라 warmup 없이도 안정적이지만, moving target에서는 초기 LR이 EMA lag와 맞물려 발산 유발

### V-JEPA mask ratio 과다로 2차도 loss 상승 (2026-04-12)
- **증상**: 2차 학습(32867645, warmup 추가)에서도 epoch 6까지 loss 단조 상승 (0.28 → 0.42). Warmup 종료(epoch 5) 후에도 하강 없음
- **원인**: mask ratio 80%(short) / 85%(long)로 visible 토큰이 30-40개뿐. 원래 V-JEPA는 16프레임의 temporal redundancy 덕에 90% masking이 가능하지만, 2-frame spatial-only 세팅에서는 그 redundancy가 없음 → predictor가 충분한 context 없이 과도한 예측을 요구받아 학습 실패
- **수정**: mask ratio 50%(short) / 60%(long)로 완화. visible 토큰 ~78-98개로 확보. 블록 스케일도 축소 (short 0.10-0.15, long 0.30-0.40)
- **교훈**: V-JEPA의 높은 masking ratio는 temporal redundancy(16프레임)가 전제. 2-frame 적응 시 masking을 대폭 완화해야 함
