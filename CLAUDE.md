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

2개 모델 사전학습: **Two-Stream v10** (현재 활성), **VideoMAE** (baseline)

**v10 확정 설정** (v6 base + mask_p 0.75):
```bash
# 클러스터 (sbatch, 권장 — 8 GPU DDP)
MODEL=two-stream-v10 sbatch scripts/cluster/pretrain.sbatch

# 로컬 (Docker dev-env)
docker exec -it dev-env bash
python scripts/pretrain.py --model two-stream \
    --depth 12 --num-stages 2 --use-ape --rotation-aug \
    --mask-ratio 0.5 --mask-ratio-p 0.75 \
    --v9-p-target future --v9-loss-weight-p 1.0 \
    --max-gap 60 --sample-dist triangular --sample-center 30 \
    --egodex-splits part1,part2,part3,part4,part5 \
    --epochs 50 --batch-size 64
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
| `scripts/eval/probe_action.py` | EgoDex action probing (v6/v10) |
| `scripts/eval/probe_action_droid.py` | DROID cross-domain probing (v6/v10) |
| `scripts/eval/finetune_libero.py` | LIBERO fine-tuning (v6/v10) |
| `scripts/eval/probe_action_v11.py` | EgoDex action probing (v11) |
| `scripts/eval/probe_action_droid_v11.py` | DROID cross-domain probing (v11) |
| `scripts/eval/finetune_libero_v11.py` | LIBERO BC fine-tune (v11) |
| `scripts/eval/visualize_attn_v11.py` | v11 attention 시각화 |
| `scripts/eval/visualize_inference.py` | 단일/다중 모델 비교 시각화 |
| `scripts/eval/visualize_sample_detail.py` | 단일 샘플 상세 (M/P channel + attention) |
| `src/models/two_stream.py` | Two-Stream 모델 (v6/v10) + 2D RoPE + MAE masking |
| `src/models/two_stream_v11.py` | Two-Stream v11 (motion-routing + dual-target) |
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
- **DROID** (180x320): 리사이즈 → 256x256 (crop 없음)
- **Ego4D** (가변): 다운로드 진행 중, 전처리 결정 미정

## 현재 Phase (2026-04-26)

**Phase 1.5 — v10 saturate (R²=0.221), v11 ep12 도달 (+0.219), DROID/LIBERO cross-domain 검증 중**

**완료 (Two-Stream lineup)**:
- **v4** (RoPE, mask 0.3/0.5): 48ep. Probing R²=0.197
- **v6** (APE + mask 0.5/0.5 + rotation aug): ep23 scancel, ep8 peak R²=0.259 (현 챔피언)
- **VideoMAE-ours**: 50ep, R²=0.326
- **V-JEPA-ours**: 3차 발산, negative result
- **폐기 라인 (v7-big / v8 / v9)**: 모두 P stream collapse. 상세는 `docs/RESEARCH_PLAN.md`

**Two-Stream v10** (v6 base + mask_p 0.75) — **종료, ep40 plateau (+0.221)**:
- 50ep 거의 완주 (JobID 33570871, AIP_long 2노드×4 H100)
- ep4-8: 1차 peak +0.206 (ep8) → ep12-20: 점진 collapse (+0.137 ep20) → ep24-40: W-shape 회복 → ep40 peak **+0.221** (`patch_mean_concat`) → ep44/ep48 plateau (+0.221, +0.222)
- **v6 챔피언 (+0.259) 추월 실패 확정**. P-stream 내부 강화 방식의 한계로 결론

**진행 중 — Two-Stream v11** (Motion-Guided Attention Routing + Dual-Target):
- **구조**: M encoder (6-layer) + P encoder (12-layer) + M decoder (3-layer, loss 없음) + P decoder 3-phase (interpreter_1 → motion-routing × 2 → interpreter_2) + shared recon_head
- **Loss**: L_t (Phase 1) + L_tk (Phase 3), masked positions only
- **Total params**: 250.9M (decoder까지 downstream 사용 시 ~204M)
- **시작**: 2026-04-25 01:10:50 KST (JobID 33594155, AIP_long 2노드×4 H100, 50ep, `--time=3d`)
- Sanity (33591381, 10:26): forward/backward OK, full scale에서 healthy

**v11 학습 추이** (loss + 표현 진단):

| Epoch | L_total | L_t | L_tk | std_p | cos_intra_p |
|-------|---------|-----|------|-------|-------------|
| 1 | 0.0196 | 0.0109 | 0.0088 | 0.349 | 0.866 |
| 4 | 0.0057 | 0.0044 | 0.0014 | 0.210 | 0.897 |
| 8 | 0.0043 | 0.0038 | 0.00052 | 0.009 | 1.000 |
| 12 | 0.0024 | 0.00197 | 0.00039 | 0.004 | 1.000 |

→ Loss 단조 감소. P encoder CLS는 collapse (cos_intra_p≈1.0) 그러나 patches는 healthy — 75% MAE 복구가 작동

**v11 ep12 — Representation 비교 (12 mode)**

4 위치: A (M encoder), B (P encoder), D' (motion-routing 후), D (Phase 3 final)

| Mode | ep4 | ep8 | **ep12** |
|------|-----|-----|----------|
| `patch_mean_m_enc` (A) | +0.170 | +0.176 | **+0.208** |
| `patch_mean_p_enc` (B) | -0.041 | -0.025 | 0.000 |
| `patch_mean_p_state_after_routing` (D') | +0.121 | +0.066 | +0.072 |
| `patch_mean_p_features_tk` (D) | +0.023 | +0.055 | +0.054 |
| `patch_mean_concat_enc_only` (A+B) | +0.160 | +0.168 | +0.200 |
| `patch_mean_concat_enc_phase3` (A+D) | +0.143 | +0.194 | **+0.219** ★ |
| `patch_mean_concat_enc_d_prime` (A+D') | +0.149 | +0.166 | +0.153 |
| `patch_mean_concat_p_enc_d_prime` (B+D') | +0.135 | +0.011 | +0.076 |
| `patch_mean_concat_all` (A+B+D') | +0.114 | +0.094 | +0.178 |
| `cls_m_enc` (A CLS) | +0.066 | +0.155 | +0.162 |
| `cls_p_enc` (B CLS) | -0.059 | -0.011 | -0.008 |
| `cls_concat_enc` (A+B CLS) | -0.048 | +0.092 | +0.148 |

**핵심 결론**:
- **ep12 A+D = +0.219** ≈ v10 ep40 plateau (+0.221). v11이 12 epoch만에 v10 50 epoch 도달
- **사용자 통찰 검증**: interpreter는 decoder의 reconstruction wrapper (D' < D 역전 ep8에)
- M encoder 단독이 강력 (+0.208) — task가 motion-biased (hand pose ≈ motion)
- P encoder 단독 약함, 단 motion routing 거치면 살아남
- Loss와 R² 정직 상관 (L_total 0.0057 → 0.0024 절반 → A+D R² +0.143 → +0.219)

**v11 Cross-domain DROID probing** (사용자 직감 검증)

| Gap (DROID 15Hz) | VideoMAE | v11 best | 격차 |
|------------------|----------|----------|------|
| 1 (0.07초) | -0.006 | -0.005 | +0.001 |
| 10 (0.67초) | -0.006 | +0.006 (A+B) | +0.012 |
| **15 (1초)** ★ | **-0.035** | **+0.005 (A+B)** | **+0.040** |
| 30 (2초) | -0.028 | -0.010 | +0.018 |

- 모든 gap에서 v11이 VideoMAE보다 일관 우위
- gap=15 (EgoDex 학습 분포 1초와 일치)에서 격차 가장 큼 (+0.040)
- VideoMAE는 in-domain (EgoDex +0.326) 강력하지만 cross-domain 음수
- 절대 R² 작음 (~0.005) — DROID action probing 자체 한계
- **방향성 검증**: v11이 cross-domain 일반화 우수

**Cross-domain LIBERO** (진행 중):
- BC fine-tune 도구 신규 작성 (`scripts/eval/finetune_libero_v11.py`)
- libero_spatial 30 epoch 시작 (33600616 VideoMAE, 33600617 v11 ep12 A+D)
- 4-6h 후 결과 (val MSE 비교)

**다음 작업**:
1. v11 ep16/ep20 probing 계속 (수렴 추세 확인)
2. LIBERO BC val MSE 결과 확인 → rollout 수행 여부 결정
3. DROID 추가 gap (5/20) 보강
4. Phase 3 LIBERO 본 평가 준비

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
