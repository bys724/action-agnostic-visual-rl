# Claude Development Guide

**세션 시작 시 필독**: 아래 핵심 문서를 먼저 읽고 작업 맥락을 파악하세요.

## 핵심 문서 (업데이트 우선)

1. **`docs/RESEARCH_PLAN.md`** - 전체 연구 계획 및 현재 phase (마스터 문서)
2. **`docs/AWS_INSTANCE_GUIDE.md`** - AWS 학습 환경 설정 및 troubleshooting
3. **`docs/PROBING_GUIDE.md`** - Action probing 실험 계획

**문서 작성 원칙**:
- 새 문서를 만들기보다 기존 핵심 문서 **업데이트** 우선
- 핵심 계획은 `RESEARCH_PLAN.md`에 집중 관리
- 일회성 정보는 git commit message에 기록

## 프로젝트 개요

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?

**핵심 아이디어**:
- EgoDex로 행동-독립적(action-agnostic) 시각 표현 사전학습
- LIBERO에서 로봇 조작 태스크로 평가

## 워크플로우

### 1. EgoDex Pre-training

2개 모델 사전학습: Two-Stream, VideoMAE (Single-Stream 제외)

**학습 목적**: 행동 라벨 없이 비디오만으로 범용 시각 표현 학습
- Two-Stream: 미래 프레임 예측
- VideoMAE: 마스크된 패치 복원

```bash
# H100 워크스테이션 (권장) — Docker dev 환경
docker compose up -d dev
docker exec -it dev-env bash
bash scripts/pretrain_local.sh                     # two-stream → videomae 순차
bash scripts/pretrain_local.sh --model two-stream   # 특정 모델만
bash scripts/pretrain_local.sh --sanity             # Sanity test

# AWS EC2 (대안)
bash scripts/pretrain_aws.sh --model two-stream
```

**데이터 다운로드** (EgoDex CDN 직접):
```bash
bash scripts/download_egodex.sh part2 part3 part5
```

### 2. Action Probing (사전학습 완료 후)

학습된 표현이 행동 정보를 인코딩하는지 검증

```bash
# DROID 데이터셋으로 linear probe (primary)
python scripts/eval/probe_action.py \
    --checkpoint data/checkpoints/two_stream/latest.pt \
    --dataset droid

# Bridge V2 (secondary, cross-embodiment 보강용)
python scripts/eval/probe_action_bridge.py \
    --checkpoint data/checkpoints/two_stream/latest.pt
```

자세한 내용은 `docs/PROBING_GUIDE.md` 참고

### 3. LIBERO Fine-tuning & Evaluation

학습된 인코더로 로봇 조작 태스크 수행

```bash
# Docker 환경에서 실행
docker compose up -d libero openvla-libero
docker exec libero-eval python src/eval_libero.py \
    --model openvla \
    --encoder two-stream \
    --checkpoint data/checkpoints/two_stream/latest.pt
```

자세한 내용은 `docs/setup/LIBERO_TEST_GUIDE.md` 참고

## 주요 파일

| 파일 | 용도 | 비고 |
|------|------|------|
| `scripts/pretrain.py` | Pre-training 메인 스크립트 | Self-supervised |
| `scripts/pretrain_local.sh` | H100 로컬 학습 런처 | 권장 |
| `scripts/pretrain_aws.sh` | AWS EC2 학습 런처 | 대안 |
| `scripts/download_egodex.sh` | EgoDex CDN 다운로드+추출 | |
| `scripts/data/extract_frames.py` | EgoDex 프레임 추출 | |
| `scripts/data/extract_bridge_frames.py` | Bridge V2 프레임 추출 | |
| `scripts/data/extract_droid_frames.py` | DROID 프레임 추출 (TFRecord→JPG) | |
| `scripts/eval/probe_action.py` | Action probing 평가 | |
| `scripts/eval/finetune_libero.py` | LIBERO fine-tuning | |
| `src/models/two_stream.py` | Two-Stream 모델 | 미래 프레임 예측 |
| `src/models/videomae.py` | VideoMAE 모델 | 마스크 복원 |
| `src/models/single_stream.py` | Single-Stream 모델 | 현재 미사용 |
| `src/datasets/egodex.py` | EgoDex 데이터셋 | |
| `src/datasets/droid.py` | DROID 데이터셋 | |
| `src/datasets/bridge.py` | Bridge V2 데이터셋 | |
| `src/training/pretrain.py` | Pre-training 루프 | |
| `src/eval_libero.py` | LIBERO 평가 | |

## 개발 원칙

1. **문서 업데이트 우선**: 새 문서보다 핵심 문서 업데이트
2. **간결한 코드**: 실험 우선, 과도한 추상화 피하기
3. **환경 분리**: H100 Docker (pretraining) vs Docker (evaluation), AWS는 대안
4. **검증 필수**: 새 스크립트는 sanity test로 검증 후 배포
5. **Best Practice 참고 필수**:
   - 새로운 구현 전에 **반드시** 공식 문서/권장 방법/샘플 코드 조사
   - 특히 데이터 로딩, 학습 파이프라인 등 성능에 영향을 주는 부분
   - 검색 키워드 예시: "pytorch video dataset best practices", "efficient video loading"
   - 비효율적 구현으로 재작업하지 않도록 사전 조사 우선

## 데이터셋 전처리 워크플로우

새 데이터셋을 학습에 사용하기 전, 아래 프로세스를 따름:

### 1. 샘플 테스트
- 소수 영상(3~5개)으로 crop/resize 옵션별 결과 비교
- 다양한 task에서 샘플링하여 대표성 확보

### 2. 결과 기록
- 비교 이미지 + 결정 근거를 `docs/preprocessing/` 하위에 기록
- 예: `docs/preprocessing/egodex.md`, `docs/preprocessing/bridge_v2.md`

### 3. 전체 추출
- 결정된 설정으로 전체 데이터셋 프레임 추출
- 추출 스크립트: `scripts/data/extract_frames.py` (또는 데이터셋별 스크립트)

### 4. 업로드 및 검증
- S3 업로드 후 샘플 다운로드하여 품질 확인

### 기존 사례
- **EgoDex**: 센터크롭(1080x1080) → 256x256 (`docs/preprocessing/egodex/`)
- **Bridge V2**: 리사이즈(480x640 → 256x256, crop 없음) (`docs/preprocessing/bridge_v2/`)
- **DROID**: 리사이즈(180x320 → 256x256, crop 없음) (`docs/preprocessing/droid/`)

## 현재 Phase (2026-03-24)

**Phase 1 실행 중** — Two-Stream v3 + VideoMAE 병렬 학습 중

- **Two-Stream 아키텍처 변천**:
  - v1: CLS-only decoder → 평균 이미지만 출력 (실패, `docs/architecture/two_stream_v1_cls_only_result.png`)
  - v2: per-stream PatchDecoder → 구조 복원 확인 (`docs/architecture/two_stream_v2_epoch1.png`)
  - v3 (현재): M channel 재설계 (ΔL + Sobel(ΔL) = 3ch, RGB diff 제거)
- Two-Stream v3 epoch 1부터 학습 시작 (GPU 0, ~15h/epoch)
- VideoMAE epoch 8 완료, epoch 9 진행 중 (GPU 1, ~3.5h/epoch)
- 아키텍처 다이어그램: `docs/architecture/two_stream_v2.png`

**다음 단계**: epoch별 시각화 결과 확인 → action probing → Phase 1 Go/No-Go

자세한 일정은 `docs/RESEARCH_PLAN.md` 참고

## 참고

### 활성 문서
- `docs/RESEARCH_PLAN.md` - 연구 계획 (마스터)
- `docs/AWS_INSTANCE_GUIDE.md` - AWS 환경 설정
- `docs/PROBING_GUIDE.md` - Action probing
- `docs/setup/LIBERO_TEST_GUIDE.md` - LIBERO 평가

### Archive (참고용)
- `docs/archive/` - 과거 코드 리뷰, 개발 상태, 마이그레이션 기록

## 트러블슈팅 로그

### EgoDex test symlink 문제 (2026-03-23)
- **증상**: eval 시 `FileNotFoundError: .../part2/.../frame_000000.jpg`로 학습 크래시
- **원인**: `pretrain_local.sh`가 test symlink을 `data/egodex/test/` (원본 hdf5/mp4)로 생성.
  추출된 프레임은 `/mnt/data/egodex_frames/test_frames/`에 별도 존재.
  학습 시작 시점에 잘못된 symlink으로 eval dataset 로드 → 프레임 없는 디렉토리 참조
- **수정**: symlink을 `test_frames`로 교체, `pretrain_local.sh` 수정
- **교훈**: 학습 시작 전 eval dataset 경로 확인 필수.
  dataset 객체는 시작 시 경로를 캐싱하므로, 학습 중 symlink 수정해도 효과 없음 → 재시작 필요

### 학습 프로세스 안정성 (2026-03-23)
- `evaluate()`와 `save_epoch_samples()`를 try/except로 보호
- eval/시각화 실패 시 WARNING 출력 후 학습 계속 (크래시 방지)
- `model.train()` 복구를 finally 블록으로 보장
