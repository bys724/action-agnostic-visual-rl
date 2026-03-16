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

### 1. EgoDex Pre-training (AWS EC2)

3개 모델 사전학습: Two-Stream, Single-Stream, VideoMAE

**학습 목적**: 행동 라벨 없이 비디오만으로 범용 시각 표현 학습
- Two-Stream/Single-Stream: 미래 프레임 예측
- VideoMAE: 마스크된 패치 복원

```bash
# AWS 인스턴스 SSH 접속
ssh ubuntu@<INSTANCE_IP>
cd /workspace/action-agnostic-visual-rl

# 환경 설정 (Deep Learning AMI)
/opt/pytorch/bin/pip install timm transformers tqdm matplotlib tensorboard opencv-python
git submodule update --init --recursive external/VideoMAE

# 학습 실행
bash scripts/pretrain_aws.sh  # 3개 모델 순차 학습
bash scripts/pretrain_aws.sh --model two-stream  # 특정 모델만
bash scripts/pretrain_aws.sh --sanity --no-shutdown  # Sanity test
```

**주의사항**:
- Deep Learning AMI는 드라이버만 제공 → `/opt/pytorch` 가상환경 필수
- System `python3` 사용 금지 (패키지 없음)
- VideoMAE 서브모듈 초기화 필수

자세한 내용은 `docs/AWS_INSTANCE_GUIDE.md` 참고

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

| 파일 | 용도 | 학습 방식 |
|------|------|----------|
| `scripts/pretrain.py` | Pre-training 스크립트 | Self-supervised (비디오만) |
| `scripts/pretrain_aws.sh` | AWS pre-training 자동화 | Self-supervised (비디오만) |
| `scripts/eval/probe_action.py` | Action probing 평가 | 분석 전용 |
| `scripts/eval/finetune_libero.py` | LIBERO fine-tuning | Supervised (비디오+행동) |
| `scripts/data/extract_frames.py` | EgoDex 프레임 추출 | 데이터 전처리 |
| `scripts/data/extract_bridge_frames.py` | Bridge V2 프레임 추출 | 데이터 전처리 |
| `scripts/data/extract_droid_frames.py` | DROID 프레임 추출 (TFRecord→JPG) | 데이터 전처리 |
| `scripts/process_egodex_part.sh` | EgoDex S3 다운로드→추출→업로드 | 데이터 파이프라인 |
| `scripts/process_egodex_all.sh` | EgoDex 전체 part 순차 처리 | 데이터 파이프라인 |
| `scripts/process_droid.sh` | DROID 추출→tar→S3 업로드 | 데이터 파이프라인 |
| `src/models/two_stream.py` | Two-Stream 모델 | 미래 프레임 예측 |
| `src/models/single_stream.py` | Single-Stream 모델 | 미래 프레임 예측 |
| `src/models/videomae.py` | VideoMAE 모델 | 마스크 복원 |
| `src/datasets/egodex.py` | EgoDex 데이터셋 | Pre-training용 |
| `src/training/pretrain.py` | Pre-training 루프 | Self-supervised |
| `src/eval_libero.py` | LIBERO 평가 | Evaluation |

## 개발 원칙

1. **문서 업데이트 우선**: 새 문서보다 핵심 문서 업데이트
2. **간결한 코드**: 실험 우선, 과도한 추상화 피하기
3. **환경 분리**: AWS (pretraining) vs Docker (evaluation)
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

## 현재 Phase (2026-03-16)

**Phase 1 실행 중** — AWS g5.12xlarge에서 사전학습

- EgoDex part1~5 + test S3 업로드 완료 (tarball 변환 진행 중)
- Bridge V2 프레임 S3 업로드 완료 (24,827 traj)
- DROID v1.0.1 다운로드 완료, 프레임 추출 진행 중 (3카메라, 95,658 ep)
- 학습 코드 점검 완료: scheduler resume 버그 수정, TensorBoard 로깅 추가

**다음 단계**: Phase 1 결과 분석 → Go/No-Go → Phase 1.5 또는 Phase 2

자세한 일정은 `docs/RESEARCH_PLAN.md` 참고

## 참고

### 활성 문서
- `docs/RESEARCH_PLAN.md` - 연구 계획 (마스터)
- `docs/AWS_INSTANCE_GUIDE.md` - AWS 환경 설정
- `docs/PROBING_GUIDE.md` - Action probing
- `docs/setup/LIBERO_TEST_GUIDE.md` - LIBERO 평가

### Archive (참고용)
- `docs/archive/` - 과거 코드 리뷰, 개발 상태, 마이그레이션 기록
