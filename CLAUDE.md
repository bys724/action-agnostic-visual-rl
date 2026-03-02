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

### 1. EgoDex Pretraining (AWS EC2)

3개 모델 사전학습: Two-Stream, Single-Stream, VideoMAE

```bash
# AWS 인스턴스 SSH 접속
ssh ubuntu@<INSTANCE_IP>
cd /workspace/action-agnostic-visual-rl

# 환경 설정 (Deep Learning AMI)
/opt/pytorch/bin/pip install timm transformers tqdm matplotlib tensorboard opencv-python
git submodule update --init --recursive external/VideoMAE

# 학습 실행
./scripts/run_aws_training.sh  # 3개 모델 순차 학습
./scripts/run_aws_training.sh --model two-stream  # 특정 모델만
./scripts/run_aws_training.sh --sanity --no-shutdown  # Sanity test
```

**주의사항**:
- Deep Learning AMI는 드라이버만 제공 → `/opt/pytorch` 가상환경 필수
- System `python3` 사용 금지 (패키지 없음)
- VideoMAE 서브모듈 초기화 필수

자세한 내용은 `docs/AWS_INSTANCE_GUIDE.md` 참고

### 2. Action Probing (사전학습 완료 후)

학습된 표현이 행동 정보를 인코딩하는지 검증

```bash
# Bridge V2 데이터셋으로 linear probe
python scripts/action_probing.py \
    --checkpoint data/checkpoints/two_stream/latest.pt \
    --dataset bridge_v2
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

| 파일 | 용도 | 환경 |
|------|------|------|
| `scripts/train_long.py` | EgoDex 사전학습 | AWS |
| `scripts/run_aws_training.sh` | AWS 학습 자동화 | AWS |
| `scripts/action_probing.py` | Action probing 평가 | 로컬/AWS |
| `src/models/two_stream.py` | Two-Stream 모델 | 공통 |
| `src/models/baselines.py` | Single-Stream 모델 | 공통 |
| `src/models/videomae_wrapper.py` | VideoMAE wrapper | 공통 |
| `src/models/openvla_encoder.py` | OpenVLA 인코더 wrapper | 공통 |
| `src/eval_libero.py` | LIBERO 평가 | Docker |

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

## 현재 Phase (2026-03-01)

**Phase 1: EgoDex Part1 Pretraining** (진행 중)

- Instance: g5.12xlarge (35.86.118.174)
- Dataset: EgoDex part1 (336GB)
- Status: Two-Stream epoch 2/30 진행 중

**다음 단계**: Phase 2 Action Probing

자세한 일정은 `docs/RESEARCH_PLAN.md` 참고

## 참고

### 활성 문서
- `docs/RESEARCH_PLAN.md` - 연구 계획 (마스터)
- `docs/AWS_INSTANCE_GUIDE.md` - AWS 환경 설정
- `docs/PROBING_GUIDE.md` - Action probing
- `docs/setup/LIBERO_TEST_GUIDE.md` - LIBERO 평가

### Legacy (참고용)
- `docs/legacy/SIMPLER_GUIDE.md` - SIMPLER 환경 (deprecated)
- `docs/development/TODO_MODEL_INTEGRATION.md` - 모델 통합 TODO (archive)
