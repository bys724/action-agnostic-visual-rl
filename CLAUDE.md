# Claude Development Guide

## 프로젝트 개요

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?

행동-독립적(action-agnostic) 시각 표현 학습을 통해 로봇 조작 성능 향상을 탐구합니다.

## Docker 환경 필수

모든 실행 코드는 Docker 컨테이너에서 실행합니다.

```bash
# 올바른 실행 방법
docker exec simpler-dev python src/eval_simpler.py --model openvla --api-url http://localhost:8001

# 잘못된 방법 (호스트에서 직접 실행)
python src/eval_simpler.py  # 이렇게 하면 안됨
```

### 빠른 시작 (SIMPLER)
```bash
docker compose up -d eval openvla  # 컨테이너 시작
docker exec simpler-dev python src/test_simpler_demo.py  # 환경 테스트
docker exec simpler-dev python src/eval_simpler.py --model openvla --api-url http://localhost:8001
```

### 빠른 시작 (LIBERO)
```bash
docker compose up -d libero openvla-libero  # 컨테이너 시작
docker exec libero-eval python src/eval_libero.py --model openvla --host localhost --port 18010
python3 src/compare_results.py --results-dir data/libero/results  # 결과 비교
```

### Two-Stream 모델 학습
```bash
# GPU 1 사용 (GPU 0이 다른 작업에 사용 중일 경우)
docker exec -e CUDA_VISIBLE_DEVICES=1 simpler-eval python /workspace/scripts/train_long.py \
    --epochs 500 --batch-size 32 --checkpoint-dir /workspace/data/checkpoints/two_stream

# 체크포인트에서 재개
docker exec -e CUDA_VISIBLE_DEVICES=1 simpler-eval python /workspace/scripts/train_long.py \
    --resume /workspace/data/checkpoints/two_stream/<timestamp>/latest.pt
```

## 주요 파일

| 파일 | 용도 |
|------|------|
| `src/models/two_stream.py` | Two-Stream 모델 구현 |
| `scripts/train_long.py` | Two-Stream 장기 학습 스크립트 |
| `src/eval_simpler.py` | SIMPLER 모델 평가 |
| `src/eval_libero.py` | LIBERO 벤치마크 평가 (OpenVLA, Pi0) |
| `src/compare_results.py` | LIBERO 결과 비교 |
| `src/policies/` | 정책 구현 (openvla, lapa) |
| `docker/` | 모델 API 서버 |
| `configs/` | 평가 설정 |

## 개발 원칙

1. **Docker 우선**: 모든 코드는 컨테이너에서 실행
2. **API 분리**: 모델별 독립 컨테이너로 의존성 충돌 방지
3. **간결한 코드**: 실험 우선, 과도한 추상화 피하기
4. **간결한 문서**: 핵심 정보만 유지, 중복 제거

## 참고

- 프로젝트 구조: `docs/PROJECT_ORGANIZATION.md`
- SIMPLER 테스트: `docs/setup/TEST_GUIDE.md`
- LIBERO 테스트: `docs/setup/LIBERO_TEST_GUIDE.md`
- 모델 통합 현황: `docs/development/TODO_MODEL_INTEGRATION.md`
- 연구 문서: `references/` 폴더
