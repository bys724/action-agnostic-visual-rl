# Development Guide for Claude

## 프로젝트 핵심

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?

이 프로젝트는 로봇 조작에서 행동-독립적(action-agnostic) 시각 표현 학습을 탐구합니다.

## 주요 실험 방향

### 1. 베이스라인 비교
- 기존 모델들(RT-1, Octo, OpenVLA)의 SimplerEnv 성능 측정
- Zero-shot vs Few-shot 학습 효율성 비교

### 2. 표현 학습 방법론
- **시간적 일관성**: 연속 프레임 간 contrastive learning
- **공간적 이해**: Masked autoencoding으로 장면 구조 학습
- **다중 시점**: 여러 카메라 뷰 간 일관성 학습

### 3. 전이 학습 실험
- 다른 로봇 형태로 전이 (WidowX → Franka)
- 새로운 작업으로 일반화 (pick → push, stack)
- 시뮬레이션 → 실제 로봇 전이

## 🚨 Docker 환경 사용 필수

**중요**: 이 프로젝트의 모든 실행 코드는 Docker 컨테이너 내에서 실행되어야 합니다!

### 올바른 실행 방법
```bash
# ✅ 올바른 방법: 컨테이너 내에서 실행
docker exec simpler-dev python src/eval_simpler.py --model openvla/openvla-7b

# ✅ 또는 컨테이너에 진입 후 실행
docker exec -it simpler-dev bash
cd /workspace && python src/eval_simpler.py --model openvla/openvla-7b
```

### 자주하는 실수
```bash
# ❌ 잘못된 방법: 호스트에서 직접 실행
python src/eval_simpler.py  # 이렇게 하면 안됨!

# ❌ 호스트에서 ManiSkill assets 설치 시도
python scripts/setup_openvla.py  # assets는 컨테이너에서만 설치됨
```

### 환경별 실행 위치
- **호스트에서 실행**: 
  - Docker 명령 (`docker compose`, `docker exec`)
  - 모델 다운로드 (`python3 scripts/setup_openvla.py`)
  
- **컨테이너에서 실행**:
  - 모든 평가 코드 (`src/eval_simpler.py`)
  - ManiSkill assets 설치 (`python scripts/setup_openvla.py --skip-deps`)
  - 테스트 코드 (`src/test_simpler_demo.py`)
  - 데이터 수집 (`src/collect_trajectories.py`)

### 컨테이너 환경 확인
```bash
# 컨테이너 상태 확인
docker ps | grep simpler-dev

# 컨테이너 시작 (필요시)
docker compose up -d eval

# 컨테이너 접속
docker exec -it simpler-dev bash
```

### 비교 모델 라인업
- **OpenVLA**: 7B VLA 모델 (구현 완료)
- **LAPA**: Action-free 사전학습 모델 (예정)
- **Custom**: 개발 중인 모델 (예정)

## 주요 파일 위치

- **평가**: `src/eval_simpler.py` (다중 모델 비교 지원)
- **데이터 수집**: `src/collect_trajectories.py`
- **테스트**: `src/test_simpler_demo.py`
- **연구 배경**: `docs/research/RESEARCH_CONTEXT.md`
- **정책 구현**: `src/policies/`
- **평가 설정**: `configs/eval_example.json`

## 개발 원칙

1. **Docker 우선**: 모든 실행 코드는 Docker 컨테이너 내에서 테스트
2. **실험 우선**: 완벽한 코드보다 빠른 실험과 결과
3. **유연한 구조**: 필요에 따라 리팩토링, 미리 과도한 설계 피하기
4. **재현 가능성**: Docker로 환경 통일, 시드 고정
5. **점진적 개선**: 작은 실험부터 시작해 점차 확장

## 테스트 체크리스트

새로운 기능을 추가하거나 테스트할 때:
- [ ] Docker 컨테이너가 실행 중인가? (`docker ps`)
- [ ] 실행 코드는 `docker exec simpler-dev` 명령을 사용하는가?
- [ ] ManiSkill assets가 필요한 경우 컨테이너에서 설치했는가?
- [ ] 모델은 호스트에서 다운로드했는가?

## 다음 작업 제안

현재 상황에 따른 우선순위:
1. 베이스라인 모델들의 성능 측정 완료
2. DROID 데이터셋으로 사전학습 파이프라인 구축
3. 행동-독립적 표현 학습 구현
4. SimplerEnv에서 fine-tuning 및 평가
5. 결과 분석 및 ablation study

## 중요 참고사항

- **Docker 필수**: 모든 실행 코드는 컨테이너에서 (`docker exec simpler-dev`)
- SimplerEnv는 ManiSkill3 기반
- NumPy 버전은 1.x 유지 (OpenCV 호환성)
- GPU 메모리 관리: XLA_PYTHON_CLIENT_PREALLOCATE=false
- 모델 비교 시 동일한 seed 사용 권장

## 빠른 실행 가이드

```bash
# 1. 항상 Docker 컨테이너 확인
docker ps | grep simpler-dev

# 2. 컨테이너 없으면 시작
docker compose up -d eval

# 3. 실행은 항상 docker exec으로
docker exec simpler-dev python src/eval_simpler.py --model openvla/openvla-7b

# 4. 디버깅이 필요하면 컨테이너 진입
docker exec -it simpler-dev bash
```