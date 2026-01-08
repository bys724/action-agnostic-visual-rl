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

## 현재 환경 구성

```bash
# 평가 환경 실행
docker compose up -d eval
docker exec -it simpler-dev bash

# 환경 변수는 .bashrc에 자동 설정됨
# - GPU 메모리 관리
# - CUDA 설정
# - 유용한 alias (test-simple, test-all)
```

### 통합된 베이스라인
- **SimplePolicy**: 테스트용 더미 정책
- **OpenVLA**: 7B VLA 모델 (src/policies/openvla/)
- **Octo**: (의존성 충돌로 현재 비활성)
- **RT-1**: (체크포인트 필요)

## 주요 파일 위치

- **평가**: `src/eval_simpler.py`
- **데이터 수집**: `src/collect_trajectories.py`
- **테스트**: `src/test_simpler_demo.py`
- **연구 배경**: `docs/research/RESEARCH_CONTEXT.md`
- **OpenVLA 정책**: `src/policies/openvla/`
- **OpenVLA 테스트**: `scripts/test_openvla.sh`

## 개발 원칙

1. **실험 우선**: 완벽한 코드보다 빠른 실험과 결과
2. **유연한 구조**: 필요에 따라 리팩토링, 미리 과도한 설계 피하기
3. **재현 가능성**: Docker로 환경 통일, 시드 고정
4. **점진적 개선**: 작은 실험부터 시작해 점차 확장

## 다음 작업 제안

현재 상황에 따른 우선순위:
1. 베이스라인 모델들의 성능 측정 완료
2. DROID 데이터셋으로 사전학습 파이프라인 구축
3. 행동-독립적 표현 학습 구현
4. SimplerEnv에서 fine-tuning 및 평가
5. 결과 분석 및 ablation study

## 중요 참고사항

- SimplerEnv는 ManiSkill3 기반 (ManiSkill2 아님)
- NumPy 버전은 1.x 유지 (OpenCV 호환성)
- JAX/TensorFlow 메모리 사전할당 비활성화 필요
- Octo 모델은 추가 의존성 필요 (dlimp, distrax 등)