# Action-Agnostic Visual Representation Learning

## 연구 목표

**핵심 가설**: 행동 정보 없이 학습한 시각적 표현이 다양한 로봇과 작업에 더 잘 일반화된다

### 검증하고자 하는 것
1. 행동 조건부 학습 vs 행동 독립적 학습의 성능 차이
2. 다양한 로봇 형태(morphology)에 대한 전이 능력
3. 새로운 작업에 대한 적응 속도

## 실험 계획

### Phase 1: 베이스라인 구축
- SimplerEnv 벤치마크에서 기존 모델들의 성능 측정
- RT-1, Octo, OpenVLA 등 최신 모델 평가
- Zero-shot 및 few-shot 성능 기록

### Phase 2: 행동 독립적 표현 학습
- 대규모 비전 데이터셋으로 사전학습 (DROID, Open-X)
- Self-supervised learning 방법론 적용
  - Contrastive learning (시간적 일관성)
  - Masked autoencoding (공간적 이해)
  - Cross-view prediction (다중 시점 일반화)

### Phase 3: 정책 학습 및 전이
- 학습된 표현 위에 경량 정책 헤드 추가
- SimplerEnv 작업에서 fine-tuning
- 다른 로봇/작업으로 전이 실험

### Phase 4: 분석 및 검증
- 표현 공간 분석 (t-SNE, attention maps)
- Ablation studies
- 실제 로봇 검증 (가능한 경우)

## 평가 지표

### 주요 메트릭
- **Success Rate**: 작업 완료율
- **Sample Efficiency**: 필요한 학습 데이터 양
- **Transfer Performance**: 새로운 작업/로봇에서의 성능
- **Representation Quality**: 선형 평가, 최근접 이웃 검색

### 벤치마크
- SimplerEnv: 4개 조작 작업
- 추가 검증 환경 (계획 중)

## 기술 스택

- **환경**: SimplerEnv (ManiSkill3 기반)
- **비전 모델**: DINOv2, CLIP, MAE
- **정책 학습**: Behavior Cloning, RLPD
- **인프라**: Docker, PyTorch, JAX

## 프로젝트 구조

```
.
├── src/              # 핵심 구현
├── scripts/          # 실험 스크립트
├── data/            # 데이터 및 결과
├── docs/            # 상세 문서
└── third_party/     # 외부 의존성
```

## 시작하기

### 빠른 설치 (새 워크스테이션)
```bash
# 1. Docker 환경 구축
docker compose build eval
docker compose up -d eval

# 2. OpenVLA 설치 (모델 + 의존성)
python3 scripts/setup_openvla.py

# 3. 컨테이너에서 assets 설치
docker exec simpler-dev python scripts/setup_openvla.py --skip-deps

# 4. 테스트 실행
docker exec simpler-dev python src/eval_simpler.py --model openvla/openvla-7b --n-episodes 1
```

### 개발 환경
```bash
# 컨테이너 접속
docker exec -it simpler-dev bash

# 환경 테스트
python src/test_simpler_demo.py
```

## 관련 문서

- [`docs/setup/TEST_GUIDE.md`](docs/setup/TEST_GUIDE.md) - 테스트 및 평가 가이드
- [`docs/research/RESEARCH_CONTEXT.md`](docs/research/RESEARCH_CONTEXT.md) - 연구 배경
- [`CLAUDE.md`](CLAUDE.md) - AI 어시스턴트용 개발 가이드

## 현재 진행 상황

- [x] SimplerEnv 환경 구축
- [x] 베이스라인 모델 통합 (SimplePolicy, OpenVLA, LAPA)
- [x] 오프라인 체크포인트 관리 시스템
- [x] LIBERO 벤치마크 환경 구축
- [x] Pi0 통합 및 평가 파이프라인
- [x] OpenVLA vs Pi0 비교 평가 (libero_spatial: Pi0 100%, OpenVLA 40%)
- [ ] 대규모 데이터셋 준비
- [ ] 행동 독립적 표현 학습 구현
- [ ] 전이 학습 실험
- [ ] 결과 분석 및 논문 작성

---

> 이 프로젝트는 로봇 조작에서 범용적인 시각 표현을 학습하는 새로운 접근 방식을 탐구합니다.