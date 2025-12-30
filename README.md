# Action-Agnostic Visual Representation for Robotic Manipulation

## 핵심 아이디어
행동 정보 없이 학습한 시각적 표현이 다양한 로봇과 작업에서 더 잘 일반화된다는 가설 검증

## 실험 환경
- **벤치마크**: SIMPLER (시뮬레이션-실제 전환 평가)
- **시뮬레이터**: SAPIEN (레이 트레이싱 렌더링)
- **GPU 요구사항**: NVIDIA RTX (RT 코어 필요)

## 프로젝트 구조
```
action-agnostic-visual-rl/
├── docker/           # Docker 개발 환경
├── docs/             # 문서
├── src/              
│   └── envs/         # SIMPLER 환경 래퍼
└── third_party/      # 외부 의존성
    └── SimplerEnv/   # SIMPLER 환경
```

## 빠른 시작 (Docker 권장)

### Docker 환경 사용 (권장)
```bash
# 1. Docker 이미지 빌드
./docker/build.sh

# 2. 개발 환경 실행
./docker/run.sh

# 3. 컨테이너 내부에서 테스트
python docker/test_env.py
```

자세한 내용은 [docs/DOCKER_SETUP.md](docs/DOCKER_SETUP.md) 참조

## 개발 현황
- ✅ Docker 개발 환경 구성 완료
- ✅ SIMPLER 환경 통합 완료
- 🔄 시각 인코더 및 정책 네트워크 구현 예정
- 🔄 학습 알고리즘 구현 예정