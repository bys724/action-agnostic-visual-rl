# OpenVLA 통합 테스트 가이드

## 테스트 완료 항목

### ✅ 구문 및 통합 검사
```bash
python3 scripts/test_openvla_syntax.py
```
- 모든 Python 파일 구문 검사 통과
- OpenVLA import 문 확인
- 필수 메서드 존재 확인 (reset, step, get_action 등)
- eval_simpler.py와 collect_trajectories.py 통합 확인

### ✅ 코드 수준 테스트
```bash
python3 scripts/test_openvla_local.py
```
- OpenVLAPolicy import 성공
- Mock policy 인터페이스 테스트 통과
- (ManiSkill 의존성 때문에 전체 통합 테스트는 Docker 내에서 수행 필요)

## Docker 내에서 실행할 테스트

### 1. Docker 컨테이너 시작
```bash
# 이미지 빌드
docker build -t simpler-env:latest .

# 컨테이너 실행
docker compose up -d eval

# 컨테이너 접속
docker exec -it simpler-dev bash
```

### 2. SimplePolicy 테스트 (기본 동작 확인)
```bash
# 평가 테스트
python src/eval_simpler.py --model simple --n-episodes 2 --max-steps 100

# Trajectory 수집 테스트  
python src/collect_trajectories.py --model simple --n-per-task 2 --max-steps 100
```

### 3. OpenVLA 테스트 (GPU 필요)
```bash
# OpenVLA 평가 (실제 모델 로드 시 GPU와 메모리 필요)
python src/eval_simpler.py --model "openvla/openvla-7b" --n-episodes 2

# OpenVLA trajectory 수집
python src/collect_trajectories.py --model "openvla/openvla-7b" --n-per-task 2
```

## 테스트 스크립트
- `scripts/test_openvla.sh`: 전체 통합 테스트 (Docker 내)
- `scripts/test_openvla_syntax.py`: 구문 검사 (로컬)
- `scripts/test_openvla_local.py`: 부분 테스트 (로컬)
- `scripts/test_baseline.sh`: 베이스라인 테스트

## 예상 결과

### SimplePolicy
- 성공률: ~0% (랜덤 동작)
- 정상 동작 확인용

### OpenVLA (GPU 사용 시)
- Zero-shot 성공률: 20-40%
- 추론 속도: 2-5 Hz
- 메모리 요구: 24GB+ VRAM

## 문제 해결

### ImportError: No module named 'mani_skill'
→ Docker 컨테이너 내에서 실행 필요

### CUDA Out of Memory
→ --n-episodes 줄이기 또는 더 작은 모델 사용

### Model loading timeout
→ 첫 실행 시 모델 다운로드 시간 필요 (수 GB)