# LIBERO 벤치마크 테스트 가이드

## 사전 요구사항

- Docker & NVIDIA Container Toolkit
- GPU (CUDA 지원)
- 충분한 디스크 공간 (~50GB, 모델 체크포인트 포함)

## 1. Docker 이미지 빌드

```bash
cd /path/to/action-agnostic-visual-rl

# LIBERO 평가 환경 빌드
docker compose build libero

# OpenVLA-LIBERO 서버 빌드
docker compose build openvla-libero
```

## 2. OpenVLA LIBERO 테스트

### 2.1 서버 실행
```bash
# 백그라운드로 서버 시작
docker compose up -d openvla-libero

# 로그 확인 (모델 로딩 완료까지 대기)
docker logs -f openvla-libero-server

# 서버 상태 확인
curl http://localhost:18010/health
```

### 2.2 평가 실행
```bash
# 테스트용 (에피소드 1개)
docker compose run --rm libero python src/eval_libero.py \
  --model openvla \
  --host localhost \
  --port 18010 \
  --task-suite libero_spatial \
  --num-trials 1 \
  --verbose

# 전체 평가 (50 trials)
docker compose run --rm libero python src/eval_libero.py \
  --model openvla \
  --host localhost \
  --port 18010 \
  --task-suite libero_spatial \
  --num-trials 50
```

### 2.3 종료
```bash
docker compose down openvla-libero
```

## 3. Pi0 LIBERO 테스트

Pi0는 openpi 인프라를 사용합니다.

```bash
cd third_party/openpi

# 서버 + 클라이언트 실행
docker compose -f examples/libero/compose.yml up --build

# 또는 서버만 백그라운드 실행
docker compose -f examples/libero/compose.yml up -d openpi_server

# 별도로 클라이언트 실행
docker compose -f examples/libero/compose.yml run runtime \
  python examples/libero/main.py \
  --task_suite_name libero_spatial \
  --num_trials_per_task 5
```

## 4. 결과 확인

```
data/libero/results/     # JSON 결과 파일
data/libero/videos/      # 에피소드 비디오
```

## 5. 문제 해결

### 포트 충돌
```bash
# 18010 포트 사용 중인 프로세스 확인
sudo lsof -i :18010

# 기존 컨테이너 정리
docker compose down
docker rm -f openvla-libero-server
```

### LIBERO import 오류
```bash
# LIBERO 설치 확인
docker compose run --rm libero python -c "import libero; print(libero.__file__)"
```

### 서버 연결 오류
```bash
# 서버 로그 확인
docker logs openvla-libero-server

# health check
curl http://localhost:18010/health
```

## 6. Task Suite 옵션

| Suite | Tasks | Max Steps | 설명 |
|-------|-------|-----------|------|
| `libero_spatial` | 10 | 220 | 공간 추론 |
| `libero_object` | 10 | 280 | 물체 조작 |
| `libero_goal` | 10 | 300 | 목표 달성 |
| `libero_10` | 10 | 520 | 긴 horizon |
| `libero_90` | 90 | 400 | 대규모 |

## 7. 비교 평가

```bash
# OpenVLA 평가
docker compose up -d openvla-libero
docker compose run --rm libero python src/eval_libero.py \
  --model openvla --host localhost --port 18010 \
  --task-suite libero_10 --num-trials 50

# Pi0 평가
cd third_party/openpi
CLIENT_ARGS="--task_suite_name libero_10 --num_trials_per_task 50" \
  docker compose -f examples/libero/compose.yml up
```
