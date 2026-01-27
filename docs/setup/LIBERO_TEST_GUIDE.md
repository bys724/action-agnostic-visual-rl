# LIBERO 벤치마크 테스트 가이드

GUI가 있는 워크스테이션에서 LIBERO 평가를 테스트하는 가이드입니다.

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
# 터미널 1: OpenVLA-LIBERO 서버 시작
docker compose up openvla-libero

# 서버 상태 확인 (다른 터미널에서)
curl http://localhost:8004/health
```

### 2.2 평가 실행
```bash
# 터미널 2: LIBERO 평가 실행
docker compose run --rm libero python src/eval_libero.py \
  --model openvla \
  --host localhost \
  --port 8004 \
  --task-suite libero_spatial \
  --num-trials 5 \
  --verbose
```

**테스트용 간단 실행 (에피소드 1개만):**
```bash
docker compose run --rm libero python src/eval_libero.py \
  --model openvla \
  --host localhost \
  --port 8004 \
  --task-suite libero_spatial \
  --num-trials 1 \
  --verbose
```

## 3. Pi0 LIBERO 테스트

Pi0는 openpi 인프라를 사용합니다.

### 3.1 openpi 서버 및 클라이언트 실행
```bash
cd third_party/openpi

# 전체 실행 (서버 + 클라이언트)
docker compose -f examples/libero/compose.yml up --build

# 또는 서버만 실행
docker compose -f examples/libero/compose.yml up openpi_server
```

### 3.2 별도 클라이언트로 테스트
```bash
# 터미널 1: Pi0 서버
cd third_party/openpi
docker compose -f examples/libero/compose.yml up openpi_server

# 터미널 2: LIBERO 클라이언트 (openpi 내장)
docker compose -f examples/libero/compose.yml run runtime \
  python examples/libero/main.py \
  --task_suite_name libero_spatial \
  --num_trials_per_task 5
```

## 4. 결과 확인

### 평가 결과 위치
```
data/libero/results/     # JSON 결과 파일
data/libero/videos/      # 에피소드 비디오 (성공/실패)
```

### 결과 파일 예시
```json
{
  "task_suite": "libero_spatial",
  "overall_success_rate": 0.75,
  "total_successes": 150,
  "total_episodes": 200,
  "task_results": [...]
}
```

## 5. 문제 해결

### LIBERO 환경 오류
```bash
# LIBERO 설정 확인
docker compose run --rm libero cat /tmp/libero/config.yaml

# 수동으로 LIBERO 경로 설정
docker compose run --rm libero bash
export LIBERO_CONFIG_PATH=/tmp/libero
python -c "from libero.libero import benchmark; print(benchmark.get_benchmark_dict().keys())"
```

### GPU 메모리 부족
```bash
# 더 작은 배치로 실행
docker compose run --rm libero python src/eval_libero.py \
  --model openvla --task-suite libero_spatial --num-trials 1
```

### 서버 연결 오류
```bash
# 서버 로그 확인
docker logs openvla-libero-server

# 포트 확인
netstat -tlnp | grep 8004
```

## 6. Task Suite 옵션

| Suite | Tasks | Max Steps | 설명 |
|-------|-------|-----------|------|
| `libero_spatial` | 10 | 220 | 공간 추론 |
| `libero_object` | 10 | 280 | 물체 조작 |
| `libero_goal` | 10 | 300 | 목표 달성 |
| `libero_10` | 10 | 520 | 긴 horizon |
| `libero_90` | 90 | 400 | 대규모 |

## 7. 비교 평가 실행

```bash
# 1. OpenVLA 평가
docker compose up -d openvla-libero
docker compose run --rm libero python src/eval_libero.py \
  --model openvla --task-suite libero_10 \
  --num-trials 50 --output-dir data/libero/results/openvla

# 2. Pi0 평가 (별도 터미널)
cd third_party/openpi
CLIENT_ARGS="--task_suite_name libero_10 --num_trials_per_task 50" \
  docker compose -f examples/libero/compose.yml up

# 3. 결과 비교
ls -la data/libero/results/
```
