# SimplerEnv 테스트 가이드

## 빠른 시작

```bash
# 1. 컨테이너 시작
docker compose up -d eval

# 2. 기본 환경 테스트
docker exec simpler-dev python src/test_simpler_demo.py

# 3. 모델 평가 (API 모드)
docker compose up -d openvla
docker exec simpler-dev python src/eval_simpler.py --model openvla --api-url http://localhost:8001
```

## 평가 명령어

### API 모드 (권장)
```bash
# OpenVLA
docker exec simpler-dev python src/eval_simpler.py \
    --model openvla --api-url http://localhost:8001 --n-episodes 24

# LAPA
docker exec simpler-dev python src/eval_simpler.py \
    --model lapa --api-url http://localhost:8002 --n-episodes 24

# 다중 모델 비교
docker exec simpler-dev python src/eval_simpler.py --config configs/eval_api_example.json
```

### 주요 옵션
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model` | 모델 타입 | - |
| `--api-url` | API 서버 URL | - |
| `--n-episodes` | 에피소드 수 | 24 |
| `--max-steps` | 최대 스텝 | 300 |
| `--config` | JSON 설정 파일 | - |

## Docker 서비스

| 서비스 | 포트 | 용도 |
|--------|------|------|
| eval | - | SimplerEnv 평가 환경 |
| openvla | 8001 | OpenVLA 추론 서버 |
| lapa | 8002 | LAPA 추론 서버 |

```bash
docker compose up -d           # 전체 시작
docker compose logs -f openvla # 로그 확인
docker compose down            # 중지
```

## 트러블슈팅

### API 서버 연결 실패
```bash
curl http://localhost:8001/health
docker compose logs openvla
docker compose restart openvla
```

### GPU 메모리 부족
```bash
docker compose stop lapa  # 사용하지 않는 서버 중지
--n-episodes 1            # 배치 크기 줄이기
```

### ManiSkill Assets 누락
```bash
docker exec simpler-dev python -m mani_skill.utils.download_asset bridge_v2_real2sim
```
