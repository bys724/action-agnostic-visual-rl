# TODO: 모델 통합 및 향후 작업

## 현재 상태

| 모델 | 상태 | 사용법 |
|------|------|--------|
| OpenVLA | ✅ 완료 | `--model openvla --api-url http://localhost:8001` |
| LAPA | ✅ 완료 | `--model lapa --api-url http://localhost:8002` |
| Pi0 | ⏳ 미구현 | - |

## 향후 작업

### 체크포인트 관리 개선 (학습 시작 시 진행)
- [ ] 체크포인트 자동 다운로드 스크립트 개선
- [ ] 학습 중 체크포인트 저장/로드 로직
- [ ] 체크포인트 버전 관리 시스템
- [ ] configs/checkpoints.yaml 활용

### 평가 자동화
- [ ] 성능 비교 그래프 자동 생성
- [ ] 배치 추론 지원
- [ ] Visual matching vs Variant aggregation 비교

### Pi0 통합 (선택적)
- 참고: [openpi](https://github.com/Physical-Intelligence/openpi)

## 빠른 시작

```bash
# OpenVLA 평가
docker compose up -d openvla
docker exec simpler-dev python src/eval_simpler.py --model openvla --api-url http://localhost:8001

# LAPA 평가
docker compose up -d lapa
docker exec simpler-dev python src/eval_simpler.py --model lapa --api-url http://localhost:8002

# 비교 평가
docker exec simpler-dev python src/eval_simpler.py --config configs/eval_api_example.json
```
