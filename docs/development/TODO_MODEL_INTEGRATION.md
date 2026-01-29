# TODO: 모델 통합 및 향후 작업

## 현재 상태

| 모델 | SIMPLER | LIBERO | 비고 |
|------|---------|--------|------|
| OpenVLA | ✅ 완료 | ✅ 완료 | 비전 인코더 연구 타겟 |
| LAPA | ✅ 완료 | ❌ N/A | baseline 비교용 (비전 인코더 교체 불가) |
| Pi0 | ❌ N/A | ✅ 완료 | 비전 인코더 연구 타겟 |

## 최신 평가 결과 (libero_spatial, 10 trials/task)

| 모델 | 성공률 | 성공 | 에피소드 |
|------|--------|------|----------|
| **Pi0** | **100.0%** | 100 | 100 |
| **OpenVLA** | **40.0%** | 40 | 100 |

결과 파일: `data/libero/results/`

## 벤치마크 선택

**LIBERO를 통합 벤치마크로 사용**
- OpenVLA와 Pi0 모두 LIBERO에서 평가 가능
- 동일한 환경에서 공정한 비교 가능
- 체크포인트: `openvla-7b-finetuned-libero-spatial`, `pi05_libero`

## Docker 서비스

| 서비스 | 포트 | 용도 |
|--------|------|------|
| `eval` | - | SIMPLER 평가 환경 |
| `openvla` | 8001 | OpenVLA (SIMPLER용) |
| `lapa` | 8002 | LAPA |
| `libero` | - | LIBERO 평가 환경 |
| `openvla-libero` | 18010 | OpenVLA (LIBERO용) |
| `pi0` | 8000 | Pi0 (openpi 인프라 사용) |

## Pi0 통합 완료

**GitHub**: https://github.com/Physical-Intelligence/openpi
**라이선스**: Apache 2.0 / MIT
**로컬 경로**: `third_party/openpi/`

### 완료 항목
- [x] openpi 레포지토리 클론 (`third_party/openpi`)
- [x] LIBERO 환경 Docker 구성 (`docker/libero/`)
- [x] LIBERO 평가 스크립트 (`src/eval_libero.py`)
- [x] docker-compose.yml 업데이트

### Pi0 비전 인코더 구조
```python
# third_party/openpi/src/openpi/models/pi0.py
img = nnx_bridge.ToNNX(
    _siglip.Module(
        num_classes=paligemma_config.width,  # 2048
        variant="So400m/14",
        pool_type="none",
        scan=True,
        dtype_mm=config.dtype,
    )
)
self.PaliGemma = nnx.Dict(llm=llm, img=img)
```

교체 지점: `self.PaliGemma.img`를 커스텀 비전 인코더로 교체 가능

## 향후 작업

### 비전 인코더 교체 실험
- [ ] Bridge V2 데이터셋으로 커스텀 비전 인코더 학습
- [ ] OpenVLA 비전 인코더 교체 실험
- [ ] Pi0 비전 인코더 교체 실험
- [ ] LIBERO 벤치마크에서 성능 비교

### 체크포인트 관리
- [ ] 체크포인트 자동 다운로드 스크립트 개선
- [ ] configs/checkpoints.yaml 활용

### 평가 자동화
- [ ] 성능 비교 그래프 자동 생성
- [ ] 배치 추론 지원

## 빠른 시작

### SIMPLER 평가
```bash
# OpenVLA (SIMPLER)
docker compose up -d eval openvla
docker exec simpler-eval python src/eval_simpler.py \
  --model openvla --api-url http://localhost:8001
```

### LIBERO 평가
```bash
# OpenVLA (LIBERO)
docker compose up -d libero openvla-libero
docker exec libero-eval python src/eval_libero.py \
  --model openvla --host localhost --port 18010 \
  --task-suite libero_spatial --num-trials 50

# Pi0 (LIBERO) - openpi 인프라 사용
cd third_party/openpi
docker compose -f examples/libero/compose.yml up --build
```

### 비교 평가 (LIBERO)
두 모델을 순차적으로 평가 후 결과 비교:
```bash
# OpenVLA 평가
docker exec libero-eval python src/eval_libero.py \
  --model openvla --host localhost --port 18010 \
  --task-suite libero_spatial --num-trials 10

# Pi0 평가
docker exec libero-eval python src/eval_libero.py \
  --model pi0 --host localhost --port 8000 \
  --task-suite libero_spatial --num-trials 10

# 결과 비교
python3 src/compare_results.py --results-dir data/libero/results \
  --models openvla pi0 --task-suite libero_spatial
```
