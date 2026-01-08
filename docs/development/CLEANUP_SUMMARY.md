# 프로젝트 정리 및 Octo 모델 설정 완료

## 정리된 파일들

### 삭제한 불필요한 파일들:
1. **`patch_simpler.py`** - ManiSkill2용 패치 (현재 ManiSkill3 사용)
2. **`README_SIMPLER.md`** - 중복된 README 파일
3. **`src/envs/simpler_wrapper.py`** - ManiSkill2 기반 오래된 래퍼
4. **`DEVELOPMENT_GUIDE.md`** - 오래된 개발 가이드

### 유지한 파일들:
- `docker-compose.yml` - Docker 컨테이너 설정
- `Dockerfile` - Docker 이미지 정의
- `requirements.txt` - Python 의존성
- `src/eval_simpler.py` - 평가 스크립트
- `src/collect_trajectories.py` - 데이터 수집 스크립트
- `src/test_simpler_demo.py` - 테스트 데모
- `scripts/test_baseline.sh` - 테스트 자동화 스크립트
- 문서 파일들 (CLAUDE.md, TEST_GUIDE.md, etc.)

## Octo 모델 의존성 설치 완료

### 설치된 패키지들:
```bash
# 주요 패키지
- tensorflow-cpu==2.15.0
- tensorflow-datasets==4.9.9
- tensorflow-probability==0.23.0 (0.25.0에서 다운그레이드)
- flax==0.10.4
- jax==0.4.34
- jaxlib==0.4.34

# Octo 관련
- octo (GitHub에서 설치)
- dlimp (GitHub에서 설치)
- distrax==0.1.5

# 기타
- optax, orbax-checkpoint, tensorstore
- array_record, dm-tree, etils
```

### 버전 호환성 이슈 해결:
1. **NumPy 버전**: 1.26.4 유지 (OpenCV와 호환)
2. **TensorFlow-Probability**: 0.23.0 (TF 2.15와 호환)
3. **CUDNN/CUBLAS**: PyTorch와 호환되는 버전 유지

## 현재 프로젝트 구조

```
/home/bys/action-agnostic-visual-rl/
├── docker-compose.yml          # Docker 설정
├── Dockerfile                   # Docker 이미지
├── requirements.txt             # 의존성
├── src/                         # 소스 코드
│   ├── eval_simpler.py         # 평가 스크립트
│   ├── collect_trajectories.py # 데이터 수집
│   └── test_simpler_demo.py    # 테스트
├── scripts/                     # 유틸리티 스크립트
│   └── test_baseline.sh        # 자동 테스트
├── data/                        # 데이터 저장
│   ├── results/                # 평가 결과
│   └── trajectories/           # 수집된 데이터
├── docs/                        # 문서
└── third_party/                 # 서브모듈
    └── SimplerEnv/             # SimplerEnv 환경

```

## 테스트 명령어

### SimplePolicy 테스트 (작동 확인됨)
```bash
docker exec simpler-dev bash -c "cd /workspace && \
  export CUDA_VISIBLE_DEVICES=0 && \
  export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
  python src/eval_simpler.py --model simple --n-episodes 2"
```

### Octo 모델 테스트 (준비 완료)
```bash
docker exec simpler-dev bash -c "cd /workspace && \
  export CUDA_VISIBLE_DEVICES=0 && \
  export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
  python src/eval_simpler.py --model octo-small --n-episodes 2"
```

### 자동 테스트 스크립트
```bash
docker exec simpler-dev bash -c "cd /workspace && ./scripts/test_baseline.sh"
```

## 주의사항

1. **메모리 설정**: JAX/TensorFlow 사용 시 메모리 사전 할당 비활성화 필요
   ```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export TF_FORCE_GPU_ALLOW_GROWTH=true
   ```

2. **Octo 모델 다운로드**: 첫 실행 시 자동으로 Hugging Face에서 다운로드

3. **NumPy 버전**: OpenCV 호환성 때문에 1.x 버전 유지 필요

## 다음 단계

1. Octo 모델 실제 평가 실행
2. RT-1 체크포인트 다운로드 및 테스트
3. 성공률 높은 모델로 trajectory 수집
4. Fine-tuning을 위한 데이터 준비