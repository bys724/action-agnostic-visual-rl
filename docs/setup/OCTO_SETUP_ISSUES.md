# Octo 모델 설정 이슈 및 해결 방법

## 주요 문제점

Octo 모델을 SimplerEnv와 함께 사용할 때 여러 의존성 충돌이 발생합니다.

### 1. JAX 버전 충돌
- **문제**: `jax.random.KeyArray` 속성이 JAX 0.4.16+ 에서 제거됨
- **해결**: octo/utils/typing.py 파일 수정
  ```python
  # 변경 전
  PRNGKey = jax.random.KeyArray
  # 변경 후  
  PRNGKey = jax.Array
  ```

### 2. CUDA 라이브러리 충돌
- PyTorch는 CUDNN 9.10, CUBLAS 12.8.4 필요
- JAX는 CUDNN 8.9, CUBLAS 12.9 사용
- 동시 사용 시 라이브러리 충돌 발생

### 3. NumPy 버전 충돌
- OpenCV는 NumPy 1.x 필요
- JAX 최신 버전은 NumPy 2.x 요구
- TensorFlow 2.15는 NumPy <2.0 필요

### 4. Flax/JAX 호환성
- Flax 0.10.x는 JAX 0.4.27+ 필요
- Flax 0.8.0은 JAX 0.4.19+ 지원
- `register_dataclass` API 변경으로 인한 충돌

## 임시 해결 방법

```bash
# 1. JAX/Flax 다운그레이드
pip install 'jax==0.4.26' 'jaxlib==0.4.26+cuda12.cudnn89' 
pip install 'flax==0.8.0' 'optax==0.1.9' 'chex==0.1.85'

# 2. NumPy 1.x 유지
pip install --force-reinstall 'numpy<2.0' --no-deps

# 3. octo 패치
python scripts/fix_octo_jax.py
```

## 권장 사항

현재 Octo 모델은 의존성 충돌이 심각하여 안정적인 실행이 어렵습니다.

### 대안:
1. **SimplePolicy 사용**: 기본 테스트용
2. **RT-1 모델**: Google 공식 체크포인트 사용
3. **OpenVLA**: 향후 통합 예정
4. **별도 환경**: Octo 전용 Docker 컨테이너 구성

## 장기 해결 방안

1. Octo 팀의 업데이트 대기 (JAX 0.5+ 지원)
2. 별도 가상환경에서 Octo 실행
3. SimplerEnv 자체 정책 모델 개발

## 테스트 상태

- ✅ SimplePolicy: 정상 작동
- ⚠️ Octo-small: 의존성 충돌로 불안정
- ⏳ RT-1: 체크포인트 다운로드 필요
- 📝 OpenVLA: 통합 예정