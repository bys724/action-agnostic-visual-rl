# SimplerEnv 모델 통합

## 지원 모델

| 모델 | 특징 | 성능 (SimplerEnv) |
|------|------|-------------------|
| OpenVLA | 7B VLA, PyTorch | 20-40% (zero-shot) |
| LAPA | Action-free 사전학습, JAX | 50-70% |
| RT-1/RT-1-X | SimplerEnv 기본 지원 | - |
| Octo | SimplerEnv 기본 지원 | - |

## 새 모델 추가 방법

1. `src/policies/{model}/` 디렉토리 생성
2. 정책 클래스 구현 (reset, step 메서드)
3. `docker/{model}/` API 서버 추가 (의존성 분리 시)
4. `eval_simpler.py`에 모델 로딩 로직 추가

### 정책 인터페이스
```python
class YourPolicy:
    def reset(self, instruction: str): ...
    def step(self, image: np.ndarray, instruction: str) -> tuple[np.ndarray, dict]: ...
```

### API 서버 엔드포인트
- `GET /health`: 서버 상태 확인
- `POST /reset`: 에피소드 리셋
- `POST /step`: 이미지 → 액션 추론

## 참고 자료
- [SimplerEnv](https://simpler-env.github.io/)
- [OpenVLA](https://github.com/openvla/openvla)
- [LAPA](https://github.com/LatentActionPretraining/LAPA)
