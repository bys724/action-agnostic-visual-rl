# SimplerEnv 모델 통합 가이드

## 개요
SimplerEnv는 다양한 로봇 조작 정책(policy) 모델을 시뮬레이션에서 평가할 수 있는 벤치마크입니다. 현재 RT-1, RT-1-X, Octo 모델이 기본적으로 지원되며, OpenVLA, RT-2, Pi0 등의 최신 모델들도 통합이 가능합니다.

## 1. OpenVLA 통합

### 지원 현황
- **지원 가능**: OpenVLA는 SimplerEnv에 직접 적용 가능
- **공식 Fork 존재**: [DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA)

### 설치 방법
```bash
# OpenVLA 전용 패키지 설치
pip install torch==2.3.1 torchvision==0.18.1 
pip install timm==0.9.10 tokenizers==0.15.2 accelerate==0.32.1
pip install flash-attn==2.6.1 --no-build-isolation
```

### 구현 예제
```python
# simpler_env/policies/openvla/openvla_model.py
from openvla import OpenVLAModel

class OpenVLAPolicy:
    def __init__(self, checkpoint_path):
        self.model = OpenVLAModel.from_pretrained(checkpoint_path)
    
    def get_action(self, obs, instruction):
        return self.model.predict(obs, instruction)
```

### 성능 비교
- SimplerEnv에서 TraceVLA가 OpenVLA 대비 10% 향상된 성능
- 실제 로봇 작업에서 3.5배 향상
- Google Robot 작업에서 일관되게 OpenVLA를 능가

## 2. RT-2 통합

### 지원 현황
- **이론적 가능**: RT-2는 RT-1의 후속 모델로 통합 가능
- **직접 구현 필요**: 공식 SimplerEnv 지원 없음

### RT-2 특징
- Vision-Language-Action (VLA) 모델
- 웹 데이터와 로봇 데이터를 함께 학습
- RT-1 대비 성능: 32% → 62% (unseen scenarios)
- 시뮬레이션 벤치마크: 90% 성공률 (이전 SOTA 77%)

### 구현 접근법
```python
# RT-1 코드를 기반으로 RT-2 어댑터 작성
# simpler_env/policies/rt2/
class RT2Policy:
    def __init__(self, model_path):
        # RT-2 모델 로드
        # 액션을 텍스트 토큰으로 변환하는 로직 필요
        pass
    
    def tokenize_action(self, action):
        # "1 128 91 241 5 101 127 217" 형식으로 변환
        pass
```

### 참고 리소스
- [kyegomez/RT-2](https://github.com/kyegomez/RT-2): RT-2 민주화 프로젝트
- RT-2 공식 페이지: https://robotics-transformer2.github.io/

## 3. Pi0 (π₀) 통합

### 지원 현황
- **최신 모델**: Physical Intelligence에서 개발
- **오픈소스**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- **커뮤니티 구현**: [allenzren/open-pi-zero](https://github.com/allenzren/open-pi-zero)

### Pi0 특징
- 3B 파라미터 VLM 기반
- 7개 로봇 플랫폼, 68개 작업으로 사전학습
- SimplerEnv 성공률: 20-50% (zero-shot)
- PhysBrain 향상 버전: 53.9% 성공률

### 모델 변형
- **Pi0 base**: 표준 사전학습 모델
- **Pi0-FAST**: 향상된 언어 추종, 4-5배 높은 추론 비용
- **Pi0 DROID**: DROID 데이터셋에 파인튜닝

### 구현 방법
```python
# Pi0 모델 통합
from openpi import Pi0Model

class Pi0Policy:
    def __init__(self, variant="base"):
        self.model = Pi0Model.from_pretrained(variant)
    
    def predict(self, obs, instruction):
        # Flow matching 기반 액션 생성
        return self.model.generate_action(obs, instruction)
```

### 파인튜닝 요구사항
- 1-20시간의 데이터로 충분
- ALOHA, DROID 플랫폼 체크포인트 제공

## 4. 통합 구현 가이드

### 새로운 정책 추가 단계

1. **정책 구현**
```bash
simpler_env/policies/{your_model}/
├── __init__.py
├── model.py          # 모델 로드 및 추론
├── config.py         # 설정 파일
└── utils.py          # 헬퍼 함수
```

2. **메인 추론 스크립트 수정**
```python
# simpler_env/main_inference.py
if args.model == "openvla":
    from policies.openvla import OpenVLAPolicy
    policy = OpenVLAPolicy(args.checkpoint)
```

3. **평가 실행**
```bash
python simpler_env/simple_inference_visual_matching_prepackaged_envs.py \
    --model openvla \
    --env PutSpoonOnTableClothInScene-v1 \
    --num_episodes 100
```

## 5. 벤치마크 평가 메트릭

### 표준 메트릭
- **Success Rate**: 작업 완료 비율
- **Episode Reward**: 누적 보상
- **MMRV**: Mean Maximum Rank Violation
- **Pearson r**: 실제 성능과의 상관관계

### 평가 설정
- **Visual Matching**: 실제 이미지 오버레이
- **Variant Aggregation**: 다양한 환경 변형 평균

## 6. 추가 리소스

### 커뮤니티 자료
- [SimplerEnv 공식 문서](https://simpler-env.github.io/)
- [공개 Google Sheets](https://docs.google.com/spreadsheets/d/xxx): SOTA 모델 성능 기록

### 논문 및 참고문헌
```bibtex
@article{li24simpler,
  title={Evaluating Real-World Robot Manipulation Policies in Simulation},
  author={Li, Xuanlin and others},
  journal={arXiv preprint arXiv:2405.05941},
  year={2024}
}
```

## 7. 문제 해결

### 일반적인 이슈
1. **CUDA 버전**: >= 11.8 필요
2. **메모리 부족**: num_envs 줄이기
3. **액션 공간 불일치**: 액션 순서와 범위 확인

### 디버깅 팁
- `simpler_env/utils/debug/` 폴더의 실제 비디오 피드 스크립트 활용
- 액션 공간 매핑 확인 (7차원: [dx, dy, dz, rx, ry, rz, gripper])