# SIMPLER 환경 설치 및 사용 가이드

## 설치 순서

### 1. 기본 요구사항
- CUDA 11.8 ~ 12.x
- NVIDIA RTX GPU (레이 트레이싱을 위한 RT 코어 필요)
- Python 3.10 또는 3.11

### 2. Python 가상환경 생성 (conda 없이)
```bash
# Python venv 사용
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate

# pip 업그레이드
pip install --upgrade pip
```

### 3. 서브모듈 초기화
```bash
# 이미 추가됨 (third_party/SimplerEnv)
git submodule update --init --recursive
```

### 4. 의존성 설치

```bash
# numpy 버전 고정 (IK 오류 방지 - 중요!)
pip install numpy==1.24.4

# ManiSkill2 real2sim 설치
cd third_party/SimplerEnv/ManiSkill2_real2sim
pip install -e .

# SIMPLER 설치
cd ../
pip install -e .

# 프로젝트 루트로 돌아오기
cd ../../

# 프로젝트 의존성 설치
pip install -r requirements.txt
```

### 5. 추가 패키지 (선택사항)

RT-1이나 Octo 모델 사용시:
```bash
cd third_party/SimplerEnv
pip install -r requirements_full_install.txt
```

## SIMPLER 환경 구조

### 주요 환경들
- `google_robot_pick_coke_can` - Google Robot 콜라캔 집기
- `google_robot_pick_horizontal_coke_can` - 수평 콜라캔
- `widowx_spoon_on_towel` - WidowX 숟가락 놓기
- `widowx_carrot_on_plate` - WidowX 당근 놓기
- `widowx_stack_cube` - WidowX 큐브 쌓기

### 행동 공간 (Action Space)
```python
# 7차원 연속 행동
action = [
    dx, dy, dz,      # 엔드 이펙터 위치 델타 (delta xyz)
    rx, ry, rz,      # 회전 델타 (axis-angle)
    gripper          # 그리퍼 열기/닫기
]
```

### 관찰 공간 (Observation)
- RGB 이미지: (H, W, 3) uint8
- 로봇 상태 정보
- 언어 지시사항

## 주요 API

### 환경 생성
```python
import simpler_env
env = simpler_env.make('google_robot_pick_coke_can')
```

### 기본 루프
```python
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

obs, reset_info = env.reset()
instruction = env.get_language_instruction()

done, truncated = False, False
while not (done or truncated):
    # 이미지 추출
    image = get_image_from_maniskill2_obs_dict(env, obs)
    
    # 정책으로 행동 계산 (여기에 모델 추론)
    action = env.action_space.sample()  # 랜덤 액션 (교체 필요)
    
    # 환경 스텝
    obs, reward, done, truncated, info = env.step(action)
```

### 긴 시간 작업 (Long-horizon tasks)
```python
# 다음 서브태스크로 진행
env.advance_to_next_subtask()

# 최종 서브태스크인지 확인
is_final = env.is_final_subtask()

# 새 지시사항 확인
new_instruction = env.get_language_instruction()
```

## Visual Matching vs Variant Aggregation

### Visual Matching (기본)
- 실제 이미지를 시뮬레이션 배경에 오버레이
- 로봇과 객체 텍스처 조정
- `simpler_env.make()`로 자동 생성

### Variant Aggregation
- 다양한 배경, 조명, 방해물 생성
- 여러 변형의 평균 성능 계산
- 더 강건한 평가 제공

## 디버깅 팁

### 1. GPU 메모리 부족
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]
    )
```

### 2. 렌더링 문제
```python
# 헤드리스 모드
os.environ["DISPLAY"] = ""

# 또는 가상 디스플레이
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
```

### 3. 행동 공간 불일치
- SIMPLER은 axis-angle 회전 사용
- 쿼터니언 사용 모델은 변환 필요
- 그리퍼 규칙 확인 (0=열기 또는 닫기는 로봇에 따라 다름)