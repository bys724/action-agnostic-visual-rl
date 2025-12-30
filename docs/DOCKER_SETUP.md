# Docker 개발 환경 설정 가이드

## 개요
SIMPLER 환경의 렌더링 및 물리 시뮬레이션 문제를 해결하기 위해 Docker 기반 개발 환경을 사용합니다.

## 사전 요구사항

1. **Docker 설치**
   ```bash
   # Docker 설치 (Ubuntu)
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   sudo usermod -aG docker $USER  # 재로그인 필요
   ```

2. **NVIDIA Container Toolkit**
   ```bash
   # NVIDIA Docker 지원 설치
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **GPU 확인**
   ```bash
   nvidia-smi  # GPU가 정상적으로 표시되어야 함
   ```

## 빠른 시작

### 1. Docker 이미지 빌드
```bash
./docker/build.sh
```

### 2. 개발 환경 실행
```bash
./docker/run.sh
```

### 3. 환경 테스트
컨테이너 내부에서:
```bash
python docker/test_env.py
```

## Docker Compose 사용

### 개발 컨테이너 실행
```bash
docker-compose up simpler-dev
```

### Jupyter Lab 실행
```bash
docker-compose up jupyter
# 브라우저에서 http://localhost:8888 접속
```

## 컨테이너 내부 작업

### 환경 확인
```python
import simpler_env
env = simpler_env.make('google_robot_pick_coke_can')
obs, info = env.reset()
print("Environment ready!")
```

### 학습 스크립트 실행
```bash
python scripts/train.py --config configs/experiment.yaml
```

## 문제 해결

### 1. CUDA 오류
- NVIDIA 드라이버 버전 확인: `nvidia-smi`
- CUDA 11.8 이상 필요

### 2. Display 오류
- 컨테이너는 자동으로 Xvfb를 사용하여 가상 디스플레이 생성
- `DISPLAY=:99` 환경 변수가 자동 설정됨

### 3. 메모리 부족
- docker-compose.yml에서 `shm_size` 조정
- 기본값: 16GB

## 디렉토리 구조

```
docker/
├── build.sh        # 이미지 빌드 스크립트
├── run.sh          # 컨테이너 실행 스크립트
└── test_env.py     # 환경 테스트 스크립트

Dockerfile          # Docker 이미지 정의
docker-compose.yml  # Docker Compose 설정
.dockerignore       # Docker 빌드 제외 파일
```

## 주요 기능

### 포함된 패키지
- Python 3.10
- CUDA 11.8 + cuDNN 8
- SIMPLER (SimplerEnv + ManiSkill2)
- PyTorch (GPU 지원)
- Vulkan (SAPIEN 렌더링)
- Xvfb (가상 디스플레이)

### 환경 변수
- `MUJOCO_GL=egl`: EGL 렌더링 사용
- `PYOPENGL_PLATFORM=egl`: PyOpenGL EGL 백엔드
- `DISPLAY=:99`: 가상 디스플레이

## 개발 워크플로우

1. **코드 수정**: 호스트에서 편집 (VSCode 등)
2. **실행**: Docker 컨테이너 내에서 실행
3. **결과 확인**: 볼륨 마운트로 실시간 동기화

## 팁

- 컨테이너 내부의 변경사항은 `/workspace`에 저장됨
- 호스트의 프로젝트 디렉토리와 자동 동기화
- GPU 메모리 모니터링: `nvidia-smi -l 1`