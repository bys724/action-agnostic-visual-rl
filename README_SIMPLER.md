# SimplerEnv 개발 환경 및 데모

## 빠른 시작

### 1. 사전 준비
```bash
# X11 권한 설정 (GUI 사용시 필요)
xhost +local:docker
```

### 2. Docker 이미지 빌드
```bash
# 일반 빌드
docker compose build dev

# 캐시 없이 처음부터 빌드 (클린 빌드)
docker compose build --no-cache dev

# 빌드 후 기존 컨테이너 재생성이 필요한 경우
docker compose down
docker compose up -d dev
```

### 3. 컨테이너 실행 (백그라운드)
```bash
# 컨테이너 시작 (호스트 DISPLAY 사용)
DISPLAY=$DISPLAY docker compose up -d dev

# 컨테이너 접속
docker exec -it simpler-dev bash
```

### 4. 데모 실행 (컨테이너 내부)
```bash
# 환경 테스트 (GUI 없이)
python src/test_simpler_demo.py

# GUI 포함 테스트
python src/test_simpler_demo.py --gui

# 다른 환경 테스트 (ManiSkill3 환경)
python src/test_simpler_demo.py --env PutCarrotOnPlateInScene-v1 --gui

# 더 많은 스텝 실행
python src/test_simpler_demo.py --gui --steps 500
```

## 데모 옵션

- `--env`: 환경 이름 (기본: PutSpoonOnTableClothInScene-v1)
- `--gui`: GUI 시각화 활성화
- `--steps`: 실행할 스텝 수 (기본: 300)

## 사용 가능한 환경

ManiSkill3 기반 SimplerEnv 환경:
- `PutSpoonOnTableClothInScene-v1`
- `PutCarrotOnPlateInScene-v1`
- `StackGreenCubeOnYellowCubeBakedTexInScene-v1`
- `PutEggplantInBasketScene-v1`

## 파일 구조

```
프로젝트 루트/
├── docker-compose.yml     # Docker 개발 환경 설정
├── Dockerfile            # Docker 이미지 정의
├── README_SIMPLER.md     # 이 문서
├── src/                  # 개발 및 테스트 코드
│   └── test_simpler_demo.py  # SimplerEnv 테스트 데모
└── third_party/          # 서브모듈
    └── SimplerEnv/       # SIMPLER 환경
```

## 워크스페이스 구조

컨테이너 내부에서 `/workspace`로 프로젝트 전체가 마운트됩니다:
- `/workspace/` - 프로젝트 루트 (호스트와 동기화)
- `/workspace/src/` - 개발 및 테스트 코드
- `/workspace/third_party/` - 외부 라이브러리

## 개발 워크플로우

1. 호스트에서 코드 수정 (VSCode 등)
2. Docker 컨테이너에서 실행 및 테스트
3. 모든 변경사항은 실시간으로 동기화됨

## 컨테이너 관리

```bash
# 컨테이너 중지
docker compose down

# 컨테이너 재시작
docker compose restart dev

# 로그 확인
docker compose logs -f dev
```

## 주의사항

- GPU가 필요함 (NVIDIA Docker 지원 필요)
- GUI 모드 사용시 `xhost +local:docker` 실행 필수
- 컨테이너 내부에서 Xvfb가 자동으로 가상 디스플레이 생성