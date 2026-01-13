# 프로젝트 구조

```
action-agnostic-visual-rl/
├── src/
│   ├── policies/           # 정책 구현
│   │   ├── api_interface.py
│   │   ├── openvla/
│   │   └── lapa/
│   ├── eval_simpler.py     # 평가 스크립트
│   └── collect_trajectories.py
├── docker/
│   ├── openvla/            # OpenVLA 서버 (포트 8001)
│   └── lapa/               # LAPA 서버 (포트 8002)
├── configs/                # 평가 설정 파일
├── data/                   # 결과, 체크포인트 (gitignore)
├── third_party/            # 서브모듈 (SimplerEnv, LAPA)
└── docs/                   # 문서
```

## 아키텍처

```
┌─────────────┐     HTTP     ┌──────────────┐
│    eval     │◄────────────►│   openvla    │
│ (SimplerEnv)│              │  (port 8001) │
└─────────────┘              └──────────────┘
      │                      ┌──────────────┐
      └─────────────────────►│     lapa     │
                             │  (port 8002) │
                             └──────────────┘
```

의존성 충돌 방지를 위해 각 모델을 독립 컨테이너로 분리합니다.
