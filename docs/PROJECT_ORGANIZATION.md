# 프로젝트 구조

```
action-agnostic-visual-rl/
├── src/
│   ├── models/             # 자체 모델 구현
│   │   └── two_stream.py   # Two-Stream Video Predictor
│   ├── policies/           # 정책 구현
│   │   ├── api_interface.py
│   │   ├── openvla/
│   │   └── lapa/
│   ├── eval_simpler.py     # SIMPLER 평가 스크립트
│   ├── eval_libero.py      # LIBERO 평가 스크립트
│   └── collect_trajectories.py
├── scripts/
│   └── train_long.py       # Two-Stream 장기 학습 스크립트
├── docker/
│   ├── dev/                # 학습 환경 (H100 x2)
│   ├── openvla/            # OpenVLA 서버 (포트 8001)
│   └── lapa/               # LAPA 서버 (포트 8002)
├── configs/
│   ├── two_stream_train.yaml  # Two-Stream 학습 설정
│   └── ...                    # 평가 설정 파일
├── data/                   # 결과, 체크포인트, 데이터셋 (gitignore)
├── references/             # 연구 참조 문서
├── third_party/            # 서브모듈 (SimplerEnv, LAPA)
└── docs/                   # 문서
```

## 평가 아키텍처

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

## Two-Stream 모델 아키텍처

```
[img_t] ──┐                          ┌─── [img_pred]
          │   ┌───────────────────┐  │
[img_t+k] ─┴─►│ Two-Stream Encoder ├──┴─► [cls_emb]
              │   M (temporal)    │
              │   P (spatial)     │
              │   CLS Exchange    │
              └───────────────────┘
```

**핵심 구성요소:**
- **M채널 (Magnocellular)**: 시간적 변화 [ΔL, ΔR, ΔG, ΔB]
- **P채널 (Parvocellular)**: 공간적 구조 + 색상 [∂x, ∂y, R, G, B]
- **CLS Exchange**: M-P 채널 간 정보 교환 (interleaved stages)
- **Video Decoder**: 임베딩 → 이미지 재구성 (forward dynamics 학습)
