# Action-Agnostic Visual Representation Learning

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 로봇 조작에 더 범용적인가? — 그리고 그 표현을 만드는 데 무엇이 정말 필요한가 (temporal/video 구조? 생물학적 M/P 경로 분리? 단순 input prior?).

이 저장소는 두 갈래의 논문을 공유 substrate 위에서 진행한다.

## 두 갈래

### Paper 1 — Input-Prior Robot Representation (ICRA)
단일프레임 **image MAE**에 hand-crafted **input prior(Sobel edge + RGB)**를 주면, 같은 스케일의 **VideoMAE를 이긴다**. → 로봇 표현 학습에 temporal/video 아키텍처가 필수가 아니며 input prior로 충분할 수 있다. 좁게 입증된 결과에서 출발해 ablation(edge vs RGB)·실로봇으로 확정. 계획: [`docs/paper1_input_prior_plan.md`](docs/paper1_input_prior_plan.md).

### Paper 2 — Action-Agnostic (AAAI)
영장류 시각피질의 **M(magnocellular, motion) / P(parvocellular, form) 경로 분리**를 모방한 two-stream 모델 **Parvo**. M이 학습 중 P를 **scaffold**하고 배포 시 P encoder만 남는다. action label 없이도 구조적 cross-stream bias가 표현을 개선하는지 검증 중. 실험: [`docs/v15b_retraining_status.md`](docs/v15b_retraining_status.md).

## 핵심 모델 — Parvo

- **구조**: two-stream. P=appearance(form), M=motion(change). pixel reconstruction(MAE) + cross-stream routing + compositional/V-JEPA aux loss.
- **배포**: P encoder 단독 (M은 학습 시 scaffold 후 제거).
- **명명**: 논문 핵심 모델만 `Parvo` (현재 구현 = code `two_stream_v15b`); 이전/divergent 버전은 버전명(v4…v15) 유지. 명명·이력은 [`CLAUDE.md`](CLAUDE.md) "명명 · 2논문 구조".

| 모델 | 설명 | 역할 |
|------|------|------|
| **Parvo** (code v15b) | two-stream M/P, scaffolded | 제안 (Paper 2) |
| image MAE (Sobel+RGB) | 단일프레임, = Parvo의 P stream 단독 | 제안 (Paper 1) |
| VideoMAE-ours | 2-frame masked autoencoder | controlled baseline |
| DINOv2 / SigLIP / VC-1 / V-JEPA 2.1 | internet/embodied SSL | 외부 baseline |

## 평가

- **Action probing** (EgoDex within-domain, DROID/CALVIN cross-domain): 표현이 변화/행동 정보를 인코딩하는지 회귀 R²로 측정
- **로봇 조작 BC** (LIBERO BC-Transformer, CortexBench, CALVIN): frozen encoder + policy head
- **실로봇** (Paper 1 본체 lift): manipulation deploy

## 프로젝트 구조

```
├── src/
│   ├── models/          # two_stream_v15(b)=Parvo, videomae, two_stream_v11 등
│   ├── encoders/adapters/  # BC-T 어댑터 (baseline 포함)
│   ├── datasets/        # EgoDex, DROID, LIBERO, CALVIN
│   ├── cortexbench/     # CortexBench (Adroit/MetaWorld) loader·config
│   └── training/        # Pre-training 루프
├── scripts/
│   ├── pretrain.py      # Pre-training 메인 (env-agnostic)
│   ├── cluster/         # IBS 클러스터 sbatch launcher
│   ├── local/           # 로컬 워크스테이션 launcher
│   ├── eval/            # probing, BC-T fine-tune, 시각화
│   └── viz/             # PCA overlay, Grad-CAM arrow
└── docs/                # 아래 "문서"
```

## 문서

- **개발 가이드 + 현재 상태**: [`CLAUDE.md`](CLAUDE.md)
- **연구 계획 (마스터)**: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)
- **Paper 1 (ICRA)**: [`docs/paper1_input_prior_plan.md`](docs/paper1_input_prior_plan.md)
- **Paper 2 (AAAI) 실험**: [`docs/v15b_retraining_status.md`](docs/v15b_retraining_status.md)
- **Probing**: [`docs/PROBING_GUIDE.md`](docs/PROBING_GUIDE.md)
- **LIBERO 평가**: [`docs/setup/LIBERO_TEST_GUIDE.md`](docs/setup/LIBERO_TEST_GUIDE.md)

실행 명령어·환경(클러스터/로컬)은 [`CLAUDE.md`](CLAUDE.md) 워크플로우 섹션 참조.

## 상태 (2026-06)

- **Paper 1**: P단독 image MAE > VideoMAE = 좁게 입증 → ablation(edge vs RGB) + 실로봇.
- **Paper 2**: Parvo(code v15b) scaffold 검증 중. LIBERO BC avg 0.785 ≈ v15(0.777), frozen baseline 미달 — 현상 유지. 다음 = M 기여 격리(no-M ablation) + 본학습 재제출.
