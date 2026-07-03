# Action-Agnostic Visual Representation Learning

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 로봇 조작에 더 범용적인가? — 그리고 그 표현을 만드는 데 무엇이 정말 필요한가 (temporal/video 구조? 생물학적 M/P 경로 분리? 단순 input prior?).

이 저장소는 두 갈래의 논문을 공유 substrate 위에서 진행한다.

## 두 갈래

### Paper 1 — Input-Prior Robot Representation (ICRA)
단일프레임 **image MAE**에 hand-crafted **input prior(Sobel edge + RGB)**를 주면, 같은 스케일의 **VideoMAE를 이긴다**. → 로봇 표현 학습에 temporal/video 아키텍처가 필수가 아니며 input prior로 충분할 수 있다. 좁게 입증된 결과에서 출발해 ablation(edge vs RGB)·실로봇으로 확정. 계획: [`docs/paper1_input_prior_plan.md`](docs/paper1_input_prior_plan.md).

### Paper 2 — Action-Agnostic (AAAI)
영장류 시각피질의 **M(magnocellular, motion) / P(parvocellular, form) 경로 분리**를 모방한 two-stream 모델 **CoMP-MAE**. M과 P가 **대칭 cross-reconstruction**(각자 자기 채널을 복구, 상대는 helper)으로 엮여, action label 없이도 구조적 cross-stream bias가 **factored·효율적** 표현을 만드는지 검증 중. 실험·설계: [`docs/comp_mae_plan.md`](docs/comp_mae_plan.md).

## 핵심 모델 — CoMP-MAE

- **구조**: two-stream. P=appearance(form, RGB), M=motion(change, ΔL). **대칭 pixel reconstruction** — P-recon(M-routed) + M-recon(P-grouped) + future 예측. M-recon이 M-encoder를 grounding(v15 no-op 대응).
- **배포**: **P encoder 단독** (M은 학습 시 P를 shaping; M을 배포 입력으로 넣으면 causal confusion으로 유해 — 확인됨).
- **명명**: 현 ours = `CoMP-MAE`(code `two_stream_v15`+M-recon 분기, `v16`); 선행 `MS-JEPA`(code v15b); 이전/divergent는 버전명(v4…v15) 유지. 명명·이력 = [`CLAUDE.md`](CLAUDE.md) "명명 · 2논문 구조".

| 모델 | 설명 | 역할 |
|------|------|------|
| **CoMP-MAE** (code v16) | two-stream M/P, 대칭 cross-recon | 제안 (Paper 2, 현 ours) |
| MS-JEPA (code v15b) | two-stream, student-anchor | 선행 축 |
| image MAE (Sobel+RGB) | 단일프레임, = P stream 단독 | 제안 (Paper 1) |
| VideoMAE-ours | 2-frame masked autoencoder | controlled baseline |
| DINOv2 / SigLIP / VC-1 / V-JEPA 2.1 | internet/embodied SSL | 외부 baseline |

## 평가

- **Action probing** (EgoDex within-domain, DROID/CALVIN cross-domain): 표현이 변화/행동 정보를 인코딩하는지 회귀 R²로 측정
- **로봇 조작 BC** (LIBERO BC-Transformer, CortexBench, CALVIN): frozen encoder + policy head
- **실로봇** (Paper 1 본체 lift): manipulation deploy

## 프로젝트 구조

```
├── src/
│   ├── models/          # two_stream_v15(+M-recon 분기)=CoMP-MAE/MS-JEPA, videomae 등
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
- **Paper 2 (AAAI)** — 설계: [`docs/comp_mae_plan.md`](docs/comp_mae_plan.md) · STEP 1 인과 실행: [`docs/factorization_crossover_plan.md`](docs/factorization_crossover_plan.md) · 선행 MS-JEPA: [`docs/v15b_retraining_status.md`](docs/v15b_retraining_status.md)
- **Probing**: [`docs/PROBING_GUIDE.md`](docs/PROBING_GUIDE.md)
- **LIBERO 평가**: [`docs/setup/LIBERO_TEST_GUIDE.md`](docs/setup/LIBERO_TEST_GUIDE.md)

실행 명령어·환경(클러스터/로컬)은 [`CLAUDE.md`](CLAUDE.md) 워크플로우 섹션 참조.

## 상태 (2026-07)

- **Paper 1**: P단독 image MAE > VideoMAE = 좁게 입증 → ablation(edge vs RGB) + 실로봇.
- **Paper 2**: ours = **CoMP-MAE(v16)**, S/B 학습 완료. 논문 spine = 3-claim(factorization·dissociation·효율). STEP 0 완료 — slope(3a) 폐기, **3b 절대 효율 생존**(~32M CoMP-MAE-S가 86M internet SSL 상회); factorization Phase A = directional 이중분리(상관). **다음 = STEP 1 인과**(M-recon `V_M/V_P` 스칼펠 + plain baseline, 2런). 배포=P-only(P+M 유해 확인).
