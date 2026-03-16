# Action-Agnostic Visual RL Research Plan

**마지막 업데이트**: 2026-03-16
**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?

---

## 실험 로드맵 (단계별 Go/No-Go)

### Phase 1: EgoDex 사전학습 → Action Probing (EgoDex)

**목표**: 세 모델의 표현 품질 비교 (hand pose linear probe)
**비용**: ~$250-700 (part1, 30ep, g5.12xlarge)

1. EgoDex part1로 3개 모델 사전학습 (AWS g5.12xlarge)
2. 인코더 freeze → EgoDex test에서 hand pose linear probe
3. 기대 결과: **Two-Stream > Single-Stream > VideoMAE**

**Go/No-Go**:
- YES (순서 맞음) → 구조 효과 확인. Phase 1.5로
- NO → 모델 구조 또는 학습 디버깅 필요

### Phase 1.5: 데이터 스케일업 (Go 시에만)

**목표**: 전체 EgoDex로 표현 품질 극대화
**비용**: ~$2,000-3,000 (part1~5, 50+ep)
**전제**: Phase 1에서 모델 구조 효과가 검증된 상태

1. EgoDex part1~5 전체로 best 모델 재학습 (epoch 확대)
2. 같은 hand pose probe로 Phase 1 대비 성능 향상 확인

**Go/No-Go**:
- 스케일업 효과 확인 → 이 체크포인트로 Phase 2 진행
- 효과 미미 → part1 체크포인트로 Phase 2 진행 (데이터 추가 불필요)

### Phase 2: Action Probing (DROID + Bridge V2)

**목표**: EgoDex 표현의 로봇 도메인 전이 가능성 검증

1. 같은 인코더로 DROID에서 로봇 행동 linear probe (**primary**)
2. Bridge V2로 cross-embodiment 추가 검증 (secondary, 필요 시)
3. Baseline 비교: DINOv2, CLIP, R3M

**데이터셋 역할**:
- **DROID** (95k ep, Franka): 주력 probing 데이터. 규모·다양성 최대
- **Bridge V2** (25k traj, WidowX): 로봇 다양성 보강용. 리뷰어 요청 시 추가

**Go/No-Go**:
- YES (baseline 대비 competitive) → cross-domain transfer 가능. Phase 3A로
- NO (도메인 갭 확인) → 실패는 아님. Phase 3B로

### Phase 3A: LIBERO Fine-tuning (EgoDex-only 사전학습)

**목표**: EgoDex 인코더로 로봇 조작 직접 평가

1. EgoDex 사전학습 인코더로 LIBERO fine-tune
2. Baseline 비교: scratch, ImageNet pretrained

**Go/No-Go**:
- YES → 논문 메인 결과
- NO → Phase 3B 필요

### Phase 3B: 혼합 사전학습 (필요 시에만)

Phase 3A 실패 시 실행. 3가지 사전학습 조건 비교:

| 사전학습 데이터 | Probing (Bridge) | LIBERO |
|---------------|-----------------|--------|
| EgoDex only | Phase 2 결과 | Phase 3A 결과 |
| Bridge only | 새로 실험 | 새로 실험 |
| EgoDex + Bridge | 새로 실험 | 새로 실험 |

**원칙**: "EgoDex만으로 충분한가?"를 먼저 증명하고, 부족할 때만 데이터 추가.
처음부터 혼합하면 기여 분리 불가.

---

## 모델

| 모델 | 학습 방식 | 목적 |
|-----|----------|------|
| Two-Stream | Future prediction (M/P 채널 분리 + CLS 교환) | 제안 모델 |
| Single-Stream | Future prediction (9ch 단일 ViT) | Baseline (구조 효과 검증) |
| VideoMAE | Masked reconstruction (75%, 공식 Base 설정) | Comparison |

### 비교 의미
- Single-Stream vs VideoMAE → 학습 방식 차이 (prediction vs reconstruction)
- Two-Stream vs Single-Stream → 같은 M/P 입력, 구조 차이 (interleaved vs single)

---

## 학습 설정

```
인스턴스: g5.12xlarge (4x A10G, 48 vCPU, 192GB RAM)
epochs: 30
batch_size: 24 (per-GPU) → effective: 96
max_gap: 10, sample_decay: 0.3, loss_decay: 0.7
데이터: EgoDex frames (S3: egodex_frames_partN/)
```

```bash
bash scripts/pretrain_aws.sh                          # 3개 모델 순차
bash scripts/pretrain_aws.sh --model two-stream        # 특정 모델만
bash scripts/pretrain_aws.sh --sanity --no-shutdown     # Sanity test
TRAIN_PARTS=part1,part2,part3 bash scripts/pretrain_aws.sh  # 여러 part
```

---

## 현재 상태 (2026-03-16)

### 완료
- [x] Three-model 구현 + VideoMAE 공식 정합성 검증
- [x] 학습 파이프라인 (pretrain.py, pretrain_aws.sh)
- [x] 프레임 전처리 확정 (256x256 저장 → 학습 시 RandomCrop 224)
- [x] Bridge V2 프레임 S3 업로드 완료 (24,827 traj)
- [x] EgoDex part1~5, test 프레임 S3 업로드 완료
- [x] AWS 스크립트 S3 프레임 경로 반영 + 멀티 split 지원
- [x] 학습 코드 점검: scheduler resume 버그 수정, TensorBoard 로깅 추가
- [x] DROID 데이터셋 다운로드 완료 (v1.0.1, 95,658 ep, 1.7TB)

### 진행 중
- [ ] AWS Phase 1 사전학습 실행 (part1, 30ep)
- [ ] DROID 프레임 추출 (3카메라 × 256x256, ~12h)
- [ ] EgoDex tarball 변환 (part1 진행 중 → part4 → 나머지 순차)

### 다음 단계
- [ ] Phase 1 결과 분석 → Go/No-Go 판단
- [ ] DROID action probing 코드 작성

---

## Ablation Study (Phase 1 이후)

| 변형 | 설명 |
|-----|------|
| M-only | P 채널 제거 |
| P-only | M 채널 제거 |
| No Exchange | CLS 교환 없이 concat만 |

---

## 참고 문서

- [`docs/AWS_INSTANCE_GUIDE.md`](AWS_INSTANCE_GUIDE.md) — AWS 환경 설정
- [`docs/PROBING_GUIDE.md`](PROBING_GUIDE.md) — Action probing 계획
