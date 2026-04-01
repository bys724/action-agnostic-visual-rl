# Action-Agnostic Visual RL Research Plan

**마지막 업데이트**: 2026-03-19
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
인스턴스: H100 워크스테이션 (x2, 81GB each) — 로컬 권장
epochs: 30
batch_size: 32 (per-GPU, H100 최적)
max_gap: 30 (~1초), sample_decay: 0.3, loss_decay: 0.7
데이터: EgoDex frames (로컬: /mnt/data/egodex_frames/)
```

```bash
bash scripts/pretrain_local.sh                          # two-stream(GPU 0) + videomae(GPU 1) 병렬
bash scripts/pretrain_local.sh --model two-stream        # 특정 모델만
bash scripts/pretrain_local.sh --sanity                  # Sanity test
bash scripts/pretrain_local.sh --splits part1            # 특정 split만

# AWS EC2 (대안)
bash scripts/pretrain_aws.sh --model two-stream
```

---

## 현재 상태 (2026-03-19)

### 완료
- [x] Two-Stream + VideoMAE 구현 (Single-Stream은 ablation용, 현재 학습 제외)
- [x] TwoStreamModel 아키텍처 개편: CLS bottleneck + 이중 디코더 (trivial solution 해결)
  - 이전: patch-level PixelwiseFusion → decoder (trivial shortcut 가능)
  - 현재: (m_cls + p_cls) / 2 → decoder_current + decoder_future (CLS 강제)
- [x] VideoMAE 공식 정합성 검증
- [x] 학습 파이프라인 (pretrain.py, pretrain_local.sh, pretrain_aws.sh)
- [x] 프레임 전처리 확정 (256x256 저장 → 학습 시 RandomCrop 224, 독립 crop)
- [x] EgoDex part1, part4 로컬 추출 완료 (~515GB)
- [x] DROID v1.0.1 다운로드 완료 (2048 shards, 1.7TB)

### 완료 (추가)
- [x] Two-Stream v3 30 epoch 완료 (train loss 0.000885, eval loss 0.001870)
- [x] Two-Stream v3 + SSIM 1 epoch 검증 완료 (학습 정상 확인, 풀 학습은 보류)
- [x] VideoMAE 30 epoch 완료
- [x] DROID 프레임 추출 완료 (ext1/ext2/wrist, 95,658 에피소드)
- [x] Action probing 코드 개선 (MP4→JPG 프레임 로딩, --cls-mode 옵션)

### Action Probing 초기 결과 (2026-03-31, 500 videos, linear probe, 20 epochs)

**모델 간 공정 비교 (patch mean pool 기준)**:

| Encoder | Split | R² | Cosine Sim | 비고 |
|---------|-------|----|------------|------|
| VideoMAE | part1 (학습) | 0.416 | 0.494 | |
| Two-Stream (patch_mean) | part1 (학습) | 0.389 | 0.482 | gap 0.027 |
| Two-Stream (patch_mean_p) | part1 (학습) | 0.401 | 0.490 | P stream만 |
| Two-Stream (patch_mean) | part4 (미사용) | 0.237 | 0.369 | 일반화 |
| VideoMAE | part4 (미사용) | 0.137 | 0.330 | 일반화 |

**Two-Stream CLS 방식별 비교 (part1)**:

| 방식 | R² | 설명 |
|------|-----|------|
| CLS average | 0.326 | (m_cls + p_cls) / 2 |
| CLS concat | 0.157 | [m_cls; p_cls] |
| CLS m_only | 0.153 | M stream CLS |
| CLS p_only | 0.169 | P stream CLS |
| patch_mean (M+P) | 0.389 | 전체 패치 mean pool |
| patch_mean_m | 0.184 | M 패치만 |
| patch_mean_p | 0.401 | P 패치만 |

**해석**:
1. patch mean pool 기준 VideoMAE와 Two-Stream 격차가 작음 (0.416 vs 0.389)
2. **미사용 데이터(part4)에서 Two-Stream이 우세** (0.237 vs 0.137) → 일반화 성능
3. CLS average(768-dim 1개)로 patch mean(196개 평균)의 83% 달성 → CLS exchange 효과
4. P stream > M stream: 외형 정보가 hand pose probing에 더 유리
5. 모든 조건에서 R² < 0.7 — 데이터 규모 확대/MLP probe 등으로 개선 여지

### Gap별 Embedding 비교 (part4, 미사용 데이터, 2026-04-01)

| Embedding | dim | gap=1 | gap=5 | gap=10 |
|-----------|-----|-------|-------|--------|
| TS CLS average | 768 | 0.159 | 0.354 | 0.364 |
| TS CLS concat | 1536 | -0.150 | 0.330 | 0.353 |
| TS m_only (CLS) | 768 | 0.046 | 0.276 | 0.359 |
| TS p_only (CLS) | 768 | -0.266 | 0.311 | 0.325 |
| TS patch_mean | 768 | 0.225 | 0.466 | 0.532 |
| **TS patch_mean_concat** | **1536** | **0.117** | **0.489** | **0.585** |
| VM patch_mean | 768 | 0.138 | 0.474 | 0.571 |

**추가 해석**:
1. **gap 효과**: gap=1의 delta는 노이즈 수준. gap=10에서 전반적으로 R²가 2~4배 상승
2. **M stream의 temporal 특성**: gap=1 최하위(0.046) → gap=10 P를 추월(0.359 > 0.325). M/P 분리 설계가 의도대로 작동
3. **patch_mean_concat(0.585) > VideoMAE(0.571)**: M/P 분리 보존 시 Two-Stream이 역전
4. **CLS concat < CLS average**: CLS exchange로 이미 동질화되어 concat이 차원만 증가. CLS는 average 방식이 적합
5. **CLS average(0.364)는 patch_mean(0.532)의 68%**: CLS bottleneck에 개선 여지

### 진행 중
- [ ] Architecture ablation 학습 (A: d=6,s=3 / B: d=6,s=2 / C: d=4,s=2)
  - A, B: epoch 1 거의 완료, C는 A/B 완료 후 자동 시작 예정
- [ ] EgoDex part2, 3, 5 CDN 다운로드 + 추출

### 다음 단계
- [ ] Ablation A/B/C probing 비교 → depth vs exchange 효과 분리
- [ ] DROID action probing (cross-domain, 로봇 7-DoF velocity)
- [ ] Full training 설정 확정 → 본 학습 시작

---

## Architecture Ablation (진행 중)

**목적**: depth(표현력)와 CLS exchange 빈도의 효과 분리

| Config | depth | stages | blk/stage | exchange | Params |
|--------|-------|--------|-----------|----------|--------|
| 기존 | 12 | 3 | 4 | 3 | 193M |
| A | 6 | 3 | 2 | 3 | 135M |
| B | 6 | 2 | 3 | 2 | 128M |
| C | 4 | 2 | 2 | 2 | 100M |

- **A vs B** (같은 depth=6): exchange 빈도 효과 (3회 vs 2회)
- **A vs C** (같은 exchange=2회): depth 효과 (6 vs 4) — B와 C 비교

---

## Ablation Study (Phase 1 이후)

| 변형 | 설명 |
|-----|------|
| M-only | P 채널 제거 |
| P-only | M 채널 제거 |
| No Exchange | CLS 교환 없이 concat만 |

---

## 아이디어 후보 (검증 필요)

### Cross-Stream Masked Reconstruction Loss

**아이디어**: MAE 스타일의 마스킹 loss를 Two-Stream에 추가. M/P stream에서 서로 다른 위치를 마스킹하여 cross-stream 정보 교환 강화.

**동기**:
- 현재 CLS average가 patch_mean의 68% 수준 (0.364 vs 0.532) → CLS의 정보 밀도 부족
- 마스킹이 CLS에 더 구체적인 spatial info를 압축하도록 유도할 수 있음

**예상 작동 방식**:
- M stream: 위치 {1,5,9} 마스킹, P stream: 위치 {3,7,11} 마스킹
- 각 stream은 자신의 마스킹 부분을 복원할 때 (1) 주변 패치 context, (2) CLS 경유 cross-stream 정보 활용
- 기존 future prediction loss와 병행

**기대 효과**:
1. 각 stream 내 local spatial reasoning 강화 (확실)
2. CLS의 spatial info 밀도 향상 → CLS probing 성능 개선 (가능성)
3. M/P 표현의 complementarity 강화 (가능성)

**한계/주의점**:
- 현재 cross-stream 정보는 CLS exchange만 통과 (patch-level cross-attention 없음)
  → P의 특정 패치가 M의 같은 위치 복원을 "직접" 돕기 어려움
- 이미 future prediction이라는 어려운 task가 있는데 masking 추가 시 학습 난이도 증가
- 학습 안정성, loss 가중치 밸런싱 필요

**마스킹 비율 가이드** (선행 연구 참고):
- MAE: 75% (이미지, 높은 마스킹이 trivial shortcut 방지)
- VideoMAE: 90~95% (비디오 시간축 중복이 크므로 더 공격적)
- iBOT(DINOv2): ~50% (discriminative라 teacher 신호 보존 필요)
- **우리 모델 권장 시작점: 30~50%**
  - 마스킹이 보조 loss (메인은 future prediction)이므로 MAE만큼 공격적일 필요 없음
  - 실험 조건: 0% (baseline), 30%, 50% → 경향 확인

**3단계 랜덤성** (마스킹 포함 시 모델의 고유한 특성):
1. Spatial: RandomCrop(224) — 매번 다른 영역
2. Temporal: gap=1~30 — 매번 다른 시간 간격
3. Structural: M/P 서로 다른 패치 마스킹 — 매번 다른 정보 가림
→ 동일 데이터도 매번 다른 문제를 풀게 되어 overfitting 방지 + robustness 향상 기대.
   단, 랜덤성 과다 시 학습 신호가 noisy해져 수렴 저하 가능 → 비율 튜닝 중요.

**실행 조건**: Architecture ablation + full training 설정 확정 후. 현재 변수가 많은 상태에서 추가하면 효과 분리 불가.

**참고**: DINOv2의 iBOT loss도 masked patch prediction이지만 pixel reconstruction이 아닌 feature-level distillation (cross-entropy in feature space). 우리는 pixel target이 있으므로 MAE 방식(MSE in pixel space)이 더 직접적.

---

## 참고 문서

- [`docs/AWS_INSTANCE_GUIDE.md`](AWS_INSTANCE_GUIDE.md) — AWS 환경 설정
- [`docs/PROBING_GUIDE.md`](PROBING_GUIDE.md) — Action probing 계획
