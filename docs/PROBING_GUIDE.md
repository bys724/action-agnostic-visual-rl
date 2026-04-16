# Action Probing Guide

## 실험 개요

**평가 우선순위 (2026-04-14 재편)**: Two-Stream과 VideoMAE-ours만 EgoDex로 학습하므로 EgoDex within-domain probing은 **controlled comparison 전용**(Two-Stream vs VideoMAE-ours). 공개 가중치(VC-1, DINOv2, SigLIP, VideoMAE-official, V-JEPA-official)와의 main 비교는 모두 OOD인 **DROID에서 수행**.

| 평가 | 비교 대상 | 역할 |
|------|----------|------|
| EgoDex probing | Two-Stream vs VideoMAE-ours (둘 다 EgoDex 학습) | 구조적 bias 기여 sanity (축 1) |
| **DROID probing (main)** | 상기 2개 + VideoMAE-official, V-JEPA-official, VC-1, DINOv2, SigLIP | Cross-encoder fair comparison |

**핵심 주장**:
> "Action-agnostic pretraining으로 학습한 변화 임베딩은 action-informative하다. 따라서 VLM은 임베딩 공간에서만 sequence modeling을 수행하면서도 실질적으로 action planning을 할 수 있다."

### 3-Stage Framework

```
Phase 1: Action-Agnostic Pretraining
  ├─ Input: (img_prev, img_curr) - 과거와 현재
  ├─ Encoder: 변화를 압축 → embedding
  ├─ Decoder: img_prev + embedding → img_curr 예측
  └─ 의미: Decoder는 변화 정보로 미래를 생성하는 법 학습

Phase 2: Action Probing (Linear Evaluation)
  ├─ EgoDex Probing (within-domain)
  │   ├─ Input: (img_t, img_t+1) - 관찰된 변화
  │   ├─ Frozen Encoder → embedding
  │   ├─ Linear Probe → action_t (변화를 만든 action)
  │   └─ 증명: 임베딩이 action-informative함
  │
  ├─ DROID Probing (cross-domain, primary)
  │   ├─ Input: Same frozen encoder
  │   ├─ Output: Robot actions (7-DoF, Franka)
  │   └─ 증명: 로봇 도메인으로 cross-domain transfer 가능
  │
  └─ Bridge Probing (cross-embodiment, secondary)
      ├─ Input: Same frozen encoder
      ├─ Output: Robot actions (7-DoF, WidowX)
      └─ 증명: 다른 로봇에도 transfer 가능 (필요 시 추가)

Phase 3: VLM/VLA Training (Future)
  ├─ Encoder: (img_prev, img_curr) → embedding_t
  ├─ VLM: embedding_t → embedding_t+1 (미래 변화)
  └─ Action Decoder: embedding_t+1 → action_t
```

### Probing의 의미

**Without Probing**:
- Encoder가 변화를 압축한다 ✓
- 하지만 임베딩 ↔ action 연결 가능한지 미지수 ❓

**With Probing (R² > 0.7)**:
- 임베딩 → action 복원 가능 ✓
- 따라서: VLM이 임베딩만 다루면서도 action planning 가능 ✓
- Cross-embodiment: 하나의 encoder로 여러 embodiment 지원 ✓

## Phase 1: Pretraining

### 학습 실행

```bash
# Two-Stream v4 (확정 설정)
python scripts/pretrain.py --model two-stream \
    --depth 12 --num-stages 2 \
    --mask-ratio 0.3 --mask-ratio-p 0.5 \
    --max-gap 60 --sample-dist triangular --sample-center 30 \
    --epochs 30

# VideoMAE baseline
python scripts/pretrain.py --model videomae --epochs 30
```

## Phase 2: Action Probing

### 2.1 EgoDex Probing (Within-Domain)

```bash
# Docker 컨테이너에서
docker exec -it simpler-eval bash

# Two-Stream probing
python scripts/probe_action.py \
    --encoder two-stream \
    --checkpoint /workspace/data/checkpoints/two_stream/best_model.pt \
    --egodex-root /workspace/data/egodex \
    --epochs 20 \
    --probe linear

# Single-Stream probing
python scripts/probe_action.py \
    --encoder single-stream \
    --checkpoint /workspace/data/checkpoints/single_stream/best_model.pt \
    --egodex-root /workspace/data/egodex \
    --probe linear

# VideoMAE probing
python scripts/probe_action.py \
    --encoder videomae \
    --checkpoint /workspace/data/checkpoints/videomae/best_model.pt \
    --egodex-root /workspace/data/egodex \
    --probe linear

# Baseline: CLIP
python scripts/probe_action.py --encoder clip --probe linear

# Baseline: DINOv2
python scripts/probe_action.py --encoder dinov2 --probe linear
```

**출력 예시**:
```
RESULTS
==================================================
Encoder:    two-stream
Probe:      linear
R²:         0.8234  PASS (threshold: 0.7)
MSE:        0.000142
Cosine Sim: 0.9112

Per-joint R²:
  rightHand                    : 0.8521
  rightThumbTip                : 0.7892
  rightIndexFingerTip          : 0.8334
  ...
```

### 2.2 DROID Probing (Cross-Domain, Primary)

```bash
# Two-Stream on DROID
python scripts/probe_action_droid.py \
    --encoder two-stream \
    --checkpoint /workspace/data/checkpoints/two_stream/best_model.pt \
    --droid-frames /workspace/data/droid_frames \
    --epochs 20 \
    --probe linear

# Baselines
python scripts/probe_action_droid.py --encoder clip --probe linear
python scripts/probe_action_droid.py --encoder dinov2 --probe linear
```

DROID 데이터:
- 3카메라 (ext1, ext2, wrist) → ext1을 기본 사용
- 95,658 에피소드, Franka Panda, 7-DoF joint velocity
- 프레임: 256x256 리사이즈 (crop 없음, 180x320 원본)

### 2.3 Bridge Probing (Cross-Embodiment, Secondary)

DROID probing 결과가 충분하면 생략 가능. 리뷰어가 로봇 다양성을 요구할 경우 추가.

```bash
# Two-Stream on Bridge
python scripts/probe_action_bridge.py \
    --encoder two-stream \
    --checkpoint /workspace/data/checkpoints/two_stream/best_model.pt \
    --bridge-root /workspace/data/datasets/bridge_v2 \
    --epochs 20 \
    --probe linear
```

## 초기 실험 결과 (2026-03-31, d=12 s=3, 30ep, 500 videos)

### Baseline 비교 (part4, 미사용 데이터, gap=1)

| Encoder | Embedding | R² | Cosine Sim |
|---------|-----------|-----|------------|
| Two-Stream | patch_mean | 0.249 | 0.363 |
| Two-Stream | CLS average | 0.143 | 0.303 |
| VideoMAE | patch_mean | 0.090 | 0.301 |
| CLIP | CLS concat | -1.744 | 0.137 |
| DINOv2 | CLS concat | -99.154 | 0.008 |

EgoDex 학습 모델 >> pretrained baseline. CLIP/DINOv2는 hand pose delta에 무용.

### Gap별 Embedding 비교 (part4, 미사용 데이터)

| Embedding | dim | gap=1 | gap=5 | gap=10 |
|-----------|-----|-------|-------|--------|
| TS CLS average | 768 | 0.159 | 0.354 | 0.364 |
| TS CLS concat | 1536 | -0.150 | 0.330 | 0.353 |
| TS m_only (CLS) | 768 | 0.046 | 0.276 | 0.359 |
| TS p_only (CLS) | 768 | -0.266 | 0.311 | 0.325 |
| TS patch_mean | 768 | 0.225 | 0.466 | 0.532 |
| **TS patch_mean_concat** | **1536** | **0.117** | **0.489** | **0.585** |
| VM patch_mean | 768 | 0.138 | 0.474 | 0.571 |

### 해석

**1. Gap 효과**: gap=1의 delta는 노이즈 수준 (mean ~0.0006). gap이 커질수록 의미 있는 변화를 포착하게 되어 R²가 크게 상승. full training 후 평가 시 **gap=5, 10을 기본으로 사용**할 것.

**2. M stream의 temporal 특성**: gap=1에서 M은 최하위(0.046)지만 gap=10에서는 P를 추월(0.359 > 0.325). M channel이 temporal change를 인코딩하므로, 변화량이 작을 때는 정보가 빈약하지만 충분한 간격에서는 motion 정보가 드러남. **M/P 분리 설계가 의도대로 작동**한다는 근거.

**3. patch_mean_concat > patch_mean**: M/P를 분리 보존한 concat(0.585)이 혼합 mean(0.532)보다 우수. Patch level에서 M/P는 서로 다른 정보를 담고 있어서 probe가 독립적으로 활용 가능.

**4. CLS concat < CLS average**: CLS는 exchange를 거치며 이미 비슷해져서 concat이 차원만 늘림. CLS average가 768-dim으로 compact하면서 효율적 → VLA에서 CLS를 representation으로 쓸 때 average 방식이 적합.

**5. Two-Stream vs VideoMAE**: patch_mean 기준 VideoMAE가 약간 우세(0.571 vs 0.532)하지만, patch_mean_concat 시 Two-Stream이 역전(0.585 vs 0.571). M/P 분리 구조의 이점이 적절한 embedding 추출 방식에서 드러남.

**⚠️ 주의**: 위 결과는 500 videos 소규모 평가. Full test split에서 수치가 크게 달라짐 (아래 참고).

---

## Full Test Split 결과 (2026-04-16, 클러스터, part4 test split 전체)

이전 500-video 실험의 과대추정 문제를 확인하기 위해 full test split (180K train / 40K eval)으로 재평가.
모든 실험: linear probe, gap=10, 20 epochs.

### Two-Stream v4 (ep48, RoPE) cls_mode 비교

| cls_mode | dim | R² | Cosine Sim | 초기 500-video R² |
|----------|-----|----|------------|------------------|
| patch_mean_concat | 1536 | **0.197** | 0.236 | 0.585 |
| concat (CLS) | 1536 | 0.177 | 0.260 | 0.353 |
| patch_mean | 768 | 0.164 | 0.209 | 0.532 |
| average (CLS) | 768 | 0.052 | 0.197 | 0.364 |

순위 패턴 동일 (patch_mean_concat > concat > patch_mean > average), 하지만 전반적으로 500-video 대비 대폭 하락. CLS average는 사실상 무용 (0.052).

### Epoch 4 동일 조건 비교 (구조적 차이만 분리)

| 모델 | Checkpoint | R² | Cosine Sim |
|------|-----------|-----|------------|
| Two-Stream v5 (APE + mask 0.5/0.5) | ep4 | **0.208** | 0.251 |
| VideoMAE-ours (mask 0.5) | ep4 | 0.198 | 0.233 |

**ep4에서는 거의 동일** — 구조 자체의 초기 표현력은 비슷.

### 학습 진행에 따른 R² 변화

| 모델 | ep4 R² | ep28 R² | ep48 R² | 추세 |
|------|--------|---------|---------|------|
| Two-Stream v5 | 0.208 | 0.192 ↓ | — | CANCELLED (v6으로 대체) |
| Two-Stream v4 | ~0.2* | — | 0.197 | 정체/미세 하락 |
| VideoMAE-ours | 0.198 | **0.317** | (학습 중) | **+60% 상승** |

*v4 ep4 직접 측정 없음, APE diagnostic ep4 (0.219)로 추정

### 해석

**1. 초기 500-video 결과는 과대추정**: 두 모델 모두 full test split에서 R²가 크게 하락. 500 videos의 제한된 분포에서 probe가 overfit한 결과였음.

**2. 구조 차이보다 학습 동역학이 결정적**: ep4에서는 Two-Stream(0.208) ≈ VideoMAE(0.198)로 거의 동일. 그러나 VideoMAE는 ep4→28에서 R²가 0.198→0.317로 60% 상승한 반면, Two-Stream은 ep4→48에서 정체 (0.2 → 0.197).

**3. M/P 전처리가 재구성 task를 과도하게 단순화**: Two-Stream reconstruction loss가 ep3~4에서 0.002로 포화 (VideoMAE는 ep28에서도 0.23). Task가 쉬우니 표현 학습 압력이 약해져 probing 성능이 조기 포화. mask ratio 0.3→0.5, RoPE→APE 변경(v5)에도 수렴 패턴 동일.

**4. Position prior overfit 가설**: v5 ep4(0.208)→ep8(0.192)에서 R²가 소폭 하락. v4도 ep4~ep48 동안 정체(~0.197). 학습이 진행될수록 모델이 EgoDex 프레임의 전형적 구도(position prior)에 overfit하여 범용 표현이 오히려 약화되는 것으로 추정. Rotation diagnostic에서 0° 복원 퀄리티가 가장 높고 회전 시 하락하는 것이 이를 뒷받침. → **v6에서 rotation augmentation(90% 동일회전 + 10% 독립회전) 도입**하여 position prior 의존 차단. v6 초기 loss가 0.015대로 v5 동시점(0.005) 대비 3배 높아 task 난이도 상승 확인됨. ep4→ep8에서 R² 상승 추세 전환이 핵심 검증 포인트.

## 실행 방법

### 기본 사용

```bash
# Docker 컨테이너 내에서 실행
# Two-Stream probing (다양한 embedding/gap 조합)
python scripts/eval/probe_action.py \
    --encoder two-stream \
    --checkpoint /mnt/data/checkpoints/two_stream/.../best_model.pt \
    --egodex-root /mnt/data/egodex \
    --frames-root /mnt/data/egodex_frames \
    --egodex-split part4 \
    --cls-mode patch_mean_concat \
    --gap 10 \
    --max-videos 500 --epochs 20

# VideoMAE
python scripts/eval/probe_action.py \
    --encoder videomae \
    --checkpoint /mnt/data/checkpoints/videomae/.../best_model.pt \
    --egodex-root /mnt/data/egodex \
    --frames-root /mnt/data/egodex_frames \
    --egodex-split part4 \
    --gap 10 \
    --max-videos 500 --epochs 20

# Baseline (CLIP, DINOv2)
python scripts/eval/probe_action.py --encoder clip ...
python scripts/eval/probe_action.py --encoder dinov2 ...
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--cls-mode` | embedding 추출 방식 | average |
| `--gap` | 프레임 간격 | 1 |
| `--egodex-split` | 데이터 파티션 (part1~5) | part1 |
| `--depth` | Two-Stream depth (ablation용) | 12 |
| `--num-stages` | CLS exchange 횟수 (ablation용) | 3 |
| `--max-videos` | 비디오 수 제한 (디버깅/간이실험) | None (전체) |

### cls-mode 선택 가이드

| cls-mode | 차원 | 용도 |
|----------|------|------|
| `average` | 768 | CLS (m+p)/2, compact representation |
| `concat` | 1536 | CLS [m;p], stream 분리 비교용 |
| `m_only` | 768 | M stream CLS만, temporal 분석 |
| `p_only` | 768 | P stream CLS만, spatial 분석 |
| `patch_mean` | 768 | M+P 패치 전체 mean pool |
| `patch_mean_concat` | 1536 | M/P 패치 각각 mean → concat (최고 성능) |
| `patch_mean_m` | 768 | M 패치만 mean pool |
| `patch_mean_p` | 768 | P 패치만 mean pool |

### Full training 후 권장 평가 프로토콜

```bash
# 1. 공정 비교: part4 (미사용), gap=10, patch_mean_concat
#    → Two-Stream vs VideoMAE vs baseline 비교

# 2. Ablation: 같은 조건에서 cls-mode별 비교
#    → M/P 분리 효과, CLS vs patch 비교

# 3. Gap sweep: gap=1,5,10,20,30
#    → M stream의 temporal 특성 분석

# 4. Architecture ablation: --depth/--num-stages 변경
#    → 구조 효과 vs 파라미터 효과 분리
```

## 다음 단계

1. Architecture ablation (d=6s=3, d=6s=2, d=4s=2) 학습 → probing 비교
2. DROID action probing (cross-domain, 로봇 7-DoF velocity)
3. Full training 후 전체 비디오로 최종 평가
