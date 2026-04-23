# Action Probing Guide

## 실험 개요

**평가 전략 (2026-04-14 재편)**: Two-Stream과 VideoMAE-ours만 EgoDex로 학습하므로 EgoDex within-domain probing은 **controlled comparison 전용**(Two-Stream vs VideoMAE-ours). 공개 가중치(VC-1, DINOv2, SigLIP, VideoMAE-official, V-JEPA-official)와의 main 비교는 OOD인 **DROID에서 수행**.

| 평가 | 비교 대상 | 역할 |
|------|----------|------|
| EgoDex probing | Two-Stream vs VideoMAE-ours (둘 다 EgoDex 학습) | 구조적 bias 기여 sanity |
| **DROID probing (main)** | 상기 2개 + VideoMAE-official, V-JEPA-official, VC-1, DINOv2, SigLIP | Cross-encoder fair comparison |

**핵심 주장**:
> "Action-agnostic pretraining으로 학습한 변화 임베딩은 action-informative하다. 따라서 VLM은 임베딩 공간에서 sequence modeling만으로 실질적 action planning이 가능하다."

### Probing 프로토콜

- **Linear probe**: frozen encoder → linear layer → action 회귀
- **Encoder frozen**: backbone 가중치 고정, probe만 학습
- **Input**: `(img_t, img_{t+gap})` 2-frame pair
- **Target**:
  - EgoDex: hand pose delta (within-domain)
  - DROID: robot 7-DoF joint velocity (cross-domain, Franka)
- **Epochs**: 20, **Gap**: 10 (기본)

### Dataset Split

- **EgoDex**: train = part1~3 (학습 미사용 보장), **test = part4** (180K train / 40K eval pair)
- **DROID**: 95,658 에피소드, ext1 카메라 기본 사용, 256x256 리사이즈 (180x320 원본, crop 없음)

## 실행 방법

### EgoDex Probing (within-domain)

```bash
python scripts/eval/probe_action.py \
    --encoder two-stream \
    --checkpoint /mnt/data/checkpoints/two_stream/.../checkpoint_epochXXXX.pt \
    --egodex-root /mnt/data/egodex \
    --frames-root /mnt/data/egodex_frames \
    --egodex-split part4 \
    --cls-mode patch_mean_concat \
    --gap 10 --epochs 20

# VideoMAE
python scripts/eval/probe_action.py --encoder videomae \
    --checkpoint <ckpt> --egodex-split part4 --gap 10

# Baseline
python scripts/eval/probe_action.py --encoder dinov2 --egodex-split part4 --gap 10
```

### DROID Probing (cross-domain, primary)

```bash
python scripts/eval/probe_action_droid.py \
    --encoder two-stream --checkpoint <ckpt> \
    --droid-root /mnt/data/droid_frames/ext1 \
    --gap 10 --epochs 20
```

### 클러스터 실행

```bash
sbatch scripts/cluster/probe_action.sbatch  # sbatch launcher
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--cls-mode` | embedding 추출 방식 | average |
| `--gap` | 프레임 간격 | 10 |
| `--egodex-split` | 데이터 파티션 | part4 (test) |
| `--max-videos` | 비디오 수 제한 (디버깅) | None (전체) |

### cls-mode 선택 가이드

| cls-mode | 차원 | 용도 |
|----------|------|------|
| `average` | 768 | CLS (m+p)/2, compact representation |
| `concat` | 1536 | CLS [m;p], stream 분리 비교용 |
| `m_only` | 768 | M stream CLS만, temporal 분석 |
| `p_only` | 768 | P stream CLS만, spatial 분석 |
| `patch_mean` | 768 | M+P 패치 전체 mean pool |
| **`patch_mean_concat`** | **1536** | **M/P 패치 각각 mean → concat (기본 권장)** |
| `patch_mean_m` | 768 | M 패치만 mean pool (stream 진단) |
| `patch_mean_p` | 768 | P 패치만 mean pool (stream 진단) |

## Probing 결과 (EgoDex part4, gap=10, linear probe, 20ep)

### 활성 모델 lineup

| 모델 | 체크포인트 | cls_mode | R² | 비고 |
|------|-----------|----------|-----|------|
| **v6 (APE + rotaug)** | ep8 | patch_mean_concat | **+0.259** | 현재 챔피언 |
| **VideoMAE-ours** | ep50 | patch_mean | **+0.326** | 수렴 (ep28 +0.317) |
| v4 (RoPE) | ep48 | patch_mean_concat | +0.197 | 정체 |
| Two-Stream v10 (v6 base + mask_p 0.75) | ep4 | patch_mean_concat | +0.195 | v9 lineup 추월 |
| Two-Stream v10 | ep8 | patch_mean_concat | **+0.206** | 1차 peak |
| Two-Stream v10 | ep12 | patch_mean_concat | +0.148 | collapse 시작 |
| Two-Stream v10 | ep16 | patch_mean_concat | +0.144 | collapse 확인 |
| Two-Stream v10 | ep20 | patch_mean_concat | +0.137 | 단조 하락 |
| Two-Stream v10 | ep24 | patch_mean_concat | **+0.202** | **W-shape 회복** (ep8 peak 근접) |
| DINOv2 (frozen) | — | CLS concat | (ceiling 참조) | 공개 weight |
| Random-init | — | — | (floor) | 구조적 prior 측정 |

### v10 stream-별 추세 (`patch_mean_{m,p}`)

| Epoch | M | P |
|-------|---|---|
| ep4   | +0.176 | +0.126 |
| ep8   | +0.150 | **+0.152** (peak) |
| ep12  | +0.129 | +0.083 |
| ep16  | +0.125 | +0.038 (sparse pinpoint viz) |
| ep20  | +0.135 | +0.022 |
| ep24  | +0.138 | **+0.092** (peak의 60% 회복) |

### v10 (mask_p 0.75) 분석

- **1차 peak ep8 +0.206**: v6 ep8 +0.259 대비 -0.053. mask_p 0.75는 ep4-ep8까지 P 학습 강화 효과 확인 (P ep8 +0.152 > v6 ep4 수준)
- **ep8→ep20 collapse**: P가 +0.152 → +0.022로 단조 하락, attention viz에서 sparse pinpoint 회귀 (v8 static salience 패턴 재현 신호)
- **ep20→ep24 W-shape 회복**: concat +0.137 → +0.202 (+0.065 점프), P +0.022 → +0.092. ep8 peak 근접. **LR cosine decay 후반 효과 추정** (BYOL/SimSiam 후반 안정화 사례와 일치)
- **결론 (ep28 결과 대기 중)**: 회복 진위 미확정. ep28에서 추세 지속 시 v6 챔피언 추월 가능성, single-epoch noise면 다시 하락

### cls_mode 비교 (v4 ep48 기준)

| cls_mode | dim | R² | Cosine Sim |
|----------|-----|----|------------|
| patch_mean_concat | 1536 | **0.197** | 0.236 |
| concat (CLS) | 1536 | 0.177 | 0.260 |
| patch_mean | 768 | 0.164 | 0.209 |
| average (CLS) | 768 | 0.052 | 0.197 |

**순위 패턴**: patch_mean_concat > concat > patch_mean > average. CLS average는 사실상 무용 (0.052).

### 해석

1. **Rotation augmentation이 결정적**: v5(rotaug 없음) ep4→ep8 하락(0.208→0.192) vs v6(rotaug) ep8 0.259. Position prior 의존을 차단하면 표현 품질이 개선됨
2. **patch_mean > CLS**: CLS는 cls_exchange로 혼합되어 정보 희석. Patch level에서 M/P는 서로 다른 정보를 담고 있어 probe가 독립적으로 활용 가능
3. **Two-Stream vs VideoMAE**: ep4에서 거의 동등(0.208 vs 0.198). 이후 학습 효율에서 차이 — VideoMAE 수렴(→0.326), Two-Stream은 rotaug/mask 설정에 민감
4. **Gap 효과**: gap=1은 노이즈 수준(~0.0006). full training 후 평가는 **gap=10 기본**

## 권장 평가 프로토콜

```bash
# 1. 공정 비교: part4 (미사용), gap=10, patch_mean_concat
#    → Two-Stream vs VideoMAE-ours 비교

# 2. Stream 진단: cls_mode={patch_mean_m, patch_mean_p}
#    → M/P 각각의 action-informativeness 분리 측정

# 3. DROID cross-domain: probe_action_droid.py 동일 프로토콜
#    → 공개 weight 포함 main comparison
```

## 다음 단계

1. **v10 ep28 probing** — ep24 W-shape 회복 진위 판별 (single-epoch noise vs LR decay 후반 회복)
2. v10 ep32+ 추가 — 회복 추세 지속 여부 (v6 +0.259 추월 가능성)
3. DROID 프레임 추출 완료 → Phase 2 DROID probing 개시
4. 공개 weight lineup (VC-1, DINOv2, SigLIP, VideoMAE-official, V-JEPA-official) DROID 평가