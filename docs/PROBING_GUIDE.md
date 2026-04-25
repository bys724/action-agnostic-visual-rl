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

### cls-mode 선택 가이드 (Two-Stream v6/v10, `probe_action.py`)

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

### cls-mode 선택 가이드 (Two-Stream v11, `probe_action_v11.py`)

v11은 4 위치에서 representation 추출 가능: A=M encoder, B=P encoder, D'=motion-routing 후, D=Phase 3 final.

| cls-mode | 차원 | 위치 | 용도 |
|----------|------|------|------|
| `cls_m_enc` | 768 | A | M encoder CLS |
| `cls_p_enc` | 768 | B | P encoder CLS |
| `cls_concat_enc` | 1536 | A+B | M/P CLS concat |
| `patch_mean_m_enc` | 768 | A | M encoder patches mean (motion-biased task에 강력) |
| `patch_mean_p_enc` | 768 | B | P encoder patches mean (단독은 약함) |
| `patch_mean_concat_enc_only` | 1536 | A+B | M+P enc patches concat |
| `patch_mean_p_state_after_routing` | 768 | D' | motion-routing × 2 후 (interpreter_2 전) |
| `patch_mean_p_features_tk` | 768 | D | Phase 3 final (interpreter_2 후) |
| **`patch_mean_concat_enc_phase3`** | **1536** | **A+D** | **M enc + P Phase 3 (★ 기본 권장, ep12 +0.219)** |
| `patch_mean_concat_enc_d_prime` | 1536 | A+D' | M enc + D' |
| `patch_mean_concat_p_enc_d_prime` | 1536 | B+D' | P enc + D' |
| `patch_mean_concat_all` | 2304 | A+B+D' | 3-way concat |

## Probing 결과 (EgoDex part4, gap=10, linear probe, 20ep)

### 활성 모델 lineup

| 모델 | 체크포인트 | cls_mode | R² | 비고 |
|------|-----------|----------|-----|------|
| **v6 (APE + rotaug)** | ep8 | patch_mean_concat | **+0.259** | 현재 챔피언 |
| **VideoMAE-ours** | ep50 | patch_mean | **+0.326** | 수렴 (ep28 +0.317) |
| v4 (RoPE) | ep48 | patch_mean_concat | +0.197 | 정체 |
| Two-Stream v10 | ep8 | patch_mean_concat | +0.206 | 1차 peak |
| Two-Stream v10 | ep20 | patch_mean_concat | +0.137 | collapse 저점 |
| Two-Stream v10 | ep24 | patch_mean_concat | +0.202 | W-shape 회복 |
| Two-Stream v10 | ep36 | patch_mean_concat | +0.214 | new peak |
| Two-Stream v10 | ep40 | patch_mean_concat | **+0.221** | **plateau 시작** |
| Two-Stream v10 | ep44 / ep48 | patch_mean_concat | +0.221 / +0.222 | plateau 지속, **v6 추월 실패 확정** |
| Two-Stream v11 | ep4 | patch_mean_concat_enc_phase3 (A+D) | +0.143 | 학습 시작 |
| Two-Stream v11 | ep8 | patch_mean_concat_enc_phase3 (A+D) | +0.194 | |
| **Two-Stream v11** | **ep12** | **patch_mean_concat_enc_phase3 (A+D)** | **+0.219** | **v10 ep40 plateau 도달, 12 epoch만에** |
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
| ep24  | +0.138 | +0.092 (peak의 60% 회복) |
| ep36  | +0.129 | +0.141 (peak의 93%) |

### v10 분석 (종료)

- **1차 peak ep8 +0.206 → ep20 collapse +0.137 → ep36 W-shape 회복 +0.214 → ep40 plateau +0.221**
- v6 ep8 챔피언 (+0.259) 추월 실패 확정. P-stream 내부 강화 방식의 한계로 결론
- LR cosine decay 후반 효과로 W-shape 회복은 진짜였으나 ceiling +0.222에서 plateau

### v11 (Motion-Guided Routing) — ep12 12-mode 비교

4 위치: A=M encoder, B=P encoder, D'=motion-routing 후, D=Phase 3 final

| Mode | ep4 | ep8 | **ep12** |
|------|-----|-----|----------|
| `patch_mean_m_enc` (A) | +0.170 | +0.176 | **+0.208** |
| `patch_mean_p_enc` (B) | -0.041 | -0.025 | 0.000 |
| `patch_mean_p_state_after_routing` (D') | +0.121 | +0.066 | +0.072 |
| `patch_mean_p_features_tk` (D) | +0.023 | +0.055 | +0.054 |
| `patch_mean_concat_enc_only` (A+B) | +0.160 | +0.168 | +0.200 |
| `patch_mean_concat_enc_phase3` (A+D) | +0.143 | +0.194 | **+0.219** ★ |
| `patch_mean_concat_enc_d_prime` (A+D') | +0.149 | +0.166 | +0.153 |
| `patch_mean_concat_p_enc_d_prime` (B+D') | +0.135 | +0.011 | +0.076 |
| `patch_mean_concat_all` (A+B+D') | +0.114 | +0.094 | +0.178 |
| `cls_m_enc` (A CLS) | +0.066 | +0.155 | +0.162 |
| `cls_p_enc` (B CLS) | -0.059 | -0.011 | -0.008 |
| `cls_concat_enc` (A+B CLS) | -0.048 | +0.092 | +0.148 |

**핵심 결론**:
- **ep12 A+D = +0.219** ≈ v10 ep40 plateau (+0.221). v11이 12 epoch만에 v10 50 epoch 도달
- 사용자 통찰 검증: interpreter는 decoder의 reconstruction wrapper (D' < D 역전 ep8에)
- M encoder 단독(+0.208)이 강력 — task가 motion-biased (hand pose ≈ motion)
- P encoder 단독은 약함, motion routing 거치면 살아남
- Loss와 R² 정직 상관 (L_total 0.0057 → 0.0024 → A+D R² +0.143 → +0.219)

## DROID Cross-domain Probing 결과

| Gap (DROID 15Hz) | VideoMAE | v11 best (mode) | 격차 |
|------------------|----------|-----------------|------|
| 1 (0.07초) | -0.006 | -0.005 | +0.001 |
| 10 (0.67초) | -0.006 | +0.006 (A+B) | +0.012 |
| **15 (1초)** ★ | **-0.035** | **+0.005 (A+B)** | **+0.040** |
| 30 (2초) | -0.028 | -0.010 | +0.018 |

- 모든 gap에서 v11이 VideoMAE보다 일관 우위
- gap=15 (EgoDex 학습 분포 1초와 일치)에서 격차 가장 큼 (+0.040)
- VideoMAE는 in-domain (EgoDex +0.326) 강력하지만 cross-domain 음수
- 절대 R²은 작음 (~0.005) — DROID action probing 자체 한계
- **방향성 검증**: v11이 cross-domain 일반화 우수 (사용자 직감 정량 검증)

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

1. v11 ep16/ep20 probing 계속 — A+D 추세 확인 (v10 plateau 추월 여부)
2. LIBERO BC fine-tune 결과 (33600616 VideoMAE, 33600617 v11 ep12) → val MSE 비교 후 rollout 결정
3. DROID 추가 gap (5/20) 보강
4. 공개 weight lineup (VC-1, DINOv2, SigLIP, VideoMAE-official, V-JEPA-official) DROID 평가