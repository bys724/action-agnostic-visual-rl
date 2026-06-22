# Readout / Pooling 개선 — 조사 + 계획

> frozen 인코더의 **대표값(단일 토큰) readout**을 평균(GAP)보다 정보손실 적게 만드는 방향. downstream 추가학습 없이.
> 출처: 2026-06-22 Vault 세션 토론 + read-only capability survey. 개념 = Vault `Projects/Action-Agnostic Paper/1. Core Idea.md` §Invariance vs Equivariance, §일반화 프레임.
> ⚠️ 본 문서는 **계획·주의사항만**. 실제 구현(pooling 교체, 요약 토큰 학습)은 dev 세션에서.

## 1. 배경 / 동기

- 현재 **모든 인코더**(Parvo + baselines)가 patch-mean(GAP)으로 **카메라당 1토큰**을 만든다.
- 평균은 *선형 L2-최적* 단일 요약이지만 **lossy** — 패치별 정보(특히 위치/분포)를 뭉갠다.
- LIBERO **spatial 약점**(Parvo 0.644 / baseline 0.715~0.802)의 *부분* 원인일 수 있음(미확정).
- 목표: **downstream fit 없이**, 정보손실 적은 대표값을 (가능하면 사전학습에 baked) 확보.

## 2. repo 현황 (grounding, 2026-06-22 read-only survey)

### 인코더별 pooling
| Encoder | file:line | 방식 | 토큰 |
|---|---|---|---|
| DINOv2 | `src/encoders/adapters/single_frame.py:114` | patch-mean (CLS drop, `patch_mean_skip_cls`) | 1/frame |
| SigLIP | `single_frame.py:116` | patch-mean (`patch_mean_no_cls`, CLS 없음) | 1/frame |
| VC-1 | `single_frame.py:108-109` | CLS (`vc1_direct`) | 1/frame |
| VideoMAE-ours | `videomae.py:74` | patch-mean | 1/frame |
| **Parvo** | `parvo_pt_ptk.py:84-85` | patch-mean (CLS drop) | 1/frame |

- baseline은 `(prev,curr)` 각각 인코딩 후 concat → 1토큰(dim `2×hidden`). DINOv2도 **CLS 버리고 patch-mean**(간판 dense feature 미사용).

### 정책의 1토큰 가정 (`src/policies/bc_transformer_adapted.py`)
- adapter contract(`src/encoders/adapters/base.py:21,40`): `forward → (B,T,embed_dim)` **단일 벡터** (NOT `(B,T,num_tokens,embed_dim)`).
- `image_projections`(`bc_transformer_adapted.py:76-79`): 카메라당 `nn.Linear(adapter_dim, embed_size)`, 마지막 dim을 feature 벡터로 가정.
- `_encode_camera`(`:134-136`): `proj.unsqueeze(2)` → `(B,T,1,E)` = **카메라당 정확히 1토큰**.
- `spatial_encode`(`:182-184`) / `temporal_encode`(`:144-151`, `return x[:,:,0]`).
- **dense/attentive pooling 경로 없음** (spatial-softmax·attention pool은 `src/models/` SSL 내부에만, adapter→policy 경로엔 전무).

## 3. 핵심 판단 (thesis-aware)

- **DINO/iBOT CLS distillation** = 검증된 global 토큰 baking이지만 *augmentation-invariant* → **equivariance thesis와 충돌**(위치/변화를 일부러 버림 = Vault 노트가 "DINOv2 = invariance → motion 무시"로 비판한 성질). 그대로 도입 시 싸우려는 position-blindness를 주입. → 주의.
- **recon/predictive(preservation) 목적**이 thesis에 더 정합 (정보를 버리지 않고 보존 = equivariance-friendly).
- ⚠️ **단일 토큰은 spatial을 못 고친다**: "물체가 왼쪽" 류 *coarse* 위치만 한 토큰에 담김. 다물체 배치(LIBERO spatial)는 본질적으로 *분산* 정보 → 토큰 하나로 불가. ⇒ 이 트랙은 **semantic/invariant 축의 일반 readout 개선**이지 spatial 해법 아님. spatial이 목표면 dense/few-latent는 **별도 트랙**.

## 4. 계획 (비용 순서)

### Step 1 — 공짜 확인 (재학습 X)
frozen 인코더 그대로, 합치기 함수만 교체해 GAP 대비 이득을 본다.

- 후보:
  - **mean + max** (또는 mean + std) concat — 분포/salient 정보 보존.
  - **GeM** (generalized mean): `z = ( (1/N) Σ x_i^p )^(1/p)`, 파라미터 `p` 하나(고정 또는 learn). p=1→평균, p→∞→max.
- 구현 지점: 각 adapter의 `.mean(dim=1)` 교체.
- **공정성 (필수)**: pooling 변경은 **모든 비교 인코더에 동일 적용**. Parvo만 바꾸면 불공정(앞선 1프레임-vs-2프레임 concat-probe artifact 류 confound).
- 차원 주의: mean+max는 dim 2배 → `image_projections` 입력 차원 맞춤(adapter `embed_dim` 갱신).
- 측정: probing R² + LIBERO BC가 GAP 대비 오르나.

### Step 2 — 이득 있으면 사전학습에 박기 (재학습 O)
학습 가능한 요약 토큰을 인코더에 추가하고 **사전학습 중** 목적함수로 학습 (downstream fit 불필요).

- (a) **recon**: 요약 토큰 → 가벼운 디코더(+positional query) → 패치 임베딩(feature-space) 복구. preservation-biased.
- (b) **predictive (권장, predictive thesis 정합)**: 현재 global 요약 → **다음 순간 global 요약 예측**(JEPA식). 노트의 "다음 순간 예측으로 표현 빚기"와 일관.
- 개선판: 토큰 1개 말고 **소수 latent(k=4~16, Perceiver-IO식)**가 패치를 복구 → coarse spatial 유지 + GAP↔dense 중간. (단 정책 1토큰 계약 변경 필요 → spatial 트랙과 합류.)

## 5. 주의사항 (critical guards)

- **공정성**: pooling/요약 변경은 모든 비교 인코더에 동일 적용. 단일 인코더만 buff 금지.
- **단일 토큰 ≠ spatial 해법**: 기대치 명확히. 이 트랙은 semantic 축 readout 개선.
- **Step1 음성 ≠ Step2 무가치**: 둘은 다른 질문 — Step1=고정 합치기로 평균 너머 정보가 뽑히나, Step2=학습으로 인코더를 한 토큰에 정보 집중하게 빚을 수 있나. Step1은 **저비용 게이트**일 뿐, 1단계 0이어도 2단계 가능성 0 아님.
- **invariance 주입 주의**: DINO식 CLS 목적은 equivariance thesis와 충돌 → recon/predictive 우선.
- **frozen 평가**: Step2 평가도 downstream fit 없이 frozen readout만으로 (사전학습에 baked 됐는지 확인).

## 6. 검증 체크리스트

- [ ] (Step1) GeM/multi-stat을 **모든** 비교 인코더에 적용
- [ ] (Step1) `image_projections` 입력 차원 맞춤(dim 변화 반영)
- [ ] (Step1) GAP 대비 probing R² / LIBERO BC delta 측정·기록
- [ ] (Step2) 요약 토큰 목적함수 선택(recon vs predictive) + invariance 주입 안 함 확인
- [ ] (Step2) downstream fit 없이 frozen readout로 평가

## 7. Cross-references

- **개념/thesis**: Vault `Projects/Action-Agnostic Paper/1. Core Idea.md` §Invariance vs Equivariance, §일반화 프레임 / `2. Experiments.md` §4(equivariance probe·spatial 약점).
- **관련 코드**: `src/encoders/adapters/{single_frame.py, videomae.py, parvo_pt_ptk.py, base.py}`, `src/policies/bc_transformer_adapted.py`.
- **grounding 출처**: 2026-06-22 read-only capability survey (본 문서 §2).
