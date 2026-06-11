# Representation Visualization Plan (Project Page용)

> 🔴 **2026-06-11 주의**: 모델 명명 `Parvo`(= 본 문서 "v15 P encoder"), CoRL 미제출→AAAI(Paper 2). **더 중요**: 본 plan의 figure 전제("scaffolding이 P에 motion encoding을 emerge시킴")는 **scaffold 인과 철회**(v15 motion routing no-op)와 충돌 — 시각화가 retracted claim을 그리게 됨. Parvo(code v15b) 검증 결과 확정 전까지 figure 전제 재검토 필요. 정규 출처 = [`CLAUDE.md`](../CLAUDE.md) "명명 · 2논문 구조".
>
> **목적**: Paper accept 후 project page 제작 시 Parvo(v15 P encoder)의 representation을 직관적으로 보여줄 가시화 자료 생성 계획. 일반 viewer 임팩트보다는 *related-work 검색 중 들어온 동료 연구자의 빠른 이해*가 main audience.
>
> **시점**: CoRL 2026 accept 발표 이후 (paper 마감 5/29 이후 워크플로우). Anonymous review 기간 동안 public 공개 X.
>
> **상세 framing 배경**: Vault `Projects/Action-Agnostic Paper/README.md` 참조.
>
> **최종 갱신**: 2026-05-27

---

## 1. Background & 목표

학술 community에서 project page (Nerfies-style)는 사실상 표준이며, 사용자 paper의 distinctive 기여(scaffolding architecture + action-agnostic motion encoding)를 visual로 풀어내는 게 핵심이다.

Optical flow 모델들은 **dense pixel-level output 자체가 visual** 이라 화살표 overlay 같은 직관적 자료가 자연스럽다. v15는 **latent representation**이라 한 단계 변환(projection 또는 gradient analysis)이 필요하지만, 이 변환이 오히려 *"모델이 무엇을 학습했는가"*를 보여주는 mechanism-revealing visual로 만들 수 있다.

### 우선순위 2종 (본 plan 대상)

1. **#3 Patch Feature PCA Overlay** — DINOv2-style RGB overlay. 30분~1시간 prototype, 즉시 임팩트.
2. **#1 Grad-CAM for ViT Arrow** — Optical flow와 시각적으로 유사한 motion arrow overlay. 반나절~1일.

### Out of scope (별도 plan으로 분리)

- Motion-routing attention 시각화 — mechanism-revealing이지만 forward hook 등 구현 비용 큼
- Reconstruction error map, M-latent t-SNE — 보조 자료
- Real-robot demo — paper limitation, 별도 연구 확장 작업

---

## 2. 적용 대상 & 입력 데이터

### Encoder 비교군 (3 enc 강력 추천)

| Encoder | Checkpoint 위치 | 비고 |
|---|---|---|
| **v15 (P encoder)** | `<ckpt path>` ep32 (champion) | paper main |
| **DINOv2** | HuggingFace `facebook/dinov2-base` | "internet-scale SSL" 대비군 |
| **VideoMAE-ours** | `<ckpt path>` ep50 | "same corpus, no scaffolding" 대비군 |

### Input frame pair

- **LIBERO**: robot manipulation, 일반인이 보기 직관적 (object 조작 명확) — **추천**
- **EgoDex**: within-domain, 가장 자연스러운 visual
- **CALVIN**: 다양한 task chain (선택)

**권장**: LIBERO rollout 비디오에서 motion이 활발한 시점 5-10 frame pair 선별.

---

## 3. #3 Patch Feature PCA Overlay

### 핵심 아이디어

- Frozen encoder의 patch features (예: ViT-B/14 → 16×16 = 256 patches × 768d)를 PCA로 차원 축소
- 첫 3 PC를 RGB로 매핑 → input frame에 patch grid로 overlay
- 비슷한 의미·motion의 patch가 비슷한 색

### 출력 형태

- **Static comparison figure**: 3 encoder × 2 frame = 6 panel grid
- **GIF**: LIBERO rollout 따라 시간 progression (10-30 frame)
- Project page에 side-by-side carousel

### 구현 단계 (pseudocode, 실제 구현은 별도 dev session)

```python
# scripts/viz/pca_overlay.py (TODO, 신규 작성)

def extract_patch_features(encoder, frame):
    """Return (N_patch, D) feature tensor — no CLS, no mean pooling."""
    ...

def fit_pca_global(all_patches):
    """전체 video frame의 patch 모아 한 번에 PCA fit (frame 간 색 일관성)."""
    return PCA(n_components=3).fit(all_patches)

def project_to_rgb(patches, pca):
    """Patch features → PCA projection → min-max normalize → RGB."""
    proj = pca.transform(patches)
    rgb = (proj - proj.min(axis=0)) / (proj.max(axis=0) - proj.min(axis=0) + eps)
    return rgb  # (N_patch, 3) ∈ [0, 1]

def overlay_on_frame(frame, rgb_map, patch_grid_shape, alpha=0.6):
    """rgb_map을 frame resolution으로 resize + alpha-blend."""
    ...
```

### ⚠️ 주의사항 (재발 가능, 가드)

- **PCA fit scope**: 한 frame만 fit하면 frame 간 색 일관성 깨짐. **전체 video patches 모아 한 번 fit** 후 모든 frame에 동일 PCA 적용 (DINOv2 컨벤션)
- **Encoder 비교 시**: 각 encoder별로 별도 PCA fit (feature space가 다르므로). 단 한 encoder 내에서는 frame 간 fit 공유
- **CLS token 포함 여부**: patch만 사용. CLS는 PCA outlier 가능성 + visualization 의미 불명확
- **Patch grid resolution**: ViT-B/14 @ 224 input = 16×16 patches로 다소 sparse. **384 input 권장** (28×28 patches). v15의 input resolution 확인 필요
- **Normalize 방식**: min-max vs z-score. min-max가 일반적이지만 outlier patch에 민감. **percentile clip** (1~99%) 적용 권장

### 선행 사례 / Reference

- DINOv2 paper (Oquab et al., 2023) Fig. 2 — 정확히 이 visualization
- DINOv2 demo: https://dinov2.metademolab.com/
- MAE paper (He et al., 2022) — feature visualization 컨벤션

---

## 4. #1 Grad-CAM for ViT — Motion Arrow Overlay

### 핵심 아이디어

- Frozen encoder + linear action probe
- Probe의 Δx 예측 head, Δy 예측 head 각각에 대해 **patch별 contribution** 계산
- `contribution_i = (∂Δ / ∂p_i) · p_i` — Grad-CAM 원리 (gradient × activation)
- Patch별 2D vector `(cx_i, cy_i)` → input frame 위에 quiver plot

### "평균 풀링이면 patch 구분 안 되지 않나?" 의문 해결

Raw gradient `∂Δx/∂p_i = w_probe / N`은 모든 patch에 동일하므로 gradient만 보면 patch 구분 X. **하지만** `(grad × activation)`은 patch별 activation 차이로 contribution이 달라짐.

| Probe input 형태 | 시각화 방법 |
|---|---|
| `mean(patch_features)` | Grad-CAM for ViT (gradient × activation per patch) |
| `CLS token` | Attention rollout (Abnar & Zuidema 2020) — CLS의 attention chain back-trace |
| `concat(P_t, P_{t+k})` | Grad-CAM 각 frame별 분리 적용 — 두 frame motion 기여 비교 가능 (가장 motion-aware) |

**📌 사전 확인 필요**: 현재 paper의 action probe setup이 위 3종 중 어느 것인지 확인 — `src/eval/probe_action.py`, `scripts/eval/probe_action_*.py`, `docs/PROBING_GUIDE.md` 참조 후 방법 선택.

### 출력 형태

- **Static figure**: 3 encoder × 2-3 frame pair = 6-9 panel grid (arrow overlay)
- **GIF**: LIBERO rollout에서 arrow direction이 task progress 따라 변화
- Project page interactive: hover시 patch별 contribution 수치 (Plotly)

### 구현 단계 (pseudocode, 실제 구현은 별도 dev session)

```python
# scripts/viz/grad_cam_arrow.py (TODO, 신규 작성)

def compute_patch_contributions(probe, encoder, frame_pair, action_dim_idx):
    """
    Δx 또는 Δy 등 특정 action dim에 대한 patch별 contribution map 계산.

    Args:
        probe: frozen linear probe (input dim = encoder feature dim)
        encoder: frozen v15 / DINOv2 / VideoMAE-ours
        frame_pair: (frame_t, frame_t+k) tensors
        action_dim_idx: 0=Δx, 1=Δy, etc.

    Returns:
        contribution: (N_patch,) — patch별 prediction contribution
    """
    patches = encoder(frame_pair)  # (N, D)
    patches.requires_grad_(True)

    # paper probe setup에 따라 분기
    if probe_input == "mean":
        pooled = patches.mean(dim=0)
    elif probe_input == "cls":
        pooled = encoder_cls(...)  # attention rollout path
    elif probe_input == "concat":
        pooled = torch.cat([patches_t.mean(dim=0), patches_tk.mean(dim=0)])

    pred = probe(pooled)  # (action_dim_count,)
    grad = torch.autograd.grad(pred[action_dim_idx], patches)[0]
    contribution = (grad * patches).sum(dim=-1)  # (N,) per-patch
    return contribution


def quiver_overlay(frame, cx_map, cy_map, patch_grid_shape):
    """
    Patch별 (cx_i, cy_i) → input frame 위에 quiver plot.

    Args:
        cx_map, cy_map: (H_patch, W_patch) contribution to Δx, Δy
    """
    # patch center 좌표 계산
    # matplotlib quiver 사용
    ...
```

### ⚠️ 주의사항 (재발 가능, 가드)

- **Probe head 분리**: 7-dim action vector (Δx,Δy,Δz,Δrx,Δry,Δrz,gripper) → **visualization에는 Δx, Δy 둘만 사용** (화면 평면 motion). Δz, rotation은 화살표로 표현 불가
- **Gradient sign**: contribution은 positive/negative 모두 가능 → arrow direction이 sign에 따라 좌/우, 상/하
- **Normalization**: 각 frame 내 max-norm 정규화 (frame 간 magnitude scale 차이 보정)
- **Resolution**: 16×16 patches (256 arrows)면 충분, 너무 sparse 안 됨
- **View 선택**: LIBERO eye-in-hand는 confound (per-frame motion 다름). **static agentview만 사용**
- **Probe 학습 데이터**: visualization frame이 probe 학습 split에 포함 안 되도록 (test split frame 사용)

### 선행 사례 / Reference

- Chefer et al., "Transformer Interpretability Beyond Attention Visualization" (CVPR 2021)
- Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (ACL 2020)
- Park et al., "What do Self-Supervised Vision Transformers Learn?" (ICLR 2023)

---

## 5. 작업 단계 TODO (Paper accept 후 우선순위)

### Step 1 — #3 PCA overlay prototype (~30분-1시간)

- [ ] `scripts/viz/pca_overlay.py` 신규 작성 (위 pseudocode 기반)
- [ ] LIBERO `<task>` rollout에서 frame pair 10장 추출
- [ ] 3 encoder × PCA overlay → 6 panel comparison figure 저장
- [ ] `paper_artifacts/visualizations/pca_overlay/` 보관 (`paper_artifacts/` 컨벤션 따름)

### Step 2 — #1 Grad-CAM arrow prototype (~반나절-1일)

- [ ] Paper의 action probe setup 확인 (probe input pooling 형태)
- [ ] `scripts/viz/grad_cam_arrow.py` 신규 작성
- [ ] Δx, Δy probe head 각각의 gradient × activation
- [ ] 3 encoder × 2-3 frame pair → arrow overlay figure
- [ ] LIBERO rollout 따라 GIF 생성 (선택)
- [ ] `paper_artifacts/visualizations/grad_cam_arrow/` 보관

### Step 3 — Project page integration (별도 plan)

- Nerfies template / Academic Project Page Template fork
- Static figures + GIF embed
- (선택) Plotly로 interactive contribution map
- Anonymity 보호 — accept 발표 전까지 비공개

---

## 6. 검증 체크리스트 (각 viz prototype 완료 후)

### PCA overlay
- [ ] 한 video 내 frame 간 색 일관성 (PCA fit이 video 전체에서 한 번만)
- [ ] v15가 DINOv2와 *명확히 다른* patch clustering 보이는가 (예: hand/object 영역 highlight)
- [ ] VideoMAE-ours와 비교 시 차이가 visualization에서 보이는가

### Grad-CAM arrow
- [ ] Arrow direction이 ground-truth motion direction과 대체로 일치
- [ ] v15 arrow 분포가 motion-relevant region (hand, manipulated object) 집중
- [ ] DINOv2 arrow가 static 3D layout 영역 (배경, 가구) 분산 — paper §5 LIBERO-Spatial gap 설명과 정합
- [ ] Frame pair 선택이 cherry-pick 의심 없도록 random sampling 후 representative 선별

---

## 7. 가드 / 주의사항 (공통)

- **시점**: paper accept 발표 후. Anonymous review 기간 동안 public 공개 X
- **HuggingFace / 외부 호스팅**: project page launch 시점에 ckpt 공개 여부 사용자 결정. 현재는 visualization 자료만 우선
- **계산 비용**: 두 viz 모두 frozen encoder + small data → 로컬 워크스테이션에서 충분 (클러스터 X)
- **Wow factor 보강**: real-robot rollout 영상이 없는 한계 → LIBERO/CALVIN sim rollout 품질이 핵심. rollout video 품질 polish는 별도 작업
- **Cherry-pick risk**: visualization frame 선별 시 motion-active한 시점만 고르면 cherry-pick 비판. **random sample 후 representative** 보고 + appendix에 전체 frame 공개

---

## 7.5 Prototype 1차 findings (2026-05-27, cluster session)

`#3 PCA overlay`와 `#1 Grad-CAM arrow` 모두 LIBERO spatial task_0 demo (multi-demo probe 학습, holdout demo_45)로 prototype 완료. 산출: `paper_artifacts/visualizations/{pca_overlay,grad_cam_arrow}/`.

**#3 PCA overlay**: v15 / DINOv2 / VideoMAE-ours 3 encoder 비교. PC1-3 explained variance:
- v15: 19/12/7% (balanced)
- DINOv2: 19/11/7%
- VideoMAE: **24**/10/8% (PC1 dominate = 단조 패턴, paper "no scaffolding baseline" 약점 정량 evidence)

**#1 Grad-CAM arrow** (V flip 적용 — 사용자 시각 검증: 그리퍼 down=image up). paired t/tk panel + sum vector (red P_t / cyan P_tk) + GT motion (green dashed) overlay.

**핵심 발견 — Linear probe가 implicit (P_tk − P_t) subtraction 학습**:
| Encoder | cos(W_t, −W_tk) Δz | sign match \|GT\|≥10cm P_t v / P_tk v |
|---------|--------------------|---------------------------------------|
| v15 | +0.79 | 1.00 / 0.00 |
| DINOv2 | +0.76 | 1.00 / 0.03 |
| VideoMAE | +0.78 | 0.71 / 0.97 |

→ 3 encoder 공통으로 `cos(W_t, −W_tk) ≈ 0.6−0.8` for motion dims = **ViT representation의 일반 metric structure** (v15-specific 아님). encoder-specific 차이는 **patch contribution 분배 방식**:
- DINOv2: single-frame discriminative 강함 (P_t u/v 모두 GT match)
- v15: axis-specific role (P_t vertical, P_tk horizontal)
- VideoMAE: subtraction 약함 (P_tk v sign이 P_t v와 같이 감)

**paper narrative**: viz는 **mechanism description** (어떻게 motion 인코딩되나)에 사용. v15 quantitative advantage는 main BC/probing table에서 보임. axis-alignment 자체로는 DINOv2 vs v15 명확한 advantage 안 드러남 — "single vs two frame 차이는 trivial". 두 frame 정보 **역할 분담** 자체가 v15의 contribution.

**남은 작업**: 로컬 세션에서 LIBERO rollout video 적용 + GIF 생성 + 3 encoder Grad-CAM 확장 (현재 1차는 v15만 viz, 3 encoder는 alignment metric만).

---

## 8. Cross-references

- Vault: `Projects/Action-Agnostic Paper/README.md` (paper 전체 진행 상황 + project page 논의)
- Vault: `Projects/Action-Agnostic Paper/4. Paper Writing.md` (paper main framing 결정 + Edit Spec)
- Vault: `Projects/Action-Agnostic Paper/3. Experiments.md` (실험 결과 + per-dim 진단 결과 — visualization 검증 시 reference)
- Dev repo: `docs/PROBING_GUIDE.md` (probe 학습 protocol, viz 전 확인 필요)
- Dev repo: `paper_artifacts/calvin_action_probing/_diagnostic/per_dim_r2.png` (per-dim 분해 결과 figure — viz 사례 참고)
