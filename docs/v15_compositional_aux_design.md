# v15 Design Spec — Layered Specialization with Compositional Auxiliary

**Target**: `src/models/two_stream_v15.py` 변경 명세 (코드 수정 가이드)
**작성일**: 2026-05-11
**원본 design doc**: Vault `Projects/Action-Agnostic Paper/v15 - Layered Specialization (Future).md`
**상태**: 본 학습 진입 전. 변경 사항 반영 후 sanity → 본 학습.

---

## 1. 변경 요약 (v15 초안 → v15 final)

| # | 변경 | 이유 |
|---|---|---|
| 1 | **DINO 제거 → L_compose 추가** | DINO 정당화 약함 (M stream에 paradigm conflict 없음의 부재증명에 그침). L_compose는 v16 chunk-level inference의 직접 prerequisite + algebraic compositional structure 강제. |
| 2 | **V-JEPA-M target = Teacher_M_encoder only (Option B)** | V-JEPA / I-JEPA strict 패턴. M_decoder는 mask_token 입력 분포에 학습되므로 teacher가 unmasked input에 통과시키면 OOD. Teacher decoder forward 절감. |
| 3 | **Motion routing input = M_encoder unmasked output** (이미 코드 그대로) | v15 코드 line 534, 543은 이미 `m_local_unmasked` (M_encoder 출력) 사용. v14의 m_completed (M_decoder 출력)에서 의도적으로 변경. M_decoder는 V-JEPA-M Predictor 단일 역할. **변경 없음, 명세 명확화**. |
| 4 | **Pretraining input pipeline 단순화** | Raw pair 불필요 (DINO teacher view 빠짐). cropped triple (frame_t, frame_t+n, frame_t+m) 만 사용. |

---

## 2. 최종 Loss 구성

```
L_total = L_t                              ← MAE on P stream (crop_t)
        + L_tk_recon                       ← MAE on P stream (crop_t+m)
        + λ_pred · L_pred                  ← V-JEPA P (predictor-only)
        + λ_m_jepa · L_m_jepa              ← V-JEPA M (masked latent recon, Option B)
        + λ_compose · L_compose            ← Compositional structure on M_encoder (NEW)
```

**5 loss, 모두 reconstruction/structural prior 계열** (no distillation paradigm).

기본 λ = 1.0 모두. Sanity 후 조정.

---

## 3. Track별 forward 명세

### Track 1, 2 — L_t, L_tk_recon (변경 없음)

```python
# Student P stream MAE
mask_p_t  = random_mask(B, ratio=0.75)
mask_p_tm = random_mask(B, ratio=0.75)

p_t_visible  = self._student_p_encode_visible(P_channel(crop_t),    mask_p_t)
p_tm_visible = self._student_p_encode_visible(P_channel(crop_t_m),  mask_p_tm)

p_t_full  = self._build_full_seq_p(p_t_visible,  mask_p_t)   # + mask_token + dec_pos_embed
p_tm_full = self._build_full_seq_p(p_tm_visible, mask_p_tm)

p_t_decoded  = self.interpreter_1(p_t_full)
p_tm_decoded = self.interpreter_1(p_tm_full)

patch_pred_t  = self.recon_head(p_t_decoded[:, 1:])    # CLS 제외
patch_pred_tm = self.recon_head(p_tm_decoded[:, 1:])

L_t        = MSE(patch_pred_t,  patchify(P_channel(crop_t)))[masked_p_t]
L_tk_recon = MSE(patch_pred_tm, patchify(P_channel(crop_t_m)))[masked_p_tm]
```

### Track 3 — L_pred (V-JEPA P, predictor-only)

```python
# M stream UNMASKED forward — routing source
m_local_unmasked = self._encode_m_unmasked(M_channel(crop_t, crop_t_m))  # M_encoder ONLY
# shape: [B, 1+N, D]
# ★ M_decoder 통과 안 함

# V source + target 모두 Teacher P (predictor-only)
with torch.no_grad():
    p_t_repr_T  = self.teacher_p.forward_unmasked(P_channel(crop_t)).detach()    # V source
    p_tm_repr_T = self.teacher_p.forward_unmasked(P_channel(crop_t_m)).detach()  # target

# p_motion_decoder = RoutingInterpreterStep × N (interleaved routing + interp)
p_state = p_t_repr_T  # V source
for step in self.p_motion_decoder:  # N steps
    p_state = step(p_state, m_local_unmasked)
    # internally: routing(Q,K ← m_local_unmasked, V ← p_state) → interpreter

predicted_tk_repr = self.p_motion_decoder_norm(p_state)

L_pred = SmoothL1(predicted_tk_repr, sg(p_tm_repr_T)).mean()  # patch + CLS 모두
```

**Gradient 흐름**:
- ✓ p_motion_decoder, ✓ M_encoder (routing K,Q를 통해)
- ✗ P_encoder (V source가 .detach()), ✗ M_decoder (이 path에 없음)

### Track 4 — L_m_jepa (V-JEPA M, Option B)

★ **변경 사항**: target에서 Teacher_M_decoder 제거.

```python
mask_m = random_mask(B, ratio=0.5)

# Student: M_encoder masked + mask_token + M_decoder
m_visible = self._encode_m_masked(M_channel(crop_t, crop_t_m), mask_m)
m_full    = self._build_full_seq_m(m_visible, mask_m)            # + mask_token + dec_pos_embed_m
m_decoded_masked = self._decode_m(m_full)                        # [B, 1+N, D] — student decoder output

# Teacher: M_encoder ONLY (★ Option B — decoder 미통과)
with torch.no_grad():
    m_target_encoded = self.teacher_m.forward_unmasked_encoder_only(
        M_channel(crop_t, crop_t_m)
    ).detach()  # [B, 1+N, D] — encoder-level target

# Loss: student decoder output vs teacher encoder output, masked patch positions only (CLS 제외)
student_patches = m_decoded_masked[:, 1:]   # [B, N, D]
target_patches  = m_target_encoded[:, 1:]   # [B, N, D]

err = SmoothL1(student_patches, target_patches, reduction="none").mean(dim=-1)  # [B, N]
L_m_jepa = (err * mask_m.float()).sum() / mask_m.float().sum().clamp(min=1.0)
```

**M_decoder의 의미**: V-JEPA-M Predictor — masked encoder + mask_token 입력으로부터 unmasked encoder 표현(teacher)을 예측.

### Track 5 — L_compose (NEW, replaces DINO)

```python
# Sample triple: (t, t+n, t+m), 0 < n < m, both ~ triangular[1, 60]
# Loader가 frame_t, frame_t+n, frame_t+m을 모두 제공해야 함

# Student forward (gradient ON, M_encoder만 학습)
m_short = self._encode_m_unmasked(M_channel(crop_t,    crop_t_n))  # [B, 1+N, D]
m_long  = self._encode_m_unmasked(M_channel(crop_t,    crop_t_m))  # [B, 1+N, D]

# Teacher forward (EMA, no_grad)
with torch.no_grad():
    m_target = self.teacher_m.forward_unmasked_encoder_only(
        M_channel(crop_t_n, crop_t_m)
    ).detach()  # [B, 1+N, D]

# Composition head
m_predicted = self.composition_head(m_short, m_long)
# Sanity 시작값: linear residual `m_long - m_short` (parameter 0개)
# 본 학습: shallow MLP (1 hidden, hidden=embed_dim, dropout 0.1)

# Loss: SmoothL1 patch + CLS 모두 reduction
L_compose = SmoothL1(m_predicted, m_target).mean()
```

**Gradient 흐름**:
- ✓ M_encoder (직접), ✓ composition_head (직접)
- ✗ Teacher_M_encoder (EMA only)

---

## 4. 모듈 변경 사항 (코드 수정 가이드)

### 4.1 제거할 모듈
```python
# 모두 제거
self.dino_head                          # DINOHead student
self.teacher_dino_head                  # Teacher_DINOHead (TeacherMv15 내부)
self.dino_center                        # DINO center buffer

# Teacher_M_decoder 부분 제거 (TeacherMv15 class)
# - forward_unmasked_full() 메서드: encoder + decoder 통과 — 이건 유지하되 V-JEPA-M에서 사용 안 함
# - 또는 forward_unmasked_encoder_only() 새 메서드 추가 (encoder만, V-JEPA-M + L_compose target용)
```

### 4.2 추가할 모듈
```python
class CompositionHead(nn.Module):
    """L_compose: (m_short, m_long) → m_predicted in M_encoder space."""

    def __init__(self, embed_dim: int, mode: str = "linear_residual", hidden_dim: int = None):
        super().__init__()
        self.mode = mode
        if mode == "linear_residual":
            # parameter 0개. m_long - m_short
            pass
        elif mode == "linear":
            self.proj = nn.Linear(2 * embed_dim, embed_dim)
        elif mode == "mlp":
            hidden_dim = hidden_dim or embed_dim
            self.net = nn.Sequential(
                nn.Linear(2 * embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, m_short: torch.Tensor, m_long: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear_residual":
            return m_long - m_short
        x = torch.cat([m_short, m_long], dim=-1)  # [B, 1+N, 2D]
        if self.mode == "linear":
            return self.proj(x)
        return self.net(x)
```

`TwoStreamV15Model.__init__`에서:
```python
self.composition_head = CompositionHead(
    embed_dim=embed_dim,
    mode=composition_mode,           # "linear_residual" (sanity) / "mlp" (본 학습)
    hidden_dim=composition_hidden_dim,
)
self.lambda_compose = lambda_compose   # default 1.0
```

### 4.3 TeacherMv15 변경

```python
class TeacherMv15(nn.Module):
    """M_encoder + M_decoder EMA copy.
    
    v15 final: M_decoder는 유지하지만 V-JEPA-M target에서는 사용 안 함.
    Decoder가 필요한 path가 사라지면 향후 decoder 부분 제거 검토.
    """
    
    @torch.no_grad()
    def forward_unmasked_encoder_only(self, m_channel: torch.Tensor) -> torch.Tensor:
        """[B, 3, 224, 224] → [B, 1+N, D] (encoder only, no decoder).
        
        V-JEPA-M target (Option B) + L_compose target에서 사용.
        """
        m_local = self._encode_unmasked(m_channel, self.pos_embed_m)
        return m_local
    
    # 기존 forward_unmasked_full() (encoder+decoder)은 유지하되 v15에서는 호출 X
    # forward_global_full() (DINO teacher 용)은 제거
```

### 4.4 Hyperparameter 변경

```python
class TwoStreamV15Model(TwoStreamV11Model):
    def __init__(
        self,
        # ... v11 args ...
        # v15 args (변경)
        lambda_pred: float = 1.0,
        lambda_m_jepa: float = 1.0,
        lambda_compose: float = 1.0,           # NEW (replaces lambda_dino)
        mask_ratio_m_jepa: float = 0.5,
        composition_mode: str = "linear_residual",   # NEW. "linear_residual" / "linear" / "mlp"
        composition_hidden_dim: Optional[int] = None,  # NEW. mode="mlp"일 때 hidden dim
        # 제거: lambda_dino, dino_n_crop, num_prototypes, dino_teacher_temp, dino_student_temp,
        #       dino_center_momentum, dino_head_hidden_dim, dino_head_bottleneck_dim
        ...
    ):
```

### 4.5 Forward 메서드 변경

```python
def forward(
    self,
    image_current: torch.Tensor,        # crop_t (B, 3, 224, 224)
    image_future:  torch.Tensor,        # crop_t_m (B, 3, 224, 224)
    image_short:   torch.Tensor,        # crop_t_n (B, 3, 224, 224) — NEW (L_compose용)
    # 제거: image_current_global, image_future_global (DINO teacher raw view)
):
    # ... existing MAE (Track 1, 2) ...
    # ... V-JEPA P (Track 3, m_local_unmasked는 변경 없음) ...
    
    # Track 4 (V-JEPA-M, Option B)
    mask_m = self._random_mask(B, device, self.mask_ratio_m_jepa)
    m_visible = self._encode_m_masked(m_channel_cropped, mask_m)
    m_full = self._build_full_seq_m(m_visible, mask_m)
    m_decoded_masked = self._decode_m(m_full)
    
    with torch.no_grad():
        m_encoded_target = self.teacher_m.forward_unmasked_encoder_only(m_channel_cropped).detach()
    
    student_patches = m_decoded_masked[:, 1:]
    target_patches = m_encoded_target[:, 1:]
    err = F.smooth_l1_loss(student_patches.float(), target_patches.float(), reduction="none").mean(dim=-1)
    loss_m_jepa = (err * mask_m.float()).sum() / mask_m.float().sum().clamp(min=1.0)
    
    # Track 5 (L_compose, NEW)
    m_channel_short = compute_m_channel(crop_t, crop_t_n)        # M_channel(t, t+n)
    m_channel_long  = m_channel_cropped                           # M_channel(t, t+m) — Track 4와 같음, 재사용
    m_channel_target = compute_m_channel(crop_t_n, crop_t_m)     # M_channel(t+n, t+m)
    
    m_short = self._encode_m_unmasked(m_channel_short)
    m_long  = self._encode_m_unmasked(m_channel_long)             # Track 3 m_local_unmasked과 동일 — 재사용 가능
    
    with torch.no_grad():
        m_compose_target = self.teacher_m.forward_unmasked_encoder_only(m_channel_target).detach()
    
    m_predicted = self.composition_head(m_short, m_long)
    loss_compose = F.smooth_l1_loss(m_predicted, m_compose_target, reduction="mean")
    
    # Total loss
    loss = (
        loss_t + loss_tk_recon
        + self.lambda_pred * loss_pred
        + self.lambda_m_jepa * loss_m_jepa
        + self.lambda_compose * loss_compose
    )
    
    return self._build_output_dict(
        loss=loss, loss_t=loss_t, loss_tk=loss_tk_recon,
        loss_pred=loss_pred, loss_m_jepa=loss_m_jepa, loss_compose=loss_compose,
        # ... 기타 진단 필드
    )
```

### 4.6 EMA 갱신 변경

```python
# v14에서 4개 → v15 final에서 2개
@torch.no_grad()
def _update_emas(self, momentum: float):
    self._ema_update(self.teacher_p,  self.p_encoder, momentum)
    self._ema_update(self.teacher_m,  self.m_encoder, momentum)
    # 제거:
    # - self._ema_update(self.teacher_m_decoder, self.m_decoder, momentum)
    # - self._ema_update(self.teacher_dino_head, self.dino_head, momentum)
    # - self.dino_center 갱신 코드
```

(만약 `teacher_m`이 encoder만 EMA로 유지하도록 변경하면, 별도 Teacher_M_decoder는 아예 없어짐.)

---

## 5. Data loader 변경

### 5.1 Triple sampling (L_compose용)

기존: `(video, t, k)` → frame_t, frame_t+k 페어 로드.

신규: `(video, t, n, m)` → frame_t, frame_t+n, frame_t+m 트리플 로드.
- `0 < n < m`
- `n, m ~ triangular[1, 60]`
- crop은 frame당 독립 random (v14 표준 그대로)

```python
def __getitem__(self, idx):
    video, t = self._sample_video_and_anchor(idx)
    n = sample_gap_triangular(low=1, high=60)
    m = sample_gap_triangular(low=n+1, high=60)  # m > n 보장
    # 또는: 두 번 sample 후 sort
    
    frame_t   = load_frame(video, t)
    frame_t_n = load_frame(video, t + n)
    frame_t_m = load_frame(video, t + m)
    
    # Independent random crop per frame
    crop_t   = random_crop(frame_t,   self.image_size)
    crop_t_n = random_crop(frame_t_n, self.image_size)
    crop_t_m = random_crop(frame_t_m, self.image_size)
    
    return {
        "image_current": crop_t,      # (3, 224, 224)
        "image_short":   crop_t_n,    # NEW
        "image_future":  crop_t_m,
        "gap_n": n,
        "gap_m": m,
    }
    # 제거: image_current_global, image_future_global (DINO raw teacher view)
```

### 5.2 제거할 부분
- Raw pair (uncropped) 로딩 — DINO teacher 전용이었음
- Multi-crop strategy (DINO student multi-crop) — 제거

---

## 6. Sanity protocol

### Phase A.1 — Linear residual composition_head로 시작 (5 ep on small data)

```python
# Configuration
composition_mode = "linear_residual"   # parameter 0개
lambda_compose = 1.0
mask_ratio_m_jepa = 0.5
```

**체크포인트별 진단**:
1. `loss_t`, `loss_tk_recon` 단조 감소
2. `loss_pred` 단조 감소 + `cos(predicted_tk_repr, p_t_repr_T) < 0.99` (identity collapse 아님)
3. **`loss_m_jepa` 단조 감소** (V-JEPA-M Option B의 trivial 해 봉쇄 확인)
4. **`loss_compose` 단조 감소** (NEW)
5. **★ `cos(m_long − m_short, m_target)` ↑ over training** — M_encoder가 emergent additivity 학습 (가장 중요한 sanity signal)
6. **trivial collapse 확인**: `cos_intra_m_encoder < 0.95` (M_encoder 출력이 input 무관하게 같아지지 않음)

**실패 mode 분기**:
- `loss_compose` plateau 빨리 → composition_head 표현력 부족 → Phase A.2로
- `cos_intra_m_encoder > 0.99` → trivial collapse → λ_compose 축소
- 다른 loss hurt → λ_compose 축소

### Phase A.2 — Shallow MLP composition_head (Phase A.1 통과 후)

```python
composition_mode = "mlp"
composition_hidden_dim = embed_dim   # 같은 dim
```

같은 6개 진단 항목 모니터. 통과 시 본 학습.

### Phase B — 본 학습 (5-7일, EgoDex 50 ep)

12-mode probing + layer-wise diagnostics:
- patch_mean_p_enc (MAE quality)
- patch_mean_p_motion_decoder_output (V-JEPA P quality)
- **patch_mean_m_enc** (V-JEPA-M + L_compose quality)
- patch_mean_m_decoder_output (V-JEPA-M Predictor 직출)
- **★ cos(m_long − m_short, m_target)** trajectory
- **★ Probing R²**: M_encoder output → motion category classification (v14 cls_m_enc R²와 비교)

---

## 7. 변경 사항 체크리스트 (구현 시 확인)

- [ ] DINO 관련 모듈 모두 제거 (`dino_head`, `teacher_dino_head`, `dino_center`, `forward_global_full` etc.)
- [ ] DINO 관련 hyperparameter 모두 제거
- [ ] `composition_head` 추가 (linear_residual / linear / mlp 3 mode 지원)
- [ ] `lambda_compose`, `composition_mode`, `composition_hidden_dim` hyperparameter 추가
- [ ] `TeacherMv15.forward_unmasked_encoder_only()` 메서드 추가
- [ ] V-JEPA-M target을 `forward_unmasked_full()` → `forward_unmasked_encoder_only()`로 변경 (Option B)
- [ ] Forward 메서드에 `image_short` 인자 추가
- [ ] L_compose forward 코드 추가 (3개 M_encoder forward + composition_head + Teacher_M_encoder target)
- [ ] Total loss 갱신: `λ_compose · loss_compose` 추가, `λ_dino · loss_dino` 제거
- [ ] EMA 갱신: Teacher_M_decoder, Teacher_DINOHead 제거 (Teacher_P_encoder, Teacher_M_encoder만 유지)
- [ ] Data loader: triple sampling (frame_t, frame_t+n, frame_t+m), independent crop
- [ ] Data loader: raw pair 로딩 제거, multi-crop 제거
- [ ] 진단 dict에 `loss_compose`, `cos_long_minus_short_target` (additivity metric) 추가
- [ ] 본 학습 launch script (`scripts/train_two_stream_v15.py`) 갱신: λ_compose, composition_mode 인자 추가

---

## 8. Sanity 후 본 학습 시 주의사항

1. **composition_head는 Phase A.1 → A.2 거친 후만 MLP로**. 처음부터 MLP면 trivial collapse risk.
2. **gap n, m sampling**: m > n 강제. 균일하게 sampling하면 m == n 가능, 그러면 m_target이 zero motion → trivial.
3. **GPU memory**: M_encoder forward가 추가 1번 (m_short forward). Teacher_M_encoder forward도 추가 1번. M_encoder는 작아 부담은 작지만 모니터.
4. **L_compose가 loss_pred나 loss_m_jepa를 hurt하면 λ_compose를 0.5나 0.3으로 축소**. M_encoder가 너무 compose에 끌려가면 motion routing K,Q가 약해질 수 있음.

---

## 9. Reference

- Vault design doc: `~/vault_YS/Projects/Action-Agnostic Paper/v15 - Layered Specialization (Future).md`
- v16 future paper draft: `~/vault_YS/Projects/Action-Agnostic Paper/v16 - Anchor-Relative Action Inference (Future).md` (L_compose가 v16의 prerequisite)
- 현재 v15 코드: `src/models/two_stream_v15.py`
- v14 코드 (참고): `src/models/two_stream_v14.py`
