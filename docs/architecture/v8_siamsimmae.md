# Two-Stream v8 — EMA-based Temporal Prediction with M-Anchor

**작성일**: 2026-04-20
**상태**: 설계 확정 (구현 단계)

## 배경

### v7 실패 요약
- **v7-big sigma003** (ep8까지 학습): loss 0.0007 수렴, probing R² ≈ 0 → representation collapse
- **v7-big isolated** (ep4 진단 완료):
  - `cos(CLS_P_bg, CLS_P_motion) = +0.9997` → 두 CLS가 사실상 동일 벡터
  - Decoder swap ablation: 예측 MSE 변화 ~1e-6 → **CLS 분화 완전 실패**
- **근본 원인**: 대칭 구조(dual CLS + dual decoder)에서 공유 파라미터와 유사 loss target이 두 CLS를 동일 해로 끌어당김. Attention isolation만으로는 차단 불가.

### 설계 철학 재정립
- BG/Motion 구분은 action-agnostic 철학과 불일치 → **폐기**
- M(변화 감지) + P(형태/외형) 이원화 + **CLS exchange** 핵심 아이디어 유지
- 비대칭성을 **objective 레벨**에서 도입: M=pixel reconstruction 전담, P=representation prediction 전담
- Teacher가 "미래의 P encoding"을 제공 → student는 "현재 + motion으로 미래 P 추론"

## 핵심 설계

### 전체 구조
```
Student                                         Teacher (no_grad)
-------                                         ------------------
frame_t (P_ch) + ΔL(t,t+k) (M_ch)               frame_{t+k} (P_ch) + ΔL(t+k,t+k)=0 (M_ch)
  │                                               │
  ▼                                               ▼
 encoder (M+P+CLS exchange)                      encoder (shared M, EMA P+exchange)
  │ (student forward에서 CLS exchange 직전       │
  │  cls_m.detach() → L_P gradient가 M 차단)     │
  │                                               │
  ├──► m_patches ─► M decoder ─► L_M (pixel)      ▼
  │                                              p_teacher [unmasked]  (stop_grad)
  ▼                                               │
 p_student [masked positions]                     │
  │                                               │
  ▼                                               │
 prediction_head h (2-layer MLP + LN)             │
  │                                               │
  └──────────── L2 loss ─────────────────────────┘
                    = L_P

Total: L = L_M + λ(epoch) · L_P

EMA update: τ(step) schedule, P-related params만 (M은 공유)
```

### 핵심 결정 사항

| 요소 | 값 | 근거 |
|---|---|---|
| Architecture | **v4 style: single CLS_P, single M decoder** | v7 dual CLS 철학 불일치 · collapse 경험 |
| Teacher 전략 | **Option X** (P만 EMA, M shared, forward no_grad) | EMA로 stable target 확보하되 M은 일관된 연산자로 유지 |
| Teacher M 입력 | **`(frame_{t+k}, frame_{t+k})` → ΔL=0** (zero M) | Preprocessing `clamp(min=1e-6)` 덕분에 수치 안정. CLS_M은 sample-invariant 상수화 |
| Teacher P 입력 | `frame_{t+k}` (no masking) | 완전한 미래 관측 target |
| Student P masking | **block mask (~50%)** (I-JEPA-style) | 예측 난이도 확보, 단순 random 대비 구조적 prediction 유도 |
| Student M masking | 기존 **random mask 0.5** | M decoder는 pixel reconstruction, 기존 방식 유지 |
| Student CLS exchange | **매 stage 직전 `cls_m.detach()`** | L_P가 M stream으로 역류하지 않도록. v8의 role separation 설계 조항. |
| EMA 대상 | `patch_embed_p`, `cls_token_p`, `blocks_p`, `cls_exchange`, `norm_p` | P-output에 영향 주는 것 전부. M 계열은 student와 공유 |
| EMA momentum τ | **0.996 → 1.0 cosine schedule** over training | BYOL/I-JEPA 표준 |
| Prediction head h | **2-layer MLP + LayerNorm** (D → 2D → D) | BN이 DDP sync · zero-M edge case · patch-level에서 덜 적합. I-JEPA/V-JEPA/MoCo v3 모두 LN |
| L_P 위치 | student p_patches[masked] → h → teacher p_patches[matching], **CLS 제외** | Patch-level alignment (I-JEPA 패턴) |
| λ (L_P 가중치) | **0 → 0.5 cosine warmup over 5 epoch**, 이후 0.5 유지 | L_M 먼저 안정화 후 L_P 점진 도입 |
| Weight decay | **param group 분리**: bias/LN/cls_token/pos_embed/mask_token → `wd=0`, 나머지 `wd=0.01` | ViT SSL 관례. 현재 TwoStream은 uniform → **v8에서 분리 필수** |
| LR | **2e-4 시작** (v4 continuity) | 조정 여지 명시 아래 참고 |
| LR warmup | **10% of epochs, linear** → SequentialLR로 cosine decay | 기존 유지 |
| Optimizer | AdamW fused, betas=**(0.9, 0.999) 기본값** | 조정 여지 명시 아래 참고 |
| Grad clip | max_norm 1.0 | 기존 유지 |
| Mixed precision | **BF16 autocast** | 기존 유지 |
| DropPath | **파라미터만 추가 (default 0), 미래 확장 대비** | 초기엔 노이즈 요인 최소화. 필요시 --drop-path 0.1 등으로 활성 |
| Collapse monitoring | `cos(p_stu, p_tea)`, `std(p_stu)`, `std(p_tea)`, `L_M`, `L_P`, `λ_current` TB 기록 | 조기 경고 |

### 의도적으로 하지 않는 것

- **BG/Motion 분화** (v7) — 철학 불일치
- **Full EMA (Y)** — M이 zero-input teacher에서 의미 적고, student-teacher M context 불일치 생성
- **Shared+stop_grad (W, SimSiam)** — EMA 없이도 가능하지만 stability 보장 약함. EMA가 추가 안전 마진
- **VICReg / orthogonality** 정규화 — L_M anchor + input asymmetry로 충분할 것이라 가정. collapse 시 트러블슈팅 플레이북에서 도입
- **Layer-wise LR decay** — 사전학습 from scratch라 불필요 (fine-tuning 기법)
- **Attention/MLP dropout** — DropPath가 대체 (미래 확장)

### v4/v7과 비교

| 항목 | v4 | v7-big | **v8** |
|---|---|---|---|
| CLS_P 개수 | 1 | 2 (bg, motion) | **1** |
| M decoder 개수 | 1 | 2 | **1** |
| Loss | L_M (pixel) | L_M 가중 (Gaussian weighted) | **L_M + λ·L_P** |
| Teacher | 없음 | 없음 | **EMA on P (shared M)** |
| Prediction head | 없음 | 없음 | **2-layer MLP + LN** |
| CLS exchange detach | 없음 | 없음 | **있음 (v8 설계 조항)** |
| Collapse 방어층 | Masking | Masking + weighted loss | **L_M anchor + input asymmetry + prediction head + EMA target** |

## 구현 스케치

### `src/models/two_stream.py`

```python
class TwoStreamModel(nn.Module):
    def __init__(self, ..., v8_mode=False, ema_tau_base=0.996,
                 pred_head_ratio=2.0, drop_path_rate=0.0, ...):
        # 기존 초기화 (v4 style: num_p_cls=1 이면 single CLS_P)
        ...
        self.v8_mode = v8_mode
        self.drop_path_rate = drop_path_rate  # 현재 사용 안 함, 미래 확장

        if v8_mode:
            # Prediction head (2-layer MLP + LN)
            self.prediction_head_p = nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * pred_head_ratio)),
                nn.LayerNorm(int(embed_dim * pred_head_ratio)),
                nn.GELU(),
                nn.Linear(int(embed_dim * pred_head_ratio), embed_dim),
            )
            # EMA copy (P-related only)
            self._init_ema_params(tau_base=ema_tau_base)

    def _init_ema_params(self, tau_base):
        """EMA 대상: patch_embed_p, cls_token_p, blocks_p, cls_exchange, norm_p"""
        self.ema_tau_base = tau_base
        # EMA buffer — student 파라미터 미러링 (P-related만)
        # 구현: named_parameters 순회하며 "patch_embed_p|cls_token_p|blocks_p|cls_exchange|norm_p" 매칭
        # self.ema_params[name] = param.data.clone() (requires_grad=False)

    @torch.no_grad()
    def update_ema(self, tau):
        """Called after optimizer.step()"""
        for name, p in self.named_parameters():
            if self._is_ema_target(name):
                self.ema_params[name].mul_(tau).add_(p.data, alpha=1-tau)

    def forward_student(self, img_t, img_tk):
        m_ch, p_ch = self.preprocessing(img_t, img_tk)
        # encoder forward with detach in CLS exchange (see _encoder_student)
        m_patches, p_patches, mask_m, mask_p = self._encoder_student(m_ch, p_ch)
        pred_m = self.decoder_m(m_patches, p_cls)  # 기존 방식
        return pred_m, p_patches, mask_m, mask_p

    def _encoder_student(self, m_ch, p_ch):
        """매 stage CLS exchange 직전에 cls_m.detach() 적용"""
        # ... 기존 encoder forward 코드 거의 그대로 복사, 수정 지점:
        # for si in range(num_stages):
        #     for bm in blocks_m[si]: m_tokens = bm(m_tokens, ...)
        #     for bp in blocks_p[si]: p_tokens = bp(p_tokens, ...)
        #     # ==== 수정 지점 ====
        #     cls_m_in = m_tokens[:, :1].detach() if self.v8_mode else m_tokens[:, :1]
        #     cls_p_in = p_tokens[:, :1]
        #     cls_ex = cls_exchange[si](cat([cls_m_in, cls_p_in], dim=1))
        #     m_tokens = cat([cls_ex[:, :1], m_tokens[:, 1:]], dim=1)
        #     p_tokens = cat([cls_ex[:, 1:2], p_tokens[:, 1:]], dim=1)
        ...

    def forward_teacher(self, img_tk):
        """Shared M, EMA P, no_grad."""
        with torch.no_grad():
            # Teacher 입력 구성
            B = img_tk.shape[0]
            m_ch = torch.zeros(B, 3, 256, 256, device=img_tk.device)  # (t+k, t+k) → ΔL=0
            # 또는 preprocessing.compute_m_channel(img_tk, img_tk)로 0 생성
            p_ch = self.preprocessing.compute_p_channel(img_tk)
            # 가중치: M은 self.* (shared), P는 self.ema_params[*] 로 swap
            p_patches_teacher = self._encoder_teacher(m_ch, p_ch, no_mask=True)
        return p_patches_teacher

    def _encoder_teacher(self, m_ch, p_ch, no_mask=True):
        """EMA P 가중치를 functional call로 사용하거나, 임시 swap.
        간단 구현: 임시로 P 파라미터를 EMA로 교체 후 forward, 이후 복원."""
        ...

    def compute_loss_v8(self, img_t, img_tk, lam):
        pred_m, p_student, mask_m, mask_p = self.forward_student(img_t, img_tk)
        p_teacher = self.forward_teacher(img_tk)

        L_M = self._compute_recon_loss(pred_m, m_ch_target, mask_m)  # 기존
        # L_P: masked positions만
        z = self.prediction_head_p(p_student[mask_p])
        tgt = p_teacher[mask_p].detach()
        L_P = F.mse_loss(z, tgt)

        metrics = self._compute_collapse_metrics(p_student, p_teacher)
        return L_M + lam * L_P, {"L_M": L_M.item(), "L_P": L_P.item(), **metrics}
```

### `src/training/pretrain.py`

```python
# --- Param group 분리 (TwoStream에도 적용) ---
def build_param_groups(model, wd=0.01):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(kw in name for kw in [".bias", "ln.", "norm.",
                                      "cls_token", "pos_embed",
                                      "mask_token"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

# --- λ schedule (per-epoch) ---
def lambda_schedule(epoch, warmup=5, lam_max=0.5):
    if epoch < warmup:
        return lam_max * 0.5 * (1 - math.cos(math.pi * epoch / warmup))
    return lam_max

# --- EMA momentum schedule (per-step) ---
def ema_tau(step, total_steps, tau_base=0.996):
    return 1.0 - (1.0 - tau_base) * (math.cos(math.pi * step / total_steps) + 1) / 2

# --- training loop (v8 path) ---
if args.v8_mode:
    lam = lambda_schedule(epoch, args.lambda_warmup, args.lambda_max)
    loss, metrics = model.compute_loss_v8(img_t, img_tk, lam)
    # optimizer step
    optimizer.step()
    # EMA update
    tau = ema_tau(global_step, total_steps, args.ema_tau_base)
    model.update_ema(tau)

    # TB logging
    for k, v in metrics.items(): tb.add_scalar(f"train/{k}", v, global_step)
    tb.add_scalar("train/lambda", lam, global_step)
    tb.add_scalar("train/ema_tau", tau, global_step)
```

### `scripts/pretrain.py` CLI flags 추가

```python
parser.add_argument("--v8-mode", action="store_true")
parser.add_argument("--lambda-max", type=float, default=0.5)
parser.add_argument("--lambda-warmup", type=int, default=5)
parser.add_argument("--ema-tau-base", type=float, default=0.996)
parser.add_argument("--pred-head-ratio", type=float, default=2.0)
parser.add_argument("--drop-path-rate", type=float, default=0.0)  # 미래 확장
```

## Sanity check 기준 (ep4 체크포인트)

| 지표 | 정상 범위 | collapse 신호 |
|---|---|---|
| L_M | 0.01 ~ 0.02 (v4와 유사) | 급격 상승 / NaN |
| L_P | ep1: 0.3~0.5 → ep4: 0.05~0.15 감소 | 즉시 0 근접 (collapse) |
| `cos(p_stu_i, p_tea_i)` same-sample (batch mean) | 0.4 ~ 0.8 | > 0.99 (collapse) |
| `std(p_stu)` across samples | > 0.3 | < 0.05 (constant) |
| `std(p_tea)` across samples | > 0.3 | < 0.05 (teacher도 collapse) |
| Probing R² (EgoDex) | v4(0.20) 수준 이상 | < 0 (random-init 수준) |

## 조정 여지가 있는 하이퍼파라미터 (기록)

**아래 항목들은 초기 기본값으로 시작하되, 성능/안정성 이슈 시 조정 후보임을 명확히 기록함.**

### LR (현재 기본값: **2e-4**)
- 근거: v4/v7에서 검증된 값, 8 GPU × batch 64 global 512 기준 sqrt scaling 적용값
- 조정 트리거:
  - Ep1~3에서 L_P NaN or spike → **1.5e-4**로 낮춤
  - Ep10까지 L_M/L_P 모두 하강 없음 → **3e-4**로 상향 (MAE linear scaling)
  - L_M은 정상이나 L_P 정체 → LR 변경 전에 **λ 증가** 먼저 시도

### AdamW betas (현재 기본값: **(0.9, 0.999)**)
- PyTorch AdamW 기본값
- 조정 후보: **(0.9, 0.95)** (MAE/I-JEPA 관례, longer-horizon gradient 평균)
- 고려 시점: L_P가 oscillate하거나 representation quality가 정체일 때 ablation

### λ_max (현재 기본값: **0.5**)
- 너무 크면 L_P가 L_M을 압도 → M collapse 위험
- 너무 작으면 L_P 학습 신호 약함
- 조정 범위: 0.1 ~ 1.0

### EMA τ_base (현재 기본값: **0.996**)
- 0.99 (더 빠른 teacher 변화, stability ↓) ~ 0.9995 (더 느림, 수렴 ↓)
- BYOL 0.996, I-JEPA 0.996, V-JEPA 0.998

### Mask ratio (student P block mask, 현재 기본값: **0.5**)
- I-JEPA: 0.25~0.75 실험. 고mask ratio 일수록 예측 난이도 ↑
- 초기 0.5, collapse 조짐 시 0.6으로 상향

## 트러블슈팅 플레이북

### 증상 1: L_P가 즉시 0 근처로 수렴 (collapse)

**탐지**: Ep1~2에서 L_P < 0.01, `cos(p_stu_i, p_tea_i)` > 0.99, `std(p_stu)` → 0

**대응 순서**:
1. Prediction head 구현 점검 (LN 위치, hidden dim)
2. λ warmup 연장: 5 → 10 epoch
3. Student P mask ratio 상향: 0.5 → 0.65
4. Variance regularization 추가 (VICReg-lite):
   ```python
   L_var = F.relu(1.0 - p_student.std(dim=0)).mean() * 0.1
   L = L_M + λ·L_P + L_var
   ```
5. EMA τ_base 상향: 0.996 → 0.999 (더 느린 teacher)
6. Full EMA (Option Y) 전환: M도 EMA copy

### 증상 2: L_M이 악화됨 (M stream 간접 collapse)

**탐지**: L_M이 v4 대비 상승, pixel reconstruction viz가 blur/constant

**대응**:
1. λ 감소: 0.5 → 0.2 (L_P 영향 축소)
2. CLS exchange detach 구현 재확인 (혹시 누락 시 critical)
3. L_M 가중치 상향: `L = 2·L_M + λ·L_P`

### 증상 3: 수치 불안정 (NaN, exploding)

**탐지**: 첫 수백 step에서 NaN, LayerNorm std=0 경고

**대응**:
1. **Fallback B-2**: Zero M 대신 tiny Gaussian noise (σ=0.01)
2. Grad clipping 강화: 1.0 → 0.5
3. LR warmup 연장: 10% → 20%
4. Prediction head LN에 eps 상향 (`nn.LayerNorm(D, eps=1e-5)` 등)

### 증상 4: Probing R² 저하 (v4 대비)

**탐지**: Ep4 probing R² < 0.10 (v4는 0.20 수준)

**원인 가능성**: L_P가 task-irrelevant 정보(position/카메라 motion 등) 학습 유도

**대응**:
1. λ 감소 (0.5 → 0.1)
2. Block mask 크기 축소 (난이도 ↓)
3. Teacher에도 light random mask 10% 정도 도입 (robust 요구)

### 증상 5: Unseen image 일반화 여전히 나쁨

**탐지**: EgoDex probing은 개선이지만 DROID/LIBERO 평가 저조

**대응**:
1. Pre-training mixture 도입 (v8 scope 밖이지만 근본 개선책)
2. Augmentation 강화 (color jitter, random resized crop)
3. Temporal gap 범위 확대 (`sample_dist="uniform" [5, 90]`)

### 증상 6: Teacher forward 계산 비용 과다

**탐지**: Wall time이 v4 대비 +50% 이상

**대응**:
1. Teacher batch 축소 (일부만 L_P 계산)
2. BF16 autocast가 teacher forward에도 적용되는지 점검

## 의사결정 계통도

```
Ep4 sanity check
├── L_P 정상 수렴 + Probing R² ≥ 0.15
│   └── 그대로 ep30까지 진행
├── L_P collapse (즉시 0)
│   └── 증상 1 플레이북: 1→2→3→4→5→6 순
├── L_P 정상이지만 L_M 악화
│   └── 증상 2 플레이북
├── 수치 불안정/NaN
│   └── 증상 3 플레이북 (B-2 우선)
└── 모든 loss 정상이지만 Probing 저하
    └── 증상 4/5 플레이북
```

## 기록해둘 가정과 가설

1. **L_M anchor**: L_M이 M stream representation을 non-trivial 상태로 유지. 실패 시 Y 전환
2. **Input asymmetry + EMA가 collapse 방지 충분**: student(t+motion) ≠ teacher(t+k+zero). 부족 시 VICReg 추가
3. **Zero M input 수치 안정**: `clamp(1e-6)` + LayerNorm으로 안정. 실패 시 B-2 (tiny noise)
4. **Detach가 role separation 유지**: L_P가 M stream에 못 흐르므로 M은 L_M에만 반응. 미검증이지만 논리적 필연
5. **Temporal gap 30 근방 patch alignment 유효**: EgoDex head-mounted 특성. DROID에서 악화 가능성

## Reference

- **SimSiam** (Chen & He, CVPR 2021): Shared encoder + stop_grad. 본 설계와 가장 관련. BN vs LN 논의 참고점.
- **BYOL** (Grill et al., NeurIPS 2020): EMA teacher + prediction head. Option X의 원형.
- **I-JEPA** (Assran et al., CVPR 2023): Masked prediction in representation space. Block masking, LN prediction head, τ=0.996.
- **V-JEPA** (Bardes et al., 2024): Video I-JEPA. 2-frame 실패 경험(V-JEPA-ours 3차 발산) 반영.
- **MAE** (He et al., CVPR 2022): LR scaling, AdamW betas(0.9, 0.95) 근거.
- **MoCo v3** (Chen et al., ICCV 2021): LN in prediction head.

## 진행 단계

1. **v7 ep8 결과 확인** (2026-04-20 14:45 KST 전후 예상)
2. v7 폐기 확정 시 v8 구현 시작 (이 문서 작성 시점에는 구현 착수 직전)
3. Sanity (1 GPU, 1~2 epoch) → DDP sanity (2 GPU) → full training (8 GPU, 30 epoch)
4. Ep4 체크포인트 probing으로 플레이북 분기 결정
5. 30 epoch 수렴 시 최종 probing (v4, VideoMAE-ours, v8 비교)

## 실행 기록

### Full training (JobID 33346962, 2026-04-20 14:24 시작)

- 설정: 50 epoch, λ_max=0.2, λ warmup=10ep, α_var=0.1, EMA τ_base=0.996, mask 0.3/0.5
- **현재 미검증 사항**: 학습 코드에 L_M / L_P / L_var 성분별 로그가 분리되지 않음 → λ warmup 종료 후 collapse 감지 어려움. 다음 run 전에 교정 필요

### ep4 Loss 궤적

| Epoch | Loss (weighted) | LR | λ |
|---|---|---|---|
| 1 | 0.0128 | 4.08e-05 | 0 |
| 2 | 0.0065 | 8.06e-05 | 0 |
| 3 | 0.0065 | 1.20e-04 | 0 |
| 4 | 0.0139 | 1.60e-04 | 0 |

- LR linear warmup 구간(5ep) 내, λ warmup 구간(10ep) 내 → 현재 기록된 Loss는 사실상 **L_M + α·L_var**만 반영
- v7-big collapse (0.0007)와 10배 이상 차이 → L_M 단독 학습은 정상 범위

### ep4 Attention viz (JobID 33451082)

- 결과: [attn_v8_ep4.png](attn_v8_ep4.png)
- 컬럼: Frame t / Frame t+30 / Student M attn on ΔL / Student P attn on frame t / Teacher P attn on frame t+k / Pred M
- 샘플: EgoDex 2 (`put_away_set_up_board_game/95`, `stock_unstock_fridge/1473`) + DROID 2 (`ep_000009`, `ep_000006`)
- 관찰:
  - Student M / Student P / Teacher P attention 모두 샘플별로 다른 영역 focus → 균질화 없음
  - 단, 현재는 λ=0 단계라 L_P가 representation에 아직 signal을 주지 않았으므로 collapse 판정은 **ep12 이후** (λ warmup 종료 후) viz에서 결판
  - Pred M은 flat/blur → ep4는 LR warmup 중이라 복원 품질 초기 수준

### 다음 체크포인트 viz 재제출

```
CKPT=/proj/external_group/mrg/checkpoints/two_stream_v8/20260420_142651/checkpoint_epoch0008.pt \
EPOCH_TAG=ep8 \
sbatch scripts/cluster/attn_viz_v8.sbatch
```
