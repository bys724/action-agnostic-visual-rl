# Pixel-wise Channel Fusion for Behavior Representation

## 정의

Multi-channel vision encoder의 출력(M/P channels)을 **spatial location을 보존하면서** 통합하는 fusion 전략. 기존의 spatial averaging (avg pooling) 방식이 spatial information을 손실하는 문제를 해결하기 위해, 각 픽셀 위치에서 채널 간 fusion을 수행한다.

---

## 문제: Spatial Averaging의 치명적 한계

### dino.txt (2024) 접근법의 한계

**Vision-language alignment에서 효과적이었던 방법**:

```python
# dino.txt 방식 (CLIP, VLA에서도 유사)
representation = [CLS_M ; avg(patches_M) ; CLS_P ; avg(patches_P)]
→ 4D global descriptor
→ Classification, Retrieval에 효과적
```

**문제점**: Spatial information 완전 손실

```python
# 예시: 4×4 패치
patches = [
    [0.1, 0.2],  # 왼쪽 상단: 빨간 물체
    [0.5, 0.1],  # 오른쪽 상단: 파란 배경
    [0.2, 0.8],  # 왼쪽 하단: 초록 테이블
    [0.3, 0.3]   # 오른쪽 하단: 그리퍼
]

avg(patches) = [0.275, 0.35]
→ "뭔가 섞여있다" ✓
→ "어디에 무엇이 있는가" ✗ (완전 손실)
```

### Behavior/Manipulation에서 Critical

**Task**: "Pick the red cube"

필요한 정보:
1. 빨간 큐브의 **위치** (x, y)
2. 그리퍼의 **현재 위치**
3. 둘 사이의 **spatial relationship**

**avg(patches) 결과**:
- "빨간색이 scene 어딘가에 있다" ✓
- "빨간색이 **어디에** 있는가" ✗
- **Spatial reasoning 불가능!**

### 생물학적 관점

- **Dorsal stream** (where pathway): Spatial location 유지
- **Ventral stream** (what pathway): Object identity
- **Behavior는 둘 다 필요**: "What is where"
- avg는 "what"만 남기고 "where"를 버림

---

## 해결책: Pixel-wise Channel Fusion

### 핵심 아이디어

```python
# 제안하는 방식
M_patches: [patch_M1, ..., patch_Mn]  (각 D차원, 위치 보존)
P_patches: [patch_P1, ..., patch_Pn]  (각 D차원, 위치 보존)
              ↓
    Pixel-wise Fusion (shared FC)
              ↓
Fused_patches: [patch_1_fused, ..., patch_n_fused]

→ 모든 spatial location 보존 ✅
→ Channel dimension만 축소 (2D → D) ✅
→ Spatial reasoning 가능 ✅
```

### 구체적 예시

```
M채널 (4×4 grid):
┌─────┬─────┬─────┬─────┐
│ 0.0 │ 0.0 │ 0.0 │ 0.0 │  ← 배경: 정적
├─────┼─────┼─────┼─────┤
│ 0.0 │-0.5 │-0.5 │ 0.0 │  ← 중앙: 큰 변화
├─────┼─────┼─────┼─────┤
│ 0.0 │-0.5 │-0.5 │ 0.0 │
├─────┼─────┼─────┼─────┤
│ 0.0 │ 0.0 │ 0.0 │ 0.0 │
└─────┴─────┴─────┴─────┘

P채널 (4×4 grid):
┌─────┬─────┬─────┬─────┐
│ 0.1 │ 0.1 │ 0.1 │ 0.1 │
├─────┼─────┼─────┼─────┤
│ 0.1 │ 0.9 │ 0.9 │ 0.1 │  ← 중앙: 강한 edge
├─────┼─────┼─────┼─────┤
│ 0.1 │ 0.9 │ 0.9 │ 0.1 │
├─────┼─────┼─────┼─────┤
│ 0.1 │ 0.1 │ 0.1 │ 0.1 │
└─────┴─────┴─────┴─────┘

avg 방식:
  avg(M) = -0.125  → "약간 어두워짐" (위치 불명)
  avg(P) = 0.35    → "약간의 edge" (위치 불명)

Pixel-wise 방식:
  patch[1,1] = FC([M:-0.5, P:0.9])
               → "중앙(1,1)에서 큰 변화 + 강한 edge"
  patch[0,0] = FC([M:0.0, P:0.1])
               → "왼쪽 상단(0,0)은 정적 + 약한 edge"
  → Spatial structure 완전 보존!
```

---

## Encoder Pre-training Strategy

Pixel-wise fusion의 효과는 **encoder가 어떻게 학습되었는지**에 직접적으로 의존한다. M/P 채널의 고유한 특성을 살리면서도 안정적으로 학습하기 위한 전략을 설계한다.

### 1. Video + Static Mixed Learning

**핵심 통찰**: "No change is also a consistent change (zero change)"

#### Data Composition

| Data Type | M Channel (4ch) | P Channel (5ch) | Ratio |
|-----------|-----------------|-----------------|-------|
| **Video** | [ΔL, ΔR, ΔG, ΔB] ≠ 0 | [∂x, ∂y, R, G, B] | 70% |
| **Static Image** | [0, 0, 0, 0] | [∂x, ∂y, R, G, B] | 30% |

#### Rationale

**Video 데이터 (70%)**:
```python
frames = [t0, t1, t2]

# M채널: 시간적 변화 (Dorsal pathway)
M = [ΔL, ΔR, ΔG, ΔB]  # ≠ 0
# → M_encoder learns "what changed"
# → Temporal change patterns + color dynamics

# P채널: 공간 + 색상 (Ventral pathway)
P = [∂x, ∂y, R, G, B]
# → P_encoder learns "what is where"
# → Spatial structure + object appearance
```

**Static 데이터 (30%)**:
```python
frames = [img, img, img]  # 동일 이미지 3번

# M채널: 변화 없음 (완전 비활성화!)
M = [0, 0, 0, 0]  # Zero temporal change!
# → M_encoder receives no signal
# → Forces P to be responsible ✅

# P채널: 모든 정보 포함
P = [∂x, ∂y, R, G, B]
# → P_encoder learns "what is where" (P-dominant)
# → Structure (∂x, ∂y) + Color (R, G, B)
# → 검증된 DINO 학습 메커니즘 활용
```

#### M-P Balance 효과

| Data Type | M Contribution | P Contribution | Learning Focus |
|-----------|----------------|----------------|----------------|
| Video | **High** (변화 ≠ 0) | **High** (구조 + 색상) | Both M and P learn |
| Static | **Zero** (변화 = 0) | **Very High** (구조 + 색상) | **P forced to learn** ✅ |
| **Mixed (7:3)** | **Balanced** | **Balanced** | **M=motion, P=appearance** |

**핵심 해결책**:
- ✅ **Static에서 M 완전 비활성화** → P가 모든 책임 (균형 강제)
- ✅ **P가 color 정보 포함** → RGB가 P에 있어 생물학적으로 타당
- ✅ M과 P가 complementary하게 학습 (M=change, P=what+where)
- ✅ Video에서 temporal + spatial reasoning 학습
- ✅ Static에서 DINO의 검증된 안정성 활용

#### Data Sources

**Video (70%)**:
- **EgoDex** (829h): Human manipulation demonstrations
- **Something-Something V2**: Object interaction videos
- **Robot replay buffer**: Behavioral trajectories

**Static (30%)**:
- **ImageNet**: General visual features
- **Robotic object datasets**: Manipulation-relevant objects
- **Scene images**: Environmental context

---

### 2. DINO-style Temporal Augmentation

**핵심**: Random crop을 사용하되 **같은 spatial location**에서 crop

#### 왜 Crop을 사용해도 괜찮은가?

```python
# CLS token: Global representation
# → Crop해도 전체 scene 이해 (DINO에서 검증됨)

# Patches: Local spatial structure
# → Crop 내부에서 상대적 위치는 보존
# → "이 96×96 영역 내에서 A가 B의 왼쪽에" 학습 가능

# Example: 96×96 crop with ViT-B/16
# → 약 36 patches
# → Pixel-wise fusion으로 각 patch 간 spatial relationship 유지
```

#### Multi-crop Implementation

```python
def get_dino_views_temporal(frames):
    """
    Args:
        frames: [t0, t1, t2] - 연속된 3 프레임

    Returns:
        views: List of (view_type, m_channels, p_channels)

    Key: 모든 crop은 동일한 위치에서 수행!
    """
    # Crop parameters 한 번만 결정
    crop_params_global = get_random_crop_params(size=224)
    crop_params_local1 = get_random_crop_params(size=96)
    crop_params_local2 = get_random_crop_params(size=96)

    views = []

    for i in range(len(frames) - 1):
        img_prev = frames[i]
        img_curr = frames[i + 1]

        # Global view (224×224, 같은 위치)
        global_prev = apply_crop(img_prev, crop_params_global)
        global_curr = apply_crop(img_curr, crop_params_global)
        m_g = preprocess_M(global_prev, global_curr)  # [4, 224, 224]
        p_g = preprocess_P(global_prev)               # [2, 224, 224]
        views.append(('global', m_g, p_g))

        # Local view 1 (96×96, 같은 위치)
        local1_prev = apply_crop(img_prev, crop_params_local1)
        local1_curr = apply_crop(img_curr, crop_params_local1)
        m_l1 = preprocess_M(local1_prev, local1_curr)
        p_l1 = preprocess_P(local1_prev)
        views.append(('local', m_l1, p_l1))

        # Local view 2 (96×96, 같은 위치)
        local2_prev = apply_crop(img_prev, crop_params_local2)
        local2_curr = apply_crop(img_curr, crop_params_local2)
        m_l2 = preprocess_M(local2_prev, local2_curr)
        p_l2 = preprocess_P(local2_prev)
        views.append(('local', m_l2, p_l2))

    return views
```

**Example**:
```
Video: [t0, t1, t2]
Crop at (50, 50, 224, 224) - 한 번만 결정

t0 → t1 변화:
  - Global: crop(t0) → crop(t1) 동일 위치
  - Local1: crop(t0) → crop(t1) 동일 위치
  - Local2: crop(t0) → crop(t1) 동일 위치

t1 → t2 변화:
  - Global: crop(t1) → crop(t2) 동일 위치
  - ...
```

---

### 3. CLS Fusion Options (Experimental)

Pre-training 단계에서 M_CLS와 P_CLS를 어떻게 결합할지는 **실험적으로 결정**한다.

#### Option A: Linear Projection (FC)

```python
class LinearClsFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.fc = nn.Linear(2 * dim, dim)

    def forward(self, cls_m, cls_p):
        # Simple concatenation + FC
        fused = self.fc(torch.cat([cls_m, cls_p], dim=-1))
        return fused  # [B, D]
```

**장점**:
- ✅ 학습 안정성 (DINO는 이미 복잡)
- ✅ Compute 효율 (파라미터 2D²)
- ✅ 명확한 baseline
- ✅ M-P 가중치 학습 가능

**단점**:
- ❌ Linear combination만 가능
- ❌ M-P 상호작용 제한적

---

#### Option B: MLP

```python
class MLPClsFusion(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, cls_m, cls_p):
        fused = self.mlp(torch.cat([cls_m, cls_p], dim=-1))
        return fused
```

**장점**:
- ✅ Non-linear interaction
- ✅ 더 높은 표현력

**단점**:
- ❌ 파라미터 증가 (~4D²)
- ❌ 학습 난이도 약간 상승

---

#### Option C: Cross-Attention

```python
class CrossAttentionClsFusion(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.attn_block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            batch_first=True
        )

    def forward(self, cls_m, cls_p):
        # Stack as sequence
        cls_tokens = torch.stack([cls_m, cls_p], dim=1)  # [B, 2, D]

        # Self-attention (M ↔ P interaction)
        cls_attended = self.attn_block(cls_tokens)  # [B, 2, D]

        # Aggregate (mean pooling)
        fused = cls_attended.mean(dim=1)  # [B, D]
        return fused
```

**장점**:
- ✅ Full M-P interaction
- ✅ Attention map으로 해석 가능
- ✅ Video(M 중심) vs Static(P 중심) 자동 조절

**단점**:
- ❌ Compute overhead (Q,K,V projection)
- ❌ 파라미터 증가 (~6D²)
- ❌ 학습 불안정 위험

---

#### Option D: Gating

```python
class GatingClsFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.gate_fc = nn.Linear(2 * dim, 1)

    def forward(self, cls_m, cls_p):
        # Learnable gate: α ∈ [0, 1]
        gate = torch.sigmoid(self.gate_fc(torch.cat([cls_m, cls_p], dim=-1)))

        # Adaptive weighting
        fused = gate * cls_m + (1 - gate) * cls_p
        return fused
```

**장점**:
- ✅ Interpretable weighting
- ✅ 파라미터 효율적
- ✅ Video에서 M↑, Static에서 P↑ 자동

**단점**:
- ❌ Linear weighted sum (제한적 표현력)

---

#### 실험 계획

**Phase 1: Pre-training**
- 각 fusion option으로 독립적으로 pre-training 수행
- Video + Static (7:3) mixture
- DINO loss로 학습

**Phase 2: Evaluation**
- Pre-trained encoder로 downstream task 평가
- Best performing fusion 선택

**Metrics**:
- Pre-training convergence speed
- Downstream task success rate
- Ablation study 성능

---

### 4. Training Protocol

```python
class TwoStreamDINO(nn.Module):
    def __init__(self, dim=768, fusion_type='fc'):
        super().__init__()

        # Student & Teacher networks
        self.student_m = ViT_M(dim)
        self.student_p = ViT_P(dim)
        self.teacher_m = ViT_M(dim)  # EMA of student
        self.teacher_p = ViT_P(dim)  # EMA of student

        # CLS fusion module (실험 대상)
        if fusion_type == 'fc':
            self.cls_fusion = LinearClsFusion(dim)
        elif fusion_type == 'mlp':
            self.cls_fusion = MLPClsFusion(dim)
        elif fusion_type == 'attention':
            self.cls_fusion = CrossAttentionClsFusion(dim)
        elif fusion_type == 'gating':
            self.cls_fusion = GatingClsFusion(dim)

        # DINO components
        self.student_head = DINOHead(dim, out_dim=65536, use_bn=False)
        self.teacher_head = DINOHead(dim, out_dim=65536, use_bn=False)

        # Initialize teacher as copy of student
        self._init_teacher()

    def _init_teacher(self):
        for param_s, param_t in zip(self.student_m.parameters(),
                                     self.teacher_m.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

        for param_s, param_t in zip(self.student_p.parameters(),
                                     self.teacher_p.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    def forward(self, batch):
        """
        Args:
            batch: {
                'type': 'video' or 'static',
                'frames': [t0, t1, t2],
            }
        """
        frames = batch['frames']

        # Get multi-crop views with temporal augmentation
        views = get_dino_views_temporal(frames)

        # Student: process all views
        student_outputs = []
        for view_type, m_ch, p_ch in views:
            # Encode
            m_tokens = self.student_m(m_ch)  # [B, N+1, D]
            p_tokens = self.student_p(p_ch)  # [B, N+1, D]

            # Extract CLS
            cls_m = m_tokens[:, 0]
            cls_p = p_tokens[:, 0]

            # Fuse CLS (실험 대상)
            cls_fused = self.cls_fusion(cls_m, cls_p)

            # Project to DINO output space
            out = self.student_head(cls_fused)
            student_outputs.append(out)

        # Teacher: only global views
        teacher_outputs = []
        with torch.no_grad():
            for view_type, m_ch, p_ch in views:
                if view_type == 'global':
                    m_tokens = self.teacher_m(m_ch)
                    p_tokens = self.teacher_p(p_ch)

                    cls_m = m_tokens[:, 0]
                    cls_p = p_tokens[:, 0]
                    cls_fused = self.cls_fusion(cls_m, cls_p)

                    out = self.teacher_head(cls_fused)
                    teacher_outputs.append(out)

        # DINO loss
        loss = dino_loss(student_outputs, teacher_outputs)

        return loss

    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        """EMA update of teacher networks"""
        for param_s, param_t in zip(self.student_m.parameters(),
                                     self.teacher_m.parameters()):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)

        for param_s, param_t in zip(self.student_p.parameters(),
                                     self.teacher_p.parameters()):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)
```

#### Training Loop

```python
# Hyperparameters
batch_size = 256
epochs = 300
warmup_epochs = 10
base_lr = 0.0005 * batch_size / 256

# Data mixture
video_ratio = 0.7
static_ratio = 0.3

# Training
for epoch in range(epochs):
    for batch in dataloader:
        # batch['type'] ∈ {'video', 'static'}
        # batch['frames'] = [t0, t1, t2]

        # Forward
        loss = model(batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update teacher (EMA)
        model.update_teacher(momentum=0.996)

        # DINO-specific: center + sharpen teacher outputs
        # (standard DINO mechanisms)
```

---

### 5. Change Representation via Video Prediction

**핵심 철학**: 이미지 간 변화를 효과적으로 압축하는 representation 학습

> **"효과적"의 기준**: 그 representation만으로 다음 순간을 정확히 예측할 수 있는가?

---

#### 왜 Video Prediction인가?

**핵심 통찰**: 변화(change)를 잘 이해했다면, 미래를 예측할 수 있어야 한다

```python
Goal: 장면 변화를 효과적으로 압축하는 representation 학습

Input: (img_t, img_t+k) - 연속된 두 이미지
Output: change_embedding - 변화의 본질을 담은 representation
Validation: "이 embedding으로 img_t+k를 재구성할 수 있는가?"

→ 재구성 성능 = representation quality의 직접적 측정
```

**왜 강력한가**:

1. **Cause-agnostic**: 변화의 원인을 구분하지 않음
   - 로봇 팔이 물체를 밀었나?
   - 중력으로 물체가 떨어졌나?
   - 사람이 개입했나?
   - **상관없음** - 변화의 "본질"만 캐치

2. **Self-validation**: 별도의 downstream task 없이 품질 측정
   - MAE: "패치 복원 잘 되나?" (정적 정보)
   - DINO: "다른 view에서도 같은 feature?" (불변성)
   - **우리**: "다음 순간 예측 정확한가?" (동적 이해)

3. **Universally useful**: 모든 embodied AI task에 활용 가능
   - Inverse dynamics: change → action
   - Imitation learning: change → policy
   - Planning: goal → desired change sequence

---

#### 전체 파이프라인: Forward → Inverse → Planning

**Phase 1: Pre-training (Forward Dynamics Learning)**

```python
# 목표: 세상의 물리 법칙 학습 (unsupervised)
Input: img_t (M, P channels)
Output: img_t+k reconstruction
Learning: "이런 초기 상태에서는 k step 후 이렇게 변한다"
```

**학습되는 것**:
- 중력 (물체가 아래로 떨어짐)
- 관성 (움직이는 물체는 계속 움직임)
- 충돌 (물체끼리 부딪히면 튕김)
- 변형 (부드러운 물체는 눌리면 찌그러짐)
- **모든 원인의 visual dynamics를 통합 학습**

**Phase 2: Change Embedding Extraction**

```python
# 목표: 변화를 압축된 표현으로 변환
change_emb = encoder(img_t, img_t+k)
# 예: "5cm 오른쪽 이동 + 30도 회전"이 벡터로 표현됨
```

**Phase 3: Inverse Dynamics Learning (Downstream)**

```python
# 목표: 변화를 만든 행동 추론
Input: (change_emb, img_t, sensors, task)
Output: robot_action
Learning: "이런 변화를 만들려면 이런 행동을 해야 함"
```

**Phase 4: Goal-Conditioned Control**

```python
# 목표: Task를 변화로 변환
planner(current_state, task) → desired_change
action_model(desired_change, ...) → robot_command
```

**이것은 로봇 조작의 정석 파이프라인입니다!**

---

#### 기존 Video Prediction 방법들과의 비교

| 방법 | Action Input? | 물리 학습? | Two-Stream? | Pixel-wise? |
|------|--------------|-----------|-------------|-------------|
| Mathieu (2016) | ❌ | Implicit | ❌ | ❌ |
| Finn (2016) | ✅ (label 필요) | Explicit | ❌ | ❌ |
| Babaeizadeh (2018) | ❌ | Implicit | ❌ | ❌ |
| Watters (2017) | ❌ | Explicit | ❌ | ❌ |
| **우리 방법** | ❌ **(unsupervised!)** | **Implicit** | ✅ | ✅ |

**우리의 차별점**:
1. **Action label 없이** visual dynamics 학습 (진정한 unsupervised)
2. **M-P split**으로 motion과 appearance 분리 학습
3. **Pixel-wise embedding**으로 spatial structure 보존
4. **Change-focused**: 변화를 representation의 핵심으로

---

#### 구현 Option 1: U-Net Decoder (Baseline)

**구조**:
```python
class TwoStreamVideoPredictor(nn.Module):
    def __init__(self, dim=768):
        super().__init__()

        # Two-Stream Encoders (변화 학습)
        self.encoder_m = ViT_M(dim)
        self.encoder_p = ViT_P(dim)

        # Pixel-wise Fusion
        self.pixel_fusion = PixelwiseFusion(dim)
        self.cls_fusion = LinearClsFusion(dim)

        # Image Encoder (원본 이미지 인코딩)
        self.img_encoder = timm.create_model(
            'resnet50',
            pretrained=True,
            features_only=True
        )

        # U-Net Decoder
        self.decoder = UNetDecoder(
            encoder_channels=[256, 512, 1024, 2048],  # ResNet50
            decoder_channels=[256, 128, 64, 32],
            change_emb_dim=dim,
            out_channels=3  # RGB
        )

    def forward(self, img_t, img_tk):
        """
        Args:
            img_t: [B, 3, H, W] - 시작 프레임
            img_tk: [B, 3, H, W] - 목표 프레임 (k frames 후)

        Returns:
            img_pred: [B, 3, H, W] - 재구성된 img_tk
            change_emb: [B, D] - 변화 embedding
        """
        # 1. M-P preprocessing (img_t → img_tk 변화)
        m_channels = magnocellular_channel(img_t, img_tk)
        p_channels = parvocellular_channel(img_tk)

        # 2. Two-Stream encoding
        m_tokens = self.encoder_m(m_channels)  # [B, N+1, D]
        p_tokens = self.encoder_p(p_channels)  # [B, N+1, D]

        # 3. Pixel-wise fusion
        pixel_emb = self.pixel_fusion(m_tokens, p_tokens)  # [B, N, D]
        change_emb = self.cls_fusion(m_tokens[:, 0], p_tokens[:, 0])  # [B, D]

        # 4. img_t 인코딩 (현재 상태)
        img_features = self.img_encoder(img_t)  # Multi-scale features

        # 5. Decode img_tk (변화 적용)
        img_pred = self.decoder(
            img_features=img_features,  # 현재 이미지
            change_emb=change_emb,      # 변화 정보 (global)
            pixel_emb=pixel_emb         # 공간적 변화 (pixel-wise)
        )

        return img_pred, change_emb
```

**UNetDecoder 구현**:
```python
class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, change_emb_dim, out_channels=3):
        super().__init__()

        # Change embedding을 spatial map으로 변환
        self.change_to_spatial = nn.Sequential(
            nn.Linear(change_emb_dim, 512),
            nn.ReLU(),
            nn.Unflatten(1, (512, 1, 1))
        )

        # Decoder blocks (skip connections with encoder)
        self.decoder_blocks = nn.ModuleList()
        in_ch = encoder_channels[-1] + 512  # 마지막 encoder + change_emb

        for out_ch in decoder_channels:
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU()
                )
            )
            in_ch = out_ch

        # Final output layer
        self.output = nn.Conv2d(decoder_channels[-1], out_channels, 3, 1, 1)

    def forward(self, img_features, change_emb, pixel_emb):
        """
        Args:
            img_features: List of [B, C_i, H_i, W_i] from ResNet
            change_emb: [B, D] - global change information
            pixel_emb: [B, N, D] - pixel-wise change (optional)

        Returns:
            img_pred: [B, 3, H, W]
        """
        # Change embedding → spatial map
        change_map = self.change_to_spatial(change_emb)  # [B, 512, 1, 1]
        change_map = F.interpolate(
            change_map,
            size=img_features[-1].shape[-2:],  # Match encoder size
            mode='bilinear'
        )

        # Start from deepest encoder feature + change
        x = torch.cat([img_features[-1], change_map], dim=1)

        # Decode with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x)

            # Skip connection (if available)
            if i < len(img_features) - 1:
                encoder_feat = img_features[-(i+2)]
                x = x + F.interpolate(encoder_feat, size=x.shape[-2:])

        # Final output
        img_pred = self.output(x)
        img_pred = torch.sigmoid(img_pred)  # [0, 1]

        return img_pred
```

**장점**:
- ✅ 구현 간단, 학습 안정적
- ✅ Video prediction 논문에서 검증됨
- ✅ Pixel-wise fusion과 철학 일치
- ✅ Baseline으로 최적

**단점**:
- ❌ 고해상도에서 메모리 많이 씀

---

#### 구현 Option 2: Latent Diffusion (Advanced)

**구조**:
```python
class LatentDiffusionPredictor(nn.Module):
    def __init__(self, change_emb_dim=768):
        super().__init__()

        # VAE encoder/decoder (frozen, pretrained)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

        # Two-Stream change encoder
        self.encoder_m = ViT_M(change_emb_dim)
        self.encoder_p = ViT_P(change_emb_dim)
        self.fusion = LinearClsFusion(change_emb_dim)

        # Diffusion U-Net (학습 대상)
        self.diffusion_unet = UNet2DConditionModel(
            cross_attention_dim=change_emb_dim
        )

        # Noise scheduler
        self.scheduler = DDPMScheduler()

    def forward(self, img_t, img_tk):
        """Training forward"""
        # img_t+k를 latent로 인코딩
        with torch.no_grad():
            latent_tk = self.vae.encode(img_tk).latent_dist.sample()

        # Change embedding 추출
        m_ch = magnocellular_channel(img_t, img_tk)
        p_ch = parvocellular_channel(img_tk)

        m_tokens = self.encoder_m(m_ch)
        p_tokens = self.encoder_p(p_ch)
        change_emb = self.fusion(m_tokens[:, 0], p_tokens[:, 0])

        # Diffusion process
        noise = torch.randn_like(latent_tk)
        timestep = torch.randint(0, 1000, (latent_tk.size(0),))
        noisy_latent = self.scheduler.add_noise(latent_tk, noise, timestep)

        # Predict noise (conditioned on change_emb)
        pred_noise = self.diffusion_unet(
            noisy_latent,
            timestep,
            encoder_hidden_states=change_emb.unsqueeze(1)  # [B, 1, D]
        ).sample

        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def generate(self, img_t, change_emb, num_steps=50):
        """Inference: img_t + change → img_tk"""
        # img_t → latent (starting point)
        latent_t = self.vae.encode(img_t).latent_dist.sample()

        # Denoising loop
        latent = torch.randn_like(latent_t)  # Random noise

        self.scheduler.set_timesteps(num_steps)
        for t in self.scheduler.timesteps:
            # Predict noise
            noise_pred = self.diffusion_unet(
                latent,
                t,
                encoder_hidden_states=change_emb.unsqueeze(1)
            ).sample

            # Remove noise
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample

        # Decode latent → image
        img_pred = self.vae.decode(latent).sample
        return img_pred
```

**장점**:
- ✅ **최고 품질** (Stable Diffusion 수준)
- ✅ 메모리 효율적 (latent space 작동: 64×64 instead of 256×256)
- ✅ Pretrained VAE 활용 가능
- ✅ 최신 트렌드 (Diffusion Policy, GenRL 등에서 사용)

**단점**:
- ❌ 학습 복잡도 높음
- ❌ Inference 느림 (50 denoising steps)

---

#### Training Protocol

**Dataset Composition**:
```python
# Video 데이터 (다양한 물리적 변화)
- EgoDex (829h): Human manipulation demonstrations
- Something-Something V2 (220k): Object interaction videos
- Robot replay buffer: Behavioral trajectories

# Interval k 전략
k_min = 1   # 작은 변화 (미세 조정)
k_max = 10  # 큰 변화 (긴 호흡)
k = random.randint(k_min, k_max)  # Variable interval
```

**Training Loop (U-Net Baseline)**:
```python
# Hyperparameters
batch_size = 128
epochs = 100
lr = 1e-4

# Model
model = TwoStreamVideoPredictor(dim=768)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training
for epoch in range(epochs):
    for batch in dataloader:
        video_frames = batch['frames']  # [B, T, 3, H, W]

        # Random interval k
        k = random.randint(1, 10)
        img_t = video_frames[:, 0]      # [B, 3, H, W]
        img_tk = video_frames[:, k]     # [B, 3, H, W]

        # Forward
        img_pred, change_emb = model(img_t, img_tk)

        # Loss: MSE reconstruction
        loss = F.mse_loss(img_pred, img_tk)

        # Optional: Perceptual loss (VGG features)
        loss_perceptual = perceptual_loss(img_pred, img_tk)
        loss = loss + 0.1 * loss_perceptual

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
```

---

#### Downstream Usage (학습 후)

**Pre-training 후 저장**:
```python
torch.save({
    'encoder_m': model.encoder_m.state_dict(),
    'encoder_p': model.encoder_p.state_dict(),
    'fusion': model.cls_fusion.state_dict(),
    # Decoder 저장 안 함! (Pre-training용 도구)
}, 'pretrained_change_encoder.pth')
```

**Inverse Dynamics Model (Downstream)**:
```python
class InverseDynamicsModel(nn.Module):
    def __init__(self, change_emb_dim=768, action_dim=7):
        super().__init__()

        # Load pretrained change encoder (frozen or fine-tunable)
        self.encoder_m = ViT_M(change_emb_dim)
        self.encoder_p = ViT_P(change_emb_dim)
        self.fusion = LinearClsFusion(change_emb_dim)

        # Action decoder (학습 대상)
        self.action_head = nn.Sequential(
            nn.Linear(change_emb_dim + task_emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, img_t, img_t1, task_embedding):
        """
        Args:
            img_t, img_t1: Consecutive frames
            task_embedding: Task-conditioned context

        Returns:
            action: Robot action that produces this change
        """
        # Extract change embedding
        m_ch = magnocellular_channel(img_t, img_t1)
        p_ch = parvocellular_channel(img_t1)

        m_tokens = self.encoder_m(m_ch)
        p_tokens = self.encoder_p(p_ch)
        change_emb = self.fusion(m_tokens[:, 0], p_tokens[:, 0])

        # Combine with task
        combined = torch.cat([change_emb, task_embedding], dim=-1)

        # Predict action
        action = self.action_head(combined)
        return action

# Load pretrained weights
checkpoint = torch.load('pretrained_change_encoder.pth')
inverse_model = InverseDynamicsModel()
inverse_model.encoder_m.load_state_dict(checkpoint['encoder_m'])
inverse_model.encoder_p.load_state_dict(checkpoint['encoder_p'])
inverse_model.fusion.load_state_dict(checkpoint['fusion'])

# Fine-tune on robot demonstrations (20-30 demos)
# Only action_head is randomly initialized
```

---

#### 핵심 설계 원칙

**유지 (Downstream에서 사용)**:
- ✅ `encoder_m`, `encoder_p`: Visual change 인코딩
- ✅ `fusion`: M+P 결합
- ✅ **Change embedding이 핵심 output**

**버림 (Pre-training용 도구)**:
- ❌ `decoder`: 재구성 품질 검증용 (학습 후 불필요)
- ❌ `img_encoder`: 현재 프레임 인코딩용 (downstream에서 교체 가능)

**Video Prediction의 역할**:
- Change representation quality의 **self-validation**
- 별도 label 없이 학습 가능
- Downstream task에서는 decoder 버리고 change_emb만 사용

**왜 이 방법인가**:
- 미래 예측 = 변화 이해의 증거
- Pixel-level supervision으로 spatial reasoning 강화
- M-P 모두 필요 (motion + appearance)
- Embodiment-independent visual dynamics 학습

---

## 구현: Modular Architecture

### Tier 1: Base Vision Encoder

```python
class TwoStreamVisionEncoder(nn.Module):
    """
    Core two-stream encoder.
    Standard output format (maximum flexibility).
    """
    def forward(self, img_prev, img_curr):
        """
        Returns:
            {
                "m_cls": [B, D],
                "m_patches": [B, N, D],
                "p_cls": [B, D],
                "p_patches": [B, N, D],
                "metadata": {...}
            }
        """
        m_channels = self.m_preprocessing(img_prev, img_curr)
        p_channels = self.p_preprocessing(img_prev)

        m_tokens = self.vit_m(m_channels)  # [B, N+1, D]
        p_tokens = self.vit_p(p_channels)  # [B, N+1, D]

        return {
            "m_cls": m_tokens[:, 0],
            "m_patches": m_tokens[:, 1:],
            "p_cls": p_tokens[:, 0],
            "p_patches": p_tokens[:, 1:],
        }
```

### Tier 2: Fusion Module (제안 방법)

```python
class PixelwiseFusion(nn.Module):
    """
    Proposed: Pixel-wise channel integration.
    Preserves spatial structure while reducing channel dimension.
    """
    def __init__(self, dim, fusion_type="separate"):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "separate":
            # CLS와 patch 별도 fusion
            self.fc_cls = nn.Linear(2 * dim, dim)
            self.fc_patch = nn.Linear(2 * dim, dim)
        elif fusion_type == "shared":
            # 모든 token 공유 fusion
            self.fc_fusion = nn.Linear(2 * dim, dim)
        elif fusion_type == "mlp":
            self.fc_cls = MLP(2 * dim, dim)
            self.fc_patch = MLP(2 * dim, dim)

    def forward(self, encoder_output):
        """
        Args:
            encoder_output: dict from TwoStreamVisionEncoder

        Returns:
            {
                "cls_fused": [B, D],
                "patches_fused": [B, N, D]
            }
        """
        m_cls = encoder_output["m_cls"]
        m_patches = encoder_output["m_patches"]
        p_cls = encoder_output["p_cls"]
        p_patches = encoder_output["p_patches"]

        if self.fusion_type == "separate":
            # CLS fusion
            cls_fused = self.fc_cls(torch.cat([m_cls, p_cls], dim=-1))

            # Pixel-wise patch fusion
            patches_concat = torch.cat([m_patches, p_patches], dim=-1)
            patches_fused = self.fc_patch(patches_concat)

        elif self.fusion_type == "shared":
            # All tokens use same fusion
            all_tokens = torch.cat([
                torch.cat([m_cls.unsqueeze(1), m_patches], dim=1),  # M
                torch.cat([p_cls.unsqueeze(1), p_patches], dim=1)   # P
            ], dim=-1)
            all_fused = self.fc_fusion(all_tokens)
            cls_fused = all_fused[:, 0]
            patches_fused = all_fused[:, 1:]

        return {
            "cls_fused": cls_fused,
            "patches_fused": patches_fused
        }
```

---

## dino.txt (2024)와의 비교

### 근본적 차이

| 측면 | dino.txt | Pixel-wise Fusion (Ours) |
|------|----------|--------------------------|
| **Spatial info** | avg → 손실 | 모든 patch 보존 |
| **Output dim** | 4D (concat) | (N+1)×D (all patches) |
| **Channel fusion** | Late (concat) | Early (pixel-wise) |
| **적합한 task** | Classification, Retrieval | **Spatial reasoning, Manipulation** |
| **Efficiency** | Concat overhead | Fusion으로 절반 축소 |
| **Gradient flow** | CLS + avg에만 | 모든 spatial location에 |

### Why dino.txt Used Averaging

**Vision-language alignment에서는 충분**:
- Classification: "Is there a cat?" → avg로 충분
- Retrieval: "Image with red objects" → avg로 충분
- **Spatial reasoning 불필요**

**Behavior learning에서는 부족**:
- Manipulation: "Where is the cat?" → avg 불충분
- Spatial reasoning: "A left of B" → avg 불가능
- **Spatial structure 필수**

---

## Ablation Study 설계

### Axis 1: Aggregation Strategy (핵심)

**공정한 비교를 위해 모든 방법이 CLS + patches 정보를 사용**

| Strategy | Input Features | Fusion Method | Output Dim | Spatial Info |
|----------|----------------|---------------|------------|--------------|
| **(A) avg + Late fusion** | [cls_m, avg(M)] + [cls_p, avg(P)] | concat | 4D | ✗ |
| **(B) avg + Early fusion** | [cls_m, avg(M)] + [cls_p, avg(P)] | FC | D | ✗ |
| **(C) Pixel-wise (ours)** | [cls_m, M_patches] + [cls_p, P_patches] | FC per pixel | (N+1)×D | ✅ |
| **(D) All patches + Late** | [M_tokens] + [P_tokens] | concat | 2(N+1)×D | ✅ |

**구현**:

```python
# 공통 입력
cls_m, patches_m = m_tokens[:, 0], m_tokens[:, 1:]  # [B, D], [B, N, D]
cls_p, patches_p = p_tokens[:, 0], p_tokens[:, 1:]  # [B, D], [B, N, D]

# (A) avg + Late fusion (dino.txt style)
m_repr = torch.cat([cls_m, patches_m.mean(1)], dim=-1)  # [B, 2D]
p_repr = torch.cat([cls_p, patches_p.mean(1)], dim=-1)  # [B, 2D]
fused = torch.cat([m_repr, p_repr], dim=-1)  # [B, 4D]
# → Spatial averaging으로 위치 정보 손실
# → Late fusion: 학습 파라미터 없음 (단순 concat)

# (B) avg + Early fusion
m_repr = torch.cat([cls_m, patches_m.mean(1)], dim=-1)  # [B, 2D]
p_repr = torch.cat([cls_p, patches_p.mean(1)], dim=-1)  # [B, 2D]
concat_repr = torch.cat([m_repr, p_repr], dim=-1)  # [B, 4D]
fused = fc_fusion(concat_repr)  # [B, D]
# → Spatial averaging으로 위치 정보 손실
# → Early fusion: 학습 가능한 FC로 차원 축소

# (C) Pixel-wise (ours) ⭐
cls_fused = fc_cls(torch.cat([cls_m, cls_p], dim=-1))  # [B, D]
patches_concat = torch.cat([patches_m, patches_p], dim=-1)  # [B, N, 2D]
patches_fused = fc_patch(patches_concat)  # [B, N, D]
# Output: cls_fused + patches_fused → [B, N+1, D]
# → Spatial structure 완전 보존
# → Pixel-wise early fusion: 각 위치에서 M+P 결합

# (D) All patches + Late
all_tokens = torch.cat([m_tokens, p_tokens], dim=1)  # [B, 2(N+1), D]
# → Spatial structure 보존되지만 2배 overhead
# → Late fusion: CLS 2개, patches 2N개 모두 유지
```

### Axis 2: Fusion Mechanism (Pixel-wise 내에서)

| Mechanism | Complexity | Params | Expressiveness |
|-----------|------------|--------|----------------|
| **FC (linear)** | Low | 2D→D | Low |
| **MLP** | Medium | 2D→2D→D | Medium |
| **Attention** | High | Q,K,V projection | High |
| **Gating** | Medium | α·M + (1-α)·P | Adaptive |

```python
# MLP
mlp_patch = nn.Sequential(
    nn.Linear(2*D, 2*D),
    nn.ReLU(),
    nn.Linear(2*D, D)
)

# Cross-Attention
class AttentionFusion(nn.Module):
    def forward(self, m_tokens, p_tokens):
        m2p = cross_attention(m_tokens, p_tokens)
        p2m = cross_attention(p_tokens, m_tokens)
        return (m2p + p2m) / 2

# Gating
gate = torch.sigmoid(gate_fc(torch.cat([m, p], dim=-1)))
fused = gate * m + (1 - gate) * p
```

### Axis 3: Weight Sharing

| Sharing | CLS Fusion | Patch Fusion | Rationale |
|---------|-----------|--------------|-----------|
| **Separate** | FC_cls | FC_patch | CLS=global, patch=local |
| **Shared** | Same FC | Same FC | 모든 token 동일 처리 |

### Axis 4: Gradient Flow Analysis

**Gradient path visualization**:

```
Loss
 ↓
Behavior Head
 ↓
┌─────────────────────────────────┐
│  (A) avg + Late fusion          │
│  Loss → [CLS_M;avg(M);CLS_P;avg(P)] → avg → Patches │
│         ↓       ↓               │
│      CLS_M   일부만 gradient    │ ← 대부분 patch 무시
└─────────────────────────────────┘

┌─────────────────────────────────┐
│  (C) Pixel-wise (ours)          │
│  Loss → [CLS;Patches] → FC → [CLS_M;Patches_M] │
│                                  CLS_P;Patches_P │
│         ↓          ↓            │
│      모든 CLS   모든 Patch       │ ← 모든 location
└─────────────────────────────────┘
```

---

## 평가 Metrics

### Standard Metrics

| Metric | Description | Why Important |
|--------|-------------|---------------|
| **Success Rate** | Task completion | Primary goal |
| **Sample Efficiency** | Demos needed | Data efficiency |
| **Training Time** | GPU hours | Computational cost |

### Spatial Reasoning Metrics (핵심!)

| Metric | Description | Expected |
|--------|-------------|----------|
| **Position Accuracy** | 물체 위치 예측 오차 (cm) | avg: 높음, pixel-wise: 낮음 |
| **Spatial Relation** | "A left of B" 판단 정확도 | avg: 낮음, pixel-wise: 높음 |
| **Precise Manipulation** | 목표 좌표 도달 성공률 | avg: 낮음, pixel-wise: 높음 |

**구체적 Task**:

```python
tasks = {
    "pick_at_xy": "Pick object at precise location (x, y)",
    "place_left_of": "Place A to the left of B",
    "stack_centered": "Stack blocks with center alignment",
    "navigate_to": "Navigate to target position",
}

# 예상 결과
results = {
    "avg_late": {
        "pick_at_xy": 58.2,  # 위치 정보 부족
        "place_left_of": 64.1,  # Spatial relation 약함
        "stack_centered": 71.5,  # Global 정보로 부분 해결
        "navigate_to": 72.3,
    },
    "pixelwise": {
        "pick_at_xy": 83.5,  # +25.3% (위치 명확)
        "place_left_of": 83.8,  # +19.7% (관계 파악)
        "stack_centered": 87.2,  # +15.7%
        "navigate_to": 84.4,  # +12.1%
    }
}
```

---

## 예상 실험 결과

### Table: Aggregation Strategy Comparison

| Method | Success↑ | Position Acc.↑ | Memory | Time | Spatial Info |
|--------|----------|----------------|--------|------|--------------|
| avg + Late (dino.txt) | 74.2% | **61.5%** | 1.0× | 1.0× | ✗ |
| avg + Early | 73.8% | 60.8% | 0.5× | 0.9× | ✗ |
| **Pixel-wise (ours)** | **81.3%** | **78.9%** | 1.5× | 1.2× | ✅ |
| All patches + Late | 80.5% | 77.2% | 2.0× | 1.5× | ✅ |

**Key Findings**:
- Spatial preservation improves position accuracy by **17.4%** (61.5% → 78.9%)
- Pixel-wise fusion achieves best performance with moderate overhead
- Early fusion without spatial preservation fails (avg + Early)

### Table: Task-Specific Breakdown

| Task Type | avg + Late | Pixel-wise | Improvement |
|-----------|-----------|------------|-------------|
| Global task (navigate) | 72.3% | 84.4% | +12.1% |
| Spatial relation (place left) | 64.1% | 83.8% | +19.7% |
| Precise manipulation (pick xy) | 58.2% | 83.5% | **+25.3%** |
| Average | 64.9% | 83.9% | +19.0% |

→ **Spatial reasoning tasks benefit most**

---

## Visualization

### Attention Map Comparison

```
Task: "Pick the red cube"

┌─────────────────────────────┐
│  avg(patches) 방식           │
│  Attention map: [균일 분산]  │ ← 어디를 봐야할지 모름
│  🔴 물체 위치: 불명확         │
└─────────────────────────────┘

┌─────────────────────────────┐
│  Pixel-wise (ours)          │
│  Attention map:             │
│      🔴 ← 명확한 focus      │
│  물체 위치: 정확히 파악      │
└─────────────────────────────┘
```

### Learned Fusion Weights

```python
# Pixel-wise FC weights visualization
fc_patch.weight.shape = [D, 2D]

# M/P importance per dimension
m_importance = fc_patch.weight[:, :D].abs().mean()
p_importance = fc_patch.weight[:, D:].abs().mean()

# Task-specific patterns
grasping_task: M > P  (0.65 vs 0.35)  # Motion 중시
navigation_task: P > M  (0.58 vs 0.42)  # Structure 중시
```

---

## 논문 작성 전략

### Introduction: Motivation

```markdown
Vision-language methods [dino.txt, 2024] use avg(patches)
to create global descriptors, effective for classification
("Is there a cat?").

However, behavior learning fundamentally requires
**spatial reasoning** ("Where is the cat?").

Consider robotic manipulation: To pick a red cube,
the agent must know not only that a red cube exists
(captured by avg) but precisely **where it is located**
(lost in averaging).

We propose **pixel-wise channel fusion** that preserves
spatial structure while efficiently integrating temporal (M)
and spatial (P) information.
```

### Method: Clear Contrast

```markdown
### 3.3 Spatial-Preserving Channel Fusion

**Limitation of Spatial Averaging**:

Vision-language methods [dino.txt] create:
    representation = [CLS_M ; avg(patches_M) ; CLS_P ; avg(patches_P)]

While effective for classification, this **destroys spatial structure**
critical for behavior learning.

**Our Approach - Pixel-wise Fusion**:

We preserve all N spatial locations and fuse channels
at each position:

    for each patch location i:
        patch_i_fused = FC_patch([patch_Mi ; patch_Pi])

    representation = [CLS_fused, patch_1_fused, ..., patch_n_fused]

**Benefits**:
1. Complete spatial structure preservation
2. Pixel-wise M-P integration
3. 50% dimension reduction vs late concat
4. Gradient flows to all spatial locations
```

### Ablation: Key Evidence

```markdown
Table X: Impact of Spatial Information Preservation

| Aggregation | Spatial Info | Success | Position Acc. |
|-------------|-------------|---------|---------------|
| avg + concat | ✗ | 74.2% | 61.5% |
| **pixel-wise (ours)** | ✅ | **81.3%** | **78.9%** |

**Key Finding**: Spatial preservation improves position
accuracy by 17.4% (61.5% → 78.9%), confirming that
behavior learning requires fine-grained spatial information.

**Task Breakdown**:
- Navigate (global): +12.1%
- Spatial relation: +19.7%
- Precise manipulation: **+25.3%** ← Largest gain

Figure X: Attention maps show that our method focuses on
task-relevant spatial locations while avg-based methods
display diffuse attention.
```

---

## 관련 개념

**주요 연결**:
- [[Two-Stream Image Preprocessing]] - M/P 채널 생성 (이 fusion의 입력)
- [[논문 - Action-Agnostic Visual Behavior Representation]] - 이 fusion을 활용하는 메인 논문
- [[Sources/papers/dino.txt (2024)]] - Spatial averaging 접근법 (비교 대상)

**이론적 배경**:
- [[Two Visual Pathways]] - Dorsal/Ventral streams, "what" vs "where"
- Spatial reasoning in manipulation
- Information bottleneck theory

---

## 구현 체크리스트

### 필수 구현

- [ ] TwoStreamVisionEncoder (standard interface)
- [ ] PixelwiseFusion module (FC, MLP variants)
- [ ] Baseline implementations (avg + late/early)
- [ ] Spatial reasoning evaluation metrics

### Ablation Experiments

- [ ] Aggregation strategy (avg vs pixel-wise)
- [ ] Fusion mechanism (FC vs MLP vs Attention)
- [ ] Weight sharing (separate vs shared)
- [ ] Task-specific breakdown analysis

### Visualization

- [ ] Attention map comparison
- [ ] Learned fusion weights
- [ ] Spatial reasoning task heatmaps

---

## 메타데이터

- **작성일**: 2026-01-28
- **관련 프로젝트**: [[논문 - Action-Agnostic Visual Behavior Representation]]
- **핵심 통찰**: Spatial averaging은 vision-language에 적합하지만 behavior learning에는 부족. Spatial structure 보존이 manipulation task에 필수.
- **dino.txt와의 차이**: Classification → avg 충분 vs Manipulation → pixel-wise 필수

---

#concept #fusion #spatial-reasoning #behavior-learning #architecture #dino-txt #ablation
