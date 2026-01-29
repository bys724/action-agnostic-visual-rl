# Two-Stream Image Preprocessing

## 정의

생물학적 시각 시스템의 **Magnocellular/Parvocellular 경로**를 모델링한 이미지 전처리 방법. 연속된 이미지로부터 **시간적 변화 (M채널)**와 **공간적 구조 (P채널)**를 분리하여 추출한다.

---

## 핵심 원리

### 두 채널의 목적 차이

```
M채널 (Magnocellular): "무엇이 변했는가?"
  - 시간적 변화 감지 (밝기 + 색상)
  - 절대적 변화량 중요
  - 움직임, 조명 변화
  - Dorsal pathway: Motion detection

P채널 (Parvocellular): "무엇이 어디 있는가?"
  - 공간적 구조 (edge, 형태)
  - 색상 정보 (appearance)
  - 조명 불변적 형태 인식
  - Ventral pathway: Object recognition
```

**핵심 통찰**: 목적이 다르므로 **normalize 전략도 달라야 한다**

---

## M채널: 시간적 변화 (Temporal Change)

### 처리 과정

```
1. RGB → Luminance (trainable weights 공유)
2. 시간 미분: ΔL = L(t) - L(t-1)
3. 색상 변화: ΔR, ΔG, ΔB = RGB(t) - RGB(t-1)
4. 절대적 범위 유지 (고정 범위로만 clip)
5. 출력: [ΔL, ΔR, ΔG, ΔB]
```

### 구체적 구현

```python
def magnocellular_channel(img_prev, img_curr, lum_weights):
    """
    M채널: 시간적 변화 (밝기 + 색상)

    Args:
        img_prev: [B, 3, H, W], range [0, 1]
        img_curr: [B, 3, H, W], range [0, 1]
        lum_weights: [3], trainable weights for R, G, B

    Returns:
        [B, 4, H, W]: [ΔL, ΔR, ΔG, ΔB]
    """
    # 1. Luminance 계산 (trainable)
    w = torch.softmax(lum_weights, dim=0).view(1, 3, 1, 1)
    lum_prev = (img_prev * w).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    lum_curr = (img_curr * w).sum(dim=1, keepdim=True)

    # 2. 밝기 변화 (시간 미분)
    delta_L = lum_curr - lum_prev  # 범위 [-1, 1]
    delta_L = torch.clamp(delta_L, -1, 1)

    # 3. 색상 변화 (RGB 각 채널)
    delta_R = img_curr[:, 0:1] - img_prev[:, 0:1]  # [B, 1, H, W]
    delta_G = img_curr[:, 1:2] - img_prev[:, 1:2]
    delta_B = img_curr[:, 2:3] - img_prev[:, 2:3]

    # Clip to valid range
    delta_R = torch.clamp(delta_R, -1, 1)
    delta_G = torch.clamp(delta_G, -1, 1)
    delta_B = torch.clamp(delta_B, -1, 1)

    # 4. 통합 (모두 temporal change)
    output = torch.cat([delta_L, delta_R, delta_G, delta_B], dim=1)  # [B, 4, H, W]

    return output
```

### Normalize 전략: 절대적 범위 유지

**왜 이미지별 normalize를 하지 않는가?**

```
시나리오: 밝은 실내 → 어두운 실외 이동

이미지별 normalize (X):
  ΔL = (변화 - 변화평균) / 변화표준편차
  → 큰 변화도 작은 변화도 동일하게 normalize됨
  → "얼마나 변했는지" 정보 손실

절대적 범위 유지 (O):
  ΔL = L(t) - L(t-1)  # [-1, 1]
  → 큰 변화 = 큰 값 (예: -0.5)
  → 작은 변화 = 작은 값 (예: -0.05)
  → "얼마나 변했는지" 정보 보존
```

**생물학적 근거**:
- Magnocellular 세포는 **절대적 contrast**에 민감
- 전역적 적응(global adaptation)은 느림 (수초~수분)
- 밝기 변화의 절대적 크기가 의미 있음 (햇빛 vs 그늘)

**장점**:
1. ✅ 조명 정보 보존 (낮/밤, 실내/실외 구분)
2. ✅ 움직임 속도 추정 (큰 변화 = 빠른 움직임)
3. ✅ 시간적 일관성 (배치 간 비교 가능)

---

## P채널: 공간적 구조 + 색상 정보 (Spatial Appearance)

### 처리 과정

```
1. RGB → Luminance (M채널과 동일한 weights 공유)
2. 이미지별 normalize (조명 불변성)
3. Sobel 필터 적용 (공간 미분)
4. Gradient magnitude 계산
5. Max magnitude로 normalize (방향 + 선명도 보존)
6. 현재 프레임 RGB 추가 (색상 정보)
7. 출력: [∂L/∂x, ∂L/∂y, R, G, B]
```

### 구체적 구현

```python
def parvocellular_channel(img, lum_weights, sobel_x, sobel_y):
    """
    P채널: 공간 구조 + 색상 정보

    생물학적 근거: Ventral pathway (P cells)는 "what" 정보 담당
    - 형태 (structure): ∂x, ∂y
    - 색상 (appearance): R, G, B

    Args:
        img: [B, 3, H, W], range [0, 1]
        lum_weights: [3], M채널과 공유
        sobel_x, sobel_y: [1, 1, 3, 3], Sobel kernels

    Returns:
        [B, 5, H, W]: [∂L/∂x, ∂L/∂y, R, G, B], range [-1, 1] or [0, 1]
    """
    # 1. Luminance 계산 (M채널과 동일한 weights)
    w = torch.softmax(lum_weights, dim=0).view(1, 3, 1, 1)
    lum = (img * w).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    # 2. 이미지별 normalize (조명 불변성)
    mean = lum.mean(dim=(2, 3), keepdim=True)
    std = lum.std(dim=(2, 3), keepdim=True) + 1e-6
    lum_norm = (lum - mean) / std
    lum_norm = torch.clamp(lum_norm, -3, 3)  # outlier 제거

    # 3. Sobel 필터 (공간 미분)
    grad_x = F.conv2d(lum_norm, sobel_x, padding=1)  # [B, 1, H, W]
    grad_y = F.conv2d(lum_norm, sobel_y, padding=1)  # [B, 1, H, W]

    # 4. Gradient magnitude
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)  # [B, 1, H, W]

    # 5. Max magnitude로 normalize
    max_magnitude = magnitude.amax(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
    max_magnitude = torch.clamp(max_magnitude, min=1e-6)  # zero division 방지

    grad_x_norm = grad_x / max_magnitude  # 비율 보존 + 선명도 보존
    grad_y_norm = grad_y / max_magnitude

    # 6. RGB 색상 정보 (현재 프레임)
    # Ventral pathway의 color perception 모델링
    R, G, B = img[:, 0:1], img[:, 1:2], img[:, 2:3]  # [B, 1, H, W] each

    # 7. 출력: Structure + Appearance
    output = torch.cat([grad_x_norm, grad_y_norm, R, G, B], dim=1)  # [B, 5, H, W]

    return output
```

### Normalize 전략: 2단계 정규화

P채널의 정규화는 **두 가지 목적**을 가지며, **순차적으로** 적용됨:

#### 1단계: Luminance 정규화 (조명 불변성)

**목적**: 조명에 무관하게 동일한 형태를 인식

```
시나리오: 같은 물체, 다른 조명

햇빛 아래 빨간 공:
  L_raw = [0.8, 0.82, 0.78, ...]  (밝음)
  L_norm = (L - 0.8) / 0.02 = [0, 1, -1, ...]

그늘 아래 빨간 공:
  L_raw = [0.3, 0.32, 0.28, ...]  (어두움)
  L_norm = (L - 0.3) / 0.02 = [0, 1, -1, ...]

→ 동일한 상대적 대비 패턴
```

**생물학적 근거**: Parvocellular 세포는 **local contrast**에 민감, receptive field 내 상대적 밝기만 인코딩

#### 2단계: Gradient Magnitude 정규화 (방향 + 선명도 보존)

**목적**: Edge의 방향(비율)과 선명도(상대적 강도)를 모두 보존

```
문제: 통계적 정규화의 한계

잘못된 방법 (이미지별 통계 normalize):
  grad_x_norm = (grad_x - mean) / std
  grad_y_norm = (grad_y - mean) / std
  → 방향(비율) 보존 X, 선명도 정보 손실

개선된 방법 (Max magnitude normalize):
  magnitude = √(grad_x² + grad_y²)
  max_mag = max(magnitude)
  grad_x_norm = grad_x / max_mag
  grad_y_norm = grad_y / max_mag
  → 방향(비율) 보존 O, 선명도(상대적 강도) 보존 O
```

**구체적 예시**:

```
픽셀 A: grad_x=0.8, grad_y=0.6 → magnitude=1.0
픽셀 B: grad_x=0.4, grad_y=0.3 → magnitude=0.5
max_magnitude = 1.0

Normalize 후:
픽셀 A: (0.8, 0.6) → 방향: arctan(0.6/0.8), 강도: 1.0 (최대)
픽셀 B: (0.4, 0.3) → 방향: arctan(0.3/0.4) (동일!), 강도: 0.5 (절반)

→ 방향(비율) 보존: A와 B는 동일한 방향 (4:3)
→ 선명도 보존: B는 A의 절반 선명도
```

**핵심 통찰**:
- **방향 = grad_y/grad_x 비율** → 나눗셈은 분자·분모 동일 스케일 시 불변
- **선명도 = magnitude의 상대적 크기** → max로 나누면 상대 관계 유지

**장점**:
1. ✅ **조명 불변성** (1단계: Luminance normalize)
2. ✅ **방향 보존** (2단계: 비율 유지)
3. ✅ **선명도 보존** (2단계: 상대적 강도 유지)
4. ✅ **동적 범위 최적화** (항상 [-1, 1] 범위)

---

## 파라미터 공유: Luminance Weights

### 핵심 설계 원칙

```python
# M채널과 P채널이 동일한 weights 공유
lum_weights = nn.Parameter(torch.tensor([0.2126, 0.7152, 0.0722]))

# M채널에서 사용
L_m(t) = w·RGB(t)

# P채널에서도 동일하게 사용
L_p = w·RGB

→ 일관된 밝기 정의
→ 추가 파라미터 없음
```

### 초기화 및 학습

**초기값**: BT.709 표준 (인간 시각 근사)
```python
w_init = [0.2126, 0.7152, 0.0722]  # R, G, B
```

**학습 가능**: 태스크에 따라 최적화
```python
w = torch.softmax(lum_weights, dim=0)  # 합 = 1 보장
```

**예상 학습 결과**:
- 주간 시나리오: w ≈ [0.21, 0.72, 0.07] (표준 유지)
- 야간 시나리오: w ≈ [0.15, 0.75, 0.10] (녹색 강조)
- 수중 로봇: w ≈ [0.10, 0.60, 0.30] (청색 증가)

---

## 출력 형식 및 통합

### 최종 출력

```
M채널: [B, 4, H, W] - Temporal Change (Dorsal)
  - Channel 0: ΔL (시간적 밝기 변화)
  - Channel 1: ΔR (빨강 채널 변화)
  - Channel 2: ΔG (초록 채널 변화)
  - Channel 3: ΔB (파랑 채널 변화)

P채널: [B, 5, H, W] - Spatial Appearance (Ventral)
  - Channel 0: ∂L/∂x (가로 방향 edge)
  - Channel 1: ∂L/∂y (세로 방향 edge)
  - Channel 2: R (빨강 채널, 현재 프레임)
  - Channel 3: G (초록 채널, 현재 프레임)
  - Channel 4: B (파랑 채널, 현재 프레임)

Total: 9 channels
```

### ViT 입력 방식

**Option A: 통합 (Single ViT)**
```python
combined = torch.cat([m_output, p_output], dim=1)  # [B, 9, H, W]
tokens = vit(combined)
```

**Option B: 분리 (Two-Stream ViT)**
```python
m_tokens = vit_m(m_output)  # [B, N+1, D] (4채널 입력)
p_tokens = vit_p(p_output)  # [B, N+1, D] (5채널 입력)
fused = fusion([m_tokens, p_tokens])  # Pixel-wise fusion
```

**추천**: Option B (Two-Stream ViT)
- M과 P의 독립적 인코딩으로 정보 보존
- [[Pixel-wise Channel Fusion for Behavior Representation]] 참조

---

## Normalize 전략 비교표

| 측면 | M채널 (Dorsal) | P채널 (Ventral) |
|------|---------------|----------------|
| **생물학적 경로** | Magnocellular → Dorsal | Parvocellular → Ventral |
| **목적** | "무엇이 변했는가" (What changed) | "무엇이 어디 있는가" (What is where) |
| **입력** | img(t-1), img(t) | img(t-1) |
| **출력** | [ΔL, ΔR, ΔG, ΔB] | [∂x, ∂y, R, G, B] |
| **채널 수** | 4 channels | 5 channels |
| **Normalize** | 절대적 (고정 범위) | 2단계 (조명 + 크기) |
| **범위** | [-1, 1] (clipped) | [-1, 1] 또는 [0, 1] |
| **보존 대상** | 절대적 변화량 | 구조(방향+선명도) + 색상 |
| **불변성** | 시간 일관성 | 조명 불변성 |
| **예시** | "빨간색이 10% 증가" | "45° edge + 빨간색 물체" |
| **기능** | Motion detection | Object recognition |

---

## 생물학적 정당성

### Magnocellular Pathway

**특성**:
- **절대적 contrast** 민감도
- 느린 전역 적응 (global adaptation)
- 시간적 변화에 transient response

**설계 대응**:
```
절대적 밝기 변화 유지
→ 조명 정보 보존
→ 움직임 속도 정보
```

**증거**: [[Two Visual Pathways#Patient L.M.]]
- MT/V5 손상 → 움직임 인지 불가
- Temporal change detection 불가능

---

### Parvocellular Pathway

**특성**:
- **Local contrast** 민감도
- 빠른 지역 적응 (receptive field 내)
- 공간적 패턴에 sustained response

**설계 대응**:
```
이미지별 normalize
→ 상대적 대비 강조
→ 조명 불변 형태
```

**증거**: [[Two Visual Pathways#Patient D.F.]]
- LO 손상 → 형태 인지 불가
- Spatial pattern recognition 불가능

---

## 실험적 예측

### 시나리오 1: 조명 변화

```
상황: 밝은 실내 → 어두운 실외

M채널:
  ΔL = -0.5 (큰 음수)
  ΔR, ΔG, ΔB < 0 (모든 색상 감소)
  → "급격히 어두워졌다" 감지 ✅

P채널:
  ∂x, ∂y: 상대적 normalize
  → 어두운 환경에서도 edge 패턴 동일 ✅
  R, G, B: 현재 프레임 색상
  → 어두운 환경에서도 물체 색상 인식 가능
```

### 시나리오 2: 빨간 공이 정지 (Static)

```
상황: 햇빛 아래 빨간 공이 정지

M채널:
  ΔL = 0 (변화 없음)
  ΔR = ΔG = ΔB = 0 (변화 없음)
  → M채널 완전 비활성화! ✅

P채널:
  ∂x, ∂y: 공의 edge 패턴
  R, G, B: 빨간 공의 색상
  → "빨간 공이 여기 있다" 인식 ✅
  → P만으로 충분! (생물학적으로 타당)
```

### 시나리오 3: 빨간 공이 빠르게 이동 (Video)

```
상황: 빨간 공이 왼쪽으로 빠르게 이동

M채널:
  ΔL: 왼쪽 +, 오른쪽 - (밝기 변화 패턴)
  ΔR > 0 왼쪽, ΔR < 0 오른쪽 (빨간색 이동)
  → "빨간 물체가 왼쪽으로 이동" ✅

P채널:
  ∂x, ∂y: 공의 edge (현재 위치)
  R, G, B: 빨간색 (현재 프레임)
  → "현재 여기에 빨간 공" ✅
```

### 시나리오 4: "Pick the red cube" Task

```
Task: 빨간 큐브를 집어라

P채널의 역할 (핵심!):
  ∂x, ∂y: 큐브 형태 (edge pattern)
  R, G, B: 빨간색 인식
  → "빨간 큐브가 어디 있는가" 파악 ✅

M채널의 역할:
  ΔL, ΔR, ΔG, ΔB: 그리퍼 접근 중 변화
  → "그리퍼가 큐브에 접근하고 있는가" 확인 ✅

→ 두 채널이 complementary!
```

---

## 주요 응용: Change Representation Learning

**논문**: [[논문 - Action-Agnostic Visual Behavior Representation]]

이 전처리 방법은 **change representation learning**의 핵심 입력으로 설계됨.

---

### 핵심 철학: Change를 효과적으로 표현하기

> **목표**: 이미지 간 변화를 압축하여 representation 학습
> **검증**: 그 representation으로 미래 프레임을 예측할 수 있는가?

**Two-Stream의 역할**:
```
Visual Change = M (temporal dynamics) + P (spatial appearance)

M채널: "무엇이 어떻게 변했는가"
  → 미래 예측에 필수 (motion, temporal dynamics)

P채널: "현재 무엇이 어디 있는가"
  → 미래 예측의 기준점 (spatial context, object identity)

→ 둘 다 있어야 정확한 미래 예측 가능!
```

---

### 왜 M-P Split이 Change Representation에 적합한가?

#### 1. **미래 예측 예시: 공이 굴러간다**

**입력**:
- img_t: 공이 왼쪽에 있음
- img_t+k: 공이 오른쪽에 있음

**P채널만으로 예측 시도** (실패):
```python
P(img_t) = [∂x, ∂y, R, G, B]  # 왼쪽에 빨간 공
P(img_t+k) = ???

# 문제: P만으로는 "어디로 움직일지" 알 수 없음
# → 정지? 왼쪽? 오른쪽? 위? 아래? 불확실!
```

**M채널 추가 (성공)**:
```python
M(t→t+k) = [ΔL, ΔR, ΔG, ΔB]  # 왼쪽 -, 오른쪽 +
P(img_t) = [∂x, ∂y, R, G, B]  # 현재 위치 + 형태

# 해결: M이 motion direction 제공, P가 현재 상태 제공
# → "빨간 공이 오른쪽으로 이동" 예측 가능!
```

**결론**: **M + P 모두 필요!**

---

#### 2. **M-P Balance 자동 달성**

**Video Prediction 학습 중**:

```python
# 정지 이미지 (Static)
M = [0, 0, 0, 0]  # 변화 없음
P = [∂x, ∂y, R, G, B]  # 모든 정보

# Decoder: img_t+k = img_t (그대로)
# → P만으로 충분 (M 사용 안 함) ✓

# 동영상 (공 이동)
M = [ΔL, ΔR, ΔG, ΔB]  # 변화 있음
P = [∂x, ∂y, R, G, B]  # 현재 상태

# Decoder: "어디로 이동?" 알아야 함
# → M 필수! (motion 정보)
# → P 필수! (현재 위치)
# → 둘 다 학습됨 ✓
```

**Video + Static 혼합 데이터셋**:
- Static 30%: P 강제 학습 (M=0)
- Video 70%: M+P 균형 학습 (motion + appearance)
- **자동으로 M-P balance 달성!**

---

### 논문과의 연결

**논문의 2-Frame Input 설계**:
```
입력: 항상 2장 이미지 (t, t+k)
출력: Change embedding
Pre-training: Video prediction (img_t + change_emb → img_t+k)
Downstream: Inverse dynamics (change_emb → robot action)
```

**Two-Stream Preprocessing의 역할**:
```
M채널 (4ch): 시간적 변화 (Dorsal pathway)
  - [ΔL, ΔR, ΔG, ΔB]
  - "무엇이 변했는가" (What changed)
  - Forward dynamics의 핵심

P채널 (5ch): 공간적 구조 + 색상 (Ventral pathway)
  - [∂x, ∂y, R, G, B]
  - "무엇이 어디 있는가" (What is where)
  - Spatial context 제공

→ 총 9채널로 완전한 change representation 학습
```

**장점 (Change Representation 관점)**:
1. ✅ **Forward dynamics 학습 가능**: M-P 분리로 motion + appearance 모두 캐치
2. ✅ **Spatial reasoning 보존**: Pixel-wise fusion과 결합 시 위치 정보 유지
3. ✅ **Embodiment-independent**: Visual dynamics는 로봇 종류와 무관
4. ✅ **Self-validation**: Video prediction 성능 = representation quality

---

### Change Representation → Inverse Dynamics 파이프라인

**Phase 1: Pre-training (Forward)**
```python
Input: img_t, img_t+k (M-P preprocessed)
Encoder: Two-Stream ViT
Output: change_emb
Loss: Reconstruction(img_t+k)

→ "어떤 변화가 있었는지" 학습 (unsupervised)
```

**Phase 2: Downstream (Inverse)**
```python
Input: change_emb + task + sensors
Decoder: Robot-specific action head
Output: robot_action
Loss: Behavior cloning / Supervised

→ "그 변화를 만들려면 어떤 action?" 학습
```

**왜 이것이 효과적인가**:
- Forward (어려움): 220k videos로 학습 (사람 비디오 활용)
- Inverse (쉬움): 20-30 demos로 학습 (로봇 데모만)
- **Two-Stream은 forward learning을 효율화**

---

### 기타 응용 시나리오

**로봇 비전 시스템**:
- Visual servoing: Target tracking with motion prediction
- Object tracking: Combine motion (M) + appearance (P)
- Manipulation: Predict object dynamics for planning

**비디오 이해**:
- Action recognition: Temporal change pattern
- Video segmentation: Motion + appearance cues
- Temporal reasoning: Predict future frames

**Visual MPC (Model Predictive Control)**:
- Forward model: img_t + action → img_t+1 prediction
- Planning: Find action sequence that leads to goal state
- Two-Stream provides dynamics prior

---

## 관련 시스템 (DynamicNet)

DynamicNet의 Input Interface에서 이 전처리를 활용 가능:

```
Phasic 노드 (짧은 Duration): M채널 ΔL 수용 → 시간적 변화 감지
Tonic 노드 (긴 Duration): M채널 RGB + P채널 edge → 형태 유지
```

**상세**: [[Idea - DynamicNet#3.1 Input Interface]]

---

## 구현 체크리스트

### 필수 구현

- [x] Luminance weights (trainable, shared)
- [x] M채널: 시간적 변화 (ΔL, ΔR, ΔG, ΔB)
- [x] P채널: 공간 구조 + 색상 (∂x, ∂y, R, G, B)
- [x] 적절한 normalize 전략
- [x] Magnitude-based normalization (방향 + 선명도 보존)
- [x] 9채널 출력

### 선택적 개선

- [ ] Gradient magnitude를 별도 채널로 출력 (P채널 3채널로)
- [ ] Color opponent channels (RG, BY)
- [ ] Multi-scale Sobel (다양한 커널 크기)
- [ ] Learnable spatial filters (Sobel 대체)

### 모니터링

```python
# 학습 중 확인
print(f"Learned weights: {lum_weights.data}")

# 예상 변화:
# 초기: [0.2126, 0.7152, 0.0722]
# 학습 후: 태스크 의존적
```

---

## 관련 개념

**주요 연결**:
- [[논문 - Action-Agnostic Visual Behavior Representation]] - 이 전처리 방법을 사용하는 메인 논문
- [[Two Visual Pathways]] - 생물학적 근거, Dorsal/Ventral Streams, Magnocellular/Parvocellular

**부차적 연결**:
- [[Idea - DynamicNet]] - DynamicNet Input Interface 설계 (응용 예시)

---

## 참고문헌

**Normalize 전략**:
- Kingdom, F. A., & Prins, N. (2016). "Psychophysics: A Practical Introduction" - Local vs global adaptation
- Webster, M. A. (2015). "Visual Adaptation" - *Annual Review of Vision Science* - Contrast normalization

**Optical Flow와 연결**:
- Fleet, D. J., & Weiss, Y. (2006). "Optical flow estimation" - Brightness constancy
- Brox, T., et al. (2004). "High Accuracy Optical Flow Estimation" - Gradient-based methods

**생물학적 근거**:
- [[Two Visual Pathways#참고문헌]] - Magnocellular/Parvocellular pathways

---

#concept #image-processing #two-stream #preprocessing #computer-vision #biological-inspiration #action-agnostic #robot-learning
