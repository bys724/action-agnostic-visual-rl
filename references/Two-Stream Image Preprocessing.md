# Two-Stream Image Preprocessing

## 정의

생물학적 시각 시스템의 **Magnocellular/Parvocellular 경로**를 모델링한 이미지 전처리 방법. 연속된 이미지로부터 **시간적 변화 (M채널)**와 **공간적 구조 (P채널)**를 분리하여 추출한다.

---

## 핵심 원리

### 두 채널의 목적 차이

```
M채널 (Magnocellular): "무엇이 변했는가?"
  - 시간적 밝기 변화 감지
  - 절대적 변화량 중요
  - 움직임, 조명 변화

P채널 (Parvocellular): "형태가 무엇인가?"
  - 공간적 대비 (edge, 형태)
  - 상대적 대비 중요
  - 조명 불변적 형태 인식
```

**핵심 통찰**: 목적이 다르므로 **normalize 전략도 달라야 한다**

---

## M채널: 절대적 밝기 변화

### 처리 과정

```
1. RGB → Luminance (trainable weights 공유)
2. 시간 미분: ΔL = L(t) - L(t-1)
3. 절대적 범위 유지 (고정 범위로만 clip)
4. 이전 프레임 RGB 추가 (색상 정보 보존)
```

### 구체적 구현

```python
def magnocellular_channel(img_prev, img_curr, lum_weights):
    """
    M채널: 절대적 밝기 변화

    Args:
        img_prev: [B, 3, H, W], range [0, 1]
        img_curr: [B, 3, H, W], range [0, 1]
        lum_weights: [3], trainable weights for R, G, B

    Returns:
        [B, 4, H, W]: [ΔL, R(t-1), G(t-1), B(t-1)]
    """
    # 1. Luminance 계산 (trainable)
    w = torch.softmax(lum_weights, dim=0).view(1, 3, 1, 1)
    lum_prev = (img_prev * w).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    lum_curr = (img_curr * w).sum(dim=1, keepdim=True)

    # 2. 시간 미분
    delta_L = lum_curr - lum_prev  # 범위 [-1, 1]

    # 3. 절대적 변화량 보존 (고정 범위로만 clip)
    delta_L = torch.clamp(delta_L, -1, 1)

    # 4. RGB 색상 정보 유지 (이전 프레임)
    rgb_prev = img_prev  # [B, 3, H, W]

    # 5. 통합
    output = torch.cat([delta_L, rgb_prev], dim=1)  # [B, 4, H, W]

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

## P채널: 상대적 대비 (형태)

### 처리 과정

```
1. RGB → Luminance (M채널과 동일한 weights 공유)
2. 이미지별 normalize (조명 불변성)
3. Sobel 필터 적용 (공간 미분)
4. Gradient magnitude 계산
5. Max magnitude로 normalize (방향 + 선명도 보존)
6. 출력: [∂L/∂x, ∂L/∂y]
```

### 구체적 구현

```python
def parvocellular_channel(img, lum_weights, sobel_x, sobel_y):
    """
    P채널: 방향(비율) + 선명도(강도) 모두 보존

    Args:
        img: [B, 3, H, W], range [0, 1]
        lum_weights: [3], M채널과 공유
        sobel_x, sobel_y: [1, 1, 3, 3], Sobel kernels

    Returns:
        [B, 2, H, W]: [∂L/∂x, ∂L/∂y], range [-1, 1]
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

    # 4. Gradient magnitude (핵심 개선!)
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)  # [B, 1, H, W]

    # 5. Max magnitude로 normalize
    max_magnitude = magnitude.amax(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
    max_magnitude = torch.clamp(max_magnitude, min=1e-6)  # zero division 방지

    grad_x_norm = grad_x / max_magnitude  # 비율 보존 + 선명도 보존
    grad_y_norm = grad_y / max_magnitude

    # 6. 출력
    output = torch.cat([grad_x_norm, grad_y_norm], dim=1)  # [B, 2, H, W]

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
M채널: [B, 4, H, W]
  - Channel 0: ΔL (시간적 밝기 변화)
  - Channel 1-3: R, G, B (색상 정보, t-1 시점)

P채널: [B, 2, H, W]
  - Channel 0: ∂L/∂x (가로 방향 edge)
  - Channel 1: ∂L/∂y (세로 방향 edge)

Total: 6 channels
```

### ViT 입력 방식

**Option A: 통합 (Single ViT)**
```python
combined = torch.cat([m_output, p_output], dim=1)  # [B, 6, H, W]
tokens = vit(combined)
```

**Option B: 분리 (Two-Stream ViT)**
```python
m_tokens = vit_m(m_output)  # [B, N, D]
p_tokens = vit_p(p_output)  # [B, N, D]
fused = fusion([m_tokens, p_tokens])
```

**추천**: Option A로 시작 → 필요 시 Option B

---

## Normalize 전략 비교표

| 측면 | M채널 | P채널 |
|------|-------|-------|
| **목적** | 시간적 변화 감지 | 공간적 형태 인식 |
| **입력** | img(t-1), img(t) | img(t-1) |
| **Normalize** | 절대적 (고정 범위) | 2단계 (조명 + 크기) |
| **범위** | [-1, 1] (clipped) | [-1, 1] (magnitude-normalized) |
| **보존 대상** | 절대적 변화량 | 방향(비율) + 선명도(강도) |
| **불변성** | 시간 일관성 | 조명 불변성 |
| **예시** | "10% 밝아졌다" | "45° 방향의 선명한 edge" |
| **생물학적** | Global adaptation (느림) | Local contrast (빠름) |

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
  RGB = 실내 색상
  → "급격히 어두워졌다" 감지 ✅

P채널:
  L_norm = 상대적 normalize
  → 어두운 환경에서도 edge 패턴 동일 ✅
  → 형태 인식 유지
```

### 시나리오 2: 동일 물체, 다른 조명

```
상황: 햇빛 아래 vs 그늘 아래 빨간 공

M채널:
  ΔL ≈ 0 (정적)
  RGB 보존 → 빨간색 유지 ✅

P채널:
  햇빛: L_norm → edge pattern A
  그늘: L_norm → edge pattern A (동일)
  → 조명 불변 형태 인식 ✅
```

### 시나리오 3: 빠른 움직임

```
상황: 물체가 빠르게 이동

M채널:
  큰 ΔL → 빠른 움직임 감지
  RGB trail → 색상 기반 추적

P채널:
  연속 프레임의 edge 패턴
  → 형태 일관성 확인
```

---

## 주요 응용: Action-Agnostic Visual Behavior Representation

**논문**: [[논문 - Action-Agnostic Visual Behavior Representation]]

이 전처리 방법은 위 논문의 핵심 입력 처리 방식으로 설계됨.

### 논문과의 연결

**논문의 2-Frame Input 설계**:
```
입력: 항상 2장 이미지 (t, t+1)
출력: Behavior representation
목적: Speed-invariant, embodiment-independent learning
```

**Two-Stream Preprocessing의 역할**:
```
M채널 (4ch): 시간적 변화 + 색상 정보
  - ΔL: 밝기 변화 감지 (움직임 방향)
  - RGB: 물체 색상 유지 (정체성)

P채널 (2ch): 공간적 구조
  - ∂L/∂x, ∂L/∂y: 형태/edge 정보
  - 조명 불변적 (normalize)

→ 총 6채널로 시간·공간·색상 통합 표현
```

**장점 (논문에서)**:
1. ✅ **속도 불변성**: 로봇 속도 차이에 강인 (절대 변화량 보존)
2. ✅ **조명 불변성**: P채널 normalize로 형태 인식 안정
3. ✅ **Embodiment-independent**: 시각적 변화 자체를 표현
4. ✅ **생물학적 근거**: Magnocellular/Parvocellular 이중 경로

### 기타 응용 시나리오

**로봇 비전 시스템**:
- Visual servoing, Object tracking, Manipulation

**비디오 이해**:
- Action recognition, Video segmentation, Temporal reasoning

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
- [x] M채널: 절대적 밝기 변화
- [x] P채널: 상대적 대비 (Sobel)
- [x] 적절한 normalize 전략
- [x] Magnitude-based normalization (방향 + 선명도 보존)
- [x] 6채널 출력

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
