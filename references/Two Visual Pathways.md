# Two Visual Pathways

## 정의

인간 시각 시스템은 망막에서부터 **두 개의 독립적인 경로**로 정보를 처리한다. 이는 "무엇(What)"과 "어디/어떻게(Where/How)"를 분리하여 처리하는 생물학적 전략이다.

---

## Two-Stream Hypothesis

### 기본 구조

```
망막 (Retina)
    ↓
시각피질 V1 (Primary Visual Cortex)
    ↓
    ├─→ Dorsal Stream (등쪽 경로) → 두정엽 (Parietal Cortex)
    │   "Where/How" - 위치, 움직임, 공간 관계, 동작 가이드
    │
    └─→ Ventral Stream (배쪽 경로) → 측두엽 (Temporal Cortex)
        "What" - 물체 인식, 형태, 색상, 정체성
```

### 두 경로의 특성

| 특성 | Dorsal Stream | Ventral Stream |
|------|---------------|----------------|
| **목적지** | 두정엽 (Parietal) | 측두엽 (Temporal) |
| **기능** | Where/How (공간, 행동) | What (정체성) |
| **처리 내용** | 위치, 움직임, 공간 관계 | 형태, 색상, 물체 정체성 |
| **시간 해상도** | 높음 (빠른 반응) | 낮음 (느린 반응) |
| **공간 해상도** | 낮음 (거친) | 높음 (세밀) |
| **의식성** | 무의식적 (행동 가이드) | 의식적 (인지) |

---

## 신경학적 증거: 환자 케이스

### Patient L.M. - Motion Blindness (Akinetopsia)

**손상 부위**: MT/V5 영역 (중간 측두 영역, Dorsal Stream의 핵심)

**증상**:
- ✅ **형태 인식 정상**: 물체의 모양, 색상, 정체성 인식 가능
- ❌ **움직임 인지 불가**: 연속적인 움직임을 볼 수 없음

**실제 경험**:
```
"차가 텔레포트하는 것처럼 보입니다"
- 멀리 있던 차 → 갑자기 바로 앞에
- 커피를 따를 때: 빈 컵 → 갑자기 가득 찬 컵
- 사람들이 대화 중 갑자기 다른 위치에 나타남
- 길 건너기가 매우 위험 (차의 속도 판단 불가)
```

**과학적 의미**:
- **시간적 변화 감지는 별도의 뇌 영역**에서 처리됨
- 정적 형태와 동적 움직임은 독립적

---

### Patient D.F. - Visual Form Agnosia

**손상 부위**: Lateral Occipital (LO) 영역 (Ventral Stream)

**증상**:
- ❌ **형태 인지 불가**: 물체의 모양을 의식적으로 인식 못함
- ✅ **움직임/위치 정상**: 물체의 위치와 움직임 감지 가능
- ✅ **시각운동 제어 정상**: 물체를 정확히 잡을 수 있음

**실제 경험**:
```
"우편함의 슬롯이 어느 방향인지 말할 수 없습니다"
- 하지만 편지를 넣을 때는 자동으로 정확한 각도로 회전
- "무엇인지"는 모르지만 "어떻게 잡아야 하는지"는 앎
- 물체를 그릴 수 없지만, 손을 뻗어 잡기는 가능
```

**과학적 의미**:
- **"인지(Perception)"와 "행동(Action)"은 별도 경로**
- Ventral = 의식적 인식, Dorsal = 무의식적 행동 가이드

---

### 환자 케이스 비교

| 환자 | 손상 부위 | 형태 인식 | 움직임 감지 | 물체 잡기 | 시사점 |
|------|----------|----------|------------|----------|--------|
| **L.M.** | MT/V5 (Dorsal) | ✅ 가능 | ❌ 불가능 | ✅ 가능 | 시간적 변화 = 별도 처리 |
| **D.F.** | LO (Ventral) | ❌ 불가능 | ✅ 가능 | ✅ 가능 | 공간적 형태 = 별도 처리 |

---

## 생물학적 기반: Magnocellular vs Parvocellular

두 경로는 **망막에서부터** 분리되어 시작:

### M세포 vs P세포

| 특성 | Magnocellular (M세포) | Parvocellular (P세포) |
|------|----------------------|---------------------|
| **투사 경로** | → Dorsal Stream | → Ventral Stream |
| **시간 해상도** | **높음** (빠른 반응, ~80ms) | 낮음 (느린 반응, ~100ms) |
| **공간 해상도** | 낮음 (거친 디테일) | **높음** (세밀한 디테일) |
| **색상 민감도** | 낮음 (흑백, 휘도) | **높음** (색상 구분) |
| **대비 민감도** | **높음** (낮은 대비도 감지) | 낮음 (고대비 필요) |
| **수용장 크기** | 큰 수용장 | 작은 수용장 |
| **주 기능** | **움직임, 밝기 변화, 깊이** | **형태, 색상, 세부사항** |

### 처리 속도 차이

```
M세포 경로 (Magnocellular):
  자극 → 80ms → Dorsal Stream → "무언가 움직인다!" (빠른 반응)

P세포 경로 (Parvocellular):
  자극 → 100ms → Ventral Stream → "빨간 사과다" (정확한 인식)
```

**생존 전략**: 빠른 움직임 감지 → 도피/회피, 느린 세밀한 인식 → 식별/학습

---

## DynamicNet 적용

### Phasic vs Tonic 노드의 생물학적 근거

**Node Properties의 Duration 속성이 Two-Stream을 반영**:

| DynamicNet | 생물학적 대응 | 처리 내용 | 환자 증거 |
|-----------|--------------|----------|----------|
| **Phasic 노드** (짧은 지속) | Magnocellular → Dorsal | 시간적 변화 감지 (움직임, 밝기 변화) | L.M. 손상 시 불가능 |
| **Tonic 노드** (긴 지속) | Parvocellular → Ventral | 공간적 상태 유지 (형태, 색상) | D.F. 손상 시 불가능 |

### Input Interface에서의 적용

```
[Input Interface - 시각]

Phasic 노드 (짧은 Duration):
  - Magnocellular 경로 모델링
  - 밝기가 변하는 순간 활성화
  - 움직이는 물체 경계 감지
  - "무언가 움직였다!" → 즉시 반응 후 꺼짐
  - 예: Edge detection, Motion detection

Tonic 노드 (긴 Duration):
  - Parvocellular 경로 모델링
  - 물체의 형태 정보 유지
  - 색상, 질감 같은 정적 속성
  - "이건 컵이다" → 계속 활성화 상태 유지
  - 예: Object recognition, Color processing
```

### 자동 분화 메커니즘

**학습으로 자동 선택**:
```
움직임 중요한 태스크 (예: 공 추적, 장애물 회피)
  → Phasic 노드 선택적 생존
  → Dorsal-like 경로 강화

형태 중요한 태스크 (예: 물체 분류, 색상 매칭)
  → Tonic 노드 선택적 생존
  → Ventral-like 경로 강화
```

**핵심**: 두 종류의 노드가 자연스럽게 공존하며, 태스크에 따라 자동으로 적절한 경로 선택

---

## 추가 신경학적 증거

### Blindsight (맹시)

- **손상**: V1 (Primary Visual Cortex) 손상
- **증상**: 의식적으로 "보지" 못함
- **하지만**: 움직이는 물체 위치를 "추측"할 수 있음
- **설명**: Dorsal Stream이 V1 우회 경로로 작동
- **의미**: **의식 없는 시각 처리** 가능

### Optic Ataxia (시각 실행증)

- **손상**: Dorsal Stream 손상 (Parietal cortex)
- **증상**: 물체를 "보고" 인식하지만, 정확히 잡지 못함
- **의미**: D.F.와 정반대 (D.F.는 인식 못하지만 잡기 가능)

---

## 진화적 의미

### 왜 두 경로로 분리?

1. **속도 vs 정확도 Trade-off**
   - 빠른 반응 (Dorsal): 생존에 필수 (포식자 회피)
   - 정확한 인식 (Ventral): 학습과 계획

2. **병렬 처리**
   - 동시에 "어디에" + "무엇이"를 처리
   - 효율적 자원 활용

3. **행동과 인지의 분리**
   - 행동: 빠르고 자동적 (Dorsal)
   - 인지: 느리고 의식적 (Ventral)

---

## 관련 개념

- [[Node Properties]] - Phasic/Tonic Duration 속성이 Two-Stream 반영
- [[Dynamic Network Structure]] - 태스크별 경로 자동 선택
- [[Cognitive Space]] - 두 경로의 정보가 통합되는 공간
- [[Hebbian vs Predictive Plasticity]] - Bottom-up (Hebbian) vs Top-down (Predictive) 학습

---

## 관련 질문

- [[Q - DynamicNet]]

---

## 참고문헌

**Two-Stream Hypothesis**:
- Goodale, M. A., & Milner, A. D. (1992). "Separate visual pathways for perception and action." *Trends in Neurosciences*.
- Ungerleider, L. G., & Mishkin, M. (1982). "Two cortical visual systems."

**환자 케이스**:
- Zihl, J., von Cramon, D., & Mai, N. (1983). "Selective disturbance of movement vision after bilateral brain damage." *Brain* (Patient L.M.)
- Milner, A. D., et al. (1991). "Perception and action in 'visual form agnosia'." *Brain* (Patient D.F.)

**Magnocellular/Parvocellular**:
- Livingstone, M., & Hubel, D. (1988). "Segregation of form, color, movement, and depth: anatomy, physiology, and perception." *Science*.

---

#concept #neuroscience #visual-system #two-stream #dorsal #ventral #dynamicnet #biological-inspiration
