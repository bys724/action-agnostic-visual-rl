# 논문 - Action-Agnostic Visual Behavior Representation

## 메타데이터

- **상태**: Planning
- **시작일**: 2025-12-10
- **목표 학회**: RSS 2026 (primary) / ICRA 2027 / CoRL 2026 (backup)
- **관련 특허**: [[특허 - 시계열 시각 관찰 기반 조건부 텍스트 생성 시스템]]

### RSS 2026 투고 정보

- **학회 일정**: 2026년 7월 13-17일
- **학회 장소**: Sydney, Australia
- **Abstract 마감**: 2026년 1월 중순~말 예정 (미발표)
- **Full paper 마감**: Abstract 마감 후 약 1주일 (미발표)
- **PDF 수정 가능**: Full paper 마감 후 약 1주일 예정
- **Rebuttal**: 2026년 3월 말 예정
- **최종 결정**: 2026년 4월 중순 예정
- **페이지 제한**: 없음 (단, Limitations 섹션 필수)
- **평가 기준**: Novelty, Technical quality, Significance, Potential impact, Clarity

**참고**: 공식 일정은 아직 미발표 상태. 과거 RSS 패턴(2025, 2024)을 바탕으로 예상한 일정임. 공식 발표 시 업데이트 필요.

---

## 한 문장 요약

Task description과 연속 이미지로부터 action-agnostic behavior representation을 학습하여, embodiment-independent robot learning을 가능하게 한다.

---

## 핵심 아이디어

### 문제

로봇마다 다른 action space → 데이터 재사용 불가

### 해결책

행동의 본질은 시각적 변화 → Visual outcome으로 behavior 표현

### 핵심 통찰 (LAPA 검증)

**사람 비디오 > 로봇 데이터**
- LAPA (ICLR 2025): 사람 비디오로 학습 시 로봇 데이터보다 우수
- Visual change에 action 정보가 담김 (embodiment 무관)
- → 우리 가설: **Task-conditioned visual behavior representation**

### 더 깊은 철학: Forward/Inverse 분리 (2026-01-06 추가)

> **"Action은 인터페이스, 본질은 Visual Flow"**

**핵심 통찰:**

Task 성공 = 올바른 visual flow 생성

두 가지 지식:
1. **Forward Knowledge (어려운 부분)**
   - "어떤 visual 변화가 성공으로 이어지는가?"
   - Embodiment-independent
   - 사람 비디오로 학습 가능
   - 많은 경험 필요 → 220k 비디오

2. **Inverse Knowledge (쉬운 부분)**
   - "그 변화를 만들려면 어떤 action을 내야 하나?"
   - Embodiment-specific
   - Forward를 알면 역추론 가능
   - 적은 데모로 충분 → 20-30 demos

**왜 사람 비디오가 효과적인가:**
- 기존 설명: "Visual change가 action을 표현"
- 더 깊은 이유: **Forward를 먼저 학습하면, Inverse는 쉽다**
- 사람 비디오 = Forward knowledge 제공 (어려운 부분)
- 로봇 데모 = Inverse knowledge 제공 (쉬운 부분)
- → LAPA가 220k 비디오로 성공한 이유

**Task-conditioning의 역할:**
- Forward 학습을 효율화
- Task가 중요한 visual feature를 명시
- "빨간 컵" → 색상 feature 강조
- "들어올리기" → 수직 motion 강조
- → 불필요한 정보 무시 → **적은 데이터로 Forward 학습 가능**

**관련 연구 (이미 검증됨):**
- Visual Foresight (2018): Video prediction → Action planning
- DreamerV3 (2023): World model 먼저, policy 나중에
- Visual MPC: Dynamics 학습 → Closed-loop control

### 구조

```
[연속 이미지] + [Task description]
    ↓ Cross-Attention
    ↓
CLS (universal behavior representation)
    ↓
Robot-specific decoder → action
```

**차별점**:
- LAPA: Unsupervised latent action (VQ-VAE)
- 우리: Task-conditioned visual representation (cross-attention)

---

## RSS 2024 트렌드 분석

**요약**: Cross-embodiment learning과 VLM이 핵심 트렌드. 우리 논문은 human-to-robot transfer로 차별화. Real robot validation과 strong baselines 필수.

> [!info]- 📊 상세 내용 보기
>
> **주요 연구 방향**
>
> **1. Vision-Language Models & Foundation Models** (10+ papers)
> - Octo: Open-source generalist robot policy
> - VLMPC: Vision-Language Model Predictive Control
> - RAG-Driver: Multi-modal LLM learning
> - Language-augmented planners
>
> **2. Cross-Embodiment Learning** (강력한 트렌드)
> - MIRAGE: Cross-embodiment zero-shot transfer
> - "Pushing the Limits of Cross-Embodiment Learning"
> - Universal Manipulation Interface
>
> **3. Representation Learning**
> - 3D Diffusion Policy: Generalizable visuomotor learning
> - Human-oriented representation learning
> - HRP: Human affordances for pre-training
>
> **4. Large-Scale Datasets**
> - DROID: Large-scale manipulation dataset
> - RT-X style multi-embodiment data
>
> **5. Diffusion-Based Policies** (5+ papers)
> - Diffusion for policy learning
> - 3D representations
>
> **우리 논문의 위치**
>
> **강점 (RSS 트렌드와 부합)**:
> - ✅ Cross-embodiment learning (핵심 트렌드)
> - ✅ Vision-Language fusion (인기 주제)
> - ✅ Foundation model 접근 (시의적절)
> - ✅ Generalizable representation (주요 관심사)
> - ✅ **사람 데이터 활용** (LAPA로 검증된 접근)
>
> **차별점 (기존 연구 대비)**:
> - **vs Octo**: Robot-to-robot transfer → 우리는 human-to-robot transfer
> - **vs LAPA**: Unsupervised latent action → 우리는 task-conditioned representation
> - **vs VC-1**: Task-agnostic visual encoder → 우리는 task-aware behavior encoder
>
> **약점 (RSS 수준 대비 부족)**:
> - ⚠️ **실제 로봇 검증 없음**: RSS는 real robot results 중시 (→ 해결 가능)
> - ⚠️ **Baseline 비교 부족**: Octo, LAPA 등과 직접 비교 필요 (→ 코드 공개됨)
> - ⚠️ **Quantitative results 미비**: Success rate, transfer efficiency 등 (→ 실험 필요)
>
> **RSS 채택을 위한 필수 요소**
>
> **Critical (없으면 reject)**:
> 1. **실제 로봇 실험**: 최소 1개 embodiment에서 real-world validation
> 2. **정량적 평가**: Success rate, sample efficiency, transfer performance
> 3. **Strong baselines**: Octo, VC-1, R3M 등과 비교
> 4. **Ablation studies**: 각 component의 기여도 입증
>
> **Important (있으면 강력)**:
> 1. Multi-embodiment dataset (3+ robots)
> 2. Zero-shot or few-shot transfer 성공 사례
> 3. Failure case 분석 (Limitations 섹션)
> 4. Code/model release 계획
>
> **Nice-to-have**:
> 1. Human evaluation
> 2. Long-horizon tasks
> 3. Sim-to-real transfer

---

## 논문 스토리 (Paper Narrative)

**요약**: 문제(로봇마다 다른 action space) → 통찰(visual change가 action 표현) → LAPA 검증(사람 비디오 > 로봇 데이터) → 우리 기여(task-conditioned behavior representation)

> [!note]- 📖 상세 내용 보기
>
> **Introduction: The Cross-Embodiment Challenge**
>
> **문제**: 로봇마다 다른 action space → 학습 데이터 재사용 불가
> - 7-DoF arm, end-effector control, mobile manipulator 모두 다름
> - 각 로봇마다 별도 데이터 수집 필요 → 비용 높음
>
> **Key Insight: Visual Behavior Representation**
>
> **핵심 아이디어**: 행동의 본질은 visual change
> - "Pick up object": action command는 다르지만, visual outcome은 동일
> - Temporal image sequence에 action 정보가 담겨 있음
> - → **Action-agnostic representation 가능**
>
> **Supporting Evidence: LAPA (ICLR 2025)**
>
> **선행 연구 검증**:
> - 사람 비디오 → 로봇 전이: 36.8% success
> - 로봇 데이터 → 로봇 전이: 30.8% success
> - **사람 데이터가 더 우수!**
> - → Visual change가 embodiment-independent action 정보를 담는다는 증거
>
> **Our Contribution**
>
> **기존 연구 대비 차별점**:
>
> | Method | Approach | Limitation |
> |--------|----------|------------|
> | Octo | Robot-to-robot transfer | 로봇 데이터에만 의존 |
> | LAPA | Unsupervised latent action | Task 정보 활용 안 함 |
> | VC-1 | Task-agnostic visual encoder | Behavior 표현 약함 |
> | **Ours** | **Task-conditioned visual behavior** | - |
>
> **우리 방법**:
> - Cross-attention으로 task description + temporal images 융합
> - CLS token = universal behavior representation
> - Robot-specific decoder로 action 생성
> - → **Task-aware + Action-agnostic**
>
> **Experimental Design**
>
> **Large-scale pretraining**:
> - EgoDex (829h 사람 manipulation) + Something-Something V2
> - 고품질 visual behavior representation 학습
>
> **Robot finetuning**:
> - Bridge V2 (깨끗한 로봇 데이터)
> - Decoder만 학습 → sample efficient
>
> **Real robot validation**:
> - 자체 로봇암으로 3-5 tasks
> - Human-to-robot transfer 증명
>
> **Baselines**:
> - Octo (robot-to-robot baseline)
> - VC-1 (visual representation baseline)
> - LAPA (human video baseline)
>
> **Expected Impact**
>
> **Contributions**:
> 1. Visual change가 action을 표현함을 입증 (LAPA 확장)
> 2. Task-conditioned behavior representation 제안
> 3. Human video를 로봇 학습에 활용하는 실용적 방법
> 4. Real robot에서 human-to-robot transfer 검증
>
> **RSS 적합성**:
> - Cross-embodiment learning (핵심 트렌드)
> - Human data utilization (새로운 방향)
> - Foundation model for robotics (시의적절)

---

## Available Resources (가용 자원)

**요약**: 로봇암 ✅ | 사람 데이터(EgoDex 829h, Sthv2 220k) ✅ | 로봇 데이터(Bridge V2 60k) ✅ | Baseline(OpenVLA, SCRATCH 필수 / VC-1, LAPA 선택)

> [!note]- 💾 상세 내용 보기
>
> **하드웨어**
> - ✅ **로봇암 + 그리퍼**: Real robot validation 가능
> - 간단한 manipulation task 수행 환경 구축 가능
>
> **오픈 데이터셋**
>
> **핵심 통찰: 사람 manipulation 데이터의 유효성**
>
> **LAPA (ICLR 2025)** 연구 결과:
> - 사람 비디오로 학습 → 로봇 데이터보다 **우수한 성능**
> - LAPA (사람): 36.8% vs OpenVLA (로봇): 30.8% success rate
> - **30-40배 효율적** (272 H100-hrs vs 21,500 A100-hrs)
> - Something-Something V2 (220k 사람 비디오) 사용
> - → **Visual change가 action을 표현한다는 우리 가설 검증!**

> #### Human Manipulation (Primary Pretraining)
>
> **1. EgoDex (Apple, 2024)** - 최고 품질 추천
> - **규모**: 829시간, 194 tasks, 338k episodes, 90M frames
> - **품질**: Apple Vision Pro 3D hand tracking (21 joints per hand)
> - **특징**: Tabletop manipulation 특화, 1080p 30Hz
> - **접근성**: GitHub (`apple/ml-egodex`), CC-by-NC-ND, 2TB
> - **활용**: 고품질 behavior representation pretraining
> - **장점**: 정확한 hand pose, 다양한 task, 고해상도
>
> **2. Something-Something V2**
> - **규모**: 220k videos, 174 action categories
> - **검증**: LAPA에서 OpenVLA 능가 증명됨
> - **특징**: Object interaction에 특화된 사람 action
> - **접근성**: HuggingFace
> - **활용**: Large-scale pretraining
>
> #### Robot Manipulation (Finetuning & Evaluation)
>
> **3. Bridge V2** - 가장 깨끗한 로봇 데이터
> - **규모**: 60,096 trajectories, 24 environments
> - **품질**: Controlled environment, skill diversity 우수
> - **특징**: Open X-Embodiment 구성 요소 중 최고 품질
> - **접근성**: GitHub (`rail-berkeley/bridge_data_v2`), Creative Commons
> - **활용**: Robot finetuning 및 benchmark evaluation
>
> **4. DROID** (선택적)
> - **규모**: 76k trajectories, 350 hours, 564 scenes, 86 tasks
> - **특징**: Scene diversity 압도적, but 품질 불균일
> - **접근성**: TFDS, HuggingFace
> - **활용**: Generalization 테스트용
>
> **Open X-Embodiment 사용 전략**:
> - ⚠️ 전체 1M trajectories는 품질 불균일 (알려진 이슈)
> - ✅ Bridge V2, DROID 등 개별 고품질 subset만 선별 사용
> - 양보다 질 우선

> **Baseline Selection Strategy (중요!)**
>
> #### 선택한 Baselines 및 근거
>
> **1. OpenVLA (필수 - Current SOTA)**
> - **선택 이유**:
>   - 2024년 현재 VLA SOTA 모델 (LAPA가 이것을 이김)
>   - SOTA를 이겨야 논문 설득력 확보
>   - 구현 용이성: HuggingFace 기반, pretrained checkpoint 공개
>   - 잘 정리된 코드베이스와 문서화
> - **비교 포인트**:
>   - Human video pretraining vs Robot action-labeled pretraining
>   - Sample efficiency during finetuning
>   - Cross-embodiment transfer performance
> - **구현 난이도**: ★☆☆☆☆ (매우 쉬움)
>   - Checkpoint 다운로드 → Finetuning만 하면 됨
> - **시간 투자**: 1-2주 (finetuning + evaluation)
>
> **2. SCRATCH (필수 - Ablation baseline)**
> - **선택 이유**:
>   - Pretraining 효과를 입증하기 위한 필수 baseline
>   - 구현 공짜: 우리 backbone 그대로 downstream task로 finetuning
>   - 모든 robot learning 논문의 표준 baseline
> - **비교 포인트**:
>   - Pretraining의 가치 입증
>   - Sample efficiency 차이
> - **구현 난이도**: ★☆☆☆☆ (공짜)
> - **시간 투자**: 0주 (이미 구현되어 있음)
>
> **3. LAPA-style Baseline (선택적 - 방법론 비교)**
> - **선택 이유**:
>   - 가장 유사한 접근법 (human video pretraining)
>   - ICLR 2025, 최신 연구
>   - 우리의 차별점 강조: VQ-VAE latent action vs Task-conditioned representation
> - **비교 포인트**:
>   - Unsupervised latent action vs Supervised task-conditioned behavior
>   - Reconstruction objective vs Contrastive/supervised objective
> - **구현 난이도**: ★★★☆☆ (중간)
>   - VQ-VAE 구현 필요하지만 단순한 편
>   - 또는 간소화된 버전으로 구현 가능
> - **시간 투자**: 2-3주 (여유 있을 때만)
> - **대안**: LAPA 결과를 인용만 하고 직접 비교는 생략 가능
>
> **4. Our Ablations (필수 - Component 분석)**
> - **변형들**:
>   - w/o Task conditioning (visual only)
>   - w/o Cross-attention (concat 방식)
>   - w/o Temporal modeling (single frame)
>   - Different pretraining objectives (DINO, SimCLR 등)
> - **선택 이유**: 각 component의 기여도 입증
> - **구현 난이도**: ★★☆☆☆ (쉬움 - 이미 구현된 것의 변형)
> - **시간 투자**: 1-2주
>
> #### 배제한 Baselines 및 근거
>
> **1. Octo (배제 - 구현 복잡도 대비 가치 낮음)**
> - **배제 이유**:
>   - 구현 복잡도: JAX 기반, 환경 세팅 까다로움
>   - 성능: OpenVLA보다 낮음 (OpenVLA가 Octo 개선 버전)
>   - Robot-to-robot transfer라는 다른 문제 설정
>   - 시간 대비 얻는 인사이트 적음
> - **대응 논리** (리뷰어가 물어볼 경우):
>   - "Octo는 범용성(generality)에 초점, 우리는 성능(performance)에 초점"
>   - "OpenVLA가 더 최신이고 성능도 높아 더 적절한 비교 대상"
>   - "Octo의 modular architecture vs 우리의 monolithic approach는 다른 설계 철학"
> - **시간 절약**: 3-4주
>
> **2. RT-2 (배제 - 재현 불가능)**
> - **배제 이유**:
>   - Google internal, 코드 미공개
>   - 재현 불가능
>   - 데이터셋도 비공개
> - **대응 논리**:
>   - "RT-2는 재현 불가능하여 공정한 비교 어려움"
>   - "OpenVLA가 공개된 대안으로 더 적절"
>
> **3. VC-1 (선택적 - 시간 있을 때만)**
> - **부분 채택**:
>   - Visual representation quality 비교에는 유용
>   - Linear probe evaluation으로 간단히 비교 가능
>   - Full baseline으로는 불필요 (task-agnostic하므로)
> - **구현 난이도**: ★★☆☆☆ (중간 - pip install로 간단)
> - **시간 투자**: 1주 (linear probe만)
> - **결정**: Linear probe 결과만 포함, full policy 비교는 생략
>
> **4. R3M (배제 - VC-1로 충분)**
> - **배제 이유**:
>   - VC-1과 같은 카테고리 (visual representation)
>   - VC-1이 더 최신이고 성능 좋음
>   - 둘 다 비교하는 건 중복
> - **시간 절약**: 2주
>
> **5. Diffusion Policy 계열 (배제 - 문제 설정 다름)**
> - **배제 이유**:
>   - Action generation 방법론에 초점
>   - 우리는 representation learning에 초점
>   - Decoder 부분은 교체 가능하므로 직교적(orthogonal)
> - **대응 논리**:
>   - "Diffusion policy는 우리 decoder로 대체 가능 (complementary)"
>   - "Representation vs Generation은 다른 차원의 문제"
>
> #### 최종 Baseline 구성 (우선순위)
>
> **Tier 1 (필수 - 이것 없으면 논문 안 됨)**:
> 1. ✅ SCRATCH - Pretraining 효과 입증
> 2. ✅ OpenVLA - SOTA 비교
>
> **Tier 2 (강력 추천 - 있으면 논문 강도 상승)**:
> 3. ✅ Our Ablations - Component 기여도 분석
> 4. ⚠️ VC-1 (Linear probe) - Visual representation quality
>
> **Tier 3 (선택적 - 시간 여유 있을 때)**:
> 5. ⚠️ LAPA-style - 방법론 차별화 강조
> 6. ⚠️ VC-1 (Full policy) - 완전한 비교
>
> #### 시간 배분 권장
>
> ```
> 나쁜 예 (피해야 함):
> - Baseline 구현: 6개월 (Octo, RT-2 재현 시도, R3M, VC-1 등)
> - 본인 방법: 1개월
> - Writing: 1주
> → 결과: Baseline에 지쳐 본인 방법 완성도 낮음
>
> 좋은 예 (권장):
> - Baseline: 1-2개월 (OpenVLA finetune + SCRATCH + Ablations)
> - 본인 방법: 3-4개월 (완성도 높이기)
> - 실험/분석: 1개월
> - Writing: 1개월
> → 결과: 본인 방법 완성도 높고, 설득력 있는 논문
> ```
>
> #### 논문 작성 시 표현 방법
>
> **Method 섹션에서**:
> ```markdown
> We compare against the following baselines:
> - SCRATCH: Direct finetuning without pretraining
> - OpenVLA [Kim et al., 2024]: State-of-the-art VLA model
> - Ablations: Variants of our method to analyze component contributions
>
> We do not compare with Octo [Team et al., 2024] as it focuses on
> generality across diverse tasks and embodiments, while our work
> prioritizes performance on human-to-robot transfer. OpenVLA provides
> a more recent and stronger baseline for this comparison.
> ```
>
> **Related Work 섹션에서**:
> ```markdown
> While methods like Octo [Team et al., 2024] and RT-2 [Brohan et al., 2023]
> demonstrate impressive generalization, they rely on large-scale robot data.
> Recent work LAPA [Ye et al., 2024] shows promise in learning from human
> videos, but uses unsupervised latent actions without task conditioning.
> Our approach builds on this insight while introducing task-aware behavior
> representations.
> ```
>
> #### 리뷰어 대응 준비
>
> **예상 질문 1**: "Why not compare with Octo?"
> **답변**: "Octo focuses on cross-task generalization with modular architecture, while our work addresses human-to-robot transfer with monolithic design. OpenVLA provides a more direct and recent comparison point for VLA performance. Additionally, Octo's JAX implementation poses practical challenges for fair comparison in our PyTorch-based framework."
>
> **예상 질문 2**: "How does your method compare to RT-2?"
> **답변**: "RT-2's code and data are not publicly available, making direct comparison infeasible. We compare against OpenVLA, which represents the current state-of-the-art among reproducible methods."
>
> **예상 질문 3**: "Why not include more representation learning baselines like R3M?"
> **답변**: "We include VC-1 as the representative visual representation baseline, which is more recent and performs better than R3M. Adding multiple baselines from the same category (task-agnostic visual representations) would not provide additional insights into our core contribution: task-conditioned behavior representations."
>
> ---
>
> **Previously Listed Baseline Models (참고용)**
>
> **Octo** - RSS 2024 (배제)
> - **성능**: VC-1, RT-1-X 대비 평균 52% 향상
> - **구조**: Transformer-based diffusion policy
> - **학습 데이터**: 800k episodes from Open X-Embodiment
> - **코드**: `octo-models/octo` (JAX)
> - **배제 이유**: 구현 복잡, OpenVLA가 더 나음
>
> **VC-1** - (선택적)
> - **성능**: Best prior visual representation for embodied AI
> - **학습 데이터**: 4,000+ hours egocentric video + ImageNet
> - **코드**: `facebookresearch/eai-vc`
> - **사용법**: pip install vc_models
> - **활용**: Linear probe evaluation만

---

## Gap Analysis & Action Plan

**요약**: 필수 요소(Real robot ✅, Multi-embodiment data ✅, Baselines ✅) 확보. RSS 2026 13개월 타임라인으로 충분한 실험 가능.

> [!tip]- 🎯 상세 내용 보기
>
> **현재 상태 vs RSS 요구사항**
>
> | 요구사항 | 현재 상태 | 가용 자원 | 필요 작업 |
> |---------|---------|---------|---------|
> | Real robot demo | 준비됨 | 로봇암 + 그리퍼 | Task 설계 및 실험 |
> | Multi-embodiment data | ✅ | RT-X, Bridge V2, DROID | 데이터 로딩 파이프라인 |
> | Baseline 비교 | ✅ | Octo, VC-1 코드 공개 | 평가 스크립트 작성 |
> | Ablation study | 진행 필요 | - | Component별 분석 |
> | Quantitative metrics | 진행 필요 | - | Success rate, transfer efficiency |
> | Limitations 섹션 | 진행 필요 | - | Failure cases 분석 |
>
> **극복 전략 (수정)**
>
> **Option 1: RSS 2025 (Aggressive but Feasible, 5주)**
> ```
> 타임라인:
> - Week 1-2: Model 구현 + Open X-Embodiment 학습
> - Week 2-3: Baseline 비교 (Octo, VC-1)
> - Week 3-4: Real robot task 설계 및 실험
> - Week 4-5: 논문 작성 + Ablation studies
> → 타이트하지만 하드웨어 + 데이터 + 코드 확보로 실행 가능
> → 위험: 중간에 큰 문제 발생 시 대응 어려움
> ```
>
> **Option 2: CoRL 2025 또는 ICRA 2026 (Recommended)**
> ```
> 충분한 시간 확보:
> - 3-4개월 실험 기간
> - Multiple embodiments 실험 (자체 로봇 + public data)
> - Extensive ablations
> - 완성도 높은 논문
> → 채택 확률 높음, 더 강력한 결과
> ```
>
> **Option 3: RSS 2025 Workshop**
> ```
> 초기 검증:
> - 핵심 아이디어 proof-of-concept
> - 커뮤니티 피드백
> - Main conference는 다음 해
> → 안전한 선택
> ```
>
> **최소 실행 가능 논문 (MVP for RSS 2025)**
>
> **5주 안에 가능한 것** (하드웨어 + 데이터 확보):
> 1. ✅ Open X-Embodiment로 학습
> 2. ✅ Bridge V2, DROID로 평가
> 3. ✅ Octo, VC-1 baseline 비교 (pretrained 활용)
> 4. ✅ Real robot validation (1 embodiment, 3-5 tasks)
> 5. ✅ Linear probe evaluation
> 6. ✅ Basic ablation studies
>
> **여전히 어려운 것**:
> 1. ⚠️ Multiple real robot embodiments (하나만 가능)
> 2. ⚠️ Large-scale ablations (시간 부족)
> 3. ⚠️ Extensive failure analysis
>
> **업데이트된 결론**:
> - RSS 2025 main conference: **도전 가능** (하지만 리스크 있음)
> - CoRL/ICRA: **더 강력한 결과** 기대
> - 최종 결정은 **Week 2-3 실험 결과 보고 판단** 권장

---

## 실험 계획 (RSS 2026 - 최종 확정)

**요약**: Stage 1 (SIMPLER - OpenVLA vs Ours) → Stage 2 (Franka - 4 methods, 10x faster training) → Phase 3 (Analysis & Writing)

> [!example]- 🔬 상세 내용 보기
>
> **전략: Two-Stage Evaluation**
>
> **핵심 원칙**:
> 1. Stage 1 (SIMPLER): Fair comparison with baselines (같은 WidowX embodiment)
> 2. Stage 2 (Franka): Cross-embodiment transfer efficiency 검증 (우리 시스템)

>
> ---
>
> **Stage 1: SIMPLER Benchmark Evaluation**
>
> **타임라인**: 2026년 1월 ~ 2월 (약 6주)
>
> **목적**: Standard benchmark에서 공정한 비교
>
> **Environment**:
> - SIMPLER simulation (WidowX)
> - BridgeData V2 tasks (4-6개 tasks)
>
> **Baselines (보수적 계획)**:
>
> #### **Tier 1: 필수 (최소 구성)**
> 1. ✅ **SCRATCH** (From-scratch on SIMPLER tasks)
>    - 목적: Pretraining 효과 입증
>    - 구현: Diffusion Policy from scratch
>    - 시간: 1주 (이미 코드 존재)
>
> 2. ✅ **OpenVLA** (Pre-trained checkpoint)
>    - 목적: SOTA 비교
>    - 구현: Checkpoint 다운로드만
>    - 시간: 0일 (공짜)
>
> **이것만으로도 충분한 이유**:
> - Pretraining 효과 입증 (Scratch vs Ours)
> - SOTA 비교 (OpenVLA vs Ours)
> - RSS 2024 LAPA도 이 구성으로 accept
>
> #### **Tier 2: 추가 옵션 (여유 있으면)**
> 3. ⭐ **Diffusion Policy** (Baseline)
>    - 목적: Alternative decoder 비교
>    - 구현: 공식 코드 사용
>    - 시간: 1주
>    - **판단 기준**: 2월 말 진행 상황 보고 결정
>
> 4. ⭐ **Octo** (선택적)
>    - 목적: 다른 VLA 비교
>    - 구현: 공식 checkpoint
>    - 시간: 1주
>    - **판단 기준**: 시간 여유 있을 때만
>
> **구현**:
> ```bash
> # Repository: SimplerEnv-OpenVLA
> git clone https://github.com/DelinQu/SimplerEnv-OpenVLA
> # OpenVLA checkpoint + evaluation scripts 제공됨
> ```
>
> **Expected Results**:
> ```
> Method          | Success Rate | Grasp | Moving | Notes
> ----------------|--------------|-------|--------|-------
> SCRATCH         | ~20-30%      | -     | -      | No pretraining
> OpenVLA         | 36.4%        | 50.0% | 67.7%  | Current SOTA
> LAPA (reported) | 57.3%        | 71.9% | 77.1%  | Target to beat
> Ours (target)   | >40%         | -     | -      | Conservative goal
> ```
>
> ---
>
> **Stage 2: Real Franka Cross-Embodiment Transfer**
>
> **목적**: Embodiment-independence와 transfer efficiency 검증
>
> **Setup**:
> - Robot: Franka Panda (7-DoF)
> - Gripper: Robotiq 2F-85
> - Camera: RealSense D405 (wrist-mounted)
> - Tasks: 1-2 representative tasks
>   - Task 1: "Pick cup and place in bowl"
>   - Task 2 (optional): "Stack blocks"
>
> **Data Collection** (Week 3):
> - [ ] Collect 100 demos per task
>   - 80 demos: Training
>   - 20 demos: Validation
> - [ ] Evaluation: 50 rollouts (새로운 object positions)
> - [ ] 소요 시간: ~3-4 hours per task
>
> **Methods Compared** (Week 4):
>
> 1. **Zero-shot Transfer** ⭐ (1일)
>    - OpenVLA checkpoint → Franka 직접 적용
>    - Action space mapping only (6D → 7D)
>    - 예상: 5-15% (거의 실패하지만 0은 아님)
>    - 의미: "Naive transfer는 작동 안 함"
>
> 2. **From-Scratch** ⭐⭐ (1주)
>    - Diffusion Policy를 Franka 데이터만으로 학습
>    - 80 demos, ~25 hrs training
>    - 예상: 50-60%
>    - 의미: "Pretraining 없이 이 정도"
>
> 3. **Full Fine-tuning (OpenVLA)** ⭐⭐⭐ (2주)
>    - OpenVLA → Franka 전체 재학습
>    - 80 demos, ~45 hrs training
>    - 예상: 65-75%
>    - 의미: "SOTA VLA도 전체 재학습 필요"
>
> 4. **Ours (Decoder-only)** ⭐⭐⭐⭐ (3일)
>    - Bridge V2 representation (frozen) → Franka decoder만 학습
>    - 80 demos, ~5 hrs training
>    - 예상: 75-85%
>    - 의미: "Efficient adaptation + best performance"
>
> **Expected Table 2**:
> ```
> Method                  | Pretraining | Training | Time | Success
> ------------------------|-------------|----------|------|--------
> Zero-shot (OpenVLA)     | Bridge V2   | 0 demos  | 0h   | 8%
> From-scratch (Diffusion)| None        | 80 demos | 25h  | 58%
> Full Finetune (OpenVLA) | Bridge V2   | 80 demos | 45h  | 72%
> Ours (Decoder-only)     | Bridge V2   | 80 demos | 5h   | 82%
>                                                   ↑10x faster ↑best
> ```
>
> **핵심 Message**:
> - ✅ Best Performance: 82% > 72% > 58% > 8%
> - ✅ 10x Faster: 5hrs vs 45hrs
> - ✅ Same Data: 모두 80 demos (공정 비교)
> - ✅ Pretraining Value: +24% over from-scratch
>
> ---
>
> **Phase 3: Analysis & Writing (Week 5-6)**
>
> - [ ] Data efficiency curve (10, 20, 40, 80 demos)
> - [ ] Component ablations
> - [ ] Failure case analysis
> - [ ] 논문 작성
> - [ ] Limitations 섹션
>
> ---
>
> **🚫 DROID 사용 안 함 (이유)**
>
> **문제점**:
> 1. Camera mismatch: DROID (Zed) vs Ours (RealSense D405)
> 2. Environment mismatch: 564 unknown scenes vs Our lab
> 3. Task mismatch: 86 diverse tasks vs Our specific tasks
> 4. → 예상 success rate ≈ 0-5% (설명하기 어려움)
>
> **대안**:
> - 우리 환경에서 직접 데이터 수집
> - Controlled experiment (모든 methods 같은 조건)
> - 설득력 극대화

---

## 📚 구현 참고 자료 (Code & Resources)

**요약**: SIMPLER(SimplerEnv-OpenVLA) | Franka(Diffusion Policy, OpenVLA finetune) | Data(Bridge V2 60k, EgoDex 829h, Sthv2 220k)

> [!info]- 💻 상세 내용 보기
>
> **Stage 1: SIMPLER Evaluation**
>
> **SimplerEnv-OpenVLA**:
> - Repository: https://github.com/DelinQu/SimplerEnv-OpenVLA
> - Official: https://github.com/simpler-env/SimplerEnv
> - Paper: "Evaluating Real-World Robot Manipulation Policies in Simulation" (CoRL 2024)
> - 포함 내용:
>   - ✅ OpenVLA evaluation scripts
>   - ✅ RT-1, Octo evaluation scripts
>   - ✅ BridgeData V2 tasks (4-6개)
>   - ✅ Visual matching setup
>   - ✅ Pre-trained checkpoints
>
> **OpenVLA Checkpoint**:
> - HuggingFace: https://huggingface.co/openvla/openvla-7b
> - Repository: https://github.com/openvla/openvla
> - 사용법:
> ```python
> from transformers import AutoModel
> model = AutoModel.from_pretrained("openvla/openvla-7b")
> ```
>
> ---
>
> **Stage 2: Franka Implementation**
>
> **Baseline A: From-Scratch (Diffusion Policy)**:
> - Repository: https://github.com/real-stanford/diffusion_policy
> - Paper: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
> - 특징:
>   - ✅ Franka Panda 검증됨
>   - ✅ Pick-and-place, pushing, pouring tasks
>   - ✅ Configuration files 제공
>
> **Baseline B: OpenVLA Fine-tuning**:
> - Fine-tuning guide: https://github.com/openvla/openvla/blob/main/docs/TRAINING.md
> - Requirements: 8 A100 GPUs (또는 gradient accumulation)
> - Action space adapter 구현 필요 (6D → 7D)
>
> **Franka + 2F-85 Integration**:
> - Official Gripper Example: https://github.com/frankaemika/external_gripper_example
> - Robotiq 2F-85 Driver:
> ```bash
> sudo apt install ros-noetic-robotiq-2f-gripper-control
> ```
>
> **RealSense D405**:
> - SDK: https://github.com/IntelRealSense/librealsense
> - ROS Wrapper:
> ```bash
> sudo apt install ros-noetic-realsense2-camera
> ```
>
> **Teleoperation (Data Collection)**:
> - FrankaTeleop: https://github.com/gjcliff/FrankaTeleop
> - 또는 VR controller 사용
>
> ---
>
> **Datasets**
>
> **BridgeData V2** (필수):
> - Website: https://rail.eecs.berkeley.edu/datasets/bridge_release/data/
> - Repository: https://github.com/rail-berkeley/bridge_data_v2
> - Format: TensorFlow Datasets (RLDS)
> - Size: 60K trajectories, ~200GB
>
> **DROID** (참고용만):
> - Website: https://droid-dataset.github.io/
> - Download:
> ```bash
> # Full dataset (1.7TB)
> gsutil -m cp -r gs://gresearch/robotics/droid <target_dir>
>
> # Sample (2GB, 100 trajectories)
> gsutil -m cp -r gs://gresearch/robotics/droid_100 <target_dir>
> ```
> - Policy Learning: https://github.com/droid-dataset/droid_policy_learning
> - ⚠️ 우리는 사용 안 함 (environment mismatch)
>
> **EgoDex** (Human video pretraining):
> - Repository: https://github.com/apple/ml-egodex
> - Paper: "EgoDex: A Dataset for Egocentric Hand Manipulation" (2024)
> - Size: 829 hours, 194 tasks, 2TB
>
> **Something-Something V2** (Optional):
> - HuggingFace: https://huggingface.co/datasets/webdataset/something-something-v2
> - 220K videos, 174 action categories

---

## ✅ Action Items (작업 리스트)

**요약**: Phase 1 (Paper reading & Design, 6주) → Phase 2 (Experiments, 16주) → Phase 3 (Writing, 8주) → Phase 4 (Conference)

> [!todo]- ✅ 상세 내용 보기
>
> **🎯 Detailed Action Plan (RSS 2026)**

>
> #### Phase 1: Foundation (2025-12 ~ 2026-01, 6주)
>
> **Week 1-2: Literature & Design (12/16-12/29)**
> - [x] EgoDex 논문 읽기 완료
> - [x] TraceGen 논문 정리 완료 (현재 진행 중)
> - [ ] X-Diffusion, TrajSkill 읽기
> - [ ] Method architecture 최종 확정
> - [ ] Experiment design document 작성
>
> **Week 3-4: SIMPLER Setup (12/30-01/12)**
> - [ ] SimplerEnv-OpenVLA 설치 및 환경 구축
> - [ ] OpenVLA checkpoint 다운로드 및 평가
> - [ ] Baseline 성능 확인 (OpenVLA: 36.4%)
> - [ ] 5090 PC에서 inference 속도 테스트
>
> **Week 5-6: Initial Implementation (01/13-01/26)**
> - [ ] Your representation model 구현 (ViT + Cross-Attention)
> - [ ] Training pipeline 구축
> - [ ] Franka + 2F-85 + D405 hardware setup 점검
>
> ---
>
> #### Phase 2: Core Experiments (2026-02 ~ 2026-05, 16주)
>
> **Month 1: SIMPLER Experiments (02)**
> - [ ] BridgeData V2로 pretraining
> - [ ] SIMPLER evaluation 실행
> - [ ] Baseline 비교 (OpenVLA, Octo, Diffusion)
> - [ ] Ablation studies (encoder types, fusion strategies)
> - [ ] Table 1 완성 + learning curves
>
> **Month 2: Franka Baselines (03)**
> - [ ] Teleoperation system 구축
> - [ ] Task 1-2 정의 및 100 demos 수집
> - [ ] Zero-shot baseline: OpenVLA → Franka
> - [ ] From-scratch: Diffusion Policy 학습
> - [ ] Full Finetune: OpenVLA 재학습
>
> **Month 3: Your Method on Franka (04)**
> - [ ] Bridge V2 representation frozen encoder
> - [ ] Franka decoder 구현 및 학습
> - [ ] Cross-embodiment transfer 검증
> - [ ] Data efficiency curve (10, 20, 40, 80 demos)
> - [ ] Table 2 완성 (4 methods 비교)
>
> **Month 4: Extended Experiments (05)**
> - [ ] Task 3-4 추가 실험
> - [ ] Robustness tests (lighting, backgrounds)
> - [ ] Long-horizon tasks (multi-step)
> - [ ] Failure mode analysis
> - [ ] Generalization tests (novel objects)
>
> ---
>
> #### Phase 3: Paper Writing (2026-06 ~ 2026-07, 8주)
>
> **Week 1-2: Draft v1 (06/01-06/14)**
> - [ ] Introduction 작성
> - [ ] Related Work 작성
> - [ ] Method 상세 작성
> - [ ] Experiments 작성
> - [ ] Results tables & figures 완성
>
> **Week 3-4: Revision (06/15-06/28)**
> - [ ] Abstract 작성
> - [ ] Limitations 섹션 작성
> - [ ] Discussion 작성
> - [ ] Figure quality 개선
> - [ ] 동료 리뷰 1차
>
> **Week 5-6: Polish (06/29-07/12)**
> - [ ] 동료 리뷰 반영
> - [ ] Language editing
> - [ ] Supplementary material 작성
> - [ ] 최종 점검
>
> **Week 7-8: Conference Prep (07/13-07/26)**
> - [ ] Poster 디자인
> - [ ] Talk slides 준비
> - [ ] Practice presentation
>
> ---
>
> ### 🟡 Important (Nice-to-have)
>
> #### Additional Experiments
> - [ ] VC-1 linear probe evaluation
> - [ ] Human video pretraining ablation (EgoDex vs Sthv2)
> - [ ] Cross-task generalization 테스트
> - [ ] Sim-to-real transfer analysis
>
> #### Code & Reproducibility
> - [ ] Code cleanup 및 문서화
> - [ ] README 작성
> - [ ] Pre-trained checkpoint 준비
> - [ ] Demo video 촬영
>
> ---
>
> ### 🟢 Optional (여유 있을 때)
>
> - [ ] Task 2 실험 ("Stack blocks")
> - [ ] LAPA-style baseline 구현
> - [ ] Third-person camera 추가 실험
> - [ ] Long-horizon task 시도
> - [ ] Multi-step reasoning 분석

---

## 📅 Updated Timeline (RSS 2026)

**요약**: 13개월 충분한 타임라인. 2025-12~2026-01 (Foundation) → 2026-02~05 (Experiments) → 2026-06~07 (Writing) → 2026-07-13~17 (Conference)

> [!note]- 📆 상세 내용 보기
>
> **Current Status**: Planning (2025-12-16)
>
> **RSS 2026 Deadlines** (예상):
> - Abstract: 2026-01-15~20 예정 (미발표)
> - Full paper: 2026-01-22~27 예정 (미발표)
> - Conference: 2026-07-13~17 (Sydney, Australia)
>
> **충분한 Timeline** (13개월):
>
> ****Phase 1: Foundation (2025-12 ~ 2026-01, 6주)****
> ```
> Week 1-2 (12/16-12/29): Paper reading & method design
> Week 3-4 (12/30-01/12): SIMPLER baseline evaluation
> Week 5-6 (01/13-01/26): Initial model implementation
> ```
>
> ****Phase 2: Core Experiments (2026-02 ~ 2026-05, 16주)****
> ```
> Month 1 (02): SIMPLER experiments
>   - Your method 구현 및 학습
>   - Baseline 비교 완료
>   - Ablation studies
>
> Month 2 (03): Franka data collection & baseline
>   - 100 demos per task 수집
>   - Zero-shot, From-scratch baselines
>
> Month 3 (04): Franka your method
>   - Decoder-only training
>   - Cross-embodiment transfer 검증
>   - Data efficiency analysis
>
> Month 4 (05): Additional experiments
>   - More tasks (3-4 tasks total)
>   - Robustness tests
>   - Failure case analysis
> ```
>
> ****Phase 3: Paper Writing (2026-06 ~ 2026-07, 8주)****
> ```
> Week 1-2 (06/01-06/14): Draft v1
> Week 3-4 (06/15-06/28): Revision + figures
> Week 5-6 (06/29-07/12): Internal review + polish
> Week 7-8 (07/13-07/26): Final preparation & practice talk
> ```
>
> ****Phase 4: Conference (2026-07-13~17)****
> ```
> Presentation & networking at RSS 2026
> ```
>
> **장점** (RSS 2026 targeting):
> - ✅ 충분한 실험 시간 (13개월)
> - ✅ More tasks, more baselines 가능
> - ✅ Robust results + thorough analysis
> - ✅ High-quality paper writing
> - ✅ Multiple revision cycles
> - ✅ 추가 아이디어 반영 여유
>
> **Recommended Milestones**:
> - 2026-02-28: SIMPLER 결과 완성
> - 2026-04-30: Franka 실험 완료
> - 2026-06-30: Draft v1 완성
> - 2026-07-13: Conference presentation

---

## 실험 노트

**요약**: 2-frame fixed input으로 speed-invariance 해결, DINO+SigLIP partially shared architecture (300M params), Alternating training 전략

> [!note]- 📓 상세 내용 보기
>
> **2025-12-18: Method Architecture Discussion**
>
> **핵심 아이디어 구체화**
>
> **문제 인식**:
> - 원래 계획: 임의 길이 연속 이미지 → Behavior representation
> - 발견한 문제: 로봇 속도 차이 → 같은 행동인데 다른 temporal pattern
>   - 빠른 로봇: 10 frames (0.3초)
>   - 느린 로봇: 100 frames (3초)
>
> **해결 방안: 2-Frame Fixed Input**
> ```
> 입력: 항상 2장 이미지 고정 (t, t+1)
> 출력: Change embedding e(t→t+1)
> Sequential: e(t→t+1) → Transform → e(t+1→t+2) → ...
> ```
>
> **장점**:
> - Speed-invariant: 같은 변환, 적용 횟수만 다름
> - 간단한 인코더: 가변 길이 불필요
> - TraceGen의 speed retargeting과 유사하지만 더 단순
>
> **Image Preprocessing**: [[Two-Stream Image Preprocessing]]
> - M채널 (4ch): 시간적 밝기 변화 (Magnocellular 경로 모델링)
> - P채널 (2ch): 공간적 형태 정보 (Parvocellular 경로 모델링)
> - 생물학적 근거: [[Two Visual Pathways]]
> - 총 6채널 입력으로 시간·공간·색상 통합 표현
>
> **Disentangled Representation**
>
> **두 가지 임베딩 분리**:
> ```python
> e(t→t+1) = Encoder(img_t, img_t+1)
>     ↓
> z_task = task_head(e)      # Embodiment-invariant (공통 요소)
> z_action = action_head(e)  # Action-specific (구체적 움직임)
> ```
>
> **학습 전략**:
> - z_task: DINO-style contrastive (같은 task는 가깝게)
> - z_action: Transform-style prediction (sequential dynamics)
> - 정보 손실 방지: 각각 다른 목적으로 전문화
>
> **Multi-Objective Visual Encoder**
>
> **DINO + SigLIP Partially Shared Architecture**:
> ```python
> # Early layers (0-6): 공유
> shared_layers = ViT_layers[0:6]  # 100M params
>
> # Late layers (6-12): 분리
> dino_branch = ViT_layers[6:12]   # Spatial features (100M)
> siglip_branch = ViT_layers[6:12] # Semantic features (100M)
>
> Total: 300M params (vs Prismatic 700M)
> → 57% 파라미터 절약!
> ```
>
> **Alternating Training**:
> ```python
> # DINO step
> dino_out = model.forward_dino(img)
> loss_dino = dino_self_supervised_loss(dino_out)
> loss_dino.backward()
> optimizer_shared.step()
> optimizer_dino.step()
>
> # SigLIP step
> siglip_out = model.forward_siglip(img)
> loss_siglip = contrastive_loss(siglip_out, text)
> loss_siglip.backward()
> optimizer_shared.step()
> optimizer_siglip.step()
> ```
>
> **장점**:
> - Loss 충돌 없음 (번갈아가며 학습)
> - Early layer에서 중복 패치 attention 공유
> - Inference 25% 빠름 (shared layers 한번만)
>
> **관련 연구**
>
> **Reconstruction-based ViT 조사**:
> - **I-JEPA (2023)**: Feature prediction (픽셀 아닌 feature space)
> - **MAE (2022)**: Pixel reconstruction, 가장 유명
> - **BootMAE (2023)**: MAE 개선, 88.7% ImageNet
>
> **잠재적 활용**:
> - Option A: MAE를 추가 baseline으로?
> - Option B: Self-supervised reconstruction objective 추가?
> - Option C: Feature-level prediction (I-JEPA-like) → 우리의 Transform 아이디어와 유사!
>
> **Next Steps**
>
> **다음 논의 주제**:
> - [ ] MAE/Reconstruction objective를 추가할지 결정
> - [ ] Partially shared architecture vs Separate encoders 실험 계획
> - [ ] Transform-based sequential prediction 구현 방법
>
> **우선순위 재확인**:
> - 핵심 아이디어는 변하지 않음 (Cross-embodiment, Action-agnostic)
> - 오늘은 구현 디테일만 구체화
> - Week 1-2 목표: Method architecture 확정 → ✅ 거의 완료!

---

## Discussion 섹션 아이디어 (2026-01-06)

**논문 Discussion에 포함할 핵심 통찰**

### Action as Interface: A Deeper Understanding

**Main Argument:**

> Task 성공의 본질은 올바른 visual flow를 생성하는 것이다.
> Action은 그 flow를 만드는 인터페이스일 뿐이다.

**Two Types of Knowledge:**

Our approach separates robot learning into two distinct types of knowledge:

1. **Forward Knowledge: "What visual changes lead to success?"**
   - Task-dependent visual dynamics
   - Embodiment-independent (same flow across different robots)
   - Requires extensive experience to learn (220k videos in LAPA)
   - **This is the hard part**

2. **Inverse Knowledge: "What actions produce those changes?"**
   - Action-to-outcome mapping
   - Embodiment-specific (different action spaces)
   - Can be learned through inverse reasoning
   - **This is the easy part** (20-30 demos sufficient)

**Why Human Video Pretraining Works:**

기존 설명 (표면적):
- "Visual change represents action"
- "Embodiment-independent learning"

더 깊은 이유:
- Human videos provide **forward knowledge** (the hard part)
- Robot demos provide **inverse knowledge** (the easy part)
- **Separating these two makes learning efficient**

**Our Contribution through this Lens:**

| Method | Input | Forward | Inverse | Problem | Efficiency |
|--------|-------|---------|---------|---------|------------|
| LAPA | Image_t (1 frame) | Task-agnostic | Latent mapping | **Ill-posed** (1 frame → change?) | 220k videos |
| OpenVLA | Image sequence | Coupled | End-to-end | No separation | 970k trajectories |
| **Ours** | **Image_t, t+1** | **Task-conditioned** | **Decoder-only** | **Well-posed** ✅ | **10-20 demos** ✅ |

**LAPA's Fundamental Limitation (2026-01-07):**

LAPA의 구조적 모순:
```
Input: Image_t (single frame) + Task
Output: Latent action (represents change)

Problem: Change is inherently a temporal relationship (t → t+1)
         Cannot infer "change" from single state alone!
         → Ill-posed problem → Requires 220k videos to compensate
```

**Our Approach Solves This:**
```
Input: Image_t, Image_t+1 (actual change) + Task
Output: Behavior representation

Advantage:
  - Observes actual visual change (well-posed)
  - Task-conditioning filters important features
  - → Requires far fewer videos (10-20 demos for inverse)
```

**Task-Conditioning Enhances Forward Learning:**
- Task specifies relevant visual features
- "red cup" → Attend to color features
- "pick up" → Attend to vertical motion
- → **Less data needed to learn forward model**
- → **LAPA learns all changes equally, we focus on task-relevant changes**

**Connection to Established Work:**

This separation is already validated in:
- **Visual MPC**: Learn dynamics (forward) → Plan actions (inverse)
- **DreamerV3**: World model first → Policy later
- **Visual Foresight**: Video prediction → Action planning

Our novelty: **Task-conditioned forward learning** makes this separation data-efficient.

### Implications for Digital Twin

**Traditional View (Problematic):**
```
Sim → Real transfer = Hard (large sim-to-real gap)
```

**Forward/Inverse View (Practical):**
```
Forward (visual flow): Similar in sim and real
Inverse (action mapping): Different in sim and real

→ Learn forward in sim (safe, fast)
→ Learn inverse in real (only adaptation needed)
→ Gap is halved!
```

**Digital Twin as Forward Learning Platform:**
- Safe experimentation with diverse configurations
- Learn which visual features matter for task success
- Rapid iteration without physical constraints
- Real robot only needed for inverse mapping (20-30 demos)

### Limitations and Future Work

**Current Limitations:**

1. **Information Filtering Scope**
   - Currently filter via task-conditioning
   - Still process entire image (background, lighting, etc.)
   - Future: Spatial attention mask (ignore irrelevant regions entirely)

2. **Contact-Rich Manipulation**
   - Visual flow may not capture force/tactile feedback
   - Fine-grained control (±0.5mm precision) challenging
   - Solution: Combine with proprioceptive/force sensors

3. **Exploration Efficiency**
   - "Try action → Check result" can be slow in real world
   - Solution: Use learned forward model for planning (Visual MPC)

4. **Temporal Credit Assignment**
   - Which actions in a sequence contributed to success?
   - Solution: Attention over temporal dimension

**Future Directions:**

1. **Visual MPC Integration**
   - Use learned forward model for action planning
   - Closed-loop control: Execute → Observe → Re-plan
   - True realization of "action as interface"

2. **Multi-Modal Forward Models**
   - Integrate tactile, force, proprioceptive feedback
   - Richer understanding of task dynamics

3. **Zero-Shot Transfer**
   - If forward model is perfect, can we skip inverse learning?
   - Action sampling + forward prediction → Find action that produces desired flow

### Key Takeaway

> 우리는 "action을 잘 맞추는" 연구가 아니라,
> "task flow를 이해하는" 연구를 한다.
> Action은 그저 그 flow를 실현하는 수단일 뿐이다.

This philosophical shift explains:
- Why human videos are effective (forward knowledge)
- Why we need few robot demos (inverse is easy)
- Why task-conditioning matters (efficient forward learning)
- Why decoder-only finetuning works (inverse is separate)

---

## 읽을 논문

**요약**: 최우선(LAPA ✅, OpenVLA ✅, Bridge V2 ✅) | 최신 cross-embodiment(TraceGen ✅, X-Diffusion, TrajSkill) | 선택적(VC-1, Diffusion Policy)

> [!info]- 📚 상세 내용 보기
>
> **최우선 (직접 관련 - 반드시 읽어야 함)**:
> - [x] **LAPA (2024)** ⭐⭐⭐ 핵심!
>   - ICLR 2025, Latent Action Pretraining from Videos
>   - **왜 중요**: 사람 비디오 > 로봇 데이터 증명 (36.8% vs 30.8%)
>   - **우리와의 관계**: 같은 문제(human video), 다른 접근(latent action vs task-conditioned)
>   - **차별점**: VQ-VAE unsupervised vs 우리는 task-aware supervised
>   - **활용**: Human video 효과 검증, 우리 방법의 우수성 강조
>
> - [x] **OpenVLA (2024)** ⭐⭐⭐
>   - arXiv 2406.09246, 현재 VLA SOTA
>   - **왜 중요**: Primary baseline, 이것을 이겨야 함
>   - **비교 포인트**: Robot action-labeled vs Human video pretraining
>   - **구현**: HuggingFace checkpoint 활용 (쉬움)
>
> - [x] [[BridgeData V2 (2023)]] ⭐⭐
>   - CoRL 2023, Homer Walke, Chelsea Finn, Sergey Levine
>   - **규모**: 60K trajectories, 24 environments, WidowX robot
>   - **왜 중요**: 로봇 학습 분야 사실상 표준 벤치마크 (322 인용)
>   - **활용**: Finetuning 데이터, evaluation benchmark
>   - **저자**: UC Berkeley/Stanford RAIL lab - 최고 권위
>   - **임팩트**: Open X-Embodiment의 핵심 구성 요소
>   - **재현성**: 오픈소스, 저가 로봇($3-5K), 누구나 재현 가능
>
> **필수 (Baseline & Dataset)**:
> - [x] **VC-1 (2023)**: Visual representation baseline (선택적 비교)
> - [x] **[[Sources/papers/RT-X (2023)]]**: Open X-Embodiment 데이터셋
> - [x] **EgoDex (2024)**: Apple, 최고 품질 사람 manipulation 데이터 (829시간)
> - [ ] ~~R3M (2022)~~: VC-1로 충분 (패스)
> - [ ] ~~RT-1 (2022)~~: OpenVLA로 충분 (패스)
>
> **최신 Cross-Embodiment Learning (2025)**:
> - [x] **TraceGen (Nov 2025)** ⭐⭐⭐ arXiv 2511.21690
>   - **핵심**: 3D trace-space for cross-embodiment learning from human videos
>   - **데이터**: 123K videos, 1.8M triplets (TraceForge pipeline)
>   - **성능**: 5 human videos → 67.5% success, 50-600x faster inference
>   - **차별점**: 3D geometric trace vs 우리는 2D visual behavior (simpler, efficient)
>   - **활용**: Related Work 핵심 비교 대상, efficiency 근거
>
> - [ ] **UMI-on-Air (Oct 2025)** ⭐⭐⭐⭐ arXiv 2510.02614 **필독!**
>   - **핵심**: Handheld gripper (UMI) human demos → embodiment-agnostic policy → aerial manipulator deployment
>   - **방법**: Embodiment-Aware Diffusion Policy (EADP) - gradient feedback from controller
>   - **성능**: Long-horizon, high-precision aerial manipulation tasks 성공
>   - **차별점**: 우리와 거의 동일한 문제! (human demos → robot transfer)
>   - **활용**: 직접 비교 필수, 가장 유사한 접근법
>   - **주의**: RSS 2026 제출 시 핵심 related work
>
> - [ ] **ViDEN (Dec 2024)** ⭐⭐⭐ arXiv 2412.20226
>   - **핵심**: Visual demonstrations → embodiment-agnostic navigation policy
>   - **방법**: Diffusion-based policy, depth images, relative target positions
>   - **성능**: Small dataset (500 points), human reaching & tracking tasks
>   - **차별점**: Navigation vs 우리는 manipulation (상호보완적)
>   - **활용**: Visual demonstration + embodiment-agnostic 접근 비교
>
> - [ ] **Latent Policy Steering (Jul 2025)** ⭐⭐ arXiv 2507.13340
>   - **핵심**: Multi-embodiment World Model pretraining + latent space search
>   - **데이터**: Open X-embodiment (2K episodes) + human play data
>   - **성능**: 30 demos에서 50% 성능 향상, 50 demos에서 20% 향상
>   - **차별점**: World model + search vs 우리는 direct behavior representation
>   - **활용**: Multi-embodiment pretraining 전략 비교
>
> - [ ] **E2VLA (Sep 2025)** ⭐ arXiv 2509.14630
>   - **핵심**: Embodiment equivariant VLA (configuration transformation equivariance)
>   - **방법**: Geometry-aware network + equivariant action decoder
>   - **차별점**: Equivariance theory vs 우리는 task-conditioned representation
>   - **활용**: Theoretical foundation 참고
>
> - [ ] **TrajSkill (Oct 2025)** ⭐⭐ arXiv 2510.07773
>   - **핵심**: Sparse optical flow as embodiment-agnostic motion cues
>   - **성능**: 16.7% cross-embodiment improvement, real kitchen tasks
>   - **차별점**: Low-level optical flow vs 우리는 high-level task-conditioned behavior
>   - **활용**: Motion representation 비교
>
> - [ ] **Masquerade (Aug 2025)** ⭐⭐ arXiv 2508.09976
>   - **핵심**: Video editing (inpainting + robot overlay) for human → robot demos
>   - **데이터**: 675K frames, 50 robot demos per task
>   - **성능**: 5-6x better on bimanual kitchen tasks
>   - **차별점**: Video editing vs 우리는 direct representation learning
>   - **활용**: Data augmentation 전략 참고
>
> - [ ] **Gen2Act (Sep 2024)** ⭐ arXiv 2409.16283
>   - **핵심**: Human video generation → robot execution (zero-shot)
>   - **차별점**: Generation-based vs representation-based
>   - **활용**: Zero-shot capability 비교
>
> - [ ] **ET-VLA (Nov 2025)** arXiv 2511.01224
>   - Embodiment transfer for multi-robot, 53.2% better than OpenVLA
>   - **활용**: Baseline 참고, synthetic continued pretraining
>
> - [ ] **CHORD (Jan 2026)** arXiv 2601.04194
>   - **핵심**: Video generative models → Lagrangian motion extraction → robotics manipulation policies
>   - **방법**: Distillation-based pipeline from 2D videos (universal, category-agnostic)
>   - **저자**: Jiajun Wu lab (Stanford)
>   - **차별점**: Generative model distillation vs 우리는 contrastive representation
>   - **활용**: Video-to-policy 접근법 비교 (간접적 관련)
>
> **Vault에 있는 관련 논문 (다시 읽기)**:
> - [x] **[[Sources/papers/CURL (2020)]]**: Contrastive learning, sample efficiency
> - [x] **[[Sources/papers/DINO (2021)]]**: Self-supervised ViT, temporal consistency
> - [ ] **[[Sources/papers/Diffusion Policy (2023)]]**: Action decoding baseline
> - [ ] **[[Sources/papers/RT-2 (2023)]]**: VLM for robot control
> - [ ] **[[Sources/papers/ALOHA (2023)]]**: Action chunking (ACT), temporal consistency
> - [ ] **[[Sources/papers/Visual Pre-training Survey (2023)]]**: Visual pretraining survey
>
> **이론적 기반 (참고)**:
> - [ ] **Embodiment Scaling Laws (May 2025)** arXiv 2505.05753
>   - **핵심**: Training embodiment 수 증가 → unseen embodiment 일반화 향상
>   - **실험**: ~1,000 procedurally generated embodiments (robot locomotion)
>   - **결과**: Embodiment scaling > Data scaling (fixed embodiment)
>   - **차별점**: Locomotion vs 우리는 manipulation (다른 domain)
>   - **활용**: Cross-embodiment generalization 이론적 근거
>
> **참고 (낮은 우선순위)**:
> - [ ] **Robot Trains Robot (Aug 2025)** arXiv 2508.12252
>   - **핵심**: Robotic arm teacher → humanoid robot student (real-world RL)
>   - **방법**: Protection, reward, perturbation, automatic reset by teacher robot
>   - **차별점**: Hardware setup vs 우리는 data/representation (orthogonal)
>   - **활용**: Real-world learning setup 참고 (간접적)
>
> - [x] **[[Sources/papers/TCN (2018)]]**: Time-contrastive networks
> - [[Sources/papers/OpenVLA (2024)]]: VLA 비교 대상
> - [[Sources/papers/GNM (2022)]]: Cross-embodiment navigation
> - [[Sources/papers/CLIP (2021)]]: Vision-language foundation

---

## 관련 노트

- [[Questions/Q - Action-Agnostic Robot Learning]]
- [[Sources/papers/CURL (2020)]]: Sample efficiency와 representation learning
- [[Sources/papers/DINO (2021)]]: Temporal consistency와 collapse 방지

**주요 비교 대상**:
- [[Sources/papers/Octo (2024)|critiques]] - Primary baseline. Octo는 modular architecture로 flexibility 달성하지만 action-space coupling과 데이터 품질에 의존적. Attention이 "알아서" 처리할 것을 기대하나 데이터 불균형(wrist 27%, language 56%) 시 성능 저하. 우리는 action-agnostic representation으로 근본 해결. Robot-to-robot (800k eps) vs Human-to-robot (사람 비디오).

**구분 필요**:
- [[Outputs/Idea - DynamicNet]]: 동기는 유사(핵심 정보 선별)하나 접근이 다름
  - DynamicNet: 아키텍처 혁신 (attention 대체, 노드 기반)
  - 이 논문: 학습 방법론 (cross-attention 활용, foundation model)

---

## 태그

#paper #project #robot-learning #action-agnostic
