# Action-Agnostic Visual RL Research Plan

**마지막 업데이트**: 2026-04-17
**연구 질문**: **구조적 inductive bias를 가진 시각 표현 학습이, action label 없이도 시각-행동 연결 태스크에 유용한 표현을 만드는가?**

## 연구 동기

- **핵심 가설**: 영장류 시각 피질의 magnocellular / parvocellular 경로 분리는 생물학적 시각 시스템의 근본적 구조 원리. 이 구조를 비디오 표현 학습 모델에 반영하면 더 범용적인 표현을 학습할 수 있다.
- **Two-Stream 접근**: M(motion) / P(pattern) 채널 분리 + CLS exchange를 통해 구조적 inductive bias를 부여. Pixel reconstruction 목표와 결합.
- **시각-행동 연결 태스크로 검증**: 로봇 조작 태스크(LIBERO)는 "어떤 feature가 진짜 유용한지"를 가장 직접적으로 평가할 수 있는 설정. **Two-Stream은 로봇용 특수 알고리즘이 아니라, 일반 시각 표현 학습 방법의 로봇 태스크 검증**이다.
- **V-JEPA에 대한 중요 관찰**: V-JEPA (Bardes et al., 2024)는 같은 조건에서 feature prediction > pixel reconstruction을 실증 주장. 하지만 이 증거는 전부 **classification 벤치마크**(Kinetics-400, SSv2, ImageNet)에 국한. **로봇 제어(continuous control) 도메인 증거는 부재**. 우리는 이 갭을 채운다.

---

## 실험 대상 모델 (Encoder lineup 재편, 2026-04-14)

**배경 (V-JEPA-ours 제외 이유)**: V-JEPA-ours를 EgoDex 2-frame 세팅에 적응시키는 3차례 시도(LR warmup 추가, mask ratio 85%→60% 완화) 모두에서 학습 발산. 2-frame 세팅은 V-JEPA의 16-frame temporal redundancy 전제를 깨뜨려 EMA target drift를 유발 (자세한 내용은 Phase 1.5 아래 참조). **V-JEPA-ours의 negative result는 paper에 "2-frame regime에서 feature prediction의 한계"로 기록**하고, V-JEPA는 **공식 가중치(16-frame, VideoMix2M)**로만 비교군에 포함.

| 구분 | Encoder | 사전학습 데이터 | 파라미터 | 방법 철학 | 학습 주체 |
|------|---------|---------------|---------|----------|----------|
| **제안** | **Two-Stream v6 (ours)** | EgoDex (~100M frames) | ~213M | **M/P 구조 + pixel reconstruction + rotation aug** | 🔥 우리 학습 |
| **Controlled comparison** | **VideoMAE-ours (2-frame)** | **EgoDex (same)** | ~101M (ViT-B + decoder) | Vanilla MAE (구조적 bias 없음, mask 0.5) | 🔥 **우리 학습 (신규)** |
| **Native 세팅 baseline** | VideoMAE-official | Kinetics-400/SSv2 (16-frame) | ~86M | MAE (공식 세팅) | 📦 공개 가중치 |
| **Native 세팅 baseline** | V-JEPA 2.1 ViT-B | VideoMix22M (16-frame, 384px) | 86.8M | Feature prediction + dense loss (최신 video SSL) | 📦 공개 가중치 |
| **로봇 제어 SOTA** | VC-1-Base | Ego4D + 조작 (~500M frames) | 86M | MAE (embodied AI 표준) | 📦 공개 가중치 |
| Internet-scale SSL | DINOv2-Base | LVD-142M (웹 이미지) | 86M | Self-distillation | 📦 공개 가중치 |
| Internet-scale VL | SigLIP-Base | WebLI (웹 10B 이미지-텍스트) | 86M | Vision-language contrastive | 📦 공개 가중치 |

### 3-축 비교 구조

**축 1: 같은 데이터·같은 framework, 다른 구조** (Two-Stream vs VideoMAE-ours)
- 통제: EgoDex, 2-frame pixel MAE framework, 같은 compute, 같은 sampling
- 변수: encoder 구조 (M/P two-stream vs vanilla ViT)
- 답할 질문: "**M/P 구조적 bias가 표현 품질에 기여하는가?**" (단일 변수 격리)
- 주의: Two-Stream(213M) > VideoMAE(101M) 파라미터 차이는 "M/P 두 스트림 설계상 불가피"로 명시. 각 스트림이 ViT-B backbone으로 baseline과 동일한 per-stream capacity 유지

**축 1': 2-frame regime에서의 방법론 한계** (V-JEPA-ours negative result)
- V-JEPA의 feature prediction framework는 temporal redundancy를 전제. 2-frame 세팅에서 3차 hyperparameter 완화 후에도 EMA target drift로 발산
- 이 실패는 paper의 Appendix에 3-attempt loss curve overlay로 기록 (Two-Stream 수렴 곡선과 비교)

**축 2: 같은 도메인 계열, 다른 구조** (Two-Stream vs VC-1)
- 통제: Egocentric/manipulation video
- 변수: 데이터(EgoDex vs Ego4D+) + 방법(M/P structured vs standard MAE)
- 답할 질문: "M/P 구조가 기존 embodied AI baseline보다 나은가?"

**축 3: 소규모 principled vs 대규모 data-driven** (Two-Stream vs DINOv2/SigLIP)
- 통제: downstream 평가
- 변수: 데이터 규모 + 방법 (100M frames 구조 기반 vs 수십억 규모 범용 SSL/VL)
- 답할 질문: "작은 규모의 구조 기반 방법이 대규모 범용 학습과 경쟁 가능한가?"

### 공정성 원칙

- **모든 encoder는 frozen** (downstream에서 학습 안 함)
- **동일한 입력 형식**: 단일 프레임 encoder (DINOv2, SigLIP, VC-1)도 `(img_{t-1}, img_t)` 두 프레임을 각각 forward pass 후 feature concat
- **각 encoder의 공식 preprocessing 사용** (정규화, resolution)
- Downstream에서 유일하게 학습되는 것: **MLP action decoder** (Phase 3) 또는 **projection + LoRA** (Phase 3B)

### 학습 주체 요약

| 카테고리 | 작업 |
|---------|------|
| 🔥 **우리가 EgoDex에 학습** | Two-Stream v4 (resume 중), **VideoMAE-ours 2-frame (신규)** |
| 📦 **공개 가중치 그대로** | VideoMAE-official, V-JEPA-official, VC-1-Base, DINOv2-Base, SigLIP-Base |
| 🔧 **Downstream 학습 (encoder frozen)** | MLP decoder (Phase 3), Projection + LoRA (Phase 3B) |

---

## 실험 로드맵

### Phase 1: EgoDex 사전학습 → Hand Pose Probing ✅ 완료

**목표**: 제안한 Two-Stream 구조의 표현 품질 검증 (within-domain)

**작업**:
- Two-Stream v4, VideoMAE를 EgoDex part1~5 일부로 반복 ablation
- frozen encoder → EgoDex test에서 hand pose linear probe (R²)
- Architecture / MAE masking / gap 분포 / composition consistency 전체 ablation

**결과**: v4 설정 확정 (d=12, s=2, M=0.3/P=0.5, max_gap=60 triangular).
상세 ablation 결과는 하단 "Phase 1 Ablation 결과" 참고.

### Phase 1.5: 데이터 스케일업 (Full EgoDex) 🔄 진행 중

**목표**: 전체 EgoDex (part1~5, 314k videos)로 최종 체크포인트 확보

**작업**:
- **Two-Stream v4** 50 epoch 학습 — epoch 48/50 시점에서 --time=3-12h 한도 TIMEOUT (jobID 32712324). **Resume 잡 32983533 제출 (AIP, 2노드×4 H100, --time=6h, 남은 2 epoch)**
- **VideoMAE-ours (2-frame)** — V-JEPA-ours 대체 controlled comparison 모델.
  - 기존 구현 (`src/models/videomae.py`) 이미 2-frame 지원 (num_frames=2, tubelet_size=2, 196 patches)
  - **mask ratio 0.5** 채택 (공식 0.75는 16-frame temporal redundancy 전제 → 2-frame에서는 visible token 부족. V-JEPA 2-frame 실패 사례와 동일 원리)
  - Two-Stream과 동일한 gap sampling (`--max-gap 60 triangular`), 동일 EgoDex splits
  - V-JEPA-ours 학습 종료/취소 후 제출
- **V-JEPA-ours** — 3차례 시도 모두 발산 (아래 참조). 30 epoch까지 기록 후 중단, negative result로 paper 기록
- 학습 후 hand pose probing으로 품질 확인 (sanity only, main 비교는 DROID)

**산출물**: `results/checkpoints/{two_stream,videomae,v_jepa}/` 에 best_model.pt + 학습 곡선

#### V-JEPA-ours 3차례 학습 시도 요약 (negative result 보존용)

| 시도 | JobID | 설정 | 결과 | 관찰 |
|------|-------|------|------|------|
| 1차 | 32867433 | warmup 없음, mask 0.80/0.85 | epoch 4 cancel | LR=2e-4부터 시작 → EMA target drift 즉시 발산 |
| 2차 | 32867645 | warmup 5ep 추가, mask 0.80/0.85 | epoch 7 cancel | warmup 후에도 상승. mask 비율 과도 의심 |
| 3차 | 32950553 | warmup + mask 0.50/0.60 | 30 epoch 도달 후 **2026-04-15 01:16 계획대로 CANCEL** | 3단계 loss 패턴: ep1-12 진동(0.13-0.30) → ep13-20 collapse(0.002-0.003, trivial solution) → ep21-30 재발산(0.15-0.23) |

**3차 학습 Loss 패턴 상세** (32950553, 2026-04-12~15, 2일 2시간, 405.5 GPU·h):
- Ep1-12: 0.13~0.30 사이 진동, 방향 잡지 못함
- Ep13-20: 0.002~0.003으로 급락 — **representation collapse 의심** (EMA y-encoder에 x-encoder가 trivial alignment)
- Ep21-30: 0.15~0.23으로 재발산, 안정 구간 없이 상승세로 중단

**Paper 기록 방식**: 3 attempts + Two-Stream reference의 loss curve overlay (log scale) → Appendix. 메시지: "2-frame 세팅은 V-JEPA의 temporal redundancy 전제를 깨뜨려, hyperparameter 완화만으로는 EMA target learning이 안정화되지 않음. 3차 시도에서 관찰된 collapse→recovery 패턴은 predictor가 일시적으로 trivial solution에 도달 후 재이탈하는 전형적 실패 모드."

#### Attention 시각화 방향 결정 (2026-04-14)

- **학습 진화 스토리 포기**: Two-Stream은 epoch 4에 이미 복원 품질 포화, epoch 33부터 완전 plateau. "학습 진행에 따른 attention 변화" 스토리는 시각적으로 나타나지 않음 (pixel MAE 특성)
- **대체 프레이밍**: "M vs P stream 기능적 분리" — ep48 한 checkpoint에서 다양한 샘플(6-8개)에 대한 M/P attention 분포 차이를 main figure로. 학습 진화는 메인에서 제외
- **정량 보강 (선택)**: M/P attention cosine distance, spatial entropy 등으로 분리도를 숫자로 증명
- **우선순위**: 시각화 추가 작업은 paper figure 작성 시점으로 미룸. 지금은 DROID probing / LIBERO 준비가 더 중요

#### 🚨 V4 position prior 편향 문제 (2026-04-15 발견)

**증상** (정량 + 정성 둘 다 signal):
1. **Probing R² = 0.197** (test split, ep48 best_model, patch_mean_concat, gap=10) — GO/NO-GO threshold 0.7에 크게 못 미침
2. **Rotation diagnostic** ([results/viz_rotation/sweep_dustpan_159.png](../results/viz_rotation/sweep_dustpan_159.png)): 입력을 0°/90°/180°/270° 회전시켜도 **M attention이 거의 동일한 중앙 하단 grid 패턴** 유지. 입력 ΔL은 회전을 따라가는데 attention은 그대로 → content가 아니라 **position prior + mask_token grid에 attention이 고정**. Pred P도 회전 입력에서 붕괴 → "학습 분포 내 image manifold 외우기"에 가깝다는 확증.

**원인 가설 (우선순위 순)**:
1. **Mask ratio 0.3/0.5가 너무 약함** — 재구성이 쉬우면 position prior + local continuity로 풀 수 있어 content 학습 압력 약화
2. **2D RoPE의 절대좌표** — patch position 신호가 강해 content 대비 지배적
3. **CLS exchange num_stages=2** — 스트림 간 소통이 약해 M이 자기 stream prior에 갇힘
4. **EgoDex 도메인 편향** — 손이 거의 늘 화면 하단. Position prior만으로도 loss 낮아짐

**해결 전략 (제안, 우선순위 순)**:

| 전략 | 목적 | 비용 | 기대 효과 |
|------|------|------|----------|
| **A. Mask ratio 대폭 상향** (M/P 모두 0.75~0.85) | 재구성을 "보이지 않는 영역 추론"으로 강제 → content 읽기 필수 | 1회 full training (~3.5일) | 🔥 가장 유력. VideoMAE의 0.75 정신 복구 |
| **B. Mask에 spatial block 패턴 도입** | random patch 대신 block 단위 masking → local copy 불가 | 학습 로직 수정 + 1회 학습 | 🔥 MAE→VideoMAE 차이의 핵심 교훈 |
| **C. 데이터 augmentation에 회전/플립 추가** | position prior 자체를 무력화 (inductive bias로) | 사소한 코드 변경 + 재학습 | 🔸 Probe R² 직접 개선 가능 |
| **D. Position embedding 약화** (RoPE frequency 조정 or dropout) | content 상대 가중치 상승 | 코드 수정 + 재학습 | 🔸 2D RoPE 자체 재설계 필요 |
| **E. Crop augmentation 강화** (RandomResizedCrop scale 0.3~1.0) | 손 위치 다양화로 position prior 깨기 | 사소 + 재학습 | 🔸 C의 간접 버전 |

**최적 조합 예상**: A(mask 0.75) + B(block masking) + C(rotation aug). 이 세 개는 서로 중첩 방지 효과 있고 구현 부담 작음. D는 큰 수술이라 뒤로 미뤄도 됨.

**다른 시각** (paper 기여):
- 만약 A+B+C로 개선되면 "저-mask ratio + absolute position은 2-frame MAE에서 prior collapse 유발"이 방법론 교훈
- 만약 개선 안 되면 "2-frame + MAE 틀 자체의 한계" → Two-Stream 구조 가치를 기각하는 증거로 paper 축 재설계 필요

##### 검증용 Action Items

**완료 (2026-04-15)**:
- [x] **V3_APE_s3_ep29, V4_mask30_APE_s2_ep4, V4_nomask_APE_s2_ep4 rotation diagnostic** (로컬 워크스테이션, 커밋 4e59777)
  - 3개 모두 **content-driven attention** 확인. 회전 대응 정상
- [x] **V4_mask30_RoPE_s2_ep4** rotation diagnostic ([rotation_V4_mask30_RoPE_s2_ep4_4.png](architecture/rotation_V4_mask30_RoPE_s2_ep4_4.png), 클러스터)
  - ep4 RoPE는 이미 부분적 position-bias 경향 관찰 (ep48만큼 완전 고정 아니나 APE와 분명한 차이) → 사용자 판정

**진단 가설 요약 (2026-04-15 관찰 기반)**:
- V3/V4 APE 체크포인트(ep4 또는 ep29): content-driven 유지
- V4 RoPE ep4: 부분 content + 초기 prior 축적
- V4 RoPE ep48: 강한 position prior 고정
- → **RoPE + 학습량 상호작용 가설**: RoPE가 위치 단축경로를 구조적으로 허용 → 학습 진행하며 position prior가 점진적으로 심화. APE는 add 방식이라 content/pos 분리가 구조적으로 유지됨 (논리적 가설)

**검증 실험 (완료, 2026-04-16)**:
- [x] **APE 진단 학습** (JobID 33012271, ep4까지 저장 후 cancel)
  - Config: reference V4 RoPE와 완전 동일, 유일한 차이 `--use-ape`
  - ep4 rotation diagnostic: **Content-driven 유지** ([rotation_APE_diag_ep4_4.png](architecture/rotation_APE_diag_ep4_4.png))
  - ep4 action probing (33179673): **peak R² = 0.2191** (RoPE ep48의 0.1972와 거의 동일)
  - → **매트릭스 L174 적중**: "PE 바꿔도 R² 낮음 → content 학습 자체 약함, mask 0.3이 지배"

**해석 매트릭스** (APE ep5 결과 기준):

| APE ep5 rotation | APE ep5 probing R² | 결론 | 다음 스텝 |
|---|---|---|---|
| Content-driven | 0.5~0.7+ | **RoPE 단독 원인 확정** | PE를 APE로 되돌리거나 RoPE frequency/dropout 재설계 (전략 D) |
| Content-driven | < 0.3 | PE 바꿔도 R² 낮음 → **content 학습 자체가 약함** (전략 A+B+C 필요, mask 0.3이 지배 원인) | A+B+C 재학습 |
| 부분 bias | 중간 | PE 외 공범 있음 (mask ratio, M channel 설계 등) | 다변수 ablation |
| 완전 고정 | 낮음 | **PE 무관**, 원인은 mask ratio/데이터 regularity | RoPE 복귀 + A+B+C 집중 |

**실험 경과 (2026-04-16~17)**:
- [x] **v5** (APE + mask 0.5/0.5, 33179788): ep4(R²=0.208)→ep8(R²=0.192) 하락 확인. Position prior overfit은 PE/mask 변경만으로 해결 불가 → ep8에서 CANCELLED
- [x] **v6** (APE + mask 0.5/0.5 + **rotation aug**, 33222151): 학습 중. **ep8 R²=0.259** — v5 ep8(0.192) 대비 +35%. Rotation aug가 position prior 의존을 효과적으로 차단
- Rotation aug: 90% 동일회전(두 프레임 같은 각도) + 10% 독립회전(각 프레임 다른 각도), {0°,90°,180°,270°}. `compute_loss`에서 `torch.rot90` view op 적용 (오버헤드 무시 가능)
- v6 reconstruction loss도 v5와 동일 패턴(ep3~4에서 0.002대) — rotation은 epoch 평균 loss에는 큰 영향 없지만 probing R²에는 확실한 효과

##### 🔗 통합 가설: "RoPE가 V-JEPA 실패의 공범?"

**PE 사용 현황 (2026-04-15 재확인)**:
- Two-Stream V4: 2D RoPE (encoder) → 수렴 + position prior 편향
- V-JEPA-ours: 2D RoPE (encoder, `two_stream.build_2d_rope_freqs` import) + sin-cos APE (predictor) → 3회 발산
- VideoMAE-ours (학습 중 33003926): **sin-cos APE** (공식 VideoMAE) → 결과 대기

**통합 해석 가능성**:
- V-JEPA 실패 = 수렴 자체 불가 (1차 원인: mask ratio 과다 + EMA target drift)
- Two-Stream RoPE 편향 = 안정 수렴 후 점진적 position prior 축적
- 다른 실패 유형이지만 **"2D RoPE × 2-frame egocentric SSL"의 구조적 부적합**이라는 **공통 근본 원인** 가능성
- RoPE는 QK 곱셈에 절대좌표를 결합 → content와 position이 분리 불가능하게 얽힘 → sparse masking + EMA target + 강한 공간 규칙성(EgoDex)이 만나면 (a) 수렴 실패 또는 (b) 위치 단축경로 학습으로 귀결

**자연 실험 (이미 진행 중)**:
VideoMAE-ours (33003926)는 같은 2-frame EgoDex + sin-cos APE. 결과 판정:

| VideoMAE-ours (APE) 결과 | 통합 해석 |
|---|---|
| 수렴 + content-driven attention | **PE 축 확정**: RoPE가 2-frame SSL의 공통 실패 원인. V-JEPA 부정적 결과에 "RoPE confound" 주석 또는 paper 축 재구성 ("PE choice가 2-frame SSL 성패 결정") 고려 |
| 수렴 + position prior 편향 | PE 무관. 원인은 mask ratio (0.5도 부족) or 데이터 regularity |
| 수렴 실패 | 2-frame + EgoDex 자체 한계. PE 무관 |

**검증 잡**:
- 33003926 (VideoMAE-ours, ~3일 잔여) → 이 결과 + Two-Stream APE 결과(33012271) 통합 해석
- 둘 다 APE가 우수 → **paper 서사 재구성**: "2D RoPE는 2-frame egocentric self-supervised에서 구조적 실패 모드 유발. 본 연구는 이를 Two-Stream APE 전환으로 해결"
- 둘 다 APE도 같은 문제 → RoPE 무관, 원인은 mask ratio / 데이터 편향

#### VideoMAE-ours 구현 상태 (작업 불필요)

- [src/models/videomae.py](../src/models/videomae.py): 이미 2-frame (num_frames=2, tubelet_size=2, 196 patches)
- [scripts/pretrain.py](../scripts/pretrain.py): `--model videomae` 분기 OK, `--mask-ratio 0.5` CLI override 가능
- [scripts/cluster/pretrain.sbatch](../scripts/cluster/pretrain.sbatch): `MODEL=videomae`에 `--mask-ratio 0.5` 기본값 설정됨
- Two-Stream과 동일 sampling/data/compute → 단일 변수(구조) 통제 비교

### Phase 2: Cross-Domain Action Probing ⏸️ 대기

**평가 우선순위 재편 (2026-04-14)**: Two-Stream과 VideoMAE-ours만 EgoDex로 학습하므로 EgoDex within-domain probing은 **controlled comparison 전용 (Two-Stream vs VideoMAE-ours, 축 1)**. 공개 가중치는 EgoDex OOD이므로 함께 비교하면 home-field advantage 문제. 따라서 **main cross-encoder 비교는 DROID**.

| 평가 | 비교 대상 | 역할 |
|------|----------|------|
| **EgoDex probing** (축 1) | Two-Stream vs VideoMAE-ours | 구조적 bias 기여도 sanity |
| **DROID probing** (main) | Two-Stream, VideoMAE-ours, VideoMAE-official, V-JEPA-official, VC-1, DINOv2, SigLIP | Cross-embodiment 전이 비교 |
| **LIBERO** (Phase 3) | 동일 7개 encoder | Downstream robot control |

**데이터**: DROID v1.0.1 (95k episodes Franka, 다운로드 완료 3.4 TiB, 프레임 추출 진행 중)

**평가 프로토콜 (2단계)**:

**Primary — 2-frame 통제 비교 (main table)**:
- 7개 encoder 전부 동일 2프레임 입력 (frame[0], frame[gap])
- V-JEPA/VideoMAE-official도 2프레임 (정보량 통제)
- Action target: action[gap] - action[0] (7-DoF delta)
- 지표: R², cosine similarity

**Supplementary — native input 비교 (부록)**:
- 각 모델이 설계된 입력 형태로 평가:
  - CLIP/DINOv2/SigLIP/VC-1: **1프레임** (frame[gap]만, 정적 표현의 한계 노출)
  - Two-Stream/VideoMAE-ours: **2프레임** (네이티브)
  - V-JEPA-official/VideoMAE-official: **16프레임** (네이티브)
- 정보 계층 1 < 2 < 16으로 명확. "정보량 증가 대비 성능 향상"을 정량화

**이상적 결과**: Two-Stream v6가 2-frame 통제에서 최상위 + native 16-frame V-JEPA에 근접/우위 → "2-frame M/P 구조가 정보 효율적"

**Go/No-Go**:
- 2-frame 비교에서 우위 → Phase 3 순항
- 열세 → 원인 분석 (도메인 갭 vs 표현 품질)

#### Encoder 로더 구현 현황

| Encoder | load_encoder | encode_batch | 패키지 | 상태 |
|---------|:---:|:---:|--------|------|
| Two-Stream v6 | ✅ | ✅ | — | 완료 |
| VideoMAE-ours | ✅ | ✅ | — | 완료 |
| CLIP | ✅ | ✅ | transformers | 완료 |
| DINOv2 | ✅ | ✅ | transformers | 완료 |
| SigLIP | ✅ | ✅ | transformers | 완료 (2026-04-17) |
| VC-1 | ✅ | ✅ | vc_models (git) | 완료 (2026-04-17) |
| V-JEPA 2.1 ViT-B | ✅ | ✅ | torch.hub (vjepa2) | 완료 (2026-04-17), 86.8M, 384px, 2-frame 동작 확인 |
| VideoMAE-official | ✅ | ✅ | transformers | 완료 (2026-04-17), pos_embed slice로 2-frame 지원 |

### Phase 3: LIBERO BC (메인 downstream 실험) ⏸️ 대기

**목표**: "표현에 인코딩된 action 정보가 **실제 제어**에 유용한가"를 검증

**실험 구조**:
```
frozen encoder → MLP action decoder → 7-DOF action
  (5종 비교)     (학습 대상)          (task-conditioned)
```

**학습 대상**: MLP action decoder만. encoder는 모두 frozen.
**학습 데이터**: LIBERO demonstrations (BC, supervised regression)
**태스크 구분**: per-task policy 또는 task ID embedding (자연어 사용 안 함)

**평가**:
- LIBERO 시뮬레이터에서 closed-loop rollout
- Task suite: `libero_spatial` (main), `libero_object`, `libero_goal`, `libero_10` (supp)
- Task당 최소 50 trials, 3 seed 평균
- 지표: success rate (%)

**공정 비교 체크리스트**:
- [ ] 모든 encoder에 동일 MLP 아키텍처 (hidden dim, depth)
- [ ] 단일 프레임 encoder는 (img_{t-1}, img_t) feature concat으로 입력 맞춤
- [ ] 동일 action 정규화, lr, epoch, batch size
- [ ] 동일 trial 수 + seed

**기대 결과**: Two-Stream v4 > VC-1 > VideoMAE > DINOv2 ≈ SigLIP

### Phase 3B: OpenVLA 통합 (축소안, 조건부 진입) ⏸️ 대기

**목표**: Phase 3 와 동일 질문 ("표현이 실제 제어에 유용한가") 을 **Llama 7B
크기의 강력한 decoder** 조건에서 재검증. Phase 3 (작은 MLP) 결과가 "decoder
용량 때문" 이라는 반박을 차단.

**핵심 논리**: 원본 OpenVLA 의 40% 를 재현하지 못해도, 모든 encoder 가 같은
불리함을 겪으므로 **상대 순위** 는 유효하다. Phase 3 + 3B 조합으로 "encoder
우위가 decoder 용량과 무관" 이라는 더 강한 주장이 가능.

**실험 구조**:
```
encoder (frozen) → projection (학습) → Llama 7B + LoRA (학습) → action tokens
```

**축소된 실험 대상** (3 encoder × 1 task suite):

| 모델 | Encoder | 역할 |
|------|---------|------|
| OpenVLA + SigLIP (frozen) | SigLIP | Home advantage reference (Llama 와 co-train 된 encoder 를 frozen) |
| OpenVLA + V-JEPA-ours | V-JEPA-ours (frozen) | 축 1 유지 (같은 데이터, 다른 방법) |
| OpenVLA + Two-Stream | Two-Stream v4 (frozen) | **메인 제안** |

**Task suite**: libero_spatial 만 (원본 OpenVLA 결과 존재)
**Trial**: task 당 50 × 3 seed

**학습 대상**:
- Projection layer (encoder → Llama embed dim 4096): 신규 학습
- Llama 7B LoRA (rank 32, OpenVLA 논문 setup): 학습
- Vision encoder: **frozen** 통일

**공정성 정책**:
- 모든 encoder 가 **같은 frozen 조건** — 원본 OpenVLA (joint fine-tune) 와 다름
- 논문에 명시: "absolute numbers are not directly comparable to the original
  OpenVLA paper; we report encoder-only relative comparison under controlled
  freeze policy"

**진입 조건 (Phase 3 결과 보고 결정)**:
- Phase 3 에서 Two-Stream 명확히 우위 → Phase 3B 는 "강화 증거" 역할
- Phase 3 에서 박빙 → Phase 3B 결과가 판정 역할, 결과 뒤집힐 수 있음 감수

**구현 작업** (상세 TODO: [src/eval_libero.py](../src/eval_libero.py) 상단):
- [ ] OpenVLA `vision_backbone` 교체 어댑터 (PrismaticVisionBackbone 인터페이스)
- [ ] Encoder → Llama embed dim projection layer
- [ ] `vla-scripts/finetune.py` 에 encoder 교체 경로 삽입
- [ ] 공식 OpenVLA 코드로 SigLIP fine-tune 재현 먼저 (검증 step)
- [ ] SigLIP frozen 으로 downgrade 실험 (frozen 정책 단독 손실 측정)

**기존 인프라**:
- OpenVLA 공식 저장소: https://github.com/openvla/openvla
- LIBERO RLDS: `~/.cache/openvla/datasets/modified_libero_rlds/` (9.6 GB)
- Docker: `docker/openvla-libero/`
- Rollout: `src/eval_libero.py` 의 `OpenVLAClient`

**예상 비용 (Phase 2-3 완료 후)**:
- Vision backbone 교체 + projection: ~1주
- 3 encoder × LoRA FT (Llama 7B, LoRA 라 감당 가능): ~1주
- Rollout 평가 (3 × libero_spatial × 3 seed): ~2-3일
- 총 ~2.5주

**축소된 이유** (기존 5 encoder 전체 → 3 encoder):
- 원래 계획: 5 encoder × OpenVLA 교체 = 과도한 비용
- 축소 후: Two-Stream (ours) + V-JEPA-ours (직접 경쟁) + SigLIP (home advantage) 만
- VC-1 / DINOv2 는 Phase 2-3 에서 이미 비교됨 → Phase 3B 에서 중복 제거

**대안 (일정 안 맞으면)**: 메인 scope 제외, **rebuttal 추가 실험** 으로 전환.
이 경우 Two-Stream 1개만 + libero_spatial 만 (2주 작업으로 축소 가능).

**의도적 제외** (scope 관리):
- Pi0, RT-2, LAPA — 인프라 불완전 또는 비공개. Future work.
- 여러 IL 알고리즘 비교 (BC-RNN, Diffusion Policy 등) — rebuttal 에서 추가 가능.
- Full fine-tuning (non-LoRA) — 비용 대비 효과 낮음.

---

## 논문 스토리 3단계

| 단계 | 주장 | 뒷받침 Phase | 지표 |
|------|------|-------------|------|
| 1 | 표현에 action 정보가 **인코딩되어 있다** | Phase 1 (EgoDex) + Phase 2 (DROID) | R², cosine similarity (open-loop) |
| 2 | 인코딩된 정보가 **실제 제어에 유용하다** | Phase 3 (LIBERO BC rollout) | success rate (closed-loop) |
| 3 | action-agnostic 사전학습이 **범용 비전 feature보다 낫다** | Phase 3 (5종 encoder 비교) + Phase 3B (SOTA 통합) | 순위 + success rate 차이 |

---

## 요약: 학습 작업 분류

### 🔥 우리가 학습해야 하는 모델 (from scratch)
1. **Two-Stream v4** on EgoDex full (50 epoch, 8 H100) — **현재 진행 중** (jobID 32712324)
2. **V-JEPA-ours** on EgoDex full (50 epoch, 8 H100) — 구현 후 학습 예정
   - 기존 VideoMAE 계획은 V-JEPA-ours로 대체됨 (V-JEPA가 더 강한 직접 경쟁)

### 📦 공개 가중치 그대로 사용 (학습 없음, frozen 사용)
3. **VC-1-Base** (Meta eai-vc, `pip install vc_models`)
4. **DINOv2-Base** (Facebook, HuggingFace)
5. **SigLIP-Base** (Google, HuggingFace)

### 🔧 Downstream에서 학습하는 것 (encoder는 모두 frozen)
- **Phase 2**: MLP probing head (5 encoder × DROID)
- **Phase 3**: MLP action decoder (5 encoder × LIBERO task suite)
- **Phase 3B**: Projection layer + Llama 7B LoRA (5 encoder × OpenVLA backbone, libero_spatial)

---

## 모델 (우리가 구현/학습하는 것)

| 모델 | 학습 방식 | 목적 |
|-----|----------|------|
| **Two-Stream v4** | Future pixel prediction (M/P 채널 분리 + CLS 교환 + MAE masking) | 제안 모델 |
| **V-JEPA-ours** | Feature prediction (x-encoder + predictor + EMA y-encoder + stop-gradient) | 직접 경쟁 baseline (같은 EgoDex 데이터) |

### 두 모델의 비교 의미

| 차원 | Two-Stream | V-JEPA-ours |
|------|-----------|-------------|
| 데이터 | EgoDex | EgoDex (동일) |
| 아키텍처 | ViT + M/P 채널 분리 + CLS 교환 | ViT + predictor + EMA teacher |
| 학습 목표 | Pixel reconstruction (MSE) | Feature prediction (L1 in embedding space) |
| 구조적 prior | **O (생물학적 M/P)** | X (generic masked modeling) |

**축 1 직접 비교**: 같은 데이터 + 같은 학습 시간 → 구조적 bias와 학습 목표가 표현 품질에 미치는 영향 분리 측정.

### Single-Stream / 과거 VideoMAE 실험

- Phase 1 ablation에서 Single-Stream (9채널 단일 ViT)과 VideoMAE (pixel reconstruction, no M/P)를 이미 검증 완료. M/P 구조의 우위 확인 → Phase 1.5 이후 제외.
- 이전 VideoMAE 학습 계획은 V-JEPA-ours로 **대체**되었음. V-JEPA가 더 강한 직접 경쟁 baseline이기 때문.

---

## 학습 설정

```
인스턴스: H100 워크스테이션 (x2, 81GB each) — 로컬 권장
epochs: 30
batch_size: 64 (single-GPU)
max_gap: 30 (~1초)
sample_decay: -1 (linear, 큰 gap 선호 — gap에 비례한 샘플링 확률)
데이터: EgoDex frames (로컬: /mnt/data/egodex_frames/)
```

**Gap 샘플링 전략 (v4에서 변경)**:
- 기존 (v1~v3): uniform (gap 1~30 동일 확률)
- v4: **linear weighting** (`sample_decay=-1`, gap에 비례)
  - gap<=5: 3.2%, gap>=20: 59.1%
  - 이유: 작은 gap에서 M/P가 너무 비슷 → cross-stream 정보 교환 압력 부족
    → 학습 데이터의 색상/배경 bias를 학습하게 됨 (DROID unseen에서 확인)
  - 큰 gap에서 프레임 차이가 커야 CLS exchange를 통한 정보 교환이 필수적

```bash
# 로컬 (H100 x2)
bash scripts/pretrain_local.sh --model two-stream        # 특정 모델만
bash scripts/pretrain_local.sh --sanity                  # Sanity test

# 8x H100 서버 (본 학습)
torchrun --nproc_per_node=8 scripts/pretrain.py \
    --model two-stream --depth 12 --num-stages 2 \
    --mask-ratio 0.3 --sample-decay -1 \
    --egodex-splits part1 part2 part3 part4 part5 \
    --epochs 30 --batch-size 64
```

### 8x H100 서버 체크리스트 (본 학습 전)

**환경 확인:**
- [ ] PyTorch 2.4+ (torch.compile 최적화)
- [ ] NCCL 버전 + GPU 간 연결 확인 (`nvidia-smi topo -m` — NVLink 여부)
- [ ] 스토리지 I/O 속도 (프레임 JPG 로딩 — NVMe 권장)

**코드 수정:**
- [ ] DataParallel → DistributedDataParallel 전환 (8 GPU 필수)
- [ ] lr scaling: `lr = base_lr * num_gpus` (batch 512 대응)
- [ ] warmup 스케줄 조정 (큰 batch에서 안정화)
- [ ] torch.compile() 적용 테스트

**데이터 전송:**
- [ ] EgoDex part1~5 프레임 전체 전송 (S3 또는 rsync)
- [ ] DataLoader: `num_workers=8~16`, `pin_memory=True`

**리소스 분배:**
| 작업 | 서버 | 예상 시간 |
|------|------|----------|
| Two-Stream v4 사전학습 | 8x H100 | 며칠 |
| VideoMAE 사전학습 | 8x H100 | 며칠 |
| Action probing (full) | 로컬 2x H100 | 수 시간 |
| LIBERO fine-tuning | 로컬 2x H100 | 수 시간 |
| LIBERO 시뮬레이터 평가 | 로컬 1 GPU | 수 시간 |

---

## 현재 상태 (2026-04-17)

### 완료된 Phase

**Phase 1** (모델 설계 + ablation): v1~v4 iterative ablation, masking/composition/gap 실험 완료. v4 확정.

**인프라**: IBS 클러스터 환경 구축, DDP 전환, EgoDex 다운로드/추출, DROID 다운로드 완료.

### Phase 1.5 진행 중

**학습 완료**:
- Two-Stream v4 (RoPE, mask 0.3/0.5): 48ep, best_model 확정
- VideoMAE-ours (mask 0.5): 50ep 완료, best_model 확정
- V-JEPA-ours: 3차 발산, negative result

**학습 중**:
- **Two-Stream v6** (APE + mask 0.5/0.5 + rotation aug, 33222151): 50ep 진행 중

**EgoDex probing 결과** → `docs/PROBING_GUIDE.md` 참고

**다음 작업**:
1. v6 학습 완료 후 최종 probing
2. DROID 프레임 추출
3. Phase 2: DROID action probing (7 encoder)
4. Phase 3: LIBERO

### Phase 1 Ablation 결과 요약 (로컬, 500 videos, 30ep)

초기 결과는 500-video 소규모 평가로 full test split에서 과대추정 확인됨 (PROBING_GUIDE.md 참조).
여기서는 ablation 간 **상대적 비교**만 유효:

- **Masking 효과**: mask30 > nomask (모든 조건에서 우세, CLS 정보 밀도 33% 개선)
- **Composition loss 불필요**: base가 pixel loss/probing 모두 우세. Composition은 학습시간 5배 + 메모리 3배 증가 대비 이점 없음
- **max_gap 60 + triangular > gap 30 linear**: eval loss 16% 개선, probing +0.054
- **patch_mean_concat > patch_mean > CLS average**: M/P 분리 보존이 유리 (full test split에서도 순위 유지)
- **P stream self-sufficiency 문제**: P가 자기 입력만으로 복원 가능 → `--mask-ratio-p`로 P masking 상향

### 확정된 v6 설정

```
depth=12, num_stages=2
mask_ratio=0.5 (M), mask_ratio_p=0.5 (P)
max_gap=60, sample_dist=triangular, sample_center=30
APE (learnable positional embedding)
rotation_aug=True (90% 동일회전 + 10% 독립회전)
```

---

## Architecture Ablation (완료)

**목적**: depth(표현력)와 CLS exchange 빈도의 효과 분리

### 설정

| Config | depth | stages | blk/stage | exchange | Params |
|--------|-------|--------|-----------|----------|--------|
| 기존 | 12 | 3 | 4 | 3 | 193M |
| A | 6 | 3 | 2 | 3 | 135M |
| B | 6 | 2 | 3 | 2 | 128M |
| C | 4 | 2 | 2 | 2 | 100M |

### 결과 (3 epoch 간이 학습, part4 probing)

| Config | blk/stage | Epoch 3 Loss | pmc gap=5 | pmc gap=10 | cls avg gap=10 |
|--------|-----------|-------------|-----------|------------|----------------|
| 기존 (30ep) | 4 | 0.0009 | 0.489 | **0.585** | 0.364 |
| A | 2 | 0.0017 | 0.212 | 0.359 | 0.243 |
| **B** | **3** | **0.0017** | **0.256** | **0.405** | **0.255** |
| C | 2 | 0.0019 | 0.175 | 0.376 | 0.138 |

### 결론

1. **B > A** (같은 depth=6): stage당 block 수가 많을수록 좋음 (3 > 2). 충분히 처리 후 교환이 효과적
2. **C도 선전** (100M으로 A 135M에 근접): depth보다 blk/stage가 중요할 수 있음
3. **최적 설정: blk/stage를 최대화** → 기존 d=12에서 s=3→s=2로 변경 (4→6 blk/stage)

### 확정 아키텍처 (v4)

```
depth=12, num_stages=2 (6 blocks/stage, CLS exchange 2회)
→ ~186M params (기존 193M 대비 약간 감소)
```

---

## Composition Consistency 실험

### 핵심 아이디어

시간 간격(gap)은 temporal 다양성이 아니라 **변화의 크기에 대한 간접적 표현**.
(t1, t2, t3) triplet으로 **변화의 합성 구조**를 검증하고, encoder가 구조화된 change embedding을 만들도록 유도.

### 동기

- 현재 encoder는 (t1, t2) 쌍으로 단일 변화만 학습
- 실제 조작은 연속적 변화의 합성: "물체를 집고(Δ12) → 옮기는(Δ23)" 과정이 전체 변화(Δ13)
- Δ12 + Δ23 ≈ Δ13 이 embedding 공간에서 성립하면, 표현이 진정으로 변화의 구조를 포착하고 있다는 증거

### 학습 구조

```
데이터: (img_t1, img_t2, img_t3) — 기존 multi-gap 확장, 3장 샘플링

encoder(t1,t2) → cls_12   (구간 1: 변화 크기 small~medium)
encoder(t2,t3) → cls_23   (구간 2: 변화 크기 small~medium)
encoder(t1,t3) → cls_13   (전체 구간: 변화 크기 large)

[Loss 1] Both-Predict-Future × 3 (주된 학습 신호)
  구간 (1,2): M/P decoder → predict img_t2
  구간 (2,3): M/P decoder → predict img_t3
  구간 (1,3): M/P decoder → predict img_t3
  = 기존 loss를 3개 쌍에 적용

[Loss 2] Composition Consistency (auxiliary)
  compositor(cls_12, cls_23) → predicted_cls_13
  L_comp = ||predicted_cls_13 - sg(cls_13)||²

Total Loss = Loss_1 + α * Loss_2   (α=0.1로 시작)
```

### Compositor 구조: Learnable Query + Cross-Attention

입력 CLS를 변형하지 않고, learnable query가 4개 CLS를 참조하여 합성 결과 생성.
Encoder의 CLS Exchange와 동일한 철학 — 정보 훼손 없이 교환.

```
Compositor:
  learnable tokens: [query_m, query_p]  (학습 가능)

  Q = [query_m, query_p]               # 2 tokens
  K, V = [cls_m_12, cls_p_12, cls_m_23, cls_p_23]  # 4 tokens

  Cross-Attention (1~2 layers) → predicted_cls_m_13, predicted_cls_p_13
```

**sg(cls_13)** target: stop-gradient. Loss 1의 pixel target이 encoder를 잡아주므로 collapse 없음.

### PE Ablation (compositor 내)

| Variant | 설명 | 해석 |
|---------|------|------|
| A (PE 없음) | K에 순서 정보 없음 (set operation) | 교환법칙이 성립하는 합성만 가능 |
| B (순서 PE) | K에 first/second PE 추가 | 비가환적 합성(회전→이동 ≠ 이동→회전) 가능 |

- A ≈ B: change embedding 자체에 공간 구조 보존 → **encoder 품질의 강력한 증거**
- B > A: 비가환적 변화에서 순서가 필요 → **순서의 역할 증명**

### 구현 변경점

1. **데이터 로더**: 2장 → 3장 샘플링 (기존 EgoDexDataset 확장)
2. **Compositor 모듈**: learnable query + cross-attention (경량, ~수 MB)
3. **Loss 항 추가**: composition consistency loss
4. **학습 스크립트**: `--composition` 플래그로 on/off

### 실험 계획

| 실험 | 설명 | 비교 |
|------|------|------|
| v4 baseline | 기존 (t1,t2) 쌍 학습 | 기준 |
| v4 + comp (PE 없음) | triplet + composition loss (variant A) | 합성 효과 |
| v4 + comp (순서 PE) | triplet + composition loss (variant B) | 순서 효과 |

**평가**: 동일 probing 프로토콜 (EgoDex part4, gap sweep, DROID cross-domain)

---

## Ablation Study (Phase 1 ��후)

| 변형 | 설명 |
|-----|------|
| M-only | P 채널 제거 |
| P-only | M 채널 제거 |
| No Exchange | CLS 교환 없이 concat만 |

---

## MAE-style Masked Auxiliary Loss (구현 예정)

Architecture 확정(d=12, s=2)에 적용. 구현 후 full training에 포함.

### 구현 방향

```
인코더 (MAE 방식):
  M: 196 patches → 독립 random mask (30~50%) → visible만 처리
  P: 196 patches → 독립 random mask (30~50%) → visible만 처리
  ※ M/P 마스킹은 완전 독립 (종속성 없음, 겹침도 자연 발생)
  ※ CLS는 마스킹 안 함 (CLS exchange 품질 유지)
      ↓ CLS exchange (visible + CLS)
      ↓ (6 blocks/stage × 2 stages)
디코더:
  visible embeddings + learnable mask_token + positional embedding → future image 복원
      ↓
Loss:
  L_total = L_future_prediction + α * L_masked_reconstruction
  ※ L_masked: masked 위치의 복원 오차만 계산
```

### 동기
- CLS average가 patch_mean의 68% 수준 (0.364 vs 0.532) → CLS 정보 밀도 부족
- 인코더가 일부만 보고도 전체를 이해해야 → 표현력 향상 (MAE 핵심 발견)
- 부수 효과: 인코더 연산 절약 (visible만 처리)

### 3단계 랜덤성 (모델 고유 특성)
1. **Spatial**: RandomCrop(224) — 매번 다른 영역
2. **Temporal**: gap=1~30 — 매번 다른 시간 간격
3. **Structural**: stream별 독립 랜덤 마스킹 — 매번 다른 정보 가림
→ 동일 데이터도 매번 다른 문제. 단, 과도한 랜덤성은 수렴 저하 → 비율 튜닝 필요.

### 마스킹 비율 가이드
| 방법 | 비율 | 이유 |
|------|------|------|
| MAE | 75% | 이미지 redundancy 높음 |
| VideoMAE | 90~95% | 비디오 시간축 중복 |
| iBOT(DINOv2) | ~50% | discriminative, teacher 신호 보존 |
| **우리 모델** | **30~50%** | 보조 loss, 메인은 future prediction |
- 실험 조건: 0% (baseline), 30%, 50%

### 주의점
- future prediction이 메인 task, masking은 보조 → α 가중치 튜닝 필요
- 랜덤성 과다 시 학습 신호 noisy → 수렴 저하 가능
- cross-stream 정보는 CLS exchange만 통과 (patch-level cross-attention 없음)

### 참고
DINOv2(iBOT)도 masked patch prediction이지만 pixel이 아닌 feature-level distillation.
우리는 pixel reconstruction target이 있으므로 MAE 방식(MSE)이 직접적.

---

## 참고 문서

- [`docs/PROBING_GUIDE.md`](PROBING_GUIDE.md) — Action probing 가이드 + 결과
- [`docs/setup/LIBERO_TEST_GUIDE.md`](setup/LIBERO_TEST_GUIDE.md) — LIBERO 평가
