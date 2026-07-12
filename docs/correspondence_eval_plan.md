# Correspondence Label-Propagation Eval — 계획·참조 (2026-07-12)

> **목적**: general-motion 벤치 리뷰어 리스크(Vault `README.md` §열린 우려) 대응 **옵션 A** 실행 계획.
> readout-free **correspondence(경로 2)** — SSv2 linear-probe(경로 1)·optical flow(경로 3) 대신 채택.
> **상태**: 계획, 코드 미착수. 이 문서 = 참조·guard. **구현·잡 제출은 dev/클러스터 세션.**
> 결정 정본: Vault `Projects/Action-Agnostic Visual Representation (AAAI)/README.md` §열린 우려 (옵션 A/B/C).

## 0. 한 줄 요약 / 왜 이 방법

- **문제**: 리뷰어가 "motion–appearance factorization"을 *일반* 주장으로 읽고 로봇 밖 motion 벤치를 요구할 수 있음. 메커니즘이 general이라 category error가 아님 → 방패(2-frame≠16-frame)는 SSv2/K400은 막지만 **태생적 2-frame 태스크(correspondence/flow)는 못 막음**.
- **답**: 로봇 밖 **correspondence** 벤치(DAVIS/JHMDB) label-propagation **1개** = 저렴한 external validity.
- **왜 경로 2 > 경로 1(SSv2 linear-probe)**:
  - 🟢 **readout-free** (학습 head 0) → concat-probe artifact **면역**. 본 프로젝트가 2회 철회한 그 실패모드를 원천 소거.
  - 🟢 **SiamMAE 홈그라운드** → 가장 가까운 아키텍처(2-frame cross-attn predictive)와 같은 땅에서 직접 비교.
  - 🟢 데이터 작음·라이선스 자유 (SSv2 ~220GB·Qualcomm 폼 대비).

## 1. 방법 — label propagation (학습 없이 유사도 복사)

- **semi-supervised 규약**: 비디오 **첫 프레임 라벨**(mask/keypoint)이 *주어짐* → frozen feature 유사도만으로 후속 프레임에 전파. 라벨을 *예측 학습*하는 게 아니라 *복사*.
- 파이프라인 (파라미터 fit 0):
  ```
  freeze encoder
  for each frame t:  F_t = patch feature grid (H', W', D)   # pooling 전 dense
  A_t = softmax( cosine(F_t, F_context) / τ )               # context = {frame1 anchor + 최근 K프레임}
                                                            # top-k 제한 + 공간 이웃(radius) 제한
  L_t = A_t @ L_context                                     # ★ 라벨 가중 복사 (학습 가중치 W 없음)
  metric = compare(전파 L_t, GT L_t)
  ```
- **포팅 출처**: DINO `eval_video_segmentation.py` / Jabri et al. *Space-Time Correspondence as a Contrastive Random Walk* (NeurIPS 2020). SiamMAE eval 동일 계열.

## 2. 데이터 (작음, 라이선스 자유)

| 셋 | 지표 | 규모 | 다운로드 | 용도 |
|---|---|---|---|---|
| **JHMDB** | PCK@0.1/0.2 (pose) | 928클립·31,838프레임·320×240 | ~1–2 GB (RGB+GT) | **우선** — keypoint 희소 → coarse patch 관대 |
| **DAVIS 2017** | J&F (mask) | val 30 seq·480p | ~0.8–1 GB (img+mask) | 선택 — dense mask → 해상도 민감 |

- 출처: `davischallenge.org/davis2017/code.html` (DAVIS-2017-trainval-480p.zip) · JHMDB = MOC/act-detector google drive (mmaction2 `tools/data/jhmdb` prep 참조; Frames + joint_positions + split).
- 클러스터 배치 경로(제안): `/proj/external_group/mrg/datasets/{davis2017,jhmdb}/`. split1 test(JHMDB)·val(DAVIS)만 있으면 됨.
- ⚠️ `/mnt/data/ssv2/` 존재하나 그건 **경로 1(미채택)** — 혼동 금지.

## 3. 재사용 자산 (dev repo — 신규 배선 최소화)

**frozen encoder patch-token 추출이 전 encoder에 이미 배선됨** (`scripts/eval/probe_action_libero.py` / `probe_action.py`):

| encoder | 로더 | patch 토큰 추출 |
|---|---|---|
| CoMP(parvo) | `build_parvo_encoder` | `_encode_p_unmasked(frame)[:,1:]` = P / `_encode_m_unmasked(ΔL)[:,1:]` = M |
| VideoMAE-vla | `build_videomae_token_encoder` | `encode_pairs_videomae_vla` |
| DINOv2/SigLIP/VC-1 | `build_standard_encoder` (cortexbench adapters) | token 경로 |

- `preprocess_frames(frames, img_size)` = 224 unified 정규화 재사용.
- ⚠️ 위 경로들은 **pooling된 readout(mean/attentive)** 반환이 주 용도 → correspondence는 **pooling 전 dense grid** 필요. **token 추출부만 재사용, `AttentivePoolProbe` 미사용.** (CLS 제거 `[:,1:]`는 이미 처리됨.)

## 4. 신규 구현 (코드 — dev 세션에서)

1. **데이터 로더**: JHMDB(frames + `joint_positions` + split1) / DAVIS(480p JPEG + first-frame PNG mask).
2. **`scripts/eval/correspondence_labelprop.py`** (신규): frozen feature grid → affinity(top-k·공간 radius·context queue) → 전파 → 지표.
3. **지표**: JHMDB PCK@[0.1,0.2] · DAVIS J&F (`davis2017-evaluation` 패키지 재사용 가능).
4. **encoder→dense grid 어댑터**: `(n_patch, D)` → `(H', W', D)` reshape + per-encoder patch grid 기록.
5. **sbatch**: `scripts/cluster/correspondence_labelprop.sbatch` (1×1 H100; JHMDB는 작아 CPU도 가능).

## 5. 🔴 핵심 guards (참고 필수)

- 🔴 **per-frame 인코딩 규약 (2-frame encoder 주의 — 가장 중요)**:
  - **P-stream(CoMP)·DINOv2·SigLIP·VC-1**: 프레임별 **단독** 인코딩. CoMP-P = `_encode_p_unmasked(frame_t)`.
  - **M-stream(CoMP)**: `ΔL(t-1,t)` 필요 → 단일프레임 불가. = **"frame t의 motion feature"** = motion correspondence(구별되는·on-thesis한 setup). **별도 track으로 보고.**
  - **VideoMAE-vla**: 기존 self-pair `cat([p,p])` 관습 유지 or adjacent pair — parity 위해 **한 규약 고정**.
  - ∴ **CoMP는 P-stream / M-stream 각각 correspondence 측정.** factorization 이야기 = "M(motion)이 appearance baseline보다 correspondence↑" 기대. (M이 약하면 그 자체가 정직한 신호 — screen-then-report.)
- 🔴 **해상도**: ViT-S/16 @224 = 14×14=196 패치 → dense J&F엔 거칢.
  - 완화: **입력 해상도 up**(448 → 28×28) via pos-embed 보간 — **재학습 0**. DINO도 고해상도 eval.
  - **JHMDB/PCK(희소)부터** → 해상도 리스크 최소. DAVIS는 해상도 확보 후 선택.
- 🔴 **parity (`eval_protocols.md` §0 규율)**: 전 encoder **동일** 입력 해상도·patch grid·context 길이 K·top-k·τ·layer(마지막 block token). 하나라도 다르면 비교 무효.
- 🔴 **artifact 규율**: **학습 head 절대 추가 금지** — 이게 경로 2 채택의 이유. 순수 유사도만.
- 🟠 **screen-then-report gate (사전 등록 — 실행 전 X 고정)**: CoMP(P or M best) vs appearance baseline(DINOv2/SigLIP) 마진 **≥ X** (JHMDB PCK@0.1 기준) → 보고. 미달 → **서랍**(negative를 `cluster_sessions.md`에 기록), 글쓰기 옵션 B+C로 회귀.
  - **X = +0.05 로 고정 (2026-07-12, 결과 확인 전 등록)**. 근거: frame 수 ~10k지만 클립 내 상관으로 유효 n 축소 → 클립 레벨 노이즈 ~1–2%p 추정, 5%p = 노이즈 대비 명확 + 문헌에서 방법 간 유의미 격차로 통용되는 크기. 비교 대상 = max(DINOv2, SigLIP) vs max(CoMP-P, CoMP-M) (CoMP ckpt = **S**, 논문 headline 모델).
- 🟠 **절대 수치 ≠ SiamMAE 공개값**: matched-budget EgoDex pretrain·coarse patch → 절대 J&F/PCK 낮음. **상대 비교(우리 stream 간·vs baseline)만** 주장. SOTA correspondence 주장 금지.
- 🟢 **비용**: frozen forward + 유사도, 학습 0 → 저렴. 데이터 작음. 주 비용 = **harness 구현·디버깅(~1–2일 eng)**.

## 6. 판정·논문 반영 (attach-only)

### ✅ 실행·판정 완료 (2026-07-12) — **gate FAIL → 서랍**

JHMDB split1-test 268클립·8,858프레임, 전 encoder 동일 조건(14×14 grid·τ0.1·topk5·r3·n_last7, parity ✓), 잡 `36828465~469` (V100, ~0.2 GPU·h):

| Encoder | PCK@0.1 | PCK@0.2 |
|---|---:|---:|
| DINOv2 (86M, internet) | **0.363** | 0.643 |
| VideoMAE-ours (86M, EgoDex full) | 0.350 | 0.604 |
| **CoMP-S P-stream** | 0.312 | 0.539 |
| SigLIP (86M, internet) | 0.299 | 0.538 |
| **CoMP-S M-stream** | 0.294 | 0.561 |

- **판정**: max(CoMP) 0.312 − max(appearance baseline) 0.363 = **−0.052 < X(+0.05)** → 사전 등록대로 **미보고**(서랍), 글쓰기 옵션 B+C 회귀. 상세·해석 = cluster_sessions 2026-07-12.
- 정직 신호: "M이 correspondence↑" 기대 불성립(M 최약체). 추정 원인 = ΔL 기반 motion feature는 저속 구간(ΔL≈0)서 매칭용 변별력 부재. CoMP-P > SigLIP, same-data VideoMAE도 DINOv2 하회 = §5 예고된 데이터 비대칭과 일관.
- 구현 자산은 유지: `scripts/eval/correspondence_labelprop.py` + sbatch (DAVIS 확장·후속 모델 재측정 시 재사용).

- **성공** (gate PASS): §4.x general-motion external-validity 참조 문단 1개 + 표 1개. **attach-only**(spine 불변, `fulldata_scaling_plan.md` 선례).
- **실패/모호**: 서랍. limitations "general motion 미검증"(옵션 C) 유지 + framing 강화(옵션 B). negative 기록.
- **무게중심 불변**: 이건 external validity 보강이지 논문 헤드라인 아님. 7/28 전 spine 재편 금지.

## 7. 경로 1 재개 — SSv2 2-frame linear probe (2026-07-12 설계, 실행 = 로컬 세션)

> 경로 2(correspondence) gate FAIL 후 사용자 결정으로 경로 1 시도. **미채택 사유였던 artifact 위험을 완화 설계로 수용**: readout = **mean-pool + linear 한정** (attentive/concat token probe 금지 — 2회 철회 실패모드).
> 🔄 **실행 위치 = 클러스터로 변경 (2026-07-12)**: "~220GB" 추정은 오류 — SSv2 공식 배포는 **~19.4GB**(220k webm, 12fps 저해상도). HF 미러(`morpheushoc/something-something-v2`, 공식 20-part 레이아웃+어노테이션)에서 `/proj/external_group/mrg/datasets/ssv2/`로 다운로드. 라이선스: Qualcomm 연구 라이선스 — 로컬 `/mnt/data/ssv2/` 정식 보유분과 동일 데이터의 미러.

**설계 (matched 2-frame 규약)**:
- Task: SSv2 174-class 분류, frozen encoder + linear probe (mean-pool).
- 입력: 클립당 (t, t+gap) 2-frame pair — gap ≈ 1s(EgoDex 학습 분포 정합, SSv2 12fps → gap≈12). eval은 클립당 N pair(예: 3) logit 평균.
- Encoder(전부 기존 배선 재사용): CoMP-S {`p_t_m`, `p_t_p_tk`} / DINOv2 / SigLIP / VideoMAE-vla. 전 encoder 동일 pair·해상도·probe 설정 (parity §0).
- **방향성 control (on-thesis, artifact-내성)**: SSv2는 방향 민감(pushing left/right 등) → eval 시 pair 순서 반전 acc 하락폭 = motion 정보 사용의 직접 증거. `p_t_m`에서 하락 크고 `p_t_p_tk`에서 작으면 M이 방향 정보를 실제 운반.
- 비용: feature 추출 train 169k + val 25k 클립 × N pair (로컬 H100×2, 수 시간) + linear probe (분 단위).

**🟠 gate 사전 등록 (결과 확인 전 고정, 2026-07-12)**:
1. **내부 대조(주)**: top-1 acc(`p_t_m`) − acc(`p_t_p_tk`) ≥ **+2%p** — M-stream의 실질 기여.
2. **외부 대조(부)**: max(CoMP) ≥ max(DINOv2, SigLIP) − 2%p — appearance 대비 동급 이상.

### ✅ 실행·판정 완료 (2026-07-12) — **① PASS · ② FAIL → 서랍**

val 24,777클립(클립 레벨 top-1, pair 3개 logit 평균), 잡 `36828510~514` (~3.8 GPU·h):

| Encoder | top-1 | top-5 | Δdir (반전 하락) |
|---|---:|---:|---:|
| DINOv2 | **22.8%** | 49.2% | +4.4%p |
| SigLIP | 21.8% | 47.9% | +4.2%p |
| **CoMP-S `p_t_m`** | 5.7% | 18.5% | +0.3%p |
| VideoMAE-ours | 5.0% | 16.3% | +0.8%p |
| **CoMP-S `p_t_p_tk`** | 3.0% | 10.9% | +0.3%p |

- **① +2.77%p PASS** — M-stream이 motion-sensitive 분류에 실질 기여(상대 +94%). on-thesis 내부 신호는 성립.
- **② 대폭 미달** (5.7 ≪ 20.8) → 사전 등록대로 **미보고(서랍)**. correspondence와 동일 패턴의 극단형: EgoDex 학습 모델 전부(VideoMAE 포함)가 인터넷-scale semantic appearance에 대패 — mean-pool linear 분류는 object/scene semantic이 지배.
- **방향성 control 역설(정직 신호)**: Δdir가 DINOv2(+4.4%p) > CoMP(+0.3%p) — "M이 방향 정보 운반" 기대 불성립. probe가 CoMP feature의 시간 방향성을 거의 사용하지 않음.
- 종합: 경로 1·2 모두 서랍 → **글쓰기 옵션 B+C 확정** (limitations "general motion 미검증" 유지 + framing 강화: CoMP 표현은 general-vision이 아닌 action-relevant 특화). ① 신호와 dissociation 관찰(§6 B-full correspondence 포함)은 framing 논거로만 활용.

### 경로 1-b: Something-Else compositional split (2026-07-12 사전 등록)

> 사용자 문제 제기: 표준 split은 train/val이 물체를 공유 → appearance shortcut이 점수를 지배, "물체-불변 행위 semantic" 측정 도구로 부적합. **Something-Else** (Materzynska et al., CVPR 2020) compositional split = train 물체 186종 ∩ val 물체 185종 **교집합 0** (검증 완료) → "사과로 배우고 비사과로 시험"이 정확히 구현됨. train 54,919/val 57,876, 같은 174 템플릿. 어노테이션 = `datasets/ssv2/splits_something_else/compositional/`.

**🟠 사전 등록 (2026-07-12, 결과 확인 전 고정)** — 프로토콜: 표준 런과 동일(2-frame gap12·mean readout·linear·fp16), split만 교체. 5 encoder 동일 매트릭스:

1. **보고 후보 조건**: ① 내부 대조 acc(`p_t_m`)−acc(`p_t_p_tk`) ≥ +2%p 유지 **AND** ② 외부 격차 max(DINOv2,SigLIP)−max(CoMP)가 표준 split(17.0%p) 대비 **≥5%p 축소**. 둘 다 → M의 물체-불변성 증거로 보고 검토(새 gate로 옵션 A 부활 검토). 미달 → 관찰 기록.
2. 부가 관찰(판정 아님): DINOv2/SigLIP 절대치 하락폭 = appearance shortcut 크기의 추정치.

### ✅ 판정 (2026-07-12, 잡 36829354~360) — **① PASS(+2.46%p) · ② 미달(축소 3.3%p < 5%p) → 관찰 기록**

top-1 (mean·fp16; ⚠️ comp는 train 54.9k(표준 1/3)·val 57.9k로 절대치의 split 간 비교는 참고용, within-split 대조가 정본):

| Encoder | 표준 split | compositional | Δ |
|---|---:|---:|---:|
| DINOv2 | 22.76 | 20.59 | **−2.17** |
| SigLIP | 21.79 | 18.85 | **−2.94** |
| VideoMAE-ours | 4.98 | 7.70 | **+2.72** |
| CoMP-S `p_t_m` | 5.72 | 6.85 | +1.13 |
| CoMP-S **`m_only`** | 5.31 | 5.42 | +0.11 |
| CoMP-S `p_t_p_tk` | 2.95 | 4.39 | +1.44 |
| 내부 대조 | +2.77%p | +2.46%p | 유지 ✓ |
| 격차(vs DINOv2) | 17.04%p | 13.74%p | −3.3%p (②미달) |

- **방향은 가설대로**: 물체-disjoint에서 appearance 모델만 하락(−2.2/−2.9%p), EgoDex motion 모델들은 train 1/3에도 불구 **상승** — appearance shortcut 실재 확인. 특히 VideoMAE가 +2.72로 최대 수혜(comp에서 CoMP 추월 7.70>6.85). 단 축소 폭이 기준 미달 → DINOv2 우위의 대부분은 shortcut이 아닌 전이 가능한 semantic.
- **P-방해 가설 (m_only, 36829354/360): 기각**. 양 split 모두 m_only ≤ p_t_m (역전 없음 — B-part1식 오염의 SSv2 재현 없음). 단 **표준에서 P의 순기여는 +0.41%p에 불과** (m_only 5.31이 절반 차원으로 p_t_m의 93%) = CoMP의 SSv2 점수는 사실상 M이 전담. comp에서는 P 기여 +1.43%p로 커짐 — P의 전이 성분(손·공간 배치)은 물체-disjoint에서도 유효.
- Δdir는 전 조건 CoMP ≈ +0.3~0.8%p 불변.
- **종합**: 경로 1-b도 보고 기준 미달 → 서랍. 단 "appearance shortcut 실재 + M 전담 + motion 모델의 comp 상승" 3관찰은 framing(옵션 B) 논거로 축적.

### 후속 관찰 (2026-07-12, 전부 관찰 전용 — gate 불변)

1. **readout-병목 반증** (meanmax 매트릭스 `36828832~836`): "mean-pool이 국소 M 신호를 희석해 CoMP에 불리"라는 가설을 mean⊕max(파라미터 0, 전 encoder 동일)로 검정 → 이득이 전 encoder +0.6~1.0%p 균등, 외부 격차 불변(17.0→16.8%p), 내부 대조 +2.77→+3.06%p. **격차는 readout이 아니라 표현(semantic 부재) 문제로 확정** — FAIL 결론 견고. (meanmax 런 = fp16 autocast·rev 토큰 재사용, 2.7× 단축.)
2. **B-full scale 효과** (`36828700/701`, mean): 내부 대조 +2.77(S)→**+4.13%p**(B-full) — scale이 P⊕P(+0.89%p)보다 M 기여를 크게 키움. M-recon scale 단조성(action probing R² 0.401)과 정합, JHMDB(제자리)와 대조 = **"scale이 키우는 건 motion-분류 유용 정보이지 매칭용 appearance 지문이 아님"** dissociation 3번째 데이터셋 일관.
3. Δdir(방향)은 모든 조건에서 CoMP ≈ 0(+0.25~0.5%p) vs DINOv2 +4.4~4.7%p — 방향 불사용 결론 유지.
- 둘 다 충족 → 보고(attach-only, §6 규율 동일). 미달 → 서랍 + negative 기록. 절대 SOTA 주장 금지(2-frame 프로토콜은 상대 비교 전용).

## 8. Cross-refs

- Vault: `Projects/Action-Agnostic Visual Representation (AAAI)/README.md` §열린 우려 (옵션 A/B/C) · `2. Experiments.md` §4 ②correspondence label-propagation.
- dev: `siammae_baseline_plan.md` (SiamMAE=correspondence 홈) · `eval_protocols.md` §0 parity · `factorization_crossover_plan.md` (per-stream 측정 규율) · `fulldata_scaling_plan.md` §4 (attach-only 선례).
- 참고 논문: SiamMAE (arXiv 2305.14344) · DINO video segmentation eval · Jabri et al. Contrastive Random Walk (NeurIPS 2020).
