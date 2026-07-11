# Correspondence Label-Propagation Eval — 계획·참조 (2026-07-12)

> **목적**: general-motion 벤치 리뷰어 리스크(Vault `README.md` §열린 우려) 대응 **옵션 A** 실행 계획.
> readout-free **correspondence(경로 2)** — SSv2 linear-probe(경로 1)·optical flow(경로 3) 대신 채택.
> **상태**: 계획, 코드 미착수. 이 문서 = 참조·guard. **구현·잡 제출은 dev/클러스터 세션.**
> 결정 정본: Vault `Projects/Action-Agnostic Paper/README.md` §열린 우려 (옵션 A/B/C).

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

- **성공** (gate PASS): §4.x general-motion external-validity 참조 문단 1개 + 표 1개. **attach-only**(spine 불변, `fulldata_scaling_plan.md` 선례).
- **실패/모호**: 서랍. limitations "general motion 미검증"(옵션 C) 유지 + framing 강화(옵션 B). negative 기록.
- **무게중심 불변**: 이건 external validity 보강이지 논문 헤드라인 아님. 7/28 전 spine 재편 금지.

## 7. Cross-refs

- Vault: `Projects/Action-Agnostic Paper/README.md` §열린 우려 (옵션 A/B/C) · `2. Experiments.md` §4 ②correspondence label-propagation.
- dev: `siammae_baseline_plan.md` (SiamMAE=correspondence 홈) · `eval_protocols.md` §0 parity · `factorization_crossover_plan.md` (per-stream 측정 규율) · `fulldata_scaling_plan.md` §4 (attach-only 선례).
- 참고 논문: SiamMAE (arXiv 2305.14344) · DINO video segmentation eval · Jabri et al. Contrastive Random Walk (NeurIPS 2020).
