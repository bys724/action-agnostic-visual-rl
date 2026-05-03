# Phase 3-1 BC-T 2차 ckpt 로컬 sanity 디버그 (2026-05-03)

이 문서는 클러스터에서 학습한 2차 BC-T ckpt (use_joint=True fix 적용)를 로컬
H100 워크스테이션에서 LIBERO rollout sanity로 검증하면서 발견한 추가 결함과
fix, 그리고 ours 인코더 (videomae-ours, two-stream-v11) 0% SR의 진단 결과를
정리한다. 로컬은 디버깅에 집중하고, 클러스터에서 다음 학습 iteration을
진행할 때 참고.

자세한 phase context: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) Phase 3 §"3-1차"

## TL;DR

| 변경 사항 | 위치 | 영향 |
|----------|------|------|
| Adapter `prev_obs` cross-camera 오염 fix | `src/eval_libero.py` `BCTransformerClient` 재설계 | rollout pipeline 정상화 (모든 encoder 적용) |
| `LIBERO_ENV_RESOLUTION` 256→128 | `src/eval_libero.py` | HDF5 demos 동일 해상도 |
| `joint_states` shape_meta + infer obs | `src/eval_libero.py` | use_joint=True 학습 호환 |
| `two_stream_v11` adapter `checkpoint=None` 허용 | `src/encoders/adapters/two_stream_v11.py` | rollout init 가능 |
| Dummy wait 기간 obs를 history에 누적 | `src/eval_libero.py` `evaluate_libero` | motion-aware 어댑터 step-0 collapse 회피 |
| robomimic `--no-deps` | `docker/libero/Dockerfile` | egl_probe build fail 회피 (image rebuild 시) |

**Sanity SR (1 task × 5 trial, libero_spatial task 0, post-fix)**:

| Encoder | SR | BC fit ‖p−r‖ mean | Encoder feature PCA r1 (centered) |
|---------|----|---------------------|-----------------------------------|
| **vc1** | **80%** | 0.295 (worst fit) | 0.59 |
| **siglip** | **60%** | (≈ dinov2) | — |
| **dinov2** | **40%** | 0.167 | 0.54 |
| **videomae-ours** | **0%** | 0.116 (best fit) | 0.42 |
| **two-stream-v11** | **0%** | 0.139 | 0.40 (D' part) |

→ **ours encoders가 BC demo trajectory를 가장 정확히 fit하지만 rollout 0%.**
encoder representation 자체는 healthy (PCA r1 비교 baseline과 동급).
원인은 encoder collapse가 아니라 **BC overfit / covariate shift sensitivity**로
판단됨.

## 발견한 결함과 수정

### 1. Adapter `prev_obs` cache cross-camera 오염

**증상**: 모든 5 encoder가 SR 0%. 정책이 의도적으로 한 방향(주로 +y)으로만
이동하여 task 영역 이탈. 비디오 프레임에서 로봇이 위쪽으로 들려 카메라 시야를
가리는 패턴.

**원인**: `videomae`, `two_stream_v11`, `single_frame` 어댑터 모두 단일
`self.prev_obs` 슬롯 사용. BC-T policy는 동일 어댑터 인스턴스를
`agentview_rgb` + `eye_in_hand_rgb` 카메라가 **공유** (`AdaptedBCTransformerPolicy.adapter`
필드 1개). T=1 rollout 시 카메라 호출이 교차되어:
- step 0: agent 호출 (prev_obs=None, prev=curr_agent, store curr_agent) → wrist 호출 (prev_obs=curr_agent, **cross-camera contamination**)
- step 1+: agent (prev=prev_wrist, contaminated), wrist (prev=curr_agent, contaminated) ...

→ policy가 OOD 입력을 봄. 학습 시 (T=10 시퀀스)는 어댑터 `if T>1` branch가
시퀀스 내부 pair 만들어 정상.

**해결**: [`src/eval_libero.py`](../src/eval_libero.py) `BCTransformerClient` 재설계
- `latent_queue` 폐기
- raw obs history (max_seq_len=10) 누적
- 매 step `(B=1, T_acc, ...)` 시퀀스 전체로 `spatial_encode` 호출
- 어댑터 `T>1` branch 활성 → 시퀀스 내부 pair 형성 → 학습 분포 정합

### 2. Env render 해상도 불일치

**원인**: HDF5 demos `agentview_rgb`는 128×128 저장. 우리 rollout은 256×256 render.
encoder 입력 해상도는 224×224로 동일 (어댑터 내부 resize)하지만 source resolution
달라 bilinear interpolation 결과 미세 차이.

**해결**: `LIBERO_ENV_RESOLUTION` 256 → 128. HDF5와 동일.

### 3. `joint_states` modality 누락

**원인**: 1차 BC-T는 `use_joint=False`로 학습되어 joint_states 누락 → 0% SR
(이전 진단). 2차에서 학습 driver 수정 (`finetune_libero_bct.py`).

**해결**: rollout 측도 동기화 — `libero_shape_meta()`에 `joint_states: [7]` 추가,
`infer()`에서 `obs['robot0_joint_pos']` 전달.

### 4. v11 어댑터 init 시 checkpoint 필수

**원인**: `TwoStreamV11Adapter.__init__`이 `load_v11_model(checkpoint_path)`을
호출하는데 rollout 시 cluster path 무효화 (`cfg.encoder.checkpoint = None`)로
ValueError.

**해결**: `checkpoint_path is None`일 때 random init 후 외부에서
`policy.load_state_dict`로 덮어쓰는 경로 추가.

### 5. Motion-aware 어댑터 step-0 collapse

**원인**: videomae-ours / v11 같은 **motion-aware** 어댑터는 (prev, curr) pair
의 차이로 motion을 인코딩. rollout step 0에서 T_acc=1, prev=curr → motion
신호 0 → A (M encoder) PCA r1=1.000 collapse 측정됨. (real motion pair 조건에선
PCA r1≈0.40로 정상)

**해결**: `evaluate_libero` 루프에서 dummy wait 기간 마지막 max_seq_len step의
obs를 `client.observe()`로 history에 누적. 첫 inference 시 T_acc=10,
adapter `T>1` branch 활성, 진짜 motion pair 시퀀스 처리.

→ 그러나 **이것만으로는 SR 0%가 해결되지 않음**. encoder 자체는 정상 작동
하지만 ours BC policy의 generalization 자체가 약함.

## ours 0% 진단 — 인코더는 정상, BC overfit이 의심

### 인코더 representation 분석 (centered cosine + PCA)

**테스트**: HDF5 demo의 다양한 timestep 10 frame × prev/curr 인접 pair
조합으로 인코더 raw feature variance 측정.

| Encoder | centered cos (낮을수록 다양) | PCA r1 (낮을수록 정보 분산) | std/‖mean‖ |
|---------|--------------------------|------------------------|-------------|
| videomae-ours patches | 0.380 | 0.424 | 0.105 |
| two-stream-v11 A (M) | 0.348 | 0.395 | 0.285 |
| two-stream-v11 D' | 0.428 | 0.633 | 0.293 |
| dinov2 concat | 0.440 | 0.543 | 0.290 |

→ **ours encoders feature가 baseline보다 오히려 더 다양**. encoder representation
자체에 collapse / 정보 부족 없음.

### Rollout action divergence

동일 init state에서 rollout 시작 → 매 step 예측 action 출력. v11과 dinov2를
비교:

```
t=0..14의 예측 action 패턴:
  v11   : x ramps 0.08→0.93, y 0.01→0.50, z 0.10→0.10 (descent 약함)
  dinov2: x ramps 0.22→0.94, y 0.18→0.49, z 0.14→0.07 (descent 좀 더 강함)
HDF5 demo_0 recorded: x 0.06→0.84, y 0.0→0.62, z 0.08→-0.29 (실제 descent)
```

- 두 encoder 모두 z-descent가 부족 (HDF5 -0.29 vs predicted ~0.10)
- v11 z가 dinov2보다 약간 더 약함
- y가 둘 다 0.5 근처에서 saturate (HDF5는 0.62)

→ **두 encoder 모두 비슷한 action drift 패턴**을 보임. dinov2는 그래도 40% SR
달성. v11/videomae는 0% — drift가 더 누적되어 임계점에서 실패하는 것으로
추정.

### BC consistency (HDF5 obs 직접 입력)

| Encoder | t=0 \|\|p−r\|\| | t≥1 mean | t≥1 max |
|---------|------------------|----------|---------|
| videomae-ours | 0.40 | 0.116 | 0.213 |
| two-stream-v11 | 0.67 | 0.139 | 0.410 |
| dinov2 | 0.96 | 0.167 | 0.509 |
| vc1 | 0.18 | 0.295 | 0.616 |

→ ours가 demo trajectory에 가장 정확히 fit. baseline은 약간 덜 정확하지만
rollout에서 robust generalize. **classic BC overfitting 패턴.**

## 클러스터에서 이어서 할 일

`git pull` 후 **3-1차 V3 본 main table 학습**을 augmentation + multi-seed 동시
적용으로 진행. 이전 답변에서 "augmentation 단독", "multi-seed 단독" 후보로
나눴던 것은 과도하게 신중한 분리였음. 둘 다 LIBERO 공식 BC-T default /
로봇 BC paper 표준이며, 동시 적용 시 추가 비용 거의 없음. 우선순위 순서:

### Main: 3-1차 V3 학습 — augmentation + multi-seed 동시

**스코프**: 5 encoder × 3 suite × 3 seed = **45 runs** (원래 main table 계획 그대로)

**cfg 변경** (`scripts/eval/finetune_libero_bct.py`):
- `cfg.train.use_augmentation = True` (현재 `False`)
- `cfg.policy.color_aug.network = 'ColorJitterAug'` (현재 `IdentityAug`)
- `cfg.policy.translation_aug.network = 'TranslationAug'`, `affine_translate = 4`
- 모두 LIBERO 공식 BC-T default. 추가 비용 0 (학습 시간 동일, GPU bound 아님)

**🔴 2-frame pair 어댑터의 augmentation 일관성 필수 검증**:

ours 어댑터 (`videomae`, `two_stream_v11`)는 (prev, curr) 두 프레임을 함께
입력 받음. augmentation이 **prev / curr에 독립적으로** 적용되면 **가짜 motion**
이 생성되어 motion-aware encoder 학습이 망가짐. 동일 augmentation을 두 프레임에
일관되게 적용해야 함.

- 카메라 간 (agentview/wrist)는 독립 augmentation **OK** (실제 두 카메라 분리)
- 시점 간 (prev/curr 동일 카메라)는 **동일 augmentation 필수**

LIBERO 공식 `BasePolicy.preprocess_input` → `DataAugGroup` → `_get_img_tuple`은
image_tuple에 동일 augmentation을 brodcast하는 것으로 보이지만, 우리 어댑터가
`spatial_encode` 내에서 직접 받는 sequence (B, T, C, H, W)에 대해 augmentation이
T 차원에서 일관되게 적용되는지 코드 경로 확인 필요.

**검증 절차** (학습 시작 전 1회 + 학습 1 epoch 후 1회):
1. `_log_first_batch_stats`에 augmented batch sample을 시각화 저장 추가
   ```python
   # 학습 시 첫 batch에서
   from torchvision.utils import save_image
   for cam in ['agentview_rgb', 'eye_in_hand_rgb']:
       imgs = data['obs'][cam][0]  # (T, C, H, W) — 첫 batch sample, T=10
       save_image(imgs, f'/tmp/aug_check_{cam}_ep0.png', nrow=10)
   ```
2. 시각적 확인: 같은 카메라의 인접 시점 (prev=t-1, curr=t)이 **동일 augmentation**
   (같은 색상 shift, 같은 translation)을 받았는가? 다르면 augmentation 코드 수정 필요.
3. 시점 간 augmentation 독립이면 ours 학습이 망가질 가능성 高 — 반드시 fix 후
   본 학습 진행.

**Epoch별 ckpt 보존**: 현재 driver는 `best.pt`만 저장. best가 overfit ckpt일
가능성 높으므로 ep 5/10/20/best 4개 ckpt 동시 저장. 비용 0 (재학습 불필요,
V3부터 적용). 추후 어떤 epoch이 가장 robust한지 비교 가능.

**예상 비용**: 45 runs × ~25h × 1 GPU = ~1125 GPU·h. 5 GPU 병렬이면 ~9일
wall-time. 클러스터 capa로 실현 가능.

**결과 해석**:
- ours가 30%+ SR 회복 → "augmentation으로 격차 좁혀짐, paper에 양쪽 보고"
- ours가 한자리 SR 유지 → "ours encoder가 EgoDex 도메인 한계로 LIBERO transfer
  어려움" — 정당한 negative finding

### 부가 (V3 진행 중 / 결과 보고 결정)

- **encoder partial unfreeze**: V3로도 ours 약하면 마지막 4 transformer block만
  unfreeze BC fine-tune. EgoDex→LIBERO 도메인 적응. ours만 적용 (baseline은
  frozen 유지하여 fair 비교)
- **v11-VfromM ablation (33615395)**: 진행 중. 학습 종료 시 동일 V3 cfg
  (augmentation + multi-seed)로 BC-T 학습 → main table A1 ablation
- **V-JEPA**: main table 제외 결정 유지. paper에서 별도 footnote로 처리

### Ablation (paper section 5용, main table 확보 후)

- with/without augmentation (1 encoder × 1 suite × 1 seed = 6 runs)
- with/without joint_states (1차/2차 결과로 이미 가지고 있음, 추가 학습 불필요)
- with/without encoder partial unfreeze (조건부)

## 로컬에서 보존된 작업물 (git pull로 동기화)

| 파일 | 변경 |
|------|------|
| [`src/eval_libero.py`](../src/eval_libero.py) | `BCTransformerClient` 재설계 + `joint_states` + 128 res + `observe()` |
| [`src/encoders/adapters/two_stream_v11.py`](../src/encoders/adapters/two_stream_v11.py) | `checkpoint=None` 허용 |
| [`docker/libero/Dockerfile`](../docker/libero/Dockerfile) | robomimic `--no-deps` |
| [`docker-compose.yml`](../docker-compose.yml) | libero service `count: 1` → `count: all` (병렬 평가용) |
| [`docs/setup/LIBERO_TEST_GUIDE.md`](setup/LIBERO_TEST_GUIDE.md) | 트러블슈팅 항목 추가 |
| [`docs/RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | Phase 3-1에 2차 sanity 결과 + 클러스터 액션 |
| [`docs/artifacts.md`](artifacts.md) | 2차 ckpt 로컬 위치 |
| 본 문서 | 디버그 자세한 결과 |

## 보존된 sanity 산출물 (참고용)

- `data/libero/results/_sanity_2026-05-03_pre_baseline/` — 2차 ckpt sanity rollout 8개 JSON
- `data/libero/results/_archive_pre_usejoint/` — 1차 broken ckpt 0/50 결과 3개
- `data/libero/videos/_orphan_pre_2026-05-03/` — 사전 비디오 일괄 stash
- `/mnt/data/checkpoints/libero_bct/_archive_pre_usejoint/` — 1차 broken ckpt 1개

## 비용 요약

오늘 사용한 로컬 GPU 시간 (sanity + 디버그):
- 1 task × 5 trial × 5 encoder × 여러 iteration ≈ 1 GPU·h
- encoder representation 분석 ≈ 0.5 GPU·h

본 평가 (5 encoder × 10 task × 50 trial × 3 suite × 3 seed) 후속은 클러스터에서
처리 권장. 로컬 H100은 인터랙티브 디버그 / 시각화 / 빠른 iteration용으로 유지.
