# LIBERO Single-Frame Task Classification — Plan & Implementation Guide

**Created**: 2026-04-30
**Status**: Plan only. Implementation TODO — write in a separate dev-repo Claude Code session.
**Related**: Vault `3. Experiments § Phase 2 보강` § "추가 실험: P Stream Evaluation Gap 보완".

> 본 문서는 **계획 + 주의사항 + pseudocode 가이드**만 담는다. 실제 구현은 별도 세션에서 사용자가 수행. Phase 2 보강 LIBERO action probing 결과의 한계 (P stream specialization positive evidence 부재) 를 보완하기 위한 추가 evaluation.

---

## 1. 목적 (P stream evaluation gap 보완)

### 현재 evidence 한계
Phase 2 보강 LIBERO action probing 으로 M stream 강점 (controlled 12/12 + gap=1 dominance) 은 입증됐지만:
- P encoder (mode B) R² = -0.003 (action probing) — **action 정보 부재** = "appearance encoder 라서 그렇다"는 우리 의도의 *negative evidence*
- 그러나 "P 가 appearance 를 잘 capture 한다"는 **positive evidence 부재**

### Reviewer 의 자연스러운 질문
- "M/P functional differentiation 을 강조하지만, P stream 이 어떤 task 에서 우월하다는 evidence 는?"
- "P 는 그냥 'inactive computational substrate' 인가?"
- "Single M stream + auxiliary P 와 차이가 무엇?"

### 본 실험의 답
- **Single-frame task classification** — single frame 이면 motion 정보 부재 → P stream 우위 자연 영역
- 가설: B (P encoder) > A (M encoder) on single-frame task → **functional differentiation 양방향 입증**
- 가설 2 (stretch): B vs DINOv2 격차 — internet-scale image SSL 과 경쟁력?

---

## 2. Protocol

### Task
- LIBERO 의 task suite 별 10-way classification
  - libero_spatial: 10 tasks
  - libero_object: 10 tasks
  - libero_goal: 10 tasks
- (Optional) libero_10 도 있으면 추가

### Frame sampling
- **Single frame per demo** (e.g., middle frame: index = T // 2)
  - Trajectory 의 mid-point 가 task semantic 가장 명확
  - Optional: 3 frames per demo (start / middle / end) 평균 → 약간의 robustness
- Single frame ≠ 2-frame pair — 2-frame encoder (v11 / VideoMAE-ours) 도 (frame_t, frame_t) self-pair 로 forward (motion 없음 → M stream ΔL ≈ 0 자연 발생)

### Encoder × mode ablation
- 5 encoder × 3 suite = 15 cells (baseline)
- v11 mode 분기 (within v11 row 만): A 단독 / B 단독 / D' 단독 / A+B+D' = 4 modes
- Total: (4 baseline encoders + v11 × 4 modes) × 3 suites = 8 × 3 = 24 cells

### Probe
- Frozen encoder feature → **linear classifier** (W ∈ R^{d × 10}) + softmax
- Cross-entropy loss, Adam, 20 epoch
- Train/test split: demo-level 80/20 (no leakage)
- 평가 metric: top-1 accuracy + per-class F1 (10 classes)

### 가설 검증
- v11 row 의 4 modes 비교:
  - **B (P encoder) > A (M encoder)**: functional differentiation 양방향 입증 ★
  - A+B+D' > B 단독: 정보 통합 효과 (이건 strong assumption 아님 — single frame 이라 D' 도 motion-poor)
- v11 B 단독 vs DINOv2: P 가 internet-scale image SSL 과 어느 정도?

---

## 3. 주의사항 (구현 시 반드시 지킬 것)

### 데이터 측면
- [ ] **LIBERO HDF5 task label location 확인**: HDF5 의 어디에 task label 이 있는지 확인 필요. 가능성:
  - `data/<demo>/attrs/lang_instruction` — natural language description (task class 추출 위해 추가 mapping 필요)
  - `data/<demo>/attrs/task_id` — integer task ID (가장 간단)
  - File name 자체에 task 이름 (HDF5 file 별로 task 가 다른 경우 — LIBERO 표준)
  - **Sanity: 첫 demo 한 개 열어서 확인** (sanity_check_libero_action.py 패턴 재사용)
- [ ] LIBERO 의 each suite 가 정확히 10 tasks 인지 확인 (libero_10 은 다를 수 있음)
- [ ] Demo per task 분포 — 각 class 가 비슷한 sample size 인지 (50 demos per task standard)

### Encoder 측면
- [ ] **Single frame forward**: 2-frame encoder (v11 / VideoMAE-ours) 는 (frame_t, frame_t) self-pair 사용. M stream 의 ΔL = 0 자연 발생 — 자연스러운 motion-poor input
- [ ] **v11 mode 분기**: `probe_action_v11.py` 의 cls_mode 옵션 재사용. A=`patch_mean_m_enc`, B=`patch_mean_p_enc`, D'=`patch_mean_p_state_after_routing`, A+B+D'=`patch_mean_concat_all`
- [ ] 단일 frame encoder (DINOv2/SigLIP/VC-1) 는 native single-frame protocol — 이게 정상적 use case
- [ ] **HF_HUB_OFFLINE=1** 환경변수 (BC-T 학습에서 발견된 race 회피)

### Probing 측면
- [ ] Linear probe — 단순 `torch.nn.Linear(d, 10)` + softmax. MLP 사용 X (encoder 평가 순수성)
- [ ] Cross-entropy loss (action regression 의 MSE 와 다름)
- [ ] **Demo-level train/test split** — frame-level split 하면 leakage. 80% demos train, 20% test, seed=0
- [ ] Class balance 확인 (10 tasks × ~50 demos = 500 demos per suite, balanced)
- [ ] Top-1 accuracy + per-class F1 + confusion matrix optional

### 분석 측면
- [ ] **v11 mode contrast**: A vs B 가 핵심. M (motion encoder) 이 single frame task 에서 약하고 P (appearance encoder) 가 강해야 가설 입증
- [ ] B vs DINOv2 / SigLIP 격차 — 만약 B 가 baseline 절반 이하면 P specialization 약함 evidence
- [ ] B vs VC-1 (manipulation-pretrained image encoder) — 동등하거나 우위면 EgoDex appearance 학습이 효과적

---

## 4. Implementation TODO (dev session 에서 작성)

### Step 1: HDF5 task label sanity (10분)
- [ ] LIBERO HDF5 한 개 열어서 task label 위치 확인
- [ ] `h5py.File("libero_spatial/<task_name>.hdf5") as f`
- [ ] `f.attrs`, `f["data"].attrs`, `f["data/demo_0"].attrs` 모두 print
- [ ] task label 찾으면 location/format 본 plan 문서에 기록
- [ ] 없으면 file name 으로 task label 도출 (e.g., file 별 task 1개씩)

### Step 2: Probing pipeline 작성 (`scripts/eval/task_classification_libero.py` 신규)
- [ ] 기존 `scripts/eval/probe_action_libero.py` 패턴 변형:
  - Action regression → classification (cross-entropy)
  - 7-DoF target → 10-way label
  - 2-frame input → single frame (or self-pair for 2-frame encoders)
- [ ] `--encoder` (5 options), `--suite` (3 options), `--v11-mode` (4 options for v11)
- [ ] Output: `paper_artifacts/libero_task_classification/<encoder>_<suite>_<mode>_<ts>/{predictions.csv, summary.json}`

### Step 3: Cluster sbatch (`scripts/cluster/task_classification_libero.sbatch`)
- [ ] AIP 1×1 H100, --time=00:30:00 (single-frame inference 빠름)
- [ ] 5 encoder × 3 suite = 15 jobs (baselines)
- [ ] v11 × 3 suite × 4 modes = 12 jobs (mode ablation)
- [ ] Total: 27 jobs

### Step 4: 실행 + aggregate
- [ ] 27 cells 실행 + CSV/JSON aggregate
- [ ] Comparative table: encoder × suite × accuracy (+ v11 mode breakdown)

### Step 5: 결과 분석 + paper 측 hand-off
- [ ] **A vs B (v11 mode)**: M encoder vs P encoder on single-frame task — 가설 검증
- [ ] **B vs DINOv2**: P encoder가 image SSL과 어느 정도
- [ ] 4 시나리오별 결정:
  - B > A: paper §4 새 sub-section "Functional Differentiation Two Directions" 추가 검토
  - B > DINOv2: hero result. paper §4.5 강조
  - B ≈ DINOv2: P가 internet-scale 경쟁력 paragraph 추가
  - B 약함: paper main framing 약화 + Limitations 명시 ("P as substrate" framing)
- [ ] Vault `3. Experiments § Phase 2 보강 § 추가 실험` 결과 갱신

---

## 5. 비용 추정

- HDF5 frame load + encoder forward: ~0.5 GPU·h (single frame batch inference, 27 cells × ~수백 demos)
- Linear probe 학습 27 cells × ~3 min: ~1.4 GPU·h
- 총: **~2 GPU·h ≈ 5K won**

Wall-clock 1-2일 (queue 포함).

---

## 6. Pseudocode 골격 (구현 가이드)

```python
# scripts/eval/task_classification_libero.py — skeleton

import h5py
import numpy as np
import torch

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal")
NUM_TASKS_PER_SUITE = 10  # confirm in Step 1

def get_task_label(hdf5_path, demo_key):
    """Extract task label.
    
    TODO Step 1: confirm exact location after HDF5 sanity
    """
    # Possibility A: integer task_id in attrs
    # with h5py.File(hdf5_path, "r") as f:
    #     return f[f"data/{demo_key}"].attrs["task_id"]
    
    # Possibility B: file name encodes task
    # task_name = hdf5_path.stem  # e.g., "pick_up_the_red_apple_and_place_it_in_the_pot"
    # return TASK_NAME_TO_ID[task_name]
    raise NotImplementedError("TODO: confirm HDF5 task label location")


def sample_frame_for_classification(hdf5_path, demo_key):
    """Sample single representative frame from trajectory.
    
    Strategy: middle frame (index = T // 2).
    For 2-frame encoders, return (frame_t, frame_t) self-pair.
    """
    with h5py.File(hdf5_path, "r") as f:
        frames = np.asarray(f[f"data/{demo_key}/obs/agentview_rgb"])  # (T, H, W, 3) uint8
    mid_idx = len(frames) // 2
    return frames[mid_idx]  # caller may duplicate for 2-frame encoders


def run_classification(encoder, suite, v11_mode=None):
    """Linear probe for one (encoder, suite[, v11_mode]) cell.
    
    For v11, v11_mode ∈ {a_only, b_only, d_prime_only, abd_prime}.
    For other encoders, v11_mode is None (use native single-frame protocol).
    """
    X, Y = [], []
    hdf5_files = list_hdf5_files(suite)  # 10 files per suite (one per task)
    
    for hdf5_path in hdf5_files:
        for demo_key in list_demos(hdf5_path):
            mid_frame = sample_frame_for_classification(hdf5_path, demo_key)
            
            if encoder.is_2frame:
                feat = encoder.forward_pair(mid_frame, mid_frame, mode=v11_mode)
            else:
                feat = encoder.forward_single(mid_frame)
            
            label = get_task_label(hdf5_path, demo_key)
            X.append(feat)
            Y.append(label)
    
    X = np.stack(X); Y = np.array(Y)
    
    # Demo-level 80/20 split (TODO: implement; ensure no class imbalance)
    # Linear classifier: nn.Linear(d, NUM_TASKS_PER_SUITE) + softmax
    # Train: cross-entropy, Adam, lr=1e-3, 20 epoch
    # Output: top-1 accuracy + per-class F1
    ...


def main():
    # argparse:
    #   --encoder ∈ {two-stream-v11, videomae-ours, dinov2, siglip, vc1}
    #   --suite ∈ {libero_spatial, libero_object, libero_goal}
    #   --v11-mode ∈ {a_only, b_only, d_prime_only, abd_prime} (only for v11)
    #   --checkpoint (for v11 / videomae-ours)
    # Output: paper_artifacts/libero_task_classification/<encoder>_<suite>_<mode>_<ts>/summary.json
    ...
```

---

## 7. 결과 보고 형식

`paper_artifacts/libero_task_classification/<encoder>_<suite>_<mode>_<ts>/summary.json`:

```json
{
  "encoder": "two-stream-v11",
  "v11_mode": "b_only",
  "suite": "libero_spatial",
  "n_train_demos": 400,
  "n_test_demos": 100,
  "top1_accuracy": 0.X,
  "per_class_f1": [0.X, ...],
  "macro_f1": 0.X,
  "confusion_matrix": [[...], ...],
  "elapsed_sec": ...
}
```

---

## 8. Cross-references

- Vault 결정 source: `Projects/Action-Agnostic Paper/3. Experiments § Phase 2 보강 § 추가 실험: P Stream Evaluation Gap`
- 관련 plan: `docs/libero_action_probing_plan.md` (Phase 2 보강 main)
- 기존 script 재사용: `scripts/eval/probe_action_libero.py` (action regression → classification 변형)
- 기존 v11 mode 옵션 참고: `scripts/eval/probe_action_v11.py` (cls_mode argument)
