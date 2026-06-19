# LIBERO Action Probing — Plan & Implementation Guide

**Created**: 2026-04-30
**Status**: Plan only. Implementation TODO — write in a separate dev-repo Claude Code session.
**Related**: Phase 2 보강 (Vault `3. Experiments § Phase 2 보강`).

> 본 문서는 **계획 + 주의사항 + pseudocode 가이드**만 담는다. 실제 구현은 별도 세션에서 사용자가 수행. 본 문서가 검증된 근거 (action space spec, 결정 rationale)를 빠짐없이 담아 implementation의 정확성을 담보한다.

---

## 1. 목적

Phase 2 DROID action probing 이 절대 R² ~0.005 로 작은 한계를 LIBERO sim 환경에서 보완. 같은 gap-matched protocol 로 5 encoder 비교 — 절대값 0.1~0.3 range 기대.

**우리 paper 의 역할**:
- DROID 와 동일 protocol → paper Tab 2 확장 (DROID 4 gaps + LIBERO 4 gaps × 5 encoders)
- Sampling-time invariance 가설의 sim 측 evidence
- Action probing 의 "절대 R² 작음" 한계를 cleaner environment 로 해소

---

## 2. Action space 사실 관계 (web 조사 완료, 2026-04-30)

### LIBERO
- **Action**: 3D EE position delta + 3D EE rotation delta (axis-angle) + 1D gripper (총 7-DoF)
- **Control freq**: 20 Hz
- **Controller**: robosuite OSC (Operational Space Control)
- HDF5 keys:
  - `data/<demo>/obs/robot0_eef_pos` — (T, 3)
  - `data/<demo>/obs/robot0_eef_quat` — (T, 4) **xyzw** order (robosuite/scipy 호환)
  - `data/<demo>/obs/robot0_gripper_qpos` — gripper joint position (참고용)
  - `data/<demo>/actions` — (T, 7) [pos_delta(3) + rotvec(3) + gripper(1)]

### DROID (참고, 우리는 기존 protocol 유지)
- Action: 6D joint velocity + 1D gripper position (공식 schema)
- Control freq: 15 Hz

---

## 3. Target 정의 — 결정사항 (사용자 통찰 + ABA 결정)

### 핵심 원칙
**Cumulative action 직접 사용 금지. Actual pose 정보로부터 역산.**

이유:
- LIBERO action 은 OSC controller 입력 → controller scaling factor 가 cumulative sum 에 적용됨 → action sum ≠ pose change
- Actual pose `eef_pos[t+k] - eef_pos[t]` 와 `quat_diff(eef_quat[t], eef_quat[t+k])` 는 robot 이 **실제로 어떻게 움직였는지** 직접 측정 — OSC scaling 영향 무관
- Sanity check 자체가 불필요해짐 (pose 변화 = ground truth)

### Target 7-DoF 정의

```
target_pos    = eef_pos[t+k] - eef_pos[t]                              # (3,) actual translation
target_rotvec = (R(eef_quat[t]).inv() * R(eef_quat[t+k])).as_rotvec() # (3,) axis-angle
target_gripper = actions[t+k-1, 6]                                     # (1,) final command
target = concat([target_pos, target_rotvec, [target_gripper]])         # (7,)
```

### 주의사항 (구현 시 반드시 지킬 것)

- [ ] **Quaternion convention**: HDF5 `eef_quat` 은 xyzw — scipy `R.from_quat` 동일 convention. wxyz 가정 금지.
- [ ] **Rotation 역산**: `q_start.inv() * q_end` 순서 — "start 에서 end 로 가는 rotation" 의미. 반대로 하면 부호 반전.
- [ ] **Axis-angle wrap-around**: `as_rotvec()` 출력은 [-π, π] 범위. 매우 큰 rotation (gap=40 이상) 에서 wrap 가능 — gap=40 의 cumulative rotation 정상 범위인지 distribution sanity 권장.
- [ ] **Gripper 는 누적 X**: cumulative sum 의미 없음. **final state** 만 사용 (`actions[t+k-1, 6]` 또는 `gripper_qpos[t+k-1]` — 둘 중 협상 결정 필요. action 권장).
- [ ] **Translation 단위**: meters (LIBERO robosuite 표준). DROID 와 시간 척도 매칭 시에도 magnitude 자체는 dataset 마다 다름 — R² 는 scale-invariant 라 OK 단 절대값 cross-dataset 비교 회피.
- [ ] **Time-matched gaps**: LIBERO 20Hz → gap {1, 13, 20, 40} 에 해당 = {0.05, 0.65, 1.0, 2.0}초. DROID 15Hz {1, 10, 15, 30} = {0.07, 0.67, 1.0, 2.0}초. **gap=20 (LIBERO) ≈ gap=15 (DROID)** = EgoDex 1초 학습 분포 일치 핵심 gap.
- [ ] **Demo length filtering**: trajectory length ≤ max(gap)+1 인 demo 는 건너뜀. 길이 분포 P95 이상 outlier 제외 권장 (value_alignment.py 패턴 동일).

---

## 4. Implementation TODO (dev session 에서 작성)

### Step 1: HDF5 sanity (5분)
- [ ] LIBERO HDF5 path 확인 (cluster `/proj/external_group/mrg/datasets/libero_*` 또는 local)
- [ ] 첫 demo 1개 로드 → schema 확인 (`obs/robot0_eef_pos`, `obs/robot0_eef_quat` 존재 여부, shape, quaternion norm ≈ 1)
- [ ] target distribution sanity (gap=20 에서 mean/std of target_pos / rotvec / gripper magnitudes) — 비정상값 (NaN, inf, magnitude=0 등) 없는지

### Step 2: Probing pipeline 작성 (`scripts/eval/probe_action_libero.py` 신규)
- [ ] 기존 `scripts/eval/probe_action_droid_v11.py` 또는 `value_alignment.py` 패턴 재사용
- [ ] HDF5 frame access + adapter encoder forward pass
- [ ] Target 계산 (위 §3 정의)
- [ ] Linear probe 학습 (frozen encoder feature → 7-DoF target, MSE loss, Adam, 20 epoch)
- [ ] Train/test split (80/20 demo-level split, no leakage)
- [ ] Per-dim R² (sklearn) + aggregate R² + cosine sim
- [ ] Output 형식: `paper_artifacts/libero_action_probing/<encoder>_<suite>_gap<k>/{per_demo_r2.csv, summary.json}` (value_alignment.py 패턴)

### Step 3: Cluster sbatch (`scripts/cluster/probe_action_libero.sbatch`)
- [ ] AIP 1×1 H100, --time=01:30:00 (작은 batch inference + linear probe)
- [ ] 환경변수: ENCODER, TASK_SUITE, GAP, CHECKPOINT, V11_MODE
- [ ] sbatch 호출당 1 (encoder, suite, gap) — 5 × 3 × 4 = 60 jobs (또는 array)
- [ ] 또는 하나의 job 안에 4 gaps loop — wall-clock 절감

### Step 4: 실행 + aggregate
- [ ] 5 encoder × 3 suite × 4 gap = 60 cells
- [ ] 결과 CSV → 통합 → paper_artifacts/libero_action_probing/v11_libero_summary.csv (전체)
- [ ] 또는 per-encoder summary file

### Step 5: Hand-off to paper
- [ ] Paper Tab 2 확장: DROID 4 gaps + LIBERO 4 gaps × 5 encoders
- [ ] Vault 3. Experiments § Phase 2 보강 결과 추가
- [ ] (조건부) 별도 Tab 6 으로 분리 vs Tab 2 통합 — 결과 본 후 결정

---

## 5. 비용 추정

- HDF5 frame load + encoder forward: ~0.5-1 GPU·h (5 encoder × 1500 trajectory × ~150 frames)
- Linear probe 학습 60 cells × ~3 min: ~3 GPU·h
- 총: ~3-5 GPU·h ≈ ~10K won

Wall-clock 1-2일 (queue 포함).

---

## 6. Pseudocode 골격 (구현 가이드)

```python
# scripts/eval/probe_action_libero.py — skeleton

from scipy.spatial.transform import Rotation as R

def libero_action_target(eef_pos, eef_quat, actions, t, k):
    """Compute 7-DoF target for action probing at (start_idx t, gap k).
    
    eef_pos: (T, 3), eef_quat: (T, 4) xyzw, actions: (T, 7)
    """
    # Translation: actual pose change (NOT cumulative action sum)
    target_pos = eef_pos[t + k] - eef_pos[t]                       # (3,)
    
    # Rotation: actual quat diff → axis-angle (역산, NOT cumulative composition)
    q_start = R.from_quat(eef_quat[t])
    q_end = R.from_quat(eef_quat[t + k])
    target_rotvec = (q_start.inv() * q_end).as_rotvec()            # (3,)
    
    # Gripper: final action command (NOT cumulative sum)
    target_gripper = actions[t + k - 1, 6:7]                       # (1,)
    
    return np.concatenate([target_pos, target_rotvec, target_gripper])  # (7,)


def run_probing(encoder, hdf5_files, gap):
    """Linear probe for one (encoder, suite, gap) cell."""
    X, Y = [], []
    for hdf5 in hdf5_files:
        for demo_key in list_demos(hdf5):
            eef_pos, eef_quat, actions, frames = load_demo(hdf5, demo_key)
            T = len(actions)
            if T <= gap + 1:
                continue
            for t in range(0, T - gap):
                # Encoder feature
                feat = encoder.forward_pair(frames[t], frames[t + gap])  # (D,)
                target = libero_action_target(eef_pos, eef_quat, actions, t, gap)
                X.append(feat); Y.append(target)
    
    X = np.stack(X); Y = np.stack(Y)
    
    # Train/test split (demo-level, no leakage)
    # Linear probe (MSE loss, Adam, 20 epoch)
    # Output: per-dim R² + aggregate R²
    ...


def main():
    # argparse: --encoder, --suite (libero_spatial / object / goal), --gap (1, 13, 20, 40), --checkpoint
    # Output: paper_artifacts/libero_action_probing/<encoder>_<suite>_gap<k>/{per_demo_r2.csv, summary.json}
    ...
```

---

## 7. 결과 보고 형식

`paper_artifacts/libero_action_probing/<encoder>_<suite>_gap<k>/summary.json`:

```json
{
  "encoder": "two-stream-v11",
  "suite": "libero_spatial",
  "gap": 20,
  "n_demos": 500,
  "n_pairs": 75000,
  "r2_per_dim": [0.X, 0.X, 0.X, 0.X, 0.X, 0.X, 0.X],
  "r2_aggregate": 0.X,
  "cosine_sim_mean": 0.X,
  "cosine_sim_p50": 0.X,
  "target_magnitude": {"pos_mean": ..., "rotvec_mean": ..., "gripper_mean": ...},
  "elapsed_sec": ...
}
```

---

## 8. 주의: paper hand-off 전에 검증할 것

- [ ] 첫 (encoder, suite, gap=20) 결과를 보고 절대 R² 가 의미 있는 range 인지 (0.05+ 기대)
- [ ] 모든 5 encoder 가 같은 protocol 로 동일 demo split 에서 학습/평가했는지 (encoder 간 fairness)
- [ ] gap 별 R² magnitude 가 자연스러운지 (gap 1 < gap 20 < gap 40 — translation magnitude 자연 증가)
- [ ] Outlier 감지: 어떤 cell 의 R² 가 절대값 0.9+ 또는 음의 큰 값이면 이상 — debug
- [ ] 결과 도착 후 Vault `3. Experiments § Phase 2 보강` 에 정량 표 갱신

---

## 9. Cross-references

- Vault 결정 source: `Projects/Action-Agnostic Paper/3. Experiments.md § Phase 2 보강`
- Paper 측 ref: paper repo `notes/paper_writing_plan.md`
- 관련 기존 script: `scripts/eval/probe_action_droid.py`, `scripts/eval/probe_action_droid_v11.py`, `scripts/eval/value_alignment.py`
- DROID protocol (기존): `docs/PROBING_GUIDE.md § DROID Probing`
