# 평가 프로토콜 단일 출처 (Eval Protocols — Single Source of Truth)

> **목적**: 논문에서 **현재 모델(Parvo)**을 baseline과 비교하는 모든 벤치마크의 **정규 프로토콜**을 한 곳에 못박는다.
> 새 ckpt를 평가할 때 이 문서 조건과 **정확히** 일치시켜야 baseline과 직접 비교 가능. parity 체크리스트 + 오류 이력 포함.
>
> 최초 작성 2026-06-18 (6개 벤치 로그·스크립트 전수 audit). 갱신 시 "검증 일자" 표기.

## 모델 범위 (2026-06-18)

- **현재 모델 = `Parvo`** (code v15b, student-anchor M→P scaffold). 논문 핵심. 평가 대상.
- **baseline (재사용 비교군)**: **VideoMAE-ours**(controlled, EgoDex 학습) · **DINOv2 · SigLIP · VC-1**(공개 weight, cross-domain).
- **input-only baseline to beat = VideoMAE-ours** (Parvo가 이걸 넘어야 scaffold 성립).
- ⛔ **v11 / v15**: 논문 미사용. 코드·ckpt는 보존하되 본 문서·비교표에서 제외(v15 일부 수치는 "legacy 참조"로만 각주). [CLAUDE.md](../CLAUDE.md) "명명·2논문 구조" 참조.

## 채택 벤치 (4개)

| 벤치 | 역할 | 상태 |
|------|------|------|
| **EgoDex probing** | within-domain controlled (Parvo vs VideoMAE-ours) | Parvo ✅, VideoMAE ✅ |
| **CALVIN probing** | cross-domain OOD (Parvo vs 4 baseline) | baseline ✅, Parvo ⬜ |
| **CortexBench** | generic manipulation SR (Parvo vs 4 baseline) | baseline ✅, Parvo ⬜ |
| **LIBERO BC-T rollout** | **main downstream** closed-loop SR | baseline ✅, Parvo ⬜ |

⛔ **DROID**(변별력 0, 배제) · 🔽 **LIBERO probing**(BC-T가 main, 생략 가능) — §2·§3.

---

## 0. parity 점검 체크리스트 (새 ckpt 평가 전 필독)

1. **split / 평가셋**: baseline이 쓴 *정확한* split(디렉토리명까지). EgoDex `test`≠`part4`.
2. **샘플 수**: `N samples from M videos`가 baseline과 일치. 다르면 probe fit 달라져 R² 비교 불가.
3. **gap**: 동일 gap 집합. cross-gap 곡선은 각 gap이 동일 평가셋(=동일 max_episodes)인지.
4. **target 정의**: dims/단위/계산식(pos delta vs cumulative, axis-angle 등).
5. **전처리/정규화** (🔴 최빈 사고원): 학습 입력 range/channel order = inference transform. EgoDex 학습 = `/255.0`만 = `[0,1]` raw ([src/datasets/base.py:119](../src/datasets/base.py#L119)). ImageNet Normalize 추가 금지(2026-05-25 CortexBench 사고).
6. **probe/epochs/batch/metric**: linear vs MLP, best-epoch vs final, R² aggregate vs per-dim.
7. **ckpt epoch**: baseline과 동일 기준.

---

## 1. EgoDex Action Probing (within-domain, controlled)

**역할**: Parvo vs VideoMAE-ours. gripper 없는 18-dim hand-pose = motion-only 지표.
**스크립트**: [scripts/eval/probe_action.py](../scripts/eval/probe_action.py) · 런처 [scripts/cluster/probe_action.sbatch](../scripts/cluster/probe_action.sbatch)

| 항목 | 정규 값 |
|------|---------|
| **split** | **`test`** (egodex/raw/test, 3,243 vid) — 🔴 `part4`(44k) 아님 |
| **샘플 수** | **train 180,921 / eval 40,914** (target_mode=same) |
| gap / target_mode | 10 / `same` (`pose[t+gap]−pose[t]`, 18-dim, confidence 필터) |
| cls_mode | `patch_mean_concat_p_t_p_tk` (P_t⊕P_tk, 1536-d) |
| 전처리 | `/255.0`=[0,1] raw |
| probe/ep/batch/metric | linear / 20 / 256 / best-epoch R² + cosine |

### 결과 (test, p_t_p_tk, 180921/40914)

| 모델 | R² | 비고 |
|------|----:|------|
| VideoMAE-ours (input-only baseline) | **+0.4705** | 🎯 Parvo가 넘어야 할 목표 (job 34467129) |
| **Parvo** (Run B-2 cont, ep30) | **+0.2884** | ❌ baseline 미달. 붕괴본(CLS collapse, patch healthy하나 concat 약). job 36052612 |
| ~~v15 (legacy)~~ | ~~+0.39~~ | 논문 미사용. concat artifact, motion no-op |

> **판정**: VideoMAE(scaffold 없는 2-frame MAE) +0.47 > Parvo +0.29. 현 Parvo는 붕괴로 input-only baseline에 미달 → scaffold 미지지. 비붕괴 Parvo ckpt 확보 후 재평가가 핵심.

**재현**:
```bash
sbatch --job-name=probe_<tag> \
  --export=ALL,ENCODER=<parvo|videomae>,CHECKPOINT=<ckpt>,\
SPLIT=test,CLS_MODE=patch_mean_concat_p_t_p_tk,GAP=10,EPOCHS=20,BATCH_SIZE=256 \
  scripts/cluster/probe_action.sbatch
```

---

## 2. DROID Action Probing — ⛔ 논문 배제 (2026-06-18)

> **결정**: 모든 인코더(baseline 포함) 절대 R²≈0이라 변별력 없음 → **논문 제외**. cross-domain은 CALVIN(§4)으로 대체. 기록만 보존.
> (참고로 기존 측정엔 gap별 max_episodes 불일치(200/500/1000) parity 결함도 있었음 — 배제로 무의미.)

스크립트: [scripts/eval/probe_action_droid.py](../scripts/eval/probe_action_droid.py). 데이터 `droid_frames/ext1`, gap {1,10,15,30}, 7-dim delta, [0,1](per-encoder 정규화는 `encode_batch` 내부).

---

## 3. LIBERO Action Probing — 🔽 후순위 (2026-06-18)

> **결정**: LIBERO는 **BC-T rollout(§6)이 main downstream** → probing은 **생략 가능**(보강 자료). 새 ckpt에 필수 아님.

스크립트: [scripts/eval/probe_action_libero.py](../scripts/eval/probe_action_libero.py). suites spatial/object/goal, gap {1,13,20,40}(gap=20=1.0s 핵심), 7-DoF pose-derived(`ee_ori`=axis-angle), [0,1], linear 20ep. 전처리 parity OK(사고 없음). 필요 시 baseline 자산은 `paper_artifacts/libero_action_probing/` 참조.

---

## 4. CALVIN Cross-folder OOD Probing

**역할**: 진짜 OOD(training/→validation/). cross-domain probing의 main ground.
**스크립트**: [scripts/eval/probe_action_calvin.py](../scripts/eval/probe_action_calvin.py) · target [src/datasets/calvin.py](../src/datasets/calvin.py) · [probe_action_calvin.sbatch](../scripts/cluster/probe_action_calvin.sbatch)

| 항목 | 정규 값 |
|------|---------|
| dataset/view | `task_ABCD_D` 30Hz, `rgb_static` |
| split | **`--cross-folder`**: train=training/ seg(200 cap), eval=**validation/ 전체 1087 seg (OOD)** |
| 샘플 단위 | **task segment**(commit `7eb0c48`, idle/transition pair 배제). gap {10,15,20,30}, canonical=**30**(1.0s). 🔴 sbatch default `10 20 30 45`라 override 필수 |
| 샘플 수 | gap30: train 6,114 / eval 32,183 |
| target | pos Δ(3,m) + rotvec Δ(3,rad) + gripper(1, binary) |
| 전처리/probe | [0,1], linear 20ep batch256 seed42. metric R² aggregate + **per-dim** |

### baseline 결과 (per-dim R², gap=30)

| 모델 | pos avg | gripper | aggregate |
|------|--------:|--------:|----------:|
| VideoMAE-ours | **+0.553** | +0.059 | +0.056 |
| VC-1 | +0.536 | +0.022 | +0.035 |
| DINOv2 | +0.223 | **+0.359** | **+0.307** |
| SigLIP | −0.314 | +0.183 | +0.162 |
| **Parvo** | ⬜ pending | | |

> ⚠️ **aggregate R²는 binary gripper에 dominated** (CALVIN pos delta scale 극소). 모델 비교는 **per-dim(특히 pos)** 으로. ~~v15 pos +0.262(dinov2 동급)~~ legacy. **5 인코더 parity 깨끗**(동일 segment·gap·샘플·전처리).

```bash
sbatch --export=ALL,ENCODER=parvo,CHECKPOINT=<ckpt>,\
SPLIT=training,CROSS_FOLDER=1,MAX_EPISODES=200,GAPS="10 15 20 30" \
  scripts/cluster/probe_action_calvin.sbatch
# (baseline: ENCODER=dinov2/siglip/vc1 무ckpt, videomae-ours +VIDEOMAE_MODE=paired)
```

---

## 5. CortexBench (local workstation, cluster 아님)

**역할**: Meta-World + Adroit success-rate (frozen encoder + BC head). vision encoder generic capability.
**스크립트**: [scripts/local/run_cortexbench_matrix.sh](../scripts/local/run_cortexbench_matrix.sh) · [aggregate_cortexbench.py](../scripts/eval/aggregate_cortexbench.py) · 가이드 `docs/setup/CORTEXBENCH_GUIDE.md`

| 항목 | 정규 값 |
|------|---------|
| 벤치 | Adroit(pen, relocate) + Meta-World(assembly, bin-picking, button-press-topdown, drawer-open, hammer)=7 task |
| seeds | 100/200/300 |
| eval | frozen encoder → embedding → BC MLP head, ep99. metric `eval/highest_success` |
| 입력 계약 | 🔴 `[0,1]` raw RGB CHW (학습 동일). **ImageNet Normalize 금지** |

### baseline 결과 (mean over seeds)

| 모델 | Meta-World(5) | Adroit(2) |
|------|--------------:|----------:|
| SigLIP | 88.8 | 54.7 |
| VC-1 | 88.8 | 48.7 |
| DINOv2 | 85.9 | 54.0 |
| VideoMAE-ours | 82.4 | 48.0 |
| **Parvo** | ⬜ pending | |

> ✅ **정규화 사고 수정·재실행 완료**(검증 2026-06-18): 2026-05-25 `v15_loader.py`/`videomae_loader.py`의 ImageNet `T.Normalize`(→OOD, Meta-World ~53% 폭락, 30잡 무효) → commit `b80d56c`로 제거, 재실행됨. ~~v15_p_only 89.9/52.0~~ = **Paper 1 image-MAE 참조용**(legacy, 본 표 미포함). committed CSV 없음(gitignore, aggregator 재생성).

---

## 6. LIBERO BC-Transformer Fine-tuning + Rollout (★ main downstream)

**역할**: 논문 1차 downstream = closed-loop success rate.
**스크립트**: [finetune_libero_bct.py](../scripts/eval/finetune_libero_bct.py) · [src/eval_libero.py](../src/eval_libero.py) · [finetune_libero_bct.sbatch](../scripts/cluster/finetune_libero_bct.sbatch) · [run_libero_rollouts.sh](../scripts/local/run_libero_rollouts.sh) · [aggregate_libero_rollouts.py](../scripts/eval/aggregate_libero_rollouts.py)

| 항목 | 정규 값 |
|------|---------|
| suites | spatial / object / goal (각 10 task) |
| encoder | **frozen** (policy head만 학습) |
| policy / V3 cfg | `AdaptedBCTransformerPolicy`(GMM 5-mode) + `use_joint=True` + aug(`ImgColorJitter` 0.3 + `TranslationAug` 4) |
| train | 50ep, batch32, lr1e-4 AdamW, cosine, seq_len 10, **train seeds {0,1,2}** |
| rollout | 50 trial×10 task = **500 ep/seed**, replan 1, wait 10, rollout RNG 고정 seed=7 |
| 입력 | env 128×128 → [0,1] → adapter img_size 224. BC-T 레벨 ImageNet norm 없음 |
| metric | per-task SR → suite overall → **3 seed mean±std** |

### baseline 결과 (3 seed mean — spatial / object / goal)

| 모델 | spatial | object | goal |
|------|--------:|-------:|-----:|
| SigLIP | 0.802 | 0.907 | 0.855 |
| VC-1 | 0.732 | 0.875 | 0.857 |
| DINOv2 | 0.715 | 0.880 | 0.838 |
| VideoMAE-ours | 0.215 | 0.239 | 0.424 |
| **Parvo** (adapter 신규 필요) | ⬜ | ⬜ | ⬜ |

> ✅ **전처리 parity 정확**(전 인코더 [0,1] raw, per-adapter 내부 정규화=각 인코더 학습 계약). 2026-05-03 cross-camera cache·env_resolution 128 수정 반영.
> ~~v15-ptptk 0.63/0.864/0.837, v15-mp 0.43/0.39/0.67~~ = legacy(논문 미사용, ckpt 보존). **Parvo는 ptptk/mp 흐름 어댑터를 Parvo ckpt로 dispatch**해 동일 매트릭스(3 suite × 3 seed).

```bash
# finetune (cluster)
ENCODER=<parvo-ptptk> CHECKPOINT=<parvo_ckpt> TASK_SUITE=libero_spatial SEED=0 \
  sbatch scripts/cluster/finetune_libero_bct.sbatch
# rollout (local docker)
bash scripts/local/run_libero_rollouts.sh <parvo-ptptk> 50 <suffix>
```

---

## 7. 현황 요약 + 빠진 실험 (2026-06-18)

### 커버리지

| 벤치 | VideoMAE | DINOv2 | SigLIP | VC-1 | **Parvo** |
|------|:--:|:--:|:--:|:--:|:--:|
| EgoDex probing | ✅+0.47 | (controlled 밖) | — | — | ✅**+0.29** |
| CALVIN probing | ✅ | ✅ | ✅ | ✅ | ⬜ pending |
| CortexBench | ✅ | ✅ | ✅ | ✅ | ⬜ pending |
| LIBERO BC-T rollout | ✅ | ✅ | ✅ | ✅ | ⬜ pending |

→ **baseline은 4 벤치 전부 완료.** Parvo는 EgoDex만 측정(+0.29, baseline 미달).

### 빠진 실험 = 평가가 아니라 "학습"

**Paper 2 (AAAI, Parvo) — critical path, 학습에 막힘** 🔴
- Parvo 비붕괴 ckpt 부재(Run B-2 붕괴). 현 +0.29 < VideoMAE +0.47 → scaffold 미지지. **비붕괴 Parvo 확보가 전제**.
- 확보 후: CALVIN/CortexBench/LIBERO-BCT에 Parvo 행 추가(동일 프로토콜, 즉시).

**Paper 1 (ICRA, image MAE) — 별개 트랙** 🟡
- **채널 격리 ablation**: standalone image MAE × {Sobel+RGB / RGB-only / Sobel-only} → VideoMAE 비교. 신규 학습 필요. [paper1_input_prior_plan.md §Ablation](paper1_input_prior_plan.md).
- VideoMAE 공정성(mask ratio·budget) + 실로봇(클러스터 아님).

> 다음 클러스터 작업 = (1) **Parvo 본학습 성공**(최우선, 붕괴 디버깅) (2) Paper 1 channel ablation 학습. 둘 다 *새 학습*. 평가 프로토콜은 본 문서로 고정됨.
