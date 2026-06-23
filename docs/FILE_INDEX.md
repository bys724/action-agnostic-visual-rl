# 파일 인덱스 (주요 파일)

> 저장소 핵심 파일 reference. 명명·2논문 구조 = [CLAUDE.md](../CLAUDE.md) "명명 · 2논문 구조".
> **기능 서술명**(2-축: Edge-Prior/no-Sobel × Image MAE/MS-JEPA/MCP-MAE) 단일 출처 = [REFACTOR_PLAN.md](REFACTOR_PLAN.md). 코드 식별자(`v15` 등) rename은 본학습 후(deferred).

## 공통 인프라

| 파일 | 용도 |
|------|------|
| `scripts/pretrain.py` | Pre-training 메인 (env-agnostic, 모든 모델 dispatch) |
| `src/training/pretrain.py` | Pre-training 루프 (모든 모델 공유) |
| `src/models/common/blocks.py` | 공용 transformer 부품: `TransformerBlock`·`MotionRoutingBlock`·RoPE — **단일 출처** |
| `src/models/common/preprocessing.py` | M/P 채널 전처리 (`use_sobel` = Edge-Prior↔no-Sobel) |
| `src/datasets/egodex.py`, `droid.py`, `libero.py` | 데이터셋 로더 |
| `scripts/local/download_egodex.sh`, `scripts/cluster/download_egodex.sh`, `extract_egodex.sbatch` | EgoDex 다운로드·추출 |
| `scripts/data/extract_droid_frames.py`, `extract_droid_actions.py` | DROID 전처리 |
| `scripts/cluster/pretrain.sbatch`, `scripts/cluster/README.md` | 클러스터 학습 launcher |

## 🏆 Two-Stream 활성 — 한 파일이 플래그로 4 모델 생성

> `two_stream_v15.py`(`TwoStreamV15Model`). 기능명↔플래그:

| 기능명 | 채널 | 플래그 |
|------|------|------|
| **MS-JEPA** (구 Parvo, 현 main) | no-Sobel | student-anchor + `--no-sobel` |
| **MCP-MAE** (§9, 붕괴면역 pixel) | no-Sobel | `--v15-pixel-pred` |
| **Image MAE** (구 no-M) | no-Sobel | `--v15-no-motion` |
| **Edge-Prior Image MAE** (Paper1, ✓입증) | Sobel+RGB | teacher-anchor P-only |

| 파일 | 용도 |
|------|------|
| `src/models/two_stream_v15.py` | Two-Stream 구현 (4 모델 dispatch, common/blocks 부품 사용) |
| `src/encoders/adapters/parvo_pt_ptk.py` | BC-T 어댑터 (no-Sobel P_t⊕P_tk) — MS-JEPA/MCP-MAE/Image MAE 공용, 현 활성 |
| `src/cortexbench/v15_loader.py` | **Edge-Prior Image MAE(Paper1)** P-only CortexBench 로더 (입증 결과 경로) |
| `scripts/eval/probe_action_v11.py` | **활성 공유 probe util** (two-stream P/M 인코더 추출·CLS_MODES). v15_loader·viz·libero probe 공유. ⚠️'v11' 이름은 deferred rename |
| `scripts/eval/probe_action.py`, `probe_action_libero.py`, `probe_action_calvin.py` | EgoDex/LIBERO/CALVIN action probing |
| `scripts/eval/finetune_libero_bct.py`, `scripts/cluster/finetune_libero_bct.sbatch` | LIBERO BC-T 학습 (V3 cfg) |
| `scripts/local/run_libero_rollouts.sh`, `scripts/eval/aggregate_libero_rollouts.py` | closed-loop rollout |
| `scripts/cluster/sanity_v15.sbatch` | sanity |
| `scripts/eval/visualize_v15_no_mask.py`, `diagnose_v15_collapse.py`, `scripts/cluster/diagnose_v15.sbatch` | viz/붕괴 진단 |
| `scripts/viz/pca_overlay.py`, `grad_cam_arrow.py` | PCA·Grad-CAM viz (probe util 공유) |

## 신규 baseline (2026-06-22)

| 파일 | 용도 |
|------|------|
| `src/models/siammae.py` | **SiamMAE** (asymmetric 95% + cross-self decoder), 2-size(ViT-S/ViT-B), `--siammae-size` |

## Active baselines

| 파일 | 용도 |
|------|------|
| `src/models/videomae.py` | **VideoMAE-ours** (controlled comparison) |
| `src/encoders/adapters/single_frame.py` | DINOv2 / SigLIP / VC-1 어댑터 |
| `src/encoders/adapters/vjepa2.py` | V-JEPA 2.1 어댑터 (probing only) |
| `src/encoders/adapters/videomae.py` | VideoMAE BC-T 어댑터 |

## Reference (paper §method history 인용용)

| 파일 | 용도 |
|------|------|
| `src/models/two_stream.py` | Two-Stream v4~v10 (v6 ep8 = +0.259). `TransformerBlock`은 common/blocks로 이동 |
| `src/models/two_stream_v11.py` | Two-Stream v11 (motion-routing, ep44 +0.288) — `TwoStreamV15Model`의 부모 클래스 |
| `scripts/eval/value_alignment.py`, `analyze_delta_l.py` | Phase 2.5 value alignment (negative result) |

> **2026-06-23 리팩토링** ([REFACTOR_PLAN.md](REFACTOR_PLAN.md)): legacy 삭제 — 모델 v12/13/14, 어댑터(v11·v15_pt_ptk·v15_mp), v4-10/v11 viz·droid-probe 스크립트, legacy sbatch. `TransformerBlock`·`MotionRoutingBlock` → `common/blocks.py`. 결과/체크포인트/이력은 git history·`artifacts.md`·`paper_artifacts/` 보존.
