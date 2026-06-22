# 파일 인덱스 (주요 파일)

> 저장소 핵심 파일 reference. 매 세션 필수가 아니라 [CLAUDE.md](../CLAUDE.md)에서 분리. 명명·2논문 구조는 CLAUDE.md "명명 · 2논문 구조"가 정규 출처.

## 공통 인프라

| 파일 | 용도 |
|------|------|
| `scripts/pretrain.py` | Pre-training 메인 스크립트 (env-agnostic, 모든 모델 dispatch) |
| `src/training/pretrain.py` | Pre-training 루프 (모든 모델 공유) |
| `src/datasets/egodex.py`, `src/datasets/droid.py`, `src/datasets/libero.py` | 데이터셋 로더 |
| `scripts/local/download_egodex.sh`, `scripts/cluster/download_egodex.sh` | EgoDex CDN 다운로드 |
| `scripts/cluster/extract_egodex.sbatch` | 프레임 추출 sbatch (144 CPU 병렬) |
| `scripts/data/extract_droid_frames.py`, `extract_droid_actions.py` | DROID 전처리 |
| `scripts/cluster/pretrain.sbatch`, `scripts/cluster/README.md` | 클러스터 학습 launcher |

## 🏆 Paper main — Parvo (code v15b)

> `two_stream_v15.py`가 v15/v15b 공용 구현 (dispatch). Parvo = `TwoStreamV15Model(pair_mode=True, use_sobel=False, masked_anchor=True)`.

| 파일 | 용도 |
|------|------|
| `src/models/two_stream_v15.py` | Two-Stream 구현 (v15 teacher-anchor / v15b=Parvo student-anchor dispatch, V-JEPA + L_compose) |
| `scripts/cluster/sanity_v15.sbatch` | v15 sanity (단일 GPU, 50vid × 3ep) |
| `src/encoders/adapters/parvo_pt_ptk.py` | **Parvo BC-T 어댑터** (no-Sobel P=RGB 3ch, P_t ⊕ P_tk) — 현 활성 |
| `src/encoders/adapters/two_stream_v15_pt_ptk.py` | 구 v15 BC-T 어댑터 (Sobel, 옵션 B P_t ⊕ P_tk) |
| `src/encoders/adapters/two_stream_v15_mp.py` | BC-T 어댑터 (C-variant, M ⊕ P_curr) |
| `scripts/eval/probe_action.py` | EgoDex action probing (v15 mode 신규 추가: `patch_mean_concat_p_t_p_tk` 등) |
| `scripts/eval/probe_action_libero.py` | LIBERO action probing (v15 mode 신규 추가) |
| `scripts/eval/finetune_libero_bct.py` | LIBERO BC-T 학습 (V3 cfg: use_joint + augmentation) |
| `scripts/cluster/finetune_libero_bct.sbatch` | BC-T sbatch launcher |
| `scripts/local/run_libero_rollouts.sh` | 로컬 closed-loop rollout 런처 |
| `scripts/eval/aggregate_libero_rollouts.py` | Rollout 결과 → `paper_artifacts/libero_rollout/` 통합 |
| `scripts/eval/visualize_v15_no_mask.py` | v15 nomask reconstruction 시각화 |
| `scripts/eval/diagnose_v15_collapse.py`, `scripts/cluster/diagnose_v15.sbatch` | v15 collapse 진단 |

## Active baselines

| 파일 | 용도 |
|------|------|
| `src/models/videomae.py` | VideoMAE baseline (active controlled comparison) |
| `src/encoders/adapters/single_frame.py` | DINOv2 / SigLIP / VC-1 어댑터 |
| `src/encoders/adapters/vjepa2.py` | V-JEPA 2.1 어댑터 (probing only, BC main 제외) |

## Reference (paper §method history 인용용)

| 파일 | 용도 |
|------|------|
| `src/models/two_stream.py` | Two-Stream v4~v10 (v6 ep8 = 이전 챔피언 +0.259) |
| `src/models/two_stream_v11.py` | Two-Stream v11 (motion-routing + dual-target, ep44 +0.288) |
| `src/encoders/adapters/two_stream_v11.py` | v11 BC-T 어댑터 (A+D' mode) |
| `src/encoders/adapters/videomae.py` | VideoMAE BC-T 어댑터 |
| `scripts/eval/visualize_attn_v11.py`, `visualize_attn_compare.py`, `visualize_attn_rotation.py` | v6/v10/v11 attention viz |
| `scripts/eval/probe_action_v11.py`, `probe_action_droid_v11.py` | v11 별도 probing 스크립트 |
| `scripts/eval/value_alignment.py`, `analyze_delta_l.py` | Phase 2.5 value alignment (negative result) |

## Deprecated (paper 미사용, 코드만 보존)

| 파일 | 비고 |
|------|------|
| `src/models/two_stream_v12.py` | v11 + CLS semantic residual. sanity cls_p collapse |
| `src/models/two_stream_v13.py` | Dual-frame + DINO global CLS. ep10+ uniform collapse |
| `src/models/two_stream_v14.py` | Stream-wise (P=MAE+V-JEPA, M=DINO). ep20 cancel, EgoDex probing R²=-0.065 |

> v12/13/14 모델 파일은 보존(dispatch 정리는 Parvo 본학습 후). 관련 launcher/viz·구 LIBERO finetune(`sanity_v{12,13,14}.sbatch`, `visualize_{attn_,}v{13,14}*.py`, `finetune_libero{,_v11}.py`+sbatch)는 삭제됨 (2026-06-11 cleanup, git history 참조 — BC-T로 대체).
