# 리팩토링 Plan — 모델 네이밍 reorg + legacy 코드 정리

> **상태**: 계획 확정, 실행 전. 작성 2026-06-23.
> **프로세스**(사용자 지정): ① 이 문서 작성 → ② 문서 기반 단계별 리팩토링 + 각 단계 검증 → ③ 끝나면 이 문서 정리(history는 git).
> **근거**: 멀티에이전트 종합 의존성 스캔 + 2-verifier 적대적 검증 (86 findings, fold-mapper + completeness-critic + order-verifier). 두 verifier 수렴.
> ⚠️ **검증 caveat**: 로그인 노드에 torch env 없고 학습 중 → 정적 grep/read 검증만. **각 stage는 free 노드에서 import-smoke + Parvo forward-parity gate** 후 진행.

---

## 1. 확정 네이밍 (2축 체계)

**채널 축(prefix)**: `Edge-Prior` = +Sobel edge (P=Sobel+RGB 5ch, M=ΔL+Sobel(ΔL) 3ch) / `[무접두]` = no-Sobel (P=RGB 3ch, M=ΔL 1ch, edge 제외).
**목적 축(base)**: Image MAE / Motion-Scaffolded JEPA / Motion-Conditioned Predictive MAE.

| 목적 (base) | no-Sobel | Edge-Prior (Sobel) |
|---|---|---|
| **Image MAE** (단일프레임 recon) | `Image MAE` (구 no-M / 2F-MAE) | `Edge-Prior Image MAE` (구 Paper1 / EP-MAE, ✓입증·`v15_p_only`) |
| **Motion-Scaffolded JEPA** (M→P latent) | `MS-JEPA` (구 Parvo / v15b, 현 main) | `Edge-Prior MS-JEPA` (코드 옵션) |
| **Motion-Conditioned Predictive MAE** (M→P pixel) | `MCP-MAE` (구 §9, 방금 구현) | `Edge-Prior MCP-MAE` (코드 옵션) |

- 규칙: `EP-` 접두 = Sobel, 무접두 = no-Sobel. 세로=같은 채널·다른 목적, 가로=같은 목적·±Sobel.
- baseline: `SiamMAE`, `VideoMAE-ours` (외부 published, 이름 유지). legacy: v4~v14 (버전명 유지).
- **모두 `two_stream_v15.py` 한 파일이 플래그로 생성**: anchor(teacher=Edge-Prior Image MAE / student=MS-JEPA), `--v15-no-motion`(Image MAE), `--v15-pixel-pred`(MCP-MAE), `--no-sobel`(채널 축).

---

## 2. 의존성 landmine (적대적 검증이 잡은 것 — 모르고 진행 시 활성 파이프라인 파손)

1. **🔴 `probe_action_v11.py` = 공유 probing 유틸 (단순 삭제/fold 불가)**. `load_v11_model`/`_p_encoder_forward` 등을 **활성 consumer 7개**가 import:
   - `src/cortexbench/v15_loader.py:50` — **Paper-1 CortexBench P-only 로더 = 유일한 '입증된' 결과 경로** (`load_v15_p_only`). ← 누락 시 입증 결과 파손.
   - `scripts/viz/pca_overlay.py:132`, `scripts/viz/grad_cam_arrow.py:87` — keep-active sbatch(`viz_pca_overlay`, `viz_grad_cam`)가 구동.
   - `probe_action_libero.py:97/154`, `value_alignment.py:108`, `probe_action_droid_v11.py:38`.
   - ⚠️ **채널 의미 보존 필수**: `load_v11_model` = 5ch Sobel P. no-Sobel Parvo = 3ch P. fold 시 caller별 채널 유지.
2. **🔴 v12/13/14 삭제 → `scripts/pretrain.py` 파손**. `scripts/pretrain.py:26-28` eager top-level import + 387/410/466 instantiation. `src/models/__init__.py:25-27`도 eager import. → 같은 commit에서 prune 안 하면 `from src.models import X` 전부 ImportError (활성 v15/siammae/videomae 학습까지 즉사).
3. **🟠 `src/training/pretrain.py`가 v14를 활성 v15와 같은 tuple 분기에 묶음**(L311, 908). surgical-prune은 tuple을 **split**해야(v15 경로 보존).
4. **🟠 `two_stream.py` 통째 삭제 불가**. `TransformerBlock` 외 `TwoStreamModel`/`TwoStreamEncoder`(v4~10)를 활성 consumer 보유: `pretrain.py:355`(--model two-stream base), `probe_action.py:346`(TwoStreamEncoder), viz 4종. → **`TransformerBlock`(+`apply_rope`/`build_2d_rope_freqs`)만 shared로 추출**, 본체는 별도 disposition.
5. **🟠 `two_stream_v11.py` 이번 범위 삭제 불가**. v15 부모 + `pretrain.py:370` + `recon_quality_v11_vs_v15.py`(진짜 v11 ckpt 필요) + 3 missed consumer. fold는 **v15 self-containment(향후 rename 대비)용**일 뿐 v11.py는 keep.
6. **🟠 state_dict key parity = 최대 리스크**. v15 흡수 시 attribute 이름/등록순서 100% 보존. forward 미사용이나 ckpt에 존재하는 `motion_routing`/`interpreter_2`/`interpreter_2_norm` 누락 금지. **freeze 라인(v15 __init__:327-332) 보존** = DDP `find_unused_parameters=False` hang 방지.
7. **🟢 검증된 safe**: `parvo_pt_ptk.py` self-contained(`_encode_p_unmasked` 사용, probe_action_v11 무관). base.py legacy 어댑터 = lazy import(분기 prune 후 삭제 가능). `TwoStreamV11Encoder` export = consumer 0(드롭 가능).

---

## 3. disposition 분류

**이번 범위에서 즉시 삭제 안전 = v12/13/14뿐** (true dead-end). 나머지는 fold/prune/keep.

| disposition | 대상 |
|---|---|
| **delete (model)** | `two_stream_v12.py`, `two_stream_v13.py`, `two_stream_v14.py` (1578 LOC, after stage 3 prune) |
| **delete (adapter)** | `two_stream_v11.py`, `two_stream_v15_pt_ptk.py`, `two_stream_v15_mp.py` (after base.py prune) |
| **delete (eval/viz)** | `visualize_attn_v11.py`, `visualize_attn_compare.py`, `visualize_attn_rotation.py`, `visualize_inference.py`, `visualize_sample_detail.py`, `recon_quality_v11_vs_v15.py` (artifacts 보관 확인 후) |
| **delete (sbatch)** | `sanity_v10.sbatch`, `sanity_v11.sbatch`, `probe_v11.sbatch`, `probe_droid_v11.sbatch`, `viz_v11.sbatch`, `recon_quality.sbatch` |
| **fold → shared** | `probe_action_v11.py`, `probe_action_droid_v11.py` 의 공유 helper → 새 shared probing util, **7 consumer repoint** |
| **fold → shared** | `TransformerBlock`(+rope helper), `MotionRoutingBlock` → `src/models/common/blocks.py` |
| **surgical-prune** | `__init__.py`, `scripts/pretrain.py`, `src/training/pretrain.py`(v14 tuple split), `probe_action.py`(v4-10 분기), `base.py`(3분기), `finetune_libero_bct.py`(CLI choices), 멀티인코더 sbatch 6종 |
| **keep (삭제 보류)** | `two_stream.py` 본체, `two_stream_v11.py` (활성 consumer 존재 — 별도 disposition) |
| **keep-doc-record** | 문서의 legacy 결과/ckpt 경로/history (보존, 단 활성 표현은 새 naming 갱신) |

---

## 4. 단계별 실행 순서 (각 stage 후 gate)

> 두 verifier 수렴 순서. **각 stage 후 free 노드에서**: ① `from src.models import *` import-smoke, ② Parvo ckpt load(strict=False)→`parvo_pt_ptk` 1-step forward가 STEP-0 golden과 bit-동일.

- **S0 — golden snapshot**: 활성 Parvo ckpt 로드 → 고정입력 P_t/P_tk 텐서 저장(parity 기준). `import TwoStreamV15Model/V11Model/TwoStreamModel` 성공 기록.
- **S1 — TransformerBlock 추출**: `common/blocks.py` 신설, `TransformerBlock`+`apply_rope`+`build_2d_rope_freqs`(two_stream.py:70-191) 이동. two_stream.py / v11:52 / v15:46 import 교체. (상속구조 불변 → weight key 불변.)
- **S2 — MotionRoutingBlock 추출**: → `common/blocks.py`. v11/v15:47 import 교체.
- **S3 — v12/13/14 string/import prune (파일 삭제 전, 1 commit)**: `__init__.py`(import 25-27, __all__ 36-38, 표 15-20), `scripts/pretrain.py`(import 26-28, instantiation 387/410/466, choices L42, --v12/13/14-* 블록), `src/training/pretrain.py`(v12/13/14 분기 + **L311/908 v14+v15 tuple split**).
- **S4 — v12/13/14.py 삭제**: `git rm`. (importer 0 확인 후.)
- **S5 — ~~probe_action_v11 fold~~ → keep-active (교정 2026-06-23)**: 실측 결과 `probe_action_v11.py`는 legacy가 아니라 **활성 공유 probe util**(legacy 의존 0, `TwoStreamV11Model`(keep)만 사용). 5개 활성 consumer(Paper-1 `v15_loader`·viz 2종·libero·value_alignment)가 공유. → **fold/삭제 안 함, keep-active.** 이름의 'v11'은 deferred 코드 rename에서 처리. 최고위험 단계 소거.
- **S6 — 활성 eval 스크립트 legacy 분기 prune**: `probe_action.py`(v4-10 'two-stream' 345-362/482/864), libero/calvin/value_alignment/diagnose(import 출처 교체, v15-alias 분기 유지), `finetune_libero_bct.py`(choices 346-350), `base.py`(L63-73).
- **S7 — legacy 어댑터 삭제**: `two_stream_v11.py`, `two_stream_v15_pt_ptk.py`, `two_stream_v15_mp.py`.
- **S8 — v4-10/v11 viz·eval 삭제**: viz 5종 + `recon_quality_v11_vs_v15.py`. (PNG/JSON artifacts 보관 확인 후 co-delete.)
- **S9 — legacy sbatch 삭제 + 멀티인코더 sbatch prune**: 6 launcher 삭제. pretrain.sbatch(L176-365 v8-14 블록 + 464-472 alias), finetune/probe_libero/probe_calvin/value_alignment sbatch, run_calvin_probing_matrix.sh의 v11 분기/변수. **`--v11-m-depth`/`--v11-routing-mode`는 v15 활성 CLI라 유지**(코드 rename 시 함께).
- **S10 — (별도 범위, 보류) v11.py + two_stream.py 본체 삭제**: 이번 안 함. v11 계열 eval consumer 은퇴 후 재평가.
- **S11 — 문서 갱신**: 끊긴 deep-link(v15b_status:234 MotionRoutingBlock, fig1 README:33, FILE_INDEX 49-65, README:38 tree, eval/README 61-70), 삭제 스크립트 참조는 'history(재실행 불가)' 표기, 새 naming 적용. ckpt 경로/cost-log/archive history는 verbatim 보존.

---

## 5. 학습 완료 후 deferred (이번 안 함)

> CLAUDE.md 정책: 코드 식별자 rename은 본학습 후 (헛 diff·resume 충돌 방지).

**코드 식별자 rename** (v15→새 naming): `two_stream_v15.py`→`motion_mae.py`(가칭), `TwoStreamV15Model`→base+config, `--v15-*` 플래그→`--*`, `--v11-m-depth`→`--m-depth` 등. v11.py/two_stream.py 본체 삭제도 이때 재평가(S10).

**클러스터 체크포인트 rename** (인벤토리, `/proj/external_group/mrg/checkpoints/`):
- 활성 → rename: `parvo*`(9개: parvo_runB2cont 등)→`ms_jepa*`, `two_stream_v15b`→main, `two_stream_v15b_noM_cont`→`image_mae*`, `two_stream_v15_student_anchor`/`_vfromm`.
- legacy → 이름 유지(history): `two_stream_v5~v14`, `v7big*`, `v8`, `v9*`.

---

## 6. 검증 프로토콜 (load-bearing)

1. **import-smoke**: `from src.models import *` + `parvo_pt_ptk`/`v15_loader` instantiate (각 stage 후).
2. **weight parity**: 기존 Parvo ckpt `load_state_dict(strict=False)` → `parvo_pt_ptk:51-53` enc_missing=[] assert 통과 + 동일입력 P_t/P_tk **bit-동일**.
3. **3-frame forward 1-step**: loss dict 키 동일.
4. **DDP**: 흡수 후 2-GPU 1-step에서 unused-param 0 (freeze 라인 보존 확인).
5. ⚠️ 로그인 노드 torch 없음 → **free GPU 노드 또는 conda env**에서 실행. 학습 잡과 자원 충돌 회피(MIG 가능).

---

## 7. 완료 후 (process ③)

리팩토링 완료 시 이 문서를 정리: 실행 history는 git commit으로, 영구 가치 있는 결과(새 naming 표, 최종 구조)는 `FILE_INDEX.md`/`CLAUDE.md` "명명" 섹션으로 흡수, 본 문서는 archive 또는 삭제.

## 8. 실행 현황 (2026-06-23)

**✅ 완료 (gate 통과·커밋·푸시, 누적 −4841 LOC):**
- S1/S2 — `common/blocks.py` 단일 출처 (`TransformerBlock`·`MotionRoutingBlock`).
- S3/S4 — v12/13/14 모델 + dispatch/import prune + 파일 삭제.
- **S5 교정** — `probe_action_v11.py` = 활성 공유 util(keep), fold/삭제 안 함.
- S6/S7 — legacy 어댑터 3 삭제 + base.py·finetune CLI prune.
- S8/S9 — legacy viz/eval 7 + sbatch launcher 7 삭제.
- S11(부분) — FILE_INDEX·eval/README 재작성(새 naming), MotionRoutingBlock deep-link 정정, RESEARCH_PLAN 스트라이크.

**⏳ 남은 작업 (전부 lighter·non-functional — 다음 세션):**
1. **training loop dead-code**: `src/training/pretrain.py`의 v12/13/14 문자열 분기 14개 제거. ⚠️ v11/v15 공유 tuple(L311/406/767/908)은 **dead 항목만 제거, v11/v15 보존**(split). import 없어 무해하나 정리 가치.
2. **in-file sbatch prune**: `pretrain.sbatch` v8~v14 MODEL 분기(L176-365, dead — 잘못된 `--model`) + 멀티인코더 sbatch(`probe_action_libero/calvin`, `value_alignment`, `run_calvin_probing_matrix.sh`)의 `V11_MODE`·`two-stream-v11` 분기·usage 주석.
3. **전체 naming 전파**: docs 전반 `Parvo`→`MS-JEPA` 등 기능명 치환 — **deferred 코드 rename과 함께** 일괄(지금 부분 치환 시 혼란).

**🔒 deferred (학습 완료 후):**
- 코드 식별자 rename: `two_stream_v15.py`/`TwoStreamV15Model`/`--v15-*`/`--v11-*`, `probe_action_v11.py`→공유 util 새 이름. v11.py·two_stream.py 본체 삭제 재평가(S10).
- 클러스터 ckpt rename: `parvo*`→`ms_jepa*`, `two_stream_v15b_noM_cont`→`image_mae*` 등 (인벤토리 = cluster_sessions/artifacts).
