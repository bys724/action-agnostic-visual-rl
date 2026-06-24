# Paper Artifacts — Paper 1 (ICRA, Input-Prior) · Paper 2 (AAAI, Action-Agnostic)

본 디렉토리는 **dev 저장소 ↔ 논문 작성 저장소** 사이의 단일 hand-off 지점.
**2논문 구조**에 맞춰 산출물을 Paper별로 정렬 (정규 출처 = [`CLAUDE.md`](../CLAUDE.md) "명명 · 2논문 구조"):

- **Paper 1 (ICRA, Input-Prior)** — 단일프레임 image MAE(Sobel+RGB) > VideoMAE. 계획 = [`docs/paper1_input_prior_plan.md`](../docs/paper1_input_prior_plan.md).
- **Paper 2 (AAAI, Action-Agnostic)** — Parvo(code v15b) scaffold. 계획 = [`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md).

> **relabel in-place**: 물리적 폴더는 paper별로 나누지 않고 README에서 귀속만 명시(docs 경로 참조 보존). 각 figN/tab README 상단에 `Paper N` 태그.

## 목적

- Dev repo는 raw 실험 (probing, ckpt, scratch 분석) 담당
- Paper repo는 figure styling, LaTeX, 편집 iteration 담당
- 본 디렉토리만 양쪽에서 read-only로 참조 — paper 작성 흐름에 직결되는 산출물만 보관

## Paper 1 (ICRA, Input-Prior) 산출물

| Type | 폴더 | Status |
|------|------|--------|
| CortexBench (핵심 증거: v15 P-only > VideoMAE-ours) | `cortexbench/{v15_p_only, videomae_ours, siglip_base/dinov2_base/vc1_vitb}` | 🟡 21잡 보유. 정규화 사고 재실행 이력 → [`eval_protocols.md`](../docs/eval_protocols.md) |
| RGB-only vs Sobel+RGB ablation | (TODO) | 🔴 미실시 — **Paper 1 존재 여부 가름** |
| Real-robot | (TODO) | 🔴 ICRA 본체 lift |

## Paper 2 (AAAI, Action-Agnostic) 산출물

| Position | Type | 폴더 | Status |
|----------|------|------|--------|
| §3 Method | Fig 1 | `fig1_architecture/` (ms_jepa·mcp_mae) | 🟡 신규 아키텍처 figure 보유 |
| §5 Analysis ★ | Fig 2 | `fig2_catalyst/` | 🔴 미시작 |
| §4 Experiments ★ | Fig 3 + Tab 1 | `fig3_bc_main/` + `tables/tab1_libero_bc/` | 🟡 round1 보유, Parvo round2 미통합 |
| §5 Analysis | Fig 4 | `fig4_recon_quality/` | 🔴 미시작 (source 메트릭만) |
| §4 Experiments | Fig 5 | `fig5_droid/` | 🟡 v11/VideoMAE 보유 (❓ Paper 1 귀속 가능성 — 확인 필요) |
| §5.2 Analysis ★ | Fig 6 + Tab 3 | `fig6_motion_routing_ablation/` + `tables/tab3_ablation/` | 🔴 C1 학습 대기 |
| Appendix C | Fig 7 | `fig7_loss_curves/` | 🔴 wandb 추출 필요 |
| Appendix D | Fig 8 | `fig8_mp_attention/` | 🟡 v11 포맷 샘플 1개, **Parvo 분리본 미생성** |
| §4 | Tab 2 | `tables/tab2_probing/` | 🟡 baseline+v11, Parvo 미통합 |
| §5.4 | Tab 4 | `tables/tab4_12mode/` | 🔴 Parvo 12-mode export 대기 |
| Appendix B | Tab 5 | `tables/tab5_hparams/` | 🔴 config 추출 필요 |
| Appendix or §5 | Tab 6 | `tables/tab6_catalyst_evidence/` | 🔴 fig2와 동일 dependency |
| §4 (❓) | Tab 7 | `tables/tab7_view_sensitivity/` | view robustness (❓ paper 귀속 확인) |
| 보조 (probing raw) | — | `calvin_action_probing/`, `libero_action_probing/`, `egodex_action_probing/` | baseline + obsolete v11/v15 기록 (Parvo 재실행 예정) |
| 보조 (rollout raw) | — | `libero_rollout/` | LIBERO BC rollout |
| representation viz | — | `visualizations/{grad_cam_arrow, pca_overlay}/` | post-accept project-page track (Parvo 재생성), 현재 포맷 샘플 1세트만 |

## 공유 / 기타

- **baseline encoders** (`siglip / vc1 / dinov2 / videomae-ours`) = **양 논문 공유** 비교군.
- `presentation/` — concept/hero 이미지 (논문 무관, 발표용).
- `_archive/` — paper main 비사용 (v13, v3 sanity, value alignment).
- iteration 덤프는 gitignored `scratch/viz/` (커밋 안 됨) — 컨벤션 [`docs/viz_assets_refactor_plan.md`](../docs/viz_assets_refactor_plan.md).

전체 생성 작업 우선순위는 [`TODO.md`](TODO.md) 참조.

## 작업 컨벤션

- 본 디렉토리는 **수작업 편집 금지** 원칙 (CSV/PNG는 생성/추출 스크립트에서). 단 README와 TODO.md는 수동 갱신.
- CSV는 UTF-8 `,`-separated + header row
- `r2` column = EgoDex/DROID **action joint position** R²
- **현 모델 = Parvo (code v15b)**. "v11"/"v15"는 obsolete 구 모델 — 포맷 샘플·유효 probing 기록만 보존(figure는 paper 미사용).
- 색 컨벤션: **M=blue / P=red / motion routing=purple / loss=green / mask=gray hatched** (paper 전체 일관)

## Provenance

- 마스터 연구 계획 (Paper 2): [`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md)
- Paper 1 계획: [`docs/paper1_input_prior_plan.md`](../docs/paper1_input_prior_plan.md)
- 실험 list (C-series) + 평가 프로토콜: [`docs/eval_protocols.md`](../docs/eval_protocols.md)
- 클러스터 잡 → ckpt 매핑: [`docs/cluster_sessions.md`](../docs/cluster_sessions.md)
- Probing 결과 (raw): [`data/probing_results/`](../data/probing_results/) (gitignored)
- BC rollout 결과 (raw): [`data/libero/results/`](../data/libero/results/) (gitignored)

## Vault cross-link

| Vault path | 본 디렉토리 매핑 |
|------------|----------------|
| `Projects/Action-Agnostic Paper/7. Outline.md` | Paper 2 Figure/Table spec |
| `Projects/Input-Prior Robot Representation (ICRA)/` | Paper 1 계획·근거 |
| `Projects/Action-Agnostic Paper/Evolution.md` | v1→v15→Parvo 진화 (_archive history) |

(Vault 루트: `/Users/bys724/LocalVault/Obsidian Vault/`)
