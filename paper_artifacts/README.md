# Paper Artifacts (CoRL 2026, v15 Main)

본 디렉토리는 **dev 저장소 ↔ 논문 작성 저장소** 사이의 단일 hand-off 지점.
구조는 [Vault `7. Outline`](file:///Users/bys724/LocalVault/Obsidian%20Vault/Projects/Action-Agnostic%20Paper/7.%20Outline.md)의 Figure/Table spec과 1:1 매핑.

## 목적

- Dev repo는 raw 실험 (probing, ckpt, scratch 분석) 담당
- Paper repo는 figure styling, LaTeX, 편집 iteration 담당
- 본 디렉토리만 양쪽에서 read-only로 참조 — paper 작성 흐름에 직결되는 산출물만 보관

## 폴더 구조

```
paper_artifacts/
├── README.md                       # (본 문서)
├── TODO.md                         # 생성해야 할 figure/table 매트릭스 + 우선순위
│
├── fig1_architecture/              # §3 Method — v15 architecture overview
├── fig2_catalyst/                  # §5 Analysis ★ — catalyst mechanism evidence
├── fig3_bc_main/                   # §4 Experiments ★ — LIBERO BC main result
├── fig4_recon_quality/             # §5 Analysis — v11 vs v15 reconstruction 비대칭
├── fig5_droid/                     # §4 Experiments — DROID cross-domain probing
├── fig6_motion_routing_ablation/   # §5.2 Analysis ★ — V from P vs V from M (C1)
├── fig7_loss_curves/               # Appendix C — 5 loss curves
├── fig8_mp_attention/              # Appendix D — M vs P attention + 입력 channel viz
│   ├── combined/                   #   통합 grid (input + M attn + P attn)
│   ├── m_only/                     #   M stream 전용 (input + attention 분리)
│   └── p_only/                     #   P stream 전용 (input + attention 분리)
│
├── tables/
│   ├── tab1_libero_bc/             # §4 ★ — fig3_bc_main의 summary.csv 참조
│   ├── tab2_probing/               # §4 — EgoDex within + LIBERO cross probing
│   ├── tab3_ablation/              # §5.1/5.2 ★ — architecture ablation
│   ├── tab4_12mode/                # §5.4 — 12-mode probing breakdown (v15 ep32)
│   ├── tab5_hparams/               # Appendix B — hyperparameters
│   └── tab6_catalyst_evidence/     # Appendix or §5 — 4-encoder catalyst 표
│
└── _archive/                       # v15 paper main 비사용 (v13, v3 sanity, value alignment)
```

## Paper에서의 figure / table 위치 한눈에

| Position | Type | Polder | Status |
|----------|------|--------|--------|
| §3 Method | Fig 1 | `fig1_architecture/` | 🟡 v11 보유, **v15 미작성** |
| §5 Analysis ★ | Fig 2 | `fig2_catalyst/` | 🔴 미시작 |
| §4 Experiments ★ | Fig 3 + Tab 1 | `fig3_bc_main/` + `tables/tab1_libero_bc/` | 🟡 round1 보유, v15 round2 미통합 |
| §5 Analysis | Fig 4 | `fig4_recon_quality/` | 🔴 미시작 |
| §4 Experiments | Fig 5 | `fig5_droid/` | 🟡 v11/VideoMAE 보유, v15 미실시 |
| §5.2 Analysis ★ | Fig 6 + Tab 3 | `fig6_motion_routing_ablation/` + `tables/tab3_ablation/` | 🔴 C1 학습 대기 |
| Appendix C | Fig 7 | `fig7_loss_curves/` | 🔴 wandb 추출 필요 |
| Appendix D | Fig 8 | `fig8_mp_attention/` | 🟡 docs/architecture에 v11 통합본, **분리본 미생성** |
| §4 | Tab 2 | `tables/tab2_probing/` | 🟡 baseline+v11, v15 미통합 |
| §5.4 | Tab 4 | `tables/tab4_12mode/` | 🔴 v15 12-mode export 대기 |
| Appendix B | Tab 5 | `tables/tab5_hparams/` | 🔴 config 추출 필요 |
| Appendix or §5 | Tab 6 | `tables/tab6_catalyst_evidence/` | 🔴 fig2와 동일 dependency |

전체 생성 작업 우선순위는 [`TODO.md`](TODO.md) 참조.

## 작업 컨벤션

- 본 디렉토리는 **수작업 편집 금지** 원칙 (CSV/PNG는 생성/추출 스크립트에서). 단 README와 TODO.md는 수동 갱신.
- CSV는 UTF-8 `,`-separated + header row
- `r2` column = EgoDex/DROID **action joint position** R²
- "v11" = `checkpoints/two_stream_v11/20260426_014333/` ckpt (특별 명시 없는 한)
- "v15" = `checkpoints/two_stream_v15/.../latest.pt` (= ep50) (특별 명시 없는 한)
- 색 컨벤션: **M=blue / P=red / motion routing=purple / loss=green / mask=gray hatched** (paper 전체 일관)

## Provenance

- 마스터 연구 계획: [`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md)
- 실험 list (C1-C9): [`docs/paper_experiments_plan.md`](../docs/paper_experiments_plan.md)
- 클러스터 잡 → ckpt 매핑: [`docs/cluster_sessions.md`](../docs/cluster_sessions.md)
- Probing 결과 (raw): [`data/probing_results/`](../data/probing_results/) (gitignored)
- BC rollout 결과 (raw): [`data/libero/results/`](../data/libero/results/) (gitignored)

## Vault cross-link

| Vault path | 본 디렉토리 매핑 |
|------------|----------------|
| `Projects/Action-Agnostic Paper/7. Outline.md` | Figure/Table spec — 본 폴더 구조 기준 |
| `Projects/Action-Agnostic Paper/4. Paper Writing.md` | §3 Method 작성 가이드 (fig1/fig2와 직결) |
| `Projects/Action-Agnostic Paper/Evolution.md` | v1→v15 진화 (_archive 항목 history) |
| `Projects/Action-Agnostic Paper/v15 - Layered Specialization (Future).md` | v15 design detail |

(Vault 루트: `/Users/bys724/LocalVault/Obsidian Vault/`)
