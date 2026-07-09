# Paper Artifacts — Paper 1 (ICRA, Input-Prior) · Paper 2 (AAAI, Action-Agnostic)

**dev 저장소 ↔ 논문 작성 저장소** 사이의 단일 hand-off 지점. paper 작성 흐름에 직결되는
**확정 산출물**만 보관하고, 양쪽에서 read-only로 참조한다.

- 명명·2논문 구조의 정규 출처 = [`CLAUDE.md`](../CLAUDE.md) "명명 · 2논문 구조"
- **Paper 2 ours 축 = CoMP-MAE (code v16)** · 논문 spine = 3-claim (① factorization ② dissociation ③ 도메인-robust 효율)
- 진행 상태·계획 = [`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md) · [`docs/factorization_crossover_plan.md`](../docs/factorization_crossover_plan.md)

## 폴더 원칙 (2026-07-09 재편)

- **키워드 폴더**: "어떤 자료인지 / 뭘 보여주는지"로 명명. **fig/tab 넘버링 금지** —
  넘버는 논문 편집 중 수시로 바뀌므로 넘버↔폴더 매핑은 Vault `7. Outline.md`에서만 관리.
- **scratch ↔ paper_artifacts 역할 분리**: gitignored `scratch/` = iteration·중간 덤프(재생성 가능, 언제든 삭제).
  확정된 자료만 여기 키워드 폴더로 **승격**(git 추적). 빈 placeholder 폴더는 만들지 않음 — 자료가 생길 때 폴더 생성.
- 폴더 내 산출물은 **수작업 편집 금지** (CSV/PNG는 생성 스크립트에서). README만 수동 갱신.

## 폴더 인덱스

| 폴더 | 무엇을 보여주나 | 귀속 | Status |
|------|----------------|------|--------|
| `ood_efficiency/` | **3b 효율 headline 표** — CoMP-S vs plain/VideoMAE/internet-scale, 4벤치 OOD probing R². 재생성 = `scripts/eval/build_step0_efficiency_table.py` | P2 | 🟢 최신 (plain 행 포함) |
| `libero_action_probing/` `calvin_action_probing/` | probing **raw** (`summary.json`/`all_gaps.csv`) — step0 효율·Phase A factorization·STEP 1 인과 판정(s1vp/s1px)·plain(s2px) + 05월 baseline | P2 | 🟢 인용 중 |
| `probing_summary/` | baseline probing aggregate CSV (`libero_all_gaps_summary.csv` — 효율 표 baseline 행의 source) | P2 | 🟢 |
| `libero_rollout/` | LIBERO BC rollout **단일 출처** `{summary,per_task,episodes}.csv` — 제어 성능 | P2 | 🟡 STEP 2(B) 18잡 rollout 대기 |
| `architecture/` | 모델 구조 다이어그램 (CoMP-MAE·MCP-MAE·MS-JEPA). 생성 = `scripts/viz/arch_figs/` | P2 | 🟢 |
| `recon_quality/` | recon 품질 증거 — `comp_mae_{s,b}_ep50/`(최종 composite + **요소별 PNG/npy**, ΔL raw 포함 재조합용) · `msjepa_runB2_samples/`(선행 v15b 계보) · v11 vs v15 구 비교 | P2 | 🟢 ep50 요소 확보 |
| `view_sensitivity/` | encoder별 view(agentview/eih) robustness 데이터+figure | P2(❓) | 🟡 |
| `droid_crossdomain/` | DROID cross-domain probing summary — 도메인 일반화 | 공유 | 🟡 |
| `cortexbench/` | **Paper 1 핵심 증거**: image MAE(Sobel+RGB) P-only > VideoMAE-ours | P1 | 🟡 ablation·real-robot 남음 |
| `presentation/` | concept/hero 이미지 (발표용, 논문 무관) | — | 🟢 |

baseline encoders(`siglip/vc1/dinov2/videomae-ours`)는 양 논문 공유 비교군 (probing raw + summary에 포함).

## 작업 컨벤션

- CSV는 UTF-8 `,`-separated + header row. `r2` column = action joint position R².
- 색 컨벤션: **M=blue / P=red / motion routing=purple / loss=green / mask=gray hatched** (paper 전체 일관)

## Provenance

- 마스터 연구 계획 (Paper 2): [`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md)
- Paper 1 계획: [`docs/paper1_input_prior_plan.md`](../docs/paper1_input_prior_plan.md)
- 평가 프로토콜 (parity 체크리스트): [`docs/eval_protocols.md`](../docs/eval_protocols.md)
- 클러스터 잡 → ckpt 매핑: [`docs/cluster_sessions.md`](../docs/cluster_sessions.md)

## Vault cross-link

| Vault path | 본 디렉토리 매핑 |
|------------|----------------|
| `Projects/Action-Agnostic Paper/7. Outline.md` | **fig/tab 넘버 ↔ 키워드 폴더 매핑** (단일 출처) |
| `Projects/Input-Prior Robot Representation (ICRA)/` | Paper 1 계획·근거 |
| `Projects/Action-Agnostic Paper/Evolution.md` | v1→v15→CoMP-MAE 진화 (구세대 산출물은 git history) |

(Vault 루트: `/Users/bys724/LocalVault/Obsidian Vault/`)

## 정리 이력

- 2026-07-09 **키워드 재편**: figN/tabN 폴더 → 키워드 폴더(rename 표는 git history), README-only
  placeholder 8개 삭제, scratch ep50 요소별 viz를 `recon_quality/`로 승격, scratch는 순수 임시로 환원.
- 2026-07-09 구세대 삭제: `_archive/`·v11 probing raw·catalyst 잔재(fig2/tab6)·구 viz 샘플·TODO.md (git history 보존).
