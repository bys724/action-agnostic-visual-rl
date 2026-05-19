# Paper Artifacts — TODO Matrix

> **Goal**: paper main (CoRL 2026, v15) 작성에 필요한 모든 figure/table 산출물 매트릭스 + 우선순위.
> **Deadline**: 2026-05-26 (Abstract) / 2026-05-28 (Paper). 오늘 2026-05-18 기준 **10일 남음**.
> **세션 시작 시 확인**: 본 문서 + [`docs/paper_experiments_plan.md`](../docs/paper_experiments_plan.md) (C1-C9 실험 list)

## 1. 우선순위 한눈에

### 🔴 P0 (paper main claim, 마감 critical)

| # | Artifact | Dependency | 소요 | 위치 |
|---|----------|-----------|------|------|
| 1 | **fig3 BC main** v15 round2 통합 | round2 tar 전송 + 로컬 rollout | 1-2일 | `fig3_bc_main/` |
| 2 | **fig1 v15 architecture diagram** | TikZ 작도 | 1-2일 | `fig1_architecture/` |
| 3 | **fig2 catalyst** schematic + bar | v15 P_t+P_tk export + VideoMAE C7 + DINOv2 controlled 결과 모으기 | 2-3일 | `fig2_catalyst/` |
| 4 | **tab2 probing** v15 12-mode export (EgoDex + LIBERO) | C4 결과 export | 0.5일 | `tables/tab2_probing/` |
| 5 | **fig6 motion routing ablation** | **C1 v15-vfromm 학습 (~43h)** + probing + BC | 3-4일 | `fig6_motion_routing_ablation/` |
| 6 | **tab1 LIBERO BC** = fig3과 자동 동기 | (fig3 의존) | — | `tables/tab1_libero_bc/` |
| 7 | **tab3 architecture ablation** | fig6 (C1) + tab2 + fig3 round2 모두 필요 | 종속 | `tables/tab3_ablation/` |

### 🟡 P1 (정직 보고 + 보완 evidence)

| # | Artifact | Dependency | 위치 |
|---|----------|-----------|------|
| 8 | **fig4 recon quality** | v11 ckpt + v15 ckpt forward 2×2 grid | `fig4_recon_quality/` |
| 9 | **fig5 DROID cross-domain** v15 추가 | C5 (v15 4-gap DROID probing) | `fig5_droid/` |
| 10 | **tab4 12-mode breakdown** | tab2 v15 export의 subset | `tables/tab4_12mode/` |
| 11 | **tab6 catalyst evidence 표** | fig2와 동일 dependency | `tables/tab6_catalyst_evidence/` |

### 🟢 P2 (Appendix, 마감 후순위)

| # | Artifact | Dependency | 위치 |
|---|----------|-----------|------|
| 12 | **fig7 loss curves** | v15 wandb run 추출 | `fig7_loss_curves/` |
| 13 | **fig8 M vs P attention** combined | docs/architecture에서 복사/심볼릭 | `fig8_mp_attention/combined/` |
| 14 | **fig8 M-only / P-only 분리본** | **viz script `--split-streams` 옵션 추가 필요** | `fig8_mp_attention/{m,p}_only/` |
| 15 | **tab5 hyperparameters** | scripts/configs 표 추출 | `tables/tab5_hparams/` |

---

## 2. C1-C9 실험 ↔ artifact 매핑

[`docs/paper_experiments_plan.md`](../docs/paper_experiments_plan.md)의 실험 ID와 본 폴더 매핑:

| Exp | 설명 | 산출 artifact | 우선순위 |
|-----|------|--------------|---------|
| C1 | v15 V from M ablation (학습 ~43h) | `fig6_motion_routing_ablation/`, `tables/tab3_ablation/` | 🔴 P0 |
| C2 | C1 ckpt head-to-head probing | `fig6_motion_routing_ablation/`, `tables/tab3_ablation/` | 🔴 P0 |
| C3 | C1 ckpt BC-T 학습 + rollout | `fig6_motion_routing_ablation/`, `fig3_bc_main/` | 🔴 P0 |
| C4 | v15 ep50 LIBERO object/goal probing | `tables/tab2_probing/` | 🔴 P0 |
| C5 | v15 ep50 DROID 4-gap probing | `fig5_droid/` | 🟡 P1 |
| C6 | recon quality v11 vs v15 viz | `fig4_recon_quality/` | 🟡 P1 |
| C7 | VideoMAE-ours P_t+P_tk catalyst | `fig2_catalyst/`, `tables/tab2_probing/`, `tables/tab6_catalyst_evidence/` | 🔴 P0 (Fig 2 핵심) |
| C8 | 선택 — Multi-gap sampling ablation | (paper 본문 제외 가능) | 🟢 |
| C9 | 선택 — L_compose ablation | (paper 본문 제외 가능) | 🟢 |

**핵심**: C1 (43h) + C3 (BC 18잡 각 22-35h) 가 timeline의 critical path. 5/18에 C1 시작 시 C3까지 완료에 ~7일 → 5/25 도착, 5/26 abstract 직전. 여유 부족.

---

## 3. Figure/Table별 세부 TODO 매트릭스

각 폴더의 `README.md`에 detail 있음. 본 문서는 cross-cut 요약.

### 데이터 export 작업 (코딩 X, 단순 export script 실행)

- [ ] **v15 EgoDex 12-mode** ep32 + ep50 → CSV
  - 위치: `tables/tab2_probing/v15_egodex_summary.csv`
  - 명령: `python scripts/analysis/export_probing_summary.py --encoder two-stream-v15 --output paper_artifacts/tables/tab2_probing/v15_egodex_summary.csv` (스크립트 인터페이스 확인 필요)
  - source: `data/probing_results/egodex/two-stream-v15/`

- [ ] **v15 LIBERO 12-cell** ep32 + ep50 → CSV
  - 위치: `tables/tab2_probing/libero_all_gaps_summary.csv` (v15 row 추가)
  - 또는 `tables/tab2_probing/v15_libero_summary.csv` 별도

- [ ] **v11 → v15 cross 4-encoder catalyst 표** 집계
  - 위치: `tables/tab6_catalyst_evidence/catalyst_evidence.csv`
  - 4 encoder × {P_t+P_tk R², BC goal SR} 한 표

### 클러스터/로컬 실행 작업 (코딩 X, sbatch/bash)

C1 학습 (5/15 시작): cluster_sessions에 기록됨, 5/18 현재 진행 중.

- [ ] **C7 VideoMAE-ours P_t+P_tk probing** — 매우 simple, 0.5 GPU·h
  - `python scripts/eval/probe_action.py --encoder videomae-ours --checkpoint <ckpt> --cls-mode patch_mean_concat_p_t_p_tk`
  - 또는 sbatch wrap
- [ ] **C5 v15 DROID 4-gap probing**
- [ ] **C4 v15 LIBERO object/goal probing**
- [ ] **v15 round2 BC rollout** (로컬, 18 ckpt × 200 episode)
  - tar 전송: `libero_bct_v15_9jobs_round2_20260514.tar` (7.6 GB, repo root)
  - 명령: `bash scripts/local/run_libero_rollouts.sh two-stream-v15-ptptk 50` (mp도 동일)
  - 결과 통합: `python scripts/eval/aggregate_libero_rollouts.py --output-dir paper_artifacts/fig3_bc_main`
- [ ] **C1 v15-vfromm 후속**: probing (cluster) + BC 학습 (cluster) + rollout (local)

### viz / 작도 작업 (코딩 또는 vector 작도)

- [ ] **fig1 v15 architecture TikZ** — `fig1_architecture/v15_architecture.tex`
- [ ] **fig2 schematic** — Panel A (3 encoder 흐름도, TikZ 또는 Inkscape)
- [ ] **fig2 bar chart 스크립트** — `scripts/viz/make_fig2_bar.py` (가칭)
- [ ] **fig3 grouped bar 스크립트** — `summary.csv` → 5 enc × 3 suite
- [ ] **fig4 2×2 recon grid 스크립트** — v11 + v15 ckpt forward + composite
- [ ] **fig5 DROID line plot 스크립트**
- [ ] **fig6 schematic + bar 스크립트**
- [ ] **fig7 loss curve 스크립트** — wandb CSV 추출 + log-scale line
- [ ] **fig8 viz script `--split-streams` 옵션 추가** — [`scripts/eval/visualize_attn_v11.py`](../scripts/eval/visualize_attn_v11.py), [`visualize_v15_no_mask.py`](../scripts/eval/visualize_v15_no_mask.py)
- [ ] **LaTeX 표 자동 생성 스크립트** — CSV → booktabs (tab1, tab2, tab3, tab4, tab5, tab6 공통)

### v11 → v15 본문 산출물 rework (2026-05-19 paper repo B-option 정리 결과)

Paper repo (`action-agnostic-paper`)의 `main/sections/experiments.tex`
§4.2 / §4.5 자리에 wire되어 있던 v11-era 산출물 4개를 본문에서 제거함 —
v11 architecture 종속 표기 (A/B/C/D = Phase-2/3 position) 와 A+B+C +0.288
champion claim 이 v15 narrative와 호환되지 않기 때문. 원본 CSV/PNG/TEX 는
본 저장소 (`paper_artifacts/probing/v11_egodex_summary.csv`,
`paper_artifacts/_archive/`) 에 그대로 보존.

v15 narrative 로 본문 §4 EgoDex within-domain probing 자리를 다시 채우려면
다음 산출물이 필요:

- [ ] **v15 epoch × R² evolution figure** (구 `fig_probing_evolution.pdf`)
  - source: v15 EgoDex 12-mode probing CSV (위 P0 entry "v15 EgoDex 12-mode")
  - v11 champion A+B+C +0.288 → v15 P_t+P_tk +0.390 catalyst signature
    중심으로 재작도. mode set 자체가 다르니 단순 swap 불가.

- [ ] **v15 12-mode bar at best epoch** (구 `fig_probing_modes_ep44.pdf`)
  - source: 동일 v15 12-mode CSV @ ep32 또는 ep50
  - v11 A/B/C/D position notation 은 v15 에 없음. v15 mode set 정의는
    `paper_artifacts/tab2_probing/README.md` 정리 후 사용.

- [ ] **v15 main probing table** (구 `tab_probing_main.tex`, 5 mode × 6 epoch)
  - v15 mode set + epoch sweep 으로 재구성.

- [ ] **v15 functional differentiation table** (구 `tab_probing_functional.tex`)
  - v11 4-position (A/B/C/D = Phase-2 motion-routing C, Phase-3 final D) 는
    v15 architecture 에 존재하지 않음.
  - v15 functional positions 새로 정의 필요 — 후보: M encoder output /
    P encoder (student) / motion-routed teacher P / p_motion decoder output.
  - 또는 student/teacher distinction 을 caption 으로 미루고 단일 position
    표기로 간략화.

진행 순서: 위 P0 entry "v15 EgoDex 12-mode export" 가 완료되면 4개
figure/table 모두 자동 생성 가능. paper repo `scripts/` 에 v15 용 build
script 신규 작성 필요 (기존 `build_probing_figures.py` 는 v11 EgoDex 전용).

---

## 4. M / P stream 분리 viz — 사용자 명시 요구

> "M stream, P stream 의 입력이나 attention 가시화 자료의 경우 하나로 통합한 것 외에 각각의 그림도 별도로 관리"

### 현황

- 통합본: [`docs/architecture/attn_v11_ep44_nomask.png`](../docs/architecture/attn_v11_ep44_nomask.png) — 4행 × 8열 grid
  - col 0/1: raw frames / col 2-5: attention overlays / col 6/7: reconstructions
- v4용 stream 분리 예시: [`docs/architecture/sample_detail/`](../docs/architecture/sample_detail/)
  - `02_m_channel_input.png`, `03_p_channel_input.png` (입력 분리)
  - `04_m_attention.png`, `05_p_attention.png` (attention 분리)
  - **v11 / v15에는 동일 분리본 미생성**

### Action

1. **viz script `--split-streams` 옵션 추가** — 한 번 forward로 5개 PNG 출력
   - `combined.png` (현 grid)
   - `m_input.png`, `p_input.png` (입력 채널만)
   - `m_attn.png`, `p_attn.png` (attention overlay만)
2. **v11 ep44 + v15 ep32 / ep50** 각각 실행 → `fig8_mp_attention/{combined, m_only, p_only}/` 채움
3. **활용 시나리오**:
   - Fig 1 architecture inset: M / P 입력 채널 mini-diagram (m_input, p_input 활용)
   - Fig 8 main: combined grid + 분리본 supplementary
   - 발표 슬라이드: 단일 stream만 강조하는 sequence 작성 시 m_only / p_only 직접 활용
   - 추가 figure (예: §5 catalyst 시각 evidence): M attention과 P attention 시간 진행을 별도 sequence로 보여줄 때

---

## 5. 5/18 ~ 5/28 timeline 제안

오늘부터 마감까지 10일 — critical path:

| Day | 작업 | 산출 |
|-----|------|------|
| 5/18 (오늘) | C1 학습 모니터링 + v15 EgoDex/LIBERO probing CSV export + C7 VideoMAE probing 잡 제출 | tab2, fig2 source 일부 |
| 5/19 | C7 결과 + DINOv2 controlled 결과 모으기, fig2 bar chart 1차 | fig2 draft |
| 5/20 | fig1 v15 TikZ 작도 시작, fig4 recon viz 스크립트 | fig1 draft, fig4 draft |
| 5/21 | C1 학습 완료 (~ 43h 종료 예상) → C2 probing 제출, fig3 v15 round2 rollout 시작 (로컬) | fig6 source 진입 |
| 5/22 | C2/C3 결과 도착 → fig6 + tab3 1차, fig5 v15 DROID probing 제출 | tab3, fig5 source |
| 5/23 | 모든 figure/table draft 완성 + LaTeX 표 자동 생성 스크립트 | paper repo에 figure push |
| 5/24-25 | §3-§5 본문 작성 (paper repo) | sections/*.tex |
| 5/26 | Abstract 제출 마감 ⚠️ | abstract.tex |
| 5/27 | Appendix (fig7, tab5) + 최종 polish | full paper |
| 5/28 | Paper 제출 마감 ⚠️ | submission |

여유 zero. 우선순위 P2는 마감 후 camera-ready 단계로 미룰 수 있음.

---

## 6. 본 문서 갱신 룰

- artifact 한 항목 채울 때마다 본 문서 status icon 갱신 (🔴 → 🟡 → ✅)
- 새 dependency 발견 시 §3 매트릭스에 추가
- C1-C9 실험 진행은 [`docs/cluster_sessions.md`](../docs/cluster_sessions.md)에 동기 기록 (잡 단위)
- timeline 변경 (마감 연장 / C1 지연 등) 시 §5 갱신
