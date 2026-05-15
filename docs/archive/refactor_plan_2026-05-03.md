# Repository Refactor Plan — 2026-05-03

**작성**: 2026-05-03 (Vault session 에서 plan 작성, 실제 실행은 별도 dev session)
**목적**: 클러스터 pull 전에 dev repo 정리. 여러 실험 (Phase 1.5 / 2 / 2.5 / 3-1 1·2차 / V3 / v12) 누적으로 임시 산출물·dated 문서·timestamped 디렉토리 비대화. 핵심 유지 + archive 분리.
**적용 시점**: V3 본 학습 클러스터 제출 직전 (`git pull` 로 클러스터 동기화 가는 직전 commit 으로 정리)

> 본 문서는 **plan only**. 실제 파일 이동·삭제·rewrite 는 별도 dev session 에서 사용자가 진행. Step 별 checklist + 영향도 + 회피해야 할 risk 명시.

---

## 1. 리팩토링 우선순위 + 영향도

| 영역 | 누적 사례 | 정리 후 효과 | 리스크 |
|------|---------|------------|--------|
| `paper_artifacts/` timestamped 디렉토리 | `libero_action_probing/` 18 dirs, `value_alignment/` 30+ dirs | 디스크 + cognitive load 감소 | raw 결과 손실 (단 `all_summary.csv` 보존됨) |
| `docs/` dated debug 문서 | `PHASE3_BCT_DEBUG_2026-05-03.md` 같은 일시 doc | 핵심 가이드 명확화 | 디버그 history 추적 어려워짐 (단 git log 로 복구) |
| `cluster_sessions.md` 누적 | 441 lines, 4월 누적 전부 | 매 세션 컨텍스트 비용 절감 | 비용 청구 history 검증 어려움 (단 sacct 로 복구) |
| `RESEARCH_PLAN.md` 비대화 | 707 lines, 폐기 phase 누적 | 핵심 로드맵 명확화 | 진화 history 손실 (단 `1. Core Idea` Vault 노트에 backup) |
| `CLAUDE.md` 비대화 | 408 lines | 매 세션 컨텍스트 비용 절감 | 정책 누락 risk |
| `data/libero/results/_*` archive prefix | 로컬 sanity 산출물, archive prefix 로 stash 됨 | (이건 gitignored, 영향 없음) | 없음 |
| `scripts/eval/` 누적 | 15 scripts, deprecated 식별 안 됨 | 사용 안 하는 script 명확화 | 의도치 않은 삭제 (단 git 으로 복구) |

---

## 2. Step-by-Step Refactor Checklist

각 Step 은 별도 commit 으로. `git pull` 충돌 회피 + 단계적 검증.

### Step A — `paper_artifacts/` timestamped 디렉토리 archive

각 evaluation 마다 같은 cell (encoder × suite) 의 여러 timestamp 산출물 누적:
- `libero_action_probing/`: 18 dirs (그 중 일부는 `_sanity` / `_streaming` 변형)
- `value_alignment/`: 30+ dirs (`_fracs`, `_v11bonly_fracs`, `_v11dprimeonly_fracs` 변형)
- `probing/` (EgoDex within-domain): 4 CSV 만 — 이미 깔끔

**정리 방침**:
- [ ] 각 (encoder × suite) cell 의 **가장 최근 정상 timestamp 1개만 유지** (sanity/_streaming 변형은 archive)
- [ ] `paper_artifacts/<eval>/_archive/` sub-folder 신설 + 폐기/sanity 산출물 이동
  ```
  paper_artifacts/libero_action_probing/
    [keep] dinov2_libero_spatial_20260430_164335/
    [keep] dinov2_libero_object_20260430_164335/
    ...
    _archive/
      [moved] videomae-ours_libero_spatial_20260430_163635_sanity/
      [moved] videomae-ours_libero_spatial_20260430_164235_sanity_streaming/
  ```
- [ ] `all_summary.csv` 는 그대로 keep (roll-up, paper hand-off contract)
- [ ] 각 evaluation 의 `README.md` 에 "최신 정상 timestamp" 표 갱신

**검증**:
- [ ] `paper_artifacts/<eval>/all_summary.csv` 가 keep 한 timestamp 의 결과와 일치하는지 (re-export 로 sanity)
- [ ] paper repo 측 figure/table 빌드 스크립트가 새 path 로 작동하는지

### Step B — `docs/` dated debug 문서 archive

dated debug 문서는 일시적 디버그 log 성격 — 핵심 가이드와 분리:
- [ ] `docs/archive/` 에 `PHASE3_BCT_DEBUG_2026-05-03.md` 이동 (기존 `docs/archive/` 에 v8_siamsimmae.md 같은 dated doc 이미 있음)
- [ ] 단 본 문서의 **핵심 정보 (cross-camera bug fix, 2차 sanity 결과, V3 계획)** 은 `RESEARCH_PLAN.md` Phase 3-1 § 또는 `cluster_sessions.md` 에 통합 후 archive
- [ ] `PROBING_GUIDE.md` 의 LIBERO Action Probing § 결과는 그대로 keep (gauge doc 성격)

**구체 통합 매핑**:
```
PHASE3_BCT_DEBUG_2026-05-03.md (249 lines) →
  RESEARCH_PLAN.md § Phase 3-1 (cross-camera fix + sanity 결과 + V3 계획) ← 핵심
  cluster_sessions.md § "2026-05-03 BC-T 2차 sanity" ← 잡 기록
  archive/ ← 원본 보존
```

### Step C — `cluster_sessions.md` 월별 archive

현재 441 lines, 4월 전부 누적:
- [ ] 4월 (2026-04 전체) 산출물 → `docs/archive/cluster_sessions_2026-04.md` 로 이동
- [ ] 본 `cluster_sessions.md` 는 **5월 이후 + 진행중 잡 + 단가/누적 표** 만 유지
- [ ] 핵심 사용 단가 / ceil 정책 / 기록 절차 (상단 ~80 lines) 는 그대로 keep — 가이드 성격

**정책 (재확인)**:
- 매월 말 직전 월 archive (5월 말에 5월 archive)
- 단 핵심 finding (Phase 1.5 v11 final champion 등) 은 archive 가 아니라 `RESEARCH_PLAN.md` 에 통합

### Step D — `RESEARCH_PLAN.md` 비대화 검토

707 lines. 폐기된 lineup (v7-big / v8 / v9) 은 이미 § "Two-Stream 설계 iteration 기록 (폐기 lineup)" 짧은 표로 압축되어 있음 — 그대로 keep.

검토 영역:
- [ ] § Phase 1.5 v6/v7-big/v8/v9/v10 진화 narrative — Vault `1. Core Idea` 에 자세한 narrative 있으므로 RESEARCH_PLAN 은 핵심 결과 표만 keep
- [ ] § Phase 3-1 (현재 진행) — 1/2차 sanity 결과 + V3 계획 갱신 (Step B 통합 포함)
- [ ] § Phase 4 (Architecture Ablation) — A1 진행 상황 갱신
- [ ] 폐기된 phase (e.g., 초기 Phase 2 task-to-change mapping) 는 archive 또는 짧은 footnote 로

목표: 707 → ~500 lines

### Step E — `CLAUDE.md` 비대화 검토

408 lines. 매 세션 컨텍스트 비용. 단 가이드 성격이라 핵심 정보 손실 시 리스크 큼:
- [ ] 폐기된 가이드 (예: 초기 폴더 구조 명세, deprecated workflow) 식별
- [ ] "현재 Phase" 섹션이 자주 갱신되는 부분 — 상위 100 lines 안에 두기 (컨텍스트 빠른 접근)
- [ ] 매 세션 비용 줄이려면 sub-doc 으로 분리 가능 (예: `docs/cluster_setup.md`, `docs/dev_workflow.md` 로 일부 이동)

**보수적 권장**: 큰 변경 없이 일단 **Phase 진행 갱신** 만. 본 refactor 의 Step D 와 같이.

### Step F — `scripts/eval/` deprecated 식별

15 scripts. 사용 안 하는 script 식별 + archive 권장:
- [ ] `analyze_delta_l.py` — 사용중인지 확인 (Phase 1 분석용?)
- [ ] `finetune_libero.py` — `finetune_libero_bct.py` 와 중복? 둘 중 하나 deprecated?
- [ ] `finetune_libero_v11.py` — `finetune_libero_bct.py` 와 중복?
- [ ] `visualize_*.py` 4개 — 모두 사용중인지 확인. 일회성이면 archive
- [ ] `probe_action.py`, `probe_action_droid.py`, `probe_action_v11.py`, `probe_action_droid_v11.py`, `probe_action_libero.py` — 5개 변형. v11 specific 과 generic 분리 OK? 통합 가능?

**정리 방침**:
- [ ] `scripts/eval/_archive/` 신설 + deprecated script 이동 (또는 git rm)
- [ ] `scripts/eval/README.md` 신설 — 각 script 의 용도 + active/archive status

### Step G — `data/libero/results/_*` (gitignored, 정보 정리만)

로컬 sanity 산출물, archive prefix 로 stash 됨:
- `_sanity_2026-05-03_pre_baseline/` (8 JSON)
- `_archive_pre_usejoint/` (1차 broken ckpt 결과)
- `_orphan_pre_2026-05-03/` (사전 비디오 stash)

**대응**: gitignored 라 commit 영향 없음. 로컬 디스크 정리는 사용자가 별도 진행. 단 `docs/artifacts.md` 에 정리 후 위치 갱신 필요.

### Step H — `src/models/` (legacy 검토)

- `two_stream.py` (v4 base?) — 현재 사용?
- `two_stream_v11.py` — active main
- `two_stream_v12.py` — post-CoRL follow-up, code only (학습 X)
- `videomae.py` — VideoMAE-ours, active

**정리 방침**:
- [ ] `two_stream.py` 가 v4/v6 legacy 면 `src/models/_archive/` 또는 그대로 keep + 주석으로 status 명시
- [ ] `two_stream_v11.py` + `two_stream_v12.py` 둘 다 keep
- [ ] `src/models/__init__.py` export 정리 (현재 v12 추가됨)

---

## 3. 클러스터 pull 전 critical TODO (V3 본 학습 직전)

본 refactor 와 별도로, **V3 본 학습 시작 전에 반드시 처리할 코드/cfg 변경**:

### 🔴 V3 학습 cfg 적용 (`scripts/eval/finetune_libero_bct.py`)
- [ ] `cfg.train.use_augmentation = True` (현재 False)
- [ ] `cfg.policy.color_aug.network = 'ColorJitterAug'` (현재 IdentityAug)
- [ ] `cfg.policy.translation_aug.network = 'TranslationAug'`, `affine_translate = 4`
- [ ] Epoch별 ckpt 보존 (ep 5/10/20/best — best 단독 → 4개 동시)

### 🔴 2-frame pair adapter augmentation 일관성 검증 (V3 학습 시작 전 1회 + 1ep 후 1회)
- [ ] `_log_first_batch_stats` 에 augmented sample 시각화 저장 코드 추가
  ```python
  # pseudocode (별도 dev session 에서 implement)
  from torchvision.utils import save_image
  for cam in ['agentview_rgb', 'eye_in_hand_rgb']:
      imgs = data['obs'][cam][0]  # (T, C, H, W) 첫 batch sample
      save_image(imgs, f'/tmp/aug_check_{cam}_ep0.png', nrow=10)
  ```
- [ ] **시각 확인**: 같은 카메라의 인접 시점 (prev=t-1, curr=t) 이 동일 augmentation 받았는가?
- [ ] 시점 간 augmentation 독립이면 → augmentation 코드 fix 후 본 학습 재개

### 🟡 V3 산출물 hand-off 위치 결정
- [ ] `paper_artifacts/libero_bct/<encoder>_<suite>_<seed>_<ts>/` 경로 컨벤션 결정
- [ ] V2 (cross-camera bug 적용 전) ckpt 와 V3 분리 표시 — `_v2_brokencam` suffix 등

### 🟡 v11-VfromM A1 ablation (33615395) 학습 진행 상황 확인
- [ ] sacct 로 진행 epoch 확인. 종료 시 V3 cfg 로 BC-T 학습 → main table A1 ablation

---

## 4. 권장 commit 순서

작업 단위별 별도 commit 으로 git history 명확화:

```
1. refactor(paper-artifacts): timestamped 디렉토리 archive 정리 (Step A)
2. docs(refactor): PHASE3 debug 정보 통합 + archive 이동 (Step B)
3. docs(cluster): cluster_sessions.md 4월 archive 분리 (Step C)
4. docs(plan): RESEARCH_PLAN.md Phase 3-1 V3 갱신 + 비대화 정리 (Step D)
5. (선택) docs(claude): CLAUDE.md 핵심만 유지 (Step E)
6. (선택) refactor(scripts): scripts/eval deprecated archive (Step F)
7. fix(libero-bct): V3 cfg + augmentation 일관성 검증 (V3 학습 직전)
```

---

## 5. 회피해야 할 risk

- [ ] ⚠️ **paper_artifacts/ raw 결과 영구 삭제 금지** — `_archive/` 로 이동만. 향후 검증 / 재분석에 필요할 수 있음
- [ ] ⚠️ **CLAUDE.md / RESEARCH_PLAN.md 의 핵심 정보 (current Phase, v11 ep44 결과 등) 손실 금지** — 갱신만, 삭제 X
- [ ] ⚠️ **클러스터 pull 시 git conflict 가능성** — 클러스터에서도 사용자가 작업 중이면 (예: 진행 중 잡의 결과 파일이 untracked 로 누적) pull 전에 충돌 검토 필요
- [ ] ⚠️ **사용자가 다른 곳에서 작업 중인 미커밋 변경사항 보호** — 본 plan 의 모든 step 은 사용자 결정 후 진행. 임의 staging 금지
- [ ] ⚠️ **archive 후 Vault cross-link 깨짐 검토** — Vault `Sources/papers/`, `Projects/Action-Agnostic Paper/` 에 dev repo 경로 link 다수 있음. 이동된 파일은 link 갱신 필요

---

## 6. 본 refactor 후 작업 흐름

1. **로컬 dev repo 에서 refactor commit 7개** (Step A~F + V3 cfg)
2. **로컬에서 push 전 검증**: `git status` 깨끗, paper_artifacts/all_summary.csv 정상, 핵심 docs 정보 손실 없음
3. **Push 후 클러스터에서 `git pull --ff-only`** — 클러스터에서 작업하던 미커밋 변경 있으면 stash 후 pull, stash apply
4. **클러스터에서 V3 본 학습 sbatch 제출** — augmentation + multi-seed cfg 로
5. **로컬 H100 은 인터랙티브 디버그 / 시각화 / 빠른 iteration 용으로 유지**

---

## 7. Cross-references

- Vault 결정 source: `Projects/Action-Agnostic Paper/3. Experiments § Phase 3-1` + `우려사항 및 대응방안 § 14`
- 통합 대상 docs: `PHASE3_BCT_DEBUG_2026-05-03.md`, `cluster_sessions.md`, `RESEARCH_PLAN.md`
- Hand-off contract: `paper_artifacts/README.md`, `docs/artifacts.md`
- V3 학습 plan: `RESEARCH_PLAN.md` § Phase 3-1 V3 + 본 문서 § 3
