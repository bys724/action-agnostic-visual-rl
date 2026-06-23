# Claude Development Guide

**세션 시작 시 필독**: 아래 핵심 문서로 작업 맥락을 파악하세요.

## 핵심 문서 (업데이트 우선)

1. **`docs/RESEARCH_PLAN.md`** — 전체 연구 계획 및 현재 phase (마스터 문서)
2. **`docs/paper1_input_prior_plan.md`** — **Paper 1 (ICRA, Input-Prior)** 계획·근거
3. **`docs/PROBING_GUIDE.md`** — Action probing 실험 가이드 + 결과
4. **`docs/eval_protocols.md`** — **평가 프로토콜 단일 출처** (6벤치 정규 조건 + parity 체크리스트 + 오류 이력). 새 모델 비교 전 필독
5. **`docs/setup/LIBERO_TEST_GUIDE.md`** — LIBERO 평가
6. **`docs/cluster_sessions.md`** — IBS 클러스터 세션 로그 (비용 청구 대조)
7. **`docs/artifacts.md`** — 클러스터/로컬 산출물·데이터셋 경로 인덱스
8. **`docs/v15b_retraining_status.md`** — Parvo(code v15b) 현황·본학습 명령(§6)·no-M ablation(§11)
9. 보조: **`docs/FILE_INDEX.md`**(파일 인덱스) · **`docs/WORKFLOW_COMMANDS.md`**(실행 명령) · **`docs/TROUBLESHOOTING.md`**(사고/재발 가드)

**문서 작성 원칙**: 새 문서보다 기존 핵심 문서 **업데이트** 우선 · 핵심 계획은 `RESEARCH_PLAN.md` · 일회성 정보는 git commit message.

## 프로젝트 개요

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?
EgoDex로 action-agnostic 시각 표현 사전학습 → LIBERO 로봇 조작으로 평가.

## 명명 · 2논문 구조 (정규 — source of truth)

> Obsidian `Projects/Action-Agnostic Paper/` + `Projects/Input-Prior Robot Representation (ICRA)/`와 1:1. 이 섹션이 명명·프레이밍의 단일 출처 — 다른 docs(eval_protocols·RESEARCH_PLAN·v15b_status)가 역참조.
> 🔄 **기능 서술명 reorg (2026-06-23, 확정)**: Parvo→**MS-JEPA**, no-M→**Image MAE**, §9→**MCP-MAE**, Paper1→**Edge-Prior Image MAE** (2-축: Edge-Prior/no-Sobel × Image MAE/MS-JEPA/MCP-MAE). 표·근거 = [docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md). 본 섹션 본문의 `Parvo` 표현 전체 치환 + 코드 식별자(`v15` 등) rename은 **본학습 후 일괄**(deferred).

**고유명은 논문 핵심 모델 하나에만 부여. 나머지(의도와 달랐던 버전 포함)는 버전명 유지.**

- **`Parvo`** — 논문 핵심 모델. two-stream M/P scaffold: M(motion)이 학습 중 P(appearance)를 scaffold, 배포 시 P만 남음. **현재 구현 = code `v15b`** (student-anchor, M→P gradient 연결), `main` 브랜치. 코드 식별자 rename은 본학습 후.
- **v15** = Parvo 직전 divergent 버전. motion routing이 student P에 **gradient=0 (no-op)** → **paper-main 아님**, `paper-corl2026` 동결. ⚠️ v15의 `+0.390(P_t⊕P_tk)`을 motion routing 인과로 귀속한 것은 **철회됨**(no-op, artifact 의심) — 재귀속 금지.

| | **Paper 1 (ICRA)** — Input-Prior | **Paper 2 (AAAI)** — Action-Agnostic |
|---|---|---|
| 핵심 모델 | 단일프레임 image MAE (Sobel+RGB) = Parvo의 P stream 단독 | **Parvo** (code: v15b) |
| 주장 | image MAE(Sobel+RGB) **> VideoMAE** | M/P 구조적 cross-stream(scaffold) bias가 표현을 개선 |
| 상태 | **좁지만 입증** (matched 아님, ablation 필요) | **미입증, 검증 중** (지지 증거 0) |

- **검증 질문** (Parvo): M→P gradient를 실제 연결한 Parvo가 **input-only baseline(Paper 1 image MAE)을 넘는가**. 못 넘으면 "multi-frame MAE concat이 강한 단순 baseline"으로 정직 재서술.
- catalyst→scaffold 용어 전환·인과 철회 history: `docs/RESEARCH_PLAN.md` · `docs/v15b_retraining_status.md` §1.

## 관련 Obsidian 노트

핵심 아이디어·논문 작성·실험 설계는 Vault에 정리됨. 코드 작업 중 맥락 보충 시 참조:

| 카테고리 | 경로 |
|---------|------|
| 프로젝트 메인 (Paper 2, AAAI) | `Projects/Action-Agnostic Paper/README.md` |
| Paper 1 (ICRA, Input-Prior) | `Projects/Input-Prior Robot Representation (ICRA)/{README, 1. Core Claim & Plan}.md` |
| 단계별 정리 | `Projects/Action-Agnostic Paper/{1. Core Idea … 5. Project Management}.md` |
| 핵심 개념 | `Concepts/{Action-Agnostic Pretraining Framework, Two Visual Pathways, Two-Stream Image Preprocessing, …}.md` |
| 논문 노트 | `Sources/papers/{EgoDex (2025), V-JEPA 2 (2025)}.md` |

(Vault 루트: `/Users/bys724/LocalVault/Obsidian Vault/`)

### Vault 편집 가드레일 (load-bearing)

- **여기(저장소)에서 Vault 노트**: 읽기 자유, 가벼운 추가(한두 줄·링크) OK. **신규 노트 / 폴더 변경 / 대규모 리팩토링은 Vault 세션에서** (컨벤션 자동 적용 위해).
- **🔁 Vault 세션에서 이 저장소**: 문서·진행 기록 편집 OK. **❌ 코드·실행 설정(`src/`, `scripts/`, sbatch, Dockerfile, requirements) 편집 금지** — 이 저장소 별도 세션. ⚠️ CLAUDE.md·README 구조/정책 변경도 이 저장소에서.

## 실행 환경

Python 코드(`scripts/pretrain.py`, `src/` 등)는 환경 무관, bash launcher만 환경별 분리.

| 환경 | 컨테이너 | GPU | 데이터 | Launcher |
|------|---------|-----|--------|----------|
| **로컬 워크스테이션** | Docker (`dev-env`) | H100 × 2 (DataParallel) | `/mnt/data/...` | `scripts/local/` |
| **IBS 클러스터** (olaf) | Apptainer (예정) | H100 × 4/node, ≤2 nodes | `/proj/external_group/mrg/...` (GPFS) | `scripts/cluster/` |

- **역할 분담**: 로컬 = probing·시각화·LIBERO finetune+rollout (빠른 turnaround) / 클러스터 = full pre-training (장시간 8 GPU DDP).
- 저장소(GPFS/scratch)·stage-in/out 상세: [scripts/cluster/README.md](scripts/cluster/README.md) · [docs/artifacts.md](docs/artifacts.md).

## 워크플로우

**활성 모델 = `Parvo`** (code v15b, student-anchor, `main`). Paper 2 검증 대상, 본학습 보류 중. VideoMAE-ours = controlled baseline. 명명은 위 "명명 · 2논문 구조".

1. **EgoDex Pre-training** — Parvo 본학습 명령은 [docs/v15b_retraining_status.md](docs/v15b_retraining_status.md) §6 단일 출처.
2. **Action Probing** — 학습 표현이 행동 정보를 인코딩하는지 R²로 검증 → [docs/PROBING_GUIDE.md](docs/PROBING_GUIDE.md).
3. **LIBERO BC-T Fine-tuning & Rollout** — frozen encoder + 공식 BCTransformerPolicy → [docs/setup/LIBERO_TEST_GUIDE.md](docs/setup/LIBERO_TEST_GUIDE.md).

실행 명령 모음(sanity·download·probe·finetune·rollout): [docs/WORKFLOW_COMMANDS.md](docs/WORKFLOW_COMMANDS.md).

## 주요 파일

전체 파일 인덱스(공통 인프라 / Parvo / baselines / reference / deprecated): [docs/FILE_INDEX.md](docs/FILE_INDEX.md).

## 개발 원칙 (프로젝트 고유)

- **검증 필수**: 새 스크립트는 sanity test로 검증 후 배포. sanity 잡은 반드시 `CHECKPOINT_SUFFIX` 사용 (auto-resume 상태 오염 방지).
- **Best Practice 사전 조사**: 새 구현(특히 데이터 로딩·학습 파이프라인) 전 공식 문서/권장 방식 조사 → 비효율 재작업 방지.
- (일반 코딩·문서·언어 원칙은 User CLAUDE.md.)

## 클러스터 세션 로깅 (필수)

- **🔴 잡 제출과 동시에 같은 turn 안에서 [docs/cluster_sessions.md](docs/cluster_sessions.md) 표 업데이트** — 보고/다음 질문 전 반드시. 컨텍스트 압축 누락 방지, 여러 잡 일괄 업데이트 금지.
- **잡 종료 시**: `sacct -j <JobID> --format=JobID,JobName,Partition,AllocTRES,Start,End,Elapsed,State` → "완료" 표로 이동 + 자원·시간(`n_gpus × elapsed_h` 또는 `n_nodes × elapsed_h`) 기입.
- 공식 단가·월누적 ceil 청구 방식은 `cluster_sessions.md` 상단이 단일 출처. 로그인 노드 활동은 미과금.
- **Slurm `--time`** (timeout 학습손실 반복 사고 가드): 본학습/finetune = partition max `2-00:00:00`; 추정 시 `총ep × ep당시간 × 1.3`; multi-encoder는 가장 느린 모델 기준 통일. 상세 5단계 → [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## 재발 사고 가드 (must-check)

- **🔴 Train↔inference preprocessing parity**: 새 eval/adapter/wrapper마다 학습 input range·channel·정규화 = inference transform 일치 점검 (EgoDex = `[0,1]` raw `/255.0`, ImageNet Normalize 금지). 위반 시 OOD inference로 SR 폭락(2026-05-25, 30잡 무효). 체크리스트 단일 출처 → [docs/eval_protocols.md](docs/eval_protocols.md) §0.
- 기타 재발 사고(DDP 함정·num_workers hang·DINO trap·3-frame mismatch 등) + 해결 완료 기록 → [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## 데이터셋 전처리

새 데이터셋은 샘플 테스트 → 결정 기록(`docs/preprocessing/`) → 전체 추출 → 검증. 절차·기존 사례(EgoDex/DROID/Ego4D) → [docs/preprocessing/README.md](docs/preprocessing/README.md).

## 현재 상태 (2026-06-22)

> 2논문 분리. 상세 phase·이력은 마스터 문서로 위임 — 본 섹션은 스냅샷. 명명 정규 출처 = 위 "명명 · 2논문 구조".

- **Paper 1 (ICRA)**: 단일프레임 image MAE(Sobel+RGB) > VideoMAE = **좁지만 입증**. 남은 일 = ablation(RGB-only vs Sobel+RGB, VideoMAE fairness) + real-robot.
- **Paper 2 (AAAI)**: scaffold 가설 **미입증**. **Parvo**(code v15b)로 M→P gradient 연결해 검증 중.
- **최신 (Parvo BC-T, 2026-06-21)**: LIBERO BC avg **0.785** ≈ v15-ptptk(0.777) — 붕괴방어·no-Sobel이 SR 무손실 유지. frozen baseline(SigLIP 0.855/VC-1 0.821/DINOv2 0.811) 미달, spatial 약점. → v15b_status §10.
- **다음**: **no-M ablation** (`--v15-no-motion`, routing-off=two-frame image MAE) 구현 완료. 본 실행 후 Parvo vs no-M 동일 LIBERO BC + OOD eval로 M 기여 격리 (v15b_status §11). 본학습 재제출(§6) 보류 중.

상세: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)(마스터) · [docs/v15b_retraining_status.md](docs/v15b_retraining_status.md) · [docs/eval_protocols.md](docs/eval_protocols.md) · [docs/cluster_sessions.md](docs/cluster_sessions.md).
