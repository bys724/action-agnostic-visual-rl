# Claude Development Guide

**세션 시작 시 필독**: 아래 핵심 문서를 먼저 읽고 작업 맥락을 파악하세요.

## 핵심 문서 (업데이트 우선)

1. **`docs/RESEARCH_PLAN.md`** - 전체 연구 계획 및 현재 phase (마스터 문서)
2. **`docs/paper1_input_prior_plan.md`** - **Paper 1 (ICRA, Input-Prior)** 계획·근거
3. **`docs/paper_experiments_plan.md`** - **Paper 2 (AAAI, Action-Agnostic)** 실험 plan (구 v15-era, Parvo 검증으로 재정렬 중)
4. **`docs/PROBING_GUIDE.md`** - Action probing 실험 가이드 + 결과
5. **`docs/setup/LIBERO_TEST_GUIDE.md`** - LIBERO 평가
6. **`docs/cluster_sessions.md`** - IBS 클러스터 sbatch/salloc 세션 로그 (비용 청구 대조)
7. **`docs/artifacts.md`** - 클러스터/로컬 워크스테이션 산출물·데이터셋 경로 인덱스 (양쪽 작업 시 참조)

**문서 작성 원칙**:
- 새 문서를 만들기보다 기존 핵심 문서 **업데이트** 우선
- 핵심 계획은 `RESEARCH_PLAN.md`에 집중 관리
- 일회성 정보는 git commit message에 기록

## 프로젝트 개요

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?

**핵심 아이디어**:
- EgoDex로 행동-독립적(action-agnostic) 시각 표현 사전학습
- LIBERO에서 로봇 조작 태스크로 평가

## 명명 · 2논문 구조 (2026-06-11 기준, 정규)

> Obsidian `Projects/Action-Agnostic Paper/` + `Projects/Input-Prior Robot Representation (ICRA)/`와 1:1 대응. 이 섹션이 명명·프레이밍의 단일 출처(source of truth). 이전 문서의 "v15 = paper main / catalyst 인과 확정" 프레임은 **폐기됨**.

**고유명은 논문 핵심 모델 하나에만 부여. 나머지(의도와 달랐던 버전 포함)는 버전명 유지.**

- **`Parvo`** — 논문 핵심 모델. two-stream M/P scaffolded 설계: M(magnocellular/motion)이 학습 중 P(parvocellular/appearance)를 **scaffold**, 배포 시 P만 남음. **현재 구현 = code `v15b`** (student-anchor, M→P gradient 연결), `main` 브랜치. 설계 개선으로 코드 버전이 올라가도 모델 정체성은 Parvo. 코드 식별자(`two_stream_v15b` 등) rename은 본학습 재제출 후 별도 단계 — 이번엔 문서·개념만.
- **이전/divergent 버전 = 버전명 유지** (v4…v15). 특히 **v15** = Parvo의 의도와 달랐던 직전 버전 — motion routing이 student P에 **gradient=0 (no-op)**, `paper-corl2026` 동결(CoRL 미제출).

**2논문 분리 (2026-06-11)** — 저장소는 단일 트리 유지, 각 실험에 `[Paper1]`/`[Paper2]` 구분 표기:

| | **Paper 1 (ICRA)** — Input-Prior | **Paper 2 (AAAI)** — Action-Agnostic |
|---|---|---|
| 핵심 모델 | 단일프레임 image MAE (Sobel+RGB) = Parvo의 P stream 단독 | **Parvo** (code: v15b) |
| 주장 | image MAE(Sobel+RGB) **> VideoMAE** | M/P 구조적 cross-stream(scaffold) bias가 표현을 개선 |
| 상태 | **좁지만 입증** (matched 아님, confound 미해소 → ablation 필요) | **미입증, 검증 중** (지지 증거 현재 0) |

**catalyst → scaffold, 그리고 인과 철회**:
- **메타포 전환** (05-20): 화학 catalyst는 kinetic only(rate만 영향)라 부정확 → "학습 중 임시 지원 + 배포 시 제거(motion decoder/routing/teacher branch)" = **scaffold**가 정확.
- **인과 철회** (06-11): **v15**(직전 버전)는 motion routing이 student P에 gradient=0 **no-op**이라, `+0.390(P_t⊕P_tk)`을 motion routing 인과로 귀속한 것은 **오류 → 철회**. 유력 대안 = **multi-frame MAE concat + probe artifact**. M-stream 기여 = **0**.
- **검증 질문** (Parvo = code v15b): M→P gradient를 실제 연결한 Parvo가 scaffold로 **input-only baseline(Paper 1의 image MAE)을 넘는가**. 못 넘으면 "multi-frame MAE concat이 강한 단순 baseline"으로 정직 재서술.

## 관련 Obsidian 노트

이 프로젝트의 핵심 아이디어·논문 작성·실험 설계는 Vault에 정리됨. 코드 작업 중 맥락 보충이 필요하면 참조:

| 카테고리 | 경로 |
|---------|------|
| 프로젝트 메인 인덱스 (Paper 2, AAAI) | `Projects/Action-Agnostic Paper/README.md` |
| Paper 1 (ICRA, Input-Prior) | `Projects/Input-Prior Robot Representation (ICRA)/{README, 1. Core Claim & Plan}.md` |
| 단계별 정리 | `Projects/Action-Agnostic Paper/{1. Core Idea, 2. Technical Details, 3. Experiments, 4. Paper Writing, 5. Project Management}.md` |
| 우려사항 | `Projects/Action-Agnostic Paper/우려사항 및 대응방안.md` |
| 핵심 개념 | `Concepts/{Action-Agnostic Pretraining Framework, Two Visual Pathways, Two-Stream Image Preprocessing, Pixel-wise Channel Fusion for Behavior Representation, Target LayerNorm (V-JEPA)}.md` |
| 논문 노트 | `Sources/papers/{EgoDex (2025), V-JEPA 2 (2025)}.md` |

(Vault 루트: `/Users/bys724/LocalVault/Obsidian Vault/`)

### Vault 편집 정책 (이 프로젝트에서 자주 발생)

이 프로젝트는 Vault 노트 ↔ 저장소의 양방향 작업이 잦음. 가드레일:

**여기(이 저장소)에서 Vault 노트 편집**:
- ✅ 읽기 자유 (filesystem MCP)
- ✅ 가벼운 추가 (한두 줄, 링크 걸기) 허용
- ⚠️ 신규 노트 / 폴더 구조 변경 / 대규모 노트 리팩토링은 Vault 작업공간에서 별도 세션
  - 이유: Vault 컨벤션(`Projects/` 숫자.md 단계 구분, 백링크 스타일, 이모지 규약)을 자동 적용받지 못함

**🔁 Vault 작업공간에서 이 저장소 편집**:
- ✅ 문서·진행 기록 OK: `docs/cluster_sessions.md` 잡 로그, `docs/RESEARCH_PLAN.md` 진행 갱신, Todo 등
- ❌ 코드·실행 설정 편집 금지: `src/`, `scripts/`, sbatch, Dockerfile, requirements 등 — 이 저장소에서 별도 세션
- ⚠️ CLAUDE.md·README의 구조/정책 변경: 이 저장소에서 처리

User CLAUDE.md의 "Vault 노트 양방향 편집 가드레일" 정책을 이 프로젝트에 구체화한 것.

## 실행 환경

이 프로젝트는 **두 개의 실행 환경**에서 동작합니다. Python 코드 (`scripts/pretrain.py`, `scripts/data/`, `scripts/eval/`, `src/`)는 환경 무관이며, bash launcher만 환경별로 분리되어 있습니다.

| 환경 | 위치 | 컨테이너 | GPU | 데이터 경로 | Launcher |
|------|------|---------|-----|------------|----------|
| **로컬 워크스테이션** | 외부 워크스테이션 | Docker (`dev-env`) | H100 × 2 (DataParallel) | `/mnt/data/...` | `scripts/local/` |
| **IBS 클러스터** (olaf) | Slurm cluster | Apptainer (예정) | H100 × 4 per node, 최대 2 nodes | `/proj/external_group/mrg/datasets/...` (GPFS) | `scripts/cluster/` |

**클러스터 저장소 요약** (관리자 확인, 2026-04-09):
- `/proj` GPFS: ~16 GB/s read/write, 모든 노드, 영구
- `/scratch/tmp/`: 로컬 NVMe (~6.8 GB/s read, ~2 GB/s write), **GPU 노드(olaf-g)에만**, 전체 쓰기 가능 (관리자 설정, 2026-04-10)
- 이 클러스터는 GPFS가 더 빠름 → 일반 작업은 `/proj` 직접 사용. scratch는 small file random I/O에서만 유리
- 자세한 내용 및 stage-in/out 패턴: [scripts/cluster/README.md](scripts/cluster/README.md)

**역할 분담** (2026-04-09):
- **로컬**: probing, 시각화, LIBERO fine-tuning + rollout (인터랙티브, 빠른 turnaround)
- **클러스터**: full pre-training (장시간 백그라운드, 8 GPU DDP)

자세한 클러스터 사용법은 [scripts/cluster/README.md](scripts/cluster/README.md) 참고.

## 워크플로우

### 1. EgoDex Pre-training

**활성 모델**: **`Parvo`** = 논문 핵심 two-stream M/P 모델 (현재 구현 code `v15b`, student-anchor, M→P gradient 연결). `main` 브랜치, Paper 2 검증 대상, 본학습 보류 중([docs/v15b_retraining_status.md](docs/v15b_retraining_status.md)). 직전 **v15**(teacher-anchor, no-op)는 `paper-corl2026` 동결. **VideoMAE-ours** = controlled baseline. v6/v10/v11은 paper §method history 인용용. 명명·2논문 구조는 상단 "명명 · 2논문 구조" 섹션 참조.

**v15 본 학습 명령** (cluster_sessions §2026-05-15 동일):
```bash
sbatch --partition=AIP_long --nodes=2 --ntasks-per-node=4 --gres=gpu:4 \
    --time=10-00:00:00 \
    --export=ALL,MODEL=two-stream-v15,EPOCHS=50,BATCH_SIZE_PER_GPU=32,\
SPLITS=part1,part2,part3,part4,part5,\
V11_ROUTING_MODE=v_from_p,\
V15_LAMBDA_PRED_WARMUP_START=0.01,V15_LAMBDA_PRED_WARMUP_EPOCHS=10,\
V15_LAMBDA_M_JEPA_WARMUP_START=0.01,V15_LAMBDA_M_JEPA_WARMUP_EPOCHS=10,\
V15_LAMBDA_COMPOSE_WARMUP_START=0.01,V15_LAMBDA_COMPOSE_WARMUP_EPOCHS=10,\
NUM_WORKERS=8 \
    scripts/cluster/pretrain.sbatch
```

- `V11_ROUTING_MODE=v_from_m` + `CHECKPOINT_SUFFIX=vfromm` → paper §5.1 main ablation (C1)
- Sanity 권장: `sbatch --export=ALL,MAX_VIDEOS=50,EPOCHS=3,BATCH_SIZE=8,NUM_WORKERS=4 scripts/cluster/sanity_v15.sbatch` (4분 28초 기준)

**데이터 다운로드** (EgoDex CDN 직접):
```bash
bash scripts/local/download_egodex.sh   part2 part3 part5  # 로컬
bash scripts/cluster/download_egodex.sh part2 part3 part5  # 클러스터
```

### 2. Action Probing (사전학습 완료 후)

학습된 표현이 행동 정보를 인코딩하는지 검증

```bash
# EgoDex (within-domain)
python scripts/eval/probe_action.py \
    --encoder two-stream \
    --checkpoint /mnt/data/checkpoints/two_stream/.../best_model.pt \
    --egodex-root /mnt/data/egodex --frames-root /mnt/data/egodex_frames \
    --egodex-split part4 --gap 10 --cls-mode patch_mean_concat

# DROID (cross-domain, primary)
python scripts/eval/probe_action_droid.py \
    --encoder two-stream --checkpoint <ckpt> \
    --droid-root /mnt/data/droid_frames/ext1 --gap 10
```

자세한 내용은 `docs/PROBING_GUIDE.md` 참고

### 3. LIBERO BC-Transformer Fine-tuning & Rollout

표준 평가 = **BC-T policy head** (frozen encoder + LIBERO 공식 `BCTransformerPolicy`). v15 신규 어댑터(`two-stream-v15-ptptk` / `two-stream-v15-mp`)로 매트릭스 학습.

```bash
# Cluster (sbatch, V3 cfg = use_joint=True + augmentation)
ENCODER=two-stream-v15-ptptk SUITE=libero_spatial SEED=0 \
    sbatch scripts/cluster/finetune_libero_bct.sbatch

# Local rollout (closed-loop SR)
bash scripts/local/run_libero_rollouts.sh two-stream-v15-ptptk 50
```

자세한 가이드는 [`docs/setup/LIBERO_TEST_GUIDE.md`](docs/setup/LIBERO_TEST_GUIDE.md) · [`scripts/eval/README.md`](scripts/eval/README.md) 참고

## 주요 파일

### 공통 인프라

| 파일 | 용도 |
|------|------|
| `scripts/pretrain.py` | Pre-training 메인 스크립트 (env-agnostic, 모든 모델 dispatch) |
| `src/training/pretrain.py` | Pre-training 루프 (모든 모델 공유) |
| `src/datasets/egodex.py`, `src/datasets/droid.py`, `src/datasets/libero.py` | 데이터셋 로더 |
| `scripts/local/download_egodex.sh`, `scripts/cluster/download_egodex.sh` | EgoDex CDN 다운로드 |
| `scripts/cluster/extract_egodex.sbatch` | 프레임 추출 sbatch (144 CPU 병렬) |
| `scripts/data/extract_droid_frames.py`, `extract_droid_actions.py` | DROID 전처리 |
| `scripts/cluster/pretrain.sbatch`, `scripts/cluster/README.md` | 클러스터 학습 launcher |

### 🏆 Paper main (v15)

| 파일 | 용도 |
|------|------|
| `src/models/two_stream_v15.py` | Two-Stream v15 (predictor-only V-JEPA + V-JEPA-M Option B + L_compose) |
| `scripts/cluster/sanity_v15.sbatch` | v15 sanity (단일 GPU, 50vid × 3ep) |
| `src/encoders/adapters/two_stream_v15_pt_ptk.py` | BC-T 어댑터 (옵션 B, P_t ⊕ P_tk) |
| `src/encoders/adapters/two_stream_v15_mp.py` | BC-T 어댑터 (C-variant, M ⊕ P_curr) |
| `scripts/eval/probe_action.py` | EgoDex action probing (v15 mode 신규 추가: `patch_mean_concat_p_t_p_tk` 등) |
| `scripts/eval/probe_action_libero.py` | LIBERO action probing (v15 mode 신규 추가) |
| `scripts/eval/finetune_libero_bct.py` | LIBERO BC-T 학습 (V3 cfg: use_joint + augmentation) |
| `scripts/cluster/finetune_libero_bct.sbatch` | BC-T sbatch launcher |
| `scripts/local/run_libero_rollouts.sh` | 로컬 closed-loop rollout 런처 |
| `scripts/eval/aggregate_libero_rollouts.py` | Rollout 결과 → `paper_artifacts/libero_rollout/` 통합 |
| `scripts/eval/visualize_v15_no_mask.py` | v15 nomask reconstruction 시각화 |
| `scripts/eval/diagnose_v15_collapse.py`, `scripts/cluster/diagnose_v15.sbatch` | v15 collapse 진단 |

### Active baselines

| 파일 | 용도 |
|------|------|
| `src/models/videomae.py` | VideoMAE baseline (active controlled comparison) |
| `src/encoders/adapters/single_frame.py` | DINOv2 / SigLIP / VC-1 어댑터 |
| `src/encoders/adapters/vjepa2.py` | V-JEPA 2.1 어댑터 (probing only, BC main 제외) |

### Reference (paper §method history 인용용)

| 파일 | 용도 |
|------|------|
| `src/models/two_stream.py` | Two-Stream v4~v10 (v6 ep8 = 이전 챔피언 +0.259) |
| `src/models/two_stream_v11.py` | Two-Stream v11 (motion-routing + dual-target, ep44 +0.288) |
| `src/encoders/adapters/two_stream_v11.py` | v11 BC-T 어댑터 (A+D' mode) |
| `src/encoders/adapters/videomae.py` | VideoMAE BC-T 어댑터 |
| `scripts/eval/visualize_attn_v11.py`, `visualize_attn_compare.py`, `visualize_attn_rotation.py` | v6/v10/v11 attention viz |
| `scripts/eval/probe_action_v11.py`, `probe_action_droid_v11.py` | v11 별도 probing 스크립트 |
| `scripts/eval/value_alignment.py`, `analyze_delta_l.py` | Phase 2.5 value alignment (negative result) |

### Deprecated (paper 미사용, 코드만 보존)

| 파일 | 비고 |
|------|------|
| `src/models/two_stream_v12.py` | v11 + CLS semantic residual. sanity cls_p collapse |
| `src/models/two_stream_v13.py` | Dual-frame + DINO global CLS. ep10+ uniform collapse |
| `src/models/two_stream_v14.py` | Stream-wise (P=MAE+V-JEPA, M=DINO). ep20 cancel, EgoDex probing R²=-0.065 |
| `scripts/cluster/sanity_v{12,13,14}.sbatch` | v12/13/14 sanity launcher |
| `scripts/eval/visualize_attn_v{13,14}.py`, `visualize_v{13,14}_no_mask.py` | v13/14 viz |
| `scripts/eval/finetune_libero.py`, `finetune_libero_v11.py` | 이전 LIBERO finetune (Phase 3 BC-T로 대체) |

## 개발 원칙

1. **문서 업데이트 우선**: 새 문서보다 핵심 문서 업데이트
2. **간결한 코드**: 실험 우선, 과도한 추상화 피하기
3. **검증 필수**: 새 스크립트는 sanity test로 검증 후 배포
4. **Best Practice 참고 필수**:
   - 새로운 구현 전에 **반드시** 공식 문서/권장 방법/샘플 코드 조사
   - 특히 데이터 로딩, 학습 파이프라인 등 성능에 영향을 주는 부분
   - 비효율적 구현으로 재작업하지 않도록 사전 조사 우선

## 클러스터 세션 로깅 (필수)

IBS 클러스터에서 sbatch/salloc 잡을 다룰 때마다 [`docs/cluster_sessions.md`](docs/cluster_sessions.md)를 업데이트해야 함. 비용 청구 대조용.

- **공식 단가** (PDF 기준, VAT 별도): H100 = 61,000원/GPU·일, CPU = 7,000원/노드·일
- **청구 방식**: 월 누적 ceil. 한 달의 모든 GPU·초(또는 노드·초) 합산 → 일수 환산 → ceil → 단가 곱셈. **잡 단위 ceil 아님** → 짧은 sanity test 누적해도 부담 적음
- **🔴 잡 제출과 동시에 같은 응답 안에서 cluster_sessions.md 업데이트**: 사용자에게 보고하기 전, 다음 질문을 받기 전에 반드시 표 추가. 컨텍스트 압축이 중간에 일어나도 누락되지 않도록 **즉시 = 같은 turn**. 여러 잡 묶어서 나중에 일괄 업데이트 금지.
- **잡 종료 확인 시**: `sacct -j <JobID> --format=JobID,JobName,Partition,AllocTRES,Start,End,Elapsed,State` 로 시각 조회 → "완료" 표로 이동, **자원·시간** (`n_gpus × elapsed_h` 또는 `n_nodes × elapsed_h`) 기입. 비용은 월말 누적 합산 시 계산
- **저장소**: mrg 그룹 10 TB 이미 할당받음 → 추적 불필요. 10 TB 초과 증설 시에만 기록
- **로그인 노드 활동(다운로드, 파일 작업 등)은 미과금**: 기록 대상 아님
- 세션 시작/마무리 시 이 문서 한 번 훑어보기 — 잊은 항목 없는지 확인

### `--time` 결정 가이드 (잡 제출 전 필수 체크)

Slurm timeout으로 학습 손실하는 사고가 반복적으로 발생. 잡 제출 전 다음 절차:

1. **Epoch당 시간 추정**: 같은 모델/데이터의 sanity test 또는 과거 잡 elapsed 확인. 추정이 어려우면 **`--time`은 partition max로** (AIP/AIP_long 모두 max 가능).
2. **계산식**: `필요 시간 = 총 ep × 평균 epoch당 시간 × 1.3 (안전 마진)`. 데이터 로딩/I/O 변동, ckpt 저장 오버헤드 흡수용.
3. **multi-encoder 동시 제출 시 주의**: encoder별 epoch당 시간이 크게 다를 수 있음 (실제 사례: BC-T 5 encoder 1082~2731s, **2.5배 차이**). 모든 잡을 가장 느린 encoder 기준으로 통일된 `--time`으로 제출.
4. **첫 1-2 epoch 끝나면 ETA 재검산**: `잔여 ep × per_ep_time + current wall_time > TIME_LIMIT` 이면 즉시 cancel + 재제출. 학습 8h 손실이 timeout으로 ckpt가 부분만 남는 것보다 paper main table 일관성에 유리.
5. **기본값 권장**: 본 학습/fine-tune은 `--time=2-00:00:00` (AIP partition 최대). probing/sanity처럼 짧은 잡만 1-3h.

## 데이터셋 전처리 워크플로우

새 데이터셋을 학습에 사용하기 전, 아래 프로세스를 따름:

1. **샘플 테스트**: 소수 영상(3~5개)으로 crop/resize 옵션별 결과 비교
2. **결과 기록**: 비교 이미지 + 결정 근거를 `docs/preprocessing/` 하위에 기록
3. **전체 추출**: 결정된 설정으로 전체 데이터셋 프레임 추출
4. **검증**: 샘플 다운로드하여 품질 확인

### 기존 사례
- **EgoDex** (1920x1080): 센터크롭 → 256x256
- **DROID** (180x320): 리사이즈 → 256x256 (crop 없음)
- **Ego4D** (가변): 다운로드 진행 중, 전처리 결정 미정

## 현재 Phase (2026-06-11)

**scaffold 인과 철회 + 2논문 분리. 활성 작업 = Parvo(code v15b) 본학습 재제출 준비.** 자세한 명명·프레이밍은 상단 "명명 · 2논문 구조" 섹션이 정규 출처.

- **CoRL 2026 미제출** (비용). 구 "v15 only main + catalyst 인과" 전략은 폐기.
- **Paper 1 (ICRA, Input-Prior)**: 단일프레임 image MAE(Sobel+RGB) > VideoMAE = **좁지만 입증**. 남은 일 = ablation(RGB-only vs Sobel+RGB, VideoMAE fairness) + real-robot.
- **Paper 2 (AAAI, Action-Agnostic)**: scaffold(M/P cross-stream) 가설 **미입증**. 직전 v15의 +0.390은 motion routing 인과 아님(no-op) → 철회. **Parvo(code v15b)** = M→P gradient 연결한 의도된 설계로 재학습 → input-only baseline 초과 여부가 성패. 본학습 보류 중, [docs/v15b_retraining_status.md](docs/v15b_retraining_status.md) §6 명령 참조.
- 5 baseline(VideoMAE-ours, DINOv2, SigLIP, VC-1) + V-JEPA 2.1(probing only)은 비교군으로 유지.

> ⚠️ 아래 Phase 1.5~4 표·기록은 **구 v15 시점의 history**. catalyst 인과·"paper main" 표현은 위 정직화에 따라 무효 — 개발 narrative 기록으로만 보존.

자세한 phase별 상황·표는 [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) · [docs/paper_experiments_plan.md](docs/paper_experiments_plan.md) · [docs/cluster_sessions.md](docs/cluster_sessions.md) · [docs/PROBING_GUIDE.md](docs/PROBING_GUIDE.md). 본 섹션은 핵심 요약·진행 상황만.

### 모델 lineup 요약 (Phase 1.5 사전학습)

| 모델 | 상태 | 핵심 결과 |
|------|------|----------|
| v4~v9 (다수) | 폐기 | RoPE/APE/EMA teacher/residual target 등 시도, 모두 실패. negative result |
| v6 (APE+mask50+rotaug) | 폐기 | ep8 R²=0.259 (이전 챔피언) |
| v10 (mask_p 0.75) | 폐기 (plateau) | ep40 +0.221, v6 추월 실패 |
| **v11** (motion-routing + dual-target) | 종료 (paper §method history) | ep44 A+B+D' +0.288 (until v15) |
| **VideoMAE-ours** | 종료 (active baseline) | 50ep, R²=0.326 |
| v12/v13/v14 | 폐기 | DINO redesign / stream-wise paradigm 시도, 모두 collapse 또는 cancel |
| **v15** (teacher-anchor: predictor-only V-JEPA + V-JEPA-M Option B + L_compose + 3-frame triple) | 동결 (`paper-corl2026`) | ep50 완주. +0.390(P_t⊕P_tk)이나 motion routing **no-op** → catalyst 인과 철회 (artifact 의심) |
| **Parvo** (code v15b, student-anchor: M→P gradient 연결 + recon-first gate) | **활성 (`main`)** | 논문 핵심 모델. Paper 2 검증 대상. sanity 통과, 본학습 보류 중 |

**v15 발견 + 정직화 (2026-06-11)**:
- `patch_mean_concat_p_t_p_tk` mode = **+0.390 EgoDex** / **+0.584 LIBERO spatial gap=20** (수치 자체는 유효).
- ~~**M stream catalyst 가설** … DINOv2 +0.006으로 catalyst 인과 확정~~ → **철회**. v15는 motion routing이 student P에 gradient=0 **no-op**이라 +0.390을 motion 인과로 귀속 불가. 유력 대안 = **multi-frame MAE concat + probe artifact** (P_t·P_tk 단독은 음수 −0.05/−0.01인데 concat에서 +0.390 → probe가 two-frame 차이를 implicit 추출). DINOv2 통제군도 single-vs-two-frame 혼재라 깨끗한 control 아님.
- 학습 종반 P/M encoder CLS는 collapse, **patch 단위에선 healthy** (75% MAE recon 작동).
- 학습 cost: AIP_long 2×4 H100 × 42.91h = **343 GPU·h** (50ep 완주).
- **인과 검증은 Parvo(code v15b)로 이관** — [docs/v15b_retraining_status.md](docs/v15b_retraining_status.md).

### Phase 3-1 V3 BC-T — 완료 (2026-05-13~14)

V3 cfg = `use_joint=True` + augmentation (`ImgColorJitter` + `TranslationAug`). v15 학습 분포(3-frame triple) mismatch 해결을 위해 **신규 어댑터 2종**:
- **`two-stream-v15-ptptk`** (옵션 B): (prev, curr) → P encoder 각자 forward → P_t patches mean ⊕ P_tk patches mean (1536-d). LIBERO probing champion mode 동치
- **`two-stream-v15-mp`** (C-variant): (prev, curr) → M encoder + P encoder (curr만) → 1536-d

**매트릭스**: 18잡 (34376001~018, v15 ep50 × {ptptk, mp} × 3 suite × 3 seed) AIP 1×1 H100 각 22~35h → 모두 COMPLETED. round1 9 ckpt (siglip / vc1 / dinov2 / v11)는 `paper_artifacts/libero_rollout/` 등록 완료. v15 18 ckpt는 로컬 전송 대기 (`libero_bct_v15_9jobs_round2_20260514.tar` 7.6 GB repo root).

### Phase 4 (paper §5.1 ablation) — 진행 중

**C1 v15 V-from-M ablation (paper §5.1 main)**: 2026-05-15 제출
- 34464714 sanity (AIP 1×1 H100, 50vid × 3ep × num_workers=4)
- 34464715 본 학습 (AIP_long 2×4 H100, 50ep × part1-5, dependency=afterok)
- fair pair: V11_ROUTING_MODE=v_from_m only different, ckpt → `/proj/external_group/mrg/checkpoints/two_stream_v15_vfromm/`
- 예상 완료: ~43h (v15 본 학습 동일 cost)

**C2~C9** (paper_experiments_plan.md): C1 ep50 후 fair head-to-head probing + BC-T (C2, C3). C4~C7 즉시 가능 (의존성 없음).

### 다음 단계

1. C1 sanity 통과 → 본 학습 자동 시작. ep4/ep8/ep12 진단 모니터링 (cluster_sessions에 기록)
2. v15 round2 BC-T 18잡 로컬 rollout (round2 tar 전송 + `bash scripts/local/run_libero_rollouts.sh two-stream-v15-ptptk 50` 등)
3. C4 (v15 ep50 LIBERO object/goal), C5 (v15 ep50 DROID gap sweep), C6 (recon quality v11 vs v15), C7 (VideoMAE P_t+P_tk catalyst) — 의존성 X, 즉시 가능

자세한 내용은 [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) · [`docs/paper_experiments_plan.md`](docs/paper_experiments_plan.md), probing 결과는 [`docs/PROBING_GUIDE.md`](docs/PROBING_GUIDE.md), 클러스터 자원·잡 추적은 [`docs/cluster_sessions.md`](docs/cluster_sessions.md) 참고

## 트러블슈팅 로그

### 재발 가능 — 숙지 필요

- **🔴 Train ↔ inference preprocessing parity (2026-05-25 CortexBench v15/videomae 사고)**: 새 평가/어댑터/wrapper를 만들 때마다 **학습 시 model input range·channel order·정규화**와 **inference pipeline의 transform**이 정확히 일치하는지 명시 점검. 사고 사례: CortexBench wrapper `v15_loader.py`/`videomae_loader.py`가 `T.Normalize(ImageNet)`을 추가했으나 EgoDex 학습 loader([src/datasets/base.py:119](src/datasets/base.py#L119))는 `/255.0`만 → `[0,1]` raw 입력. 두 모델만 OOD inference로 metaworld SR ~53% (vc1/siglip 88%대) 폭락. 30잡(`v15_p_only` 21 + `videomae_ours` 9) 전부 무효. **방지 체크리스트**: (1) 학습 dataset의 `_load_image`/`__getitem__` 마지막 단계에서 입력 텐서 range·dtype 확인, (2) 어댑터/wrapper docstring에 input contract 명시(예: `obs: [0,1] float`), (3) 평가 entry script에 첫 batch min/max/dtype 로깅(`finetune_libero_bct.py:_log_first_batch_stats` 패턴), (4) 새 wrapper에는 학습 분포 통과 sanity test 1-2개 추가
- **Slurm `--time` 부족 timeout (반복 사고)**: 2026-04-29 BC-T 5 encoder 학습에서 dinov2/siglip/vc1 모두 `--time=24h`로 제출 후 epoch당 시간 측정해보니 적자 -4.5h ~ -6.3h 확인. 8h 학습 후 cancel + 재제출 (~24 GPU·h 손실). **잡 제출 전 epoch당 시간 × 총 ep × 1.3 안전 마진 계산**, 추정 어려우면 partition max (`2-00:00:00`)로. multi-encoder 동시 제출 시 가장 느린 모델 기준 통일. 자세한 가이드는 "클러스터 세션 로깅 → `--time` 결정 가이드" 참고.
- **Sanity checkpoint → 본 학습 auto-resume 오염**: 서로 다른 학습 설정의 체크포인트가 같은 디렉토리에 있으면 scheduler/optimizer 상태 오염 (T_max, lr, batch_size). **sanity 잡은 반드시 `CHECKPOINT_SUFFIX` 사용**.
- **Slurm DDP 3대 함정**: (1) `--gpus-per-task=1` 대신 `--gres=gpu:N` 사용 (NCCL PCI 탐색 실패 방지), (2) `MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))` (포트 충돌 방지), (3) srun에서 `$CONDA_PREFIX/bin/python` 절대 경로 사용
- **로그인 노드 프로세스 제한**: 대용량 다운로드는 순차 실행, 여러 다운로드 동시 실행 금지. `gsutil -m`은 thread/process count 제한 필수 (`-o parallel_thread_count=8 -o parallel_process_count=4`)
- **Scratch stage-in 비현실적**: 소형 파일 수백만 개는 메타데이터 병목. GPFS 직접 읽기 사용
- **`num_workers=16` multiprocessing hang** (v15 sanity 사고): 47분 hang → CANCELLED. EgoDex dataset triple은 단독 OK였으나 high worker 수에서 deadlock 추정. **`num_workers=4-8`로 회피**. v15 본 학습은 8 고정
- **DDP unused-param**: 모델이 forward에 일부 parameter를 안 쓰면 DDP grad reduce가 stuck. **사용 안 하는 mask token 등은 `requires_grad_(False)`** (v14 `mask_token_m` 사고 → 같은 fix 적용). `find_unused_parameters=True`는 임시방편, 정공법은 `requires_grad_(False)`
- **DINO loss 평형 trap**: K (prototype 수)에 비해 λ가 작으면 `L=log(K)` 근처에 갇혀 prototype space 학습 안 됨 (v13 K=4096+λ=0.005 → L=8.32 plateau). **K=1024 + λ=0.01 권장 (DINOv2 default)**. K↑가 항상 좋은 게 아님
- **3-frame vs 2-frame 학습 분포 mismatch**: v15는 3-frame triple로 학습. 기존 2-frame pair BC-T 어댑터 그대로 쓰면 학습 분포 불일치. **v15 전용 신규 어댑터(`two-stream-v15-ptptk`, `two-stream-v15-mp`) 사용 필수**. 새 모델 도입 시 BC-T adapter 호환성 사전 검토

### 해결 완료 — 참고용

- **BF16 autocast**: `use_bf16` 플래그로 scaler와 분리 (GradScaler 불필요). 37.6→62.2 samples/sec
- **SSIM BF16 NaN**: SSIM 연산 FP32 강제 + sigma clamp(min=0)
- **V-JEPA 발산**: (1) EMA target 기반 학습은 LR warmup 필수, (2) 2-frame에서는 mask ratio 대폭 완화 필요 (16-frame temporal redundancy 없음). 결국 3차 모두 발산 → negative result
