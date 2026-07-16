# AAAI-27 Code & Data Supplement 준비 — dev 실행 가이드

> **마감: 7/31 AoE** (AAAI-27 supplementary). **계약 원문(무엇을 내야 하는지) = paper repo `notes/code_release_prep.md`** — 형식·익명화 규칙·제출 방식은 그쪽이 단일 출처. 이 문서 = dev 측 **어떻게 만들지**(파일 매핑·작업 순서·검증).
>
> 배경: `main/ReproducibilityChecklist.tex`의 4.2–4.5 "yes" + significance "partial"(Wilcoxon)이 **이 코드 묶음이 제출물에 실제 첨부됨을 전제**. 실물 없이 제출 불가.

## 0. 작업 환경 분담

| 단계 | 환경 | 이유 |
|---|---|---|
| 코드 추출·정리·익명화·zip | **로컬 워크스테이션** | 파일 작업 + 재현 검증과 같은 곳 |
| 재현 검증 (probing 재실행) | 로컬 워크스테이션 (H100×2) | probing·rollout은 로컬 역할 (CLAUDE.md 역할분담) |
| ckpt 반출 (필요 시) | 클러스터 → 로컬 | 이미 로컬에 있으면 생략 |
| (선택) `conda list`·VideoMAE config 확보 | 클러스터 | checklist 4.7·4.12 상향용 — camera-ready 대비, 7/31 필수 아님 |

**성공 기준**: ① zip 하나가 아래 §1 매핑 전 항목 충족 ② 익명화 grep(§3) 0건 ③ clean 트리 + ckpt로 probing 표 수치 재현 확인 로그 확보(§4). 셋 다 되면 끝.

## 1. Deliverable → dev 파일 매핑

계약의 5-item + 통계. **fork 금지 — history 없는 새 트리에 복사** 후 정리.

| # | Deliverable | dev 정본 위치 | 작업 |
|---|---|---|---|
| 1 | 모델 정의 (two-stream P·M, 마스킹, value-ownership routing) | `src/models/two_stream_v15.py` (**v16 = `--v15-comp-mae` 분기**, `_forward_pair_comp`) + `src/models/common/{blocks,preprocessing}.py` | v9–v15 dead 분기 제거한 clean 모듈로 발췌. ⚠️ v16이 v15 파일 내 분기라 발췌 시 동작 보존 검증 필수(§4가 겸함) |
| 2 | 학습 목적 (masked cross-recon loss) | 같은 파일 loss부 (M-recon floor+scale·\|ΔL\| 가중 등) | 주석에 논문 §method 수식 번호 참조 추가 |
| 3 | 학습 스크립트 + config | `scripts/pretrain.py` + **보고 run HP = `scripts/cluster/pretrain.sbatch` env + `docs/cluster_sessions.md` 해당 잡 기록** | sbatch 자체는 제외(SLURM·클러스터 경로). HP를 plain config(yaml/argparse default)로 옮기고 **paper App A 표와 대조** |
| 4 | 평가 하네스 (probing + BC) | `scripts/eval/probe_action_{libero,calvin}.py` · `build_step0_efficiency_table.py` · `finetune_libero_bct.py` · `src/eval_libero.py` · `scripts/eval/aggregate_libero_rollouts.py` · adapter `src/encoders/adapters/parvo_pt_ptk.py` | 헤드라인 표(efficiency·rollout) 산출 경로만. baseline 로더(VideoMAE/VC-1/DINOv2/SigLIP)는 표 재현에 필요한 만큼만 |
| 5 | 체크포인트 + README | ckpt: `two_stream_v15b_comp_mae_s`(CoMP-MAE-S) + `two_stream_v15b_step1_plain_xmae_s`(plain control) — 헤드라인 2종이 최소셋 | README에 requirements·결과표·정확한 재현 명령. ckpt 내부 메타·파일명에서 username/경로 제거 |
| 6 | 통계 스크립트 + 근거 CSV | paper repo `scripts/stats_libero_rollout.py` + dev `paper_artifacts/libero_rollout/per_task.csv` | 🔴 **스크립트 `DEFAULT_CSV`가 `/Users/bys724/...` 하드코딩** — zip 동봉 시 상대경로로 수정 (double-blind 위반 소지) |

**제외** (계약 그대로): 내부 sweep·SLURM/sbatch·W&B·죽은 브랜치(v9–v11, siammae 등)·private 데이터 로더. **EgoDex·LIBERO·CALVIN raw 데이터 재배포 금지** — 다운로드 안내만.

## 2. 작업 순서 (체크포인트식)

1. **zip 트리 스켈레톤 설계** — 새 디렉토리(예: scratch 영역)에 `model/ · train/ · eval/ · stats/ · checkpoints/ · README.md · LICENSE · requirements.txt` 골격. 이 시점에 포함 파일 목록을 확정하고 시작.
2. **코드 복사·정리** — §1 매핑대로. dead 분기 제거는 이 트리에서만(원 repo는 rename deferred 정책대로 불변).
3. **익명화 pass** — §3 grep 체크리스트 0건까지.
4. **재현 검증 (필수 선행)** — §4. 통과 전 checklist "yes" 상태로 제출 금지.
5. **README·LICENSE·requirements 확정** — LICENSE는 연구용 허용(MIT/Apache-2.0 등, checklist 문구 충족).
6. **zip 생성 + 용량 확인** — 한도 초과 시 코드 zip / ckpt 익명 링크 분리(계약 문서 참조; 현 단계 권장은 zip 단일).

## 3. 익명화 grep 체크리스트 (전부 0건이어야 함)

```bash
grep -rn "bys724\|/Users/\|/home/\|/proj/external_group\|/mnt/data\|mrg\|IBS\|olaf\|github.com" <zip-tree>/
```

- ckpt는 grep으로 안 잡힘 — `torch.load` 후 키·메타 별도 확인 (optimizer state·경로 문자열 잔존 여부).
- 주석 속 한국어 메모도 제거 대상(작성자 추정 단서).

## 4. 재현 검증 프로토콜 (checklist "yes"의 근거)

- **무엇을**: clean 트리 코드 + 동봉 ckpt로 **probing 헤드라인 재실행** → `step0_ood_efficiency` 표의 CoMP-S·plain 행 수치 일치 확인. BC rollout은 std 내 재현이면 통과(전 매트릭스 재실행 불요 — 스팟 체크 1 suite × 1 seed 권장).
- **parity 주의**: eval 시 `docs/eval_protocols.md` §0 preprocessing parity 체크 그대로 적용 (`[0,1]` raw, ImageNet Normalize 금지).
- 특정 수치 재현 실패 시: 그 항목만 paper checklist "partial"로 강등 + 범위 명시 (paper 세션에 전달).

## 4-b. 진행 상황 — 클러스터측 완료분 (2026-07-16)

**머지 전략**: supplement 트리 = **`release/aaai27_supplement/`** (repo 내 전용 디렉토리, 양측이 서로 다른 파일을 채워 conflict 없음 — 코드·텍스트는 git, ckpt 바이너리만 tar 채널). 최종 zip = 이 디렉토리를 압축 (git history 미포함 = §1 "새 트리" 조건 자동 충족). 익명화 grep(§3)도 이 디렉토리 대상.

**클러스터 완료 (git으로 수령)**:
- `release/aaai27_supplement/train/config_{comp_mae_s,plain_control}.yaml` — 보고 run HP를 sacct SubmitLine(36177296·36652564 실측)에서 전사. deliverable #3의 "plain config" 실물. **로컬 할 일: paper App A 표와 대조**.
- `release/aaai27_supplement/requirements.txt` — 학습 env(pip freeze) 기반 초안. **로컬 할 일: 실제 shipped 코드 import 기준으로 확정** (LIBERO/robosuite는 별도 안내로).
- `release/aaai27_supplement/checkpoints/README.md` — ckpt 목록·sha256·로딩 스니펫·parity 경고. zip 동봉 최종본 후보.

**ckpt 반출 (tar 채널, deliverable #5)**: repo root **`aaai27_supplement_ckpt.tar`** (420MB) = weights-only 사본 2종 + train_meta.json 2종 + SHA256SUMS.
- strip 내역: `latest.pt`(ep50)에서 optimizer/scheduler/history 제거 → `{epoch, model_state_dict, train_loss, eval_loss}`만. state_dict 키 익명화 grep 0건·reload 검증 완료. **로컬 §3 ckpt 메타 확인 항목은 사실상 선처리됨**(경로 문자열의 주 서식지였던 optimizer state 제거).
- 원본 매핑(불변): `comp_mae_s.pt` ← `two_stream_v15b_step1_comp_mae_s/20260629_101634/latest.pt` · `plain_xmae_s.pt` ← `two_stream_v15b_step1_plain_xmae_s/20260708_012539/latest.pt` (STEP 2(A) probe 잡이 쓴 바로 그 파일 — 재현 검증 §4와 정합).
- 수령: `rsync --partial --progress <cluster>:<repo>/aaai27_supplement_ckpt.tar .` → `tar -xf` → `sha256sum -c SHA256SUMS` → `release/aaai27_supplement/checkpoints/`에 배치(*.pt는 .gitignore 대상).

**로컬 워크스테이션 남은 작업 (§2 순서 기준)**:
1. git pull + ckpt tar 수령·검증 (위)
2. §1 매핑대로 코드 발췌 → `release/aaai27_supplement/{model,eval,stats}/` — model: `two_stream_v15.py`의 v16 분기(`_forward_pair_comp`)+`common/{blocks,preprocessing}.py` (v9–v15 dead 분기 제거, 동작 보존은 §4가 검증) / train: `scripts/pretrain.py` 정리본 / eval: §1 #4 목록 / stats: `stats_libero_rollout.py`(🔴 `DEFAULT_CSV` `/Users/` 하드코딩 상대경로화) + `per_task.csv`
3. 익명화 pass — §3 grep을 `release/aaai27_supplement/` 대상 0건까지 (한국어 주석 제거 포함)
4. 재현 검증(§4) — probing 헤드라인: efficiency 표 CoMP-S·plain 행 재현 / BC 스팟 체크 1 suite × 1 seed
5. README(루트)·LICENSE 작성, requirements 확정 → zip 생성·용량 확인

## 5. Cross-refs

- 계약·형식·주의점 원문: paper repo `notes/code_release_prep.md` (07-15)
- checklist 문안·근거: paper repo `notes/repro_checklist_review.md` · `main/ReproducibilityChecklist.tex`
- 이식 수치 정본: paper repo `notes/aaai27_dev_materials.md` (dev `paper_artifacts/` 기준)
- Camera-ready(채택 후): 실명 public repo 전환 — 계약 문서 말미 참조
