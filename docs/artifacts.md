# Artifacts & Data Locations

**목적**: 클러스터(IBS olaf) ↔ 로컬 워크스테이션 사이 산출물 이동·참조 인덱스.
새 ckpt 생성·데이터셋 위치 변경·전송 완료 시 즉시 업데이트.

---

## 빠른 참조

| 항목 | 클러스터 (IBS olaf) | 로컬 워크스테이션 |
|------|--------------------|-------------------|
| 프로젝트 root | `/proj/home/mrg/bys724/action-agnostic-visual-rl` | `/home/etri/bys/action-agnostic-visual-rl` |
| 학습 ckpt 루트 | `/proj/external_group/mrg/checkpoints/` | `/mnt/data/checkpoints/` |
| 데이터셋 루트 | `/proj/external_group/mrg/datasets/` | `/mnt/data/` |
| Slurm/실행 log | `/proj/external_group/mrg/logs/` | `/mnt/data/logs/`, `/mnt/data/*.log` |
| Conda/Docker env | `/proj/external_group/mrg/conda_envs/aavrl-train` | Docker `dev-env` (`action-agnostic-dev:latest`) |
| Launcher | `scripts/cluster/*.sbatch` | `scripts/local/*.sh` |

---

## 클러스터 (IBS olaf)

### 학습된 모델 ckpt

#### 🚚 로컬 워크스테이션 전송 대기 (repo root tar)
이 저장소 최상위 폴더에 tar 형식으로 배치됨. 로컬에서 `scp` 한 번 + `tar -xf`로 전개.

| Tar 파일 (repo root) | 크기 | 용도 | 내용 |
|---|---|---|---|
| `v11_bct_v3_rollout.tar` | 7.1 GB | v11 BC-T rollout 평가 (학습 불필요, 9 ckpt) | `two-stream-v11_libero_{spatial,object,goal}_seed{0,1,2}_<TS>_v3/{best.pt, config.yaml}` × 9 |
| `dinov2_bct_v3.tar` | 3.1 GB | DINOv2 BC-T rollout 평가 (9 ckpt, base 86M라 사이즈 작음) | `dinov2_libero_{spatial,object,goal}_seed{0,1,2}_<TS>_v3/{best.pt, config.yaml}` × 9 |
| `two_stream_v14_ep20.tar` | 2.9 GB | v14 pre-trained encoder (로컬 LIBERO BC 학습 + rollout) | `20260507_202712/{checkpoint_epoch0020.pt, config.json}` |

전송 후 로컬:
- `v11_bct_v3_rollout.tar` → `/mnt/data/checkpoints/libero_bct/` 하위에 풀기 → `src/eval_libero.py BCTransformerClient`로 rollout
- `two_stream_v14_ep20.tar` → `/mnt/data/checkpoints/two_stream_v14/` 하위에 풀기 → `scripts/local/finetune_libero_bct.sh`로 BC-T 학습

비교 목적:
- v11 BC-T 9 ckpt는 baseline 36잡 (videomae/dinov2/siglip/vc1, 33866041~76)과 **단일 변인(ENCODER)** 차이로 학습된 V3 cfg → 동일 조건 main table rollout 비교 가능
- v14는 EgoDex within-domain probing collapse (-0.065) 후 LIBERO probing은 +0.5 영역. 학습 미완 (ep20에서 stop)이지만 로컬에서 BC fine-tune 후 v11과 직접 비교 → cross-domain 적합도 검증

#### 🏆 Two-Stream v11 — final champion (ep44 A+B+D' = +0.288)
- 경로: `/proj/external_group/mrg/checkpoints/two_stream_v11/20260426_014333/`
- 파일: `latest.pt` (= ep50), `best_model.pt`, `checkpoint_epoch00{16,20,24,28,32,36,40,44,48}.pt`, `config.json`, `history.json`, `tb/`
- 추천 전송 셋 (paper/inference용): `checkpoint_epoch0044.pt` (final champion), `latest.pt`, `config.json`, `history.json` — 약 7 GB
- 이전 디렉토리 `20260425_011343/` 는 1차 학습 (ep1~12 후 TIMEOUT). 무시 가능

#### VideoMAE-ours
- 경로: `/proj/external_group/mrg/checkpoints/videomae/20260415_012017/`
- 50 epoch 완주, R²=0.326. ep4~ep48 4 epoch 간격 + best/latest

#### Two-Stream v6 (이전 챔피언)
- 경로: `/proj/external_group/mrg/checkpoints/two_stream_v6_ape_mask50_rotaug/20260416_203440/`
- ep8 R²=0.259 (ep23 scancel). 보유 ckpt: ep08/12/16/20 + best

#### Two-Stream v10
- 경로: `/proj/external_group/mrg/checkpoints/two_stream_v10/20260422_130827/`
- 50ep 완주. ep40 plateau +0.221. 보유: ep40/44/48 + best/latest

#### LIBERO BC fine-tuned
- 경로: `/proj/external_group/mrg/checkpoints/libero/`
- `videomae_libero_spatial_20260426_005309/` (best val MSE 0.0286)
- `two_stream_v11_libero_spatial_20260426_011708/` (v11 ep12 A+D, best val MSE 0.0290)

#### 폐기 라인 (참고용 보관, 전송 불필요)
v4, v5, v7-big (×3), v8, v9 (×4 dirs), V-JEPA-ours, vjepa2_official, vjepa_official 등. negative result. 자세한 내용은 `docs/RESEARCH_PLAN.md`

### 데이터셋

| 데이터셋 | 경로 | 자체 다운로드 가능? |
|---------|------|---------------------|
| EgoDex raw | `datasets/egodex/raw` (part1~5) | ✅ CDN, `scripts/cluster/download_egodex.sh` |
| EgoDex frames | `datasets/egodex/frames/part{1..5}` | 위 raw에서 추출 (`scripts/cluster/extract_egodex.sbatch`) |
| EgoDex zips | `datasets/egodex/zips` | CDN 원본 archive (보존용) |
| DROID frames | `datasets/droid_frames/ext1` | gsutil 가능 but 3.4 TB 원본 + 추출 부담. 클러스터 보유본 재활용 권장 |
| DROID frames sample | `datasets/droid_frames_sample` | sanity test용 소규모 |
| DROID raw | `datasets/droid` | gsutil 다운로드본 |
| LIBERO | `datasets/libero/` | ✅ HuggingFace |
| Ego4D | `datasets/ego4d/` | 라이선스 + gsutil. 다운로드 진행 중 |
| Nymeria | `datasets/nymeria/` | (참조만, 사용 안 함) |
| EpicKitchens-100 | `datasets/epic_kitchens_100/` | (참조만) |
| **CALVIN** (§C10) | `datasets/calvin/task_ABCD_D.zip` | 🔄 다운로드 진행 중 (2026-05-18, 166 GB 압축, Freiburg CDN, ETA ~27h) |
| **CortexBench Adroit** (§C11) | `datasets/cortexbench/adroit-expert-v1.0.zip` | 🔄 다운로드 진행 중 (2026-05-18, ~1.6 GB, Meta CDN, 빠름) |
| **CortexBench Meta-World** (§C11) | `datasets/cortexbench/metaworld-expert-v1.0.zip` | 🔄 Adroit 완료 후 자동 시작 (sequential) |

(루트: `/proj/external_group/mrg/datasets/`)

### External codebases (repo 내 submodule/clone)

| 코드 | 위치 | 용도 |
|------|------|------|
| VideoMAE | `external/VideoMAE/` | 우리 VideoMAE-ours 구현 base |
| eai-vc (§C11) | `external/eai-vc/` | CortexBench (Adroit + Meta-World BC training/eval framework) — 2026-05-18 clone |

### Probing 결과 (repo 내부, git 동기화)
- `data/probing_results/probe_v11_*.json` — ep4~ep50 × 12 mode (84개 JSON)
- `data/probing_results/probe_droid_v11_*.json` — DROID cross-domain
- repo에 포함되어 있어 `git pull`로 워크스테이션 동기화 가능

### 가시화 산출물 (repo 내부, git 동기화)
- `docs/architecture/attn_v11_ep{48,50}.png` — final champion attention
- `docs/architecture/rotation_v10_ep{8,12,16,20,24,36,40,44,48}/` — v10 rotation viz
- 기타 PNG는 `docs/architecture/` 하위

### 로그
- Slurm sbatch logs: `/proj/external_group/mrg/logs/{<jobname>_<jobid>.out,.err}`
- 실패 디버깅 시 `cat /proj/external_group/mrg/logs/aavrl_pretrain_<jobid>.err` 부터 확인

### 클러스터 SSH 정보 (필요 시 채우기)
- Host: `<cluster>` (sbatch는 login 노드에서, GPU는 olaf-g 노드)
- User: `bys724`

---

## 로컬 워크스테이션

- 호스트: `H100` (user `etri`)
- OS: Ubuntu (Linux 6.8)
- GPU: NVIDIA H100 PCIe × 2 (각 80 GB)

### 학습 ckpt

루트: `/mnt/data/checkpoints/`

**클러스터에서 전송 받은 모델**

| 모델 | 로컬 경로 | 전송일 | 비고 |
|------|----------|--------|------|
| BC-T VideoMAE-ours × spatial seed=0 (1차, 폐기) | `/mnt/data/checkpoints/libero_bct/_archive_pre_usejoint/bct_videomae-ours_libero_spatial_seed0_best.pt` | 2026-04-30 | use_joint=False로 학습된 broken 1차. 별도 archive로 격리 |
| BC-T 5 encoder × spatial seed=0 (2차, use_joint fix) | `/mnt/data/checkpoints/libero_bct/<encoder>_libero_spatial_seed0_20260430_150733_usejoint/{best.pt, config.yaml}` | 2026-05-03 | tar 일괄 전송. 5 dir: `two-stream-v11`, `videomae-ours`, `dinov2`, `siglip`, `vc1`. 각 best.pt 340 MB ~ 807 MB (v11이 큼). config.yaml에 `use_joint: true` 확인 |
| (예정) v11 ep44/ep50 | `/mnt/data/checkpoints/two_stream_v11/20260426_014333/` | — | 미전송. 전송 시 `checkpoint_epoch0044.pt`, `latest.pt`, `config.json`, `history.json` 약 7 GB |
| (예정) VideoMAE-ours 50ep | `/mnt/data/checkpoints/videomae_full/20260407_104721/` | — | 디렉토리 골격(config.json + tb)만 있음. ckpt 본체 미전송 |

**로컬에서 학습한 모델 (legacy / sanity)**

| 디렉토리 | 모델 / 용도 |
|----------|-------------|
| `two_stream/2026031{9,3,4,5}_*`, `two_stream/2026033{0}_*` | 초기 Two-Stream 실험 ckpt (v3 계열, 16개 run) |
| `two_stream_ssim/20260331_021059` | SSIM loss 실험 |
| `v4_base_b16`, `v4_base_gap60_tri`, `v4_composition`, `v4_comp_grad`, `v4_mask30`, `v4_nomask` | v4 ablation 시리즈 |
| `videomae`, `videomae_full/20260407_104721` | VideoMAE 시도 (full은 빈 골격) |
| `ablation_A`, `ablation_B`, `ablation_C` | ablation run |
| `libero_test_{dinov2,videomae,v3}` | LIBERO BC fine-tune 실험 |

→ 본 학습은 클러스터에서 진행 중이므로 위 로컬 ckpt는 sanity / 초기 탐색용. 정리 검토 대상.

### 데이터셋

| 데이터셋 | 경로 | 상태 | 비고 |
|---------|------|------|------|
| EgoDex raw | `/mnt/data/egodex/` | ✅ 보유 | `scripts/local/download_egodex.sh part1..5` |
| EgoDex frames | `/mnt/data/egodex_frames/` + `egodex_frames_part{1..5}/` | ✅ 보유 | 추출 결과 (256×256 center-crop) |
| DROID raw | `/mnt/data/droid/` | ✅ 보유 | gsutil 다운로드본 |
| DROID frames | `/mnt/data/droid_frames/{ext1,ext2,wrist}/` | ✅ 보유 | cross-domain probing용 |
| Bridge V2 | `/mnt/data/bridge_v2/` | ✅ 보유 | 256×256 리사이즈 |
| Something-Something v2 | `/mnt/data/ssv2/` | ✅ 보유 | (사용 여부 미정) |
| Tarballs | `/mnt/data/tarballs/` | ✅ 보유 | 백업/업로드용 archive |
| **LIBERO** | `/mnt/data/libero/` | ❌ **없음** | LIBERO BC/rollout 작업 시 HuggingFace에서 받아야 함 |

### Docker dev-env

- 컨테이너명: `dev-env` (이미지: `action-agnostic-dev:latest`, 5주째 가동 중)
- 진입: `docker exec -it dev-env bash`
- 마운트:
  | 호스트 | 컨테이너 |
  |-------|----------|
  | `/home/etri/bys/action-agnostic-visual-rl` | `/workspace` |
  | `/home/etri/bys/action-agnostic-visual-rl/data` | `/workspace/data` |
  | `/mnt/data` | `/mnt/data` |
  | `/home/etri/.cache/huggingface` | `/workspace/.cache/huggingface` |

### Repo 내부 산출물 (git 동기화)

- `data/probing_results/` — probe JSON (CLIP/DINOv2/two-stream/videomae 등)
- `data/eval_results/` — 평가 결과
- `data/logs/` — 로컬 학습 로그
- `data/datasets/` — 메타데이터 / 인덱스

---

## 동기화 패턴

### 클러스터 → 워크스테이션 (학습 ckpt 가져오기)

**옵션 1 (1-hop, 권장)** — 워크스테이션이 클러스터 SSH 직접 접속 가능 시:
```bash
mkdir -p /mnt/data/checkpoints/two_stream_v11/20260426_014333
rsync -avzP --inplace \
  bys724@<cluster>:/proj/external_group/mrg/checkpoints/two_stream_v11/20260426_014333/{best_model.pt,checkpoint_epoch00{44,48}.pt,latest.pt,config.json,history.json} \
  /mnt/data/checkpoints/two_stream_v11/20260426_014333/
```

**옵션 2 (3-hop, 맥북 경유)** — 워크스테이션이 클러스터 직접 접근 불가, 맥북에서 stream pipe (맥북 디스크 사용 X):
```bash
ssh bys724@<cluster> "cd /proj/external_group/mrg/checkpoints/two_stream_v11/20260426_014333 && tar c best_model.pt checkpoint_epoch00{44,48}.pt latest.pt config.json history.json" \
  | ssh user@<workstation> "tar x -C /mnt/data/checkpoints/two_stream_v11/"
```

**inference만 필요한 경우**: cluster에서 `model_state_dict`만 추출 → optimizer/scheduler 제외 → 약 60% 용량 절약

#### BC-T best.pt 단일 파일 전송 (LIBERO rollout 평가용)

LIBERO BC-T는 학습 산출물이 단일 `best.pt` (~340 MB, `policy_state_dict` +
`config` 포함). encoder ckpt path가 cluster path로 박혀 있어도 rollout 시
None override + state_dict 덮어쓰기로 처리하므로 **encoder ckpt를 별도로
가져올 필요 없음**. 이름은 본 워크스테이션 컨벤션에 맞춰:
`bct_<encoder>_<suite>_seed<N>_best.pt`

```bash
mkdir -p /mnt/data/checkpoints/libero_bct
rsync -avzP \
  bys724@<cluster>:/proj/external_group/mrg/checkpoints/libero_bct/<run_dir>/best.pt \
  /mnt/data/checkpoints/libero_bct/bct_<encoder>_<suite>_seed<N>_best.pt
```

### 워크스테이션 → 클러스터 (분석 결과)
- 작은 산출물 (probing JSON, viz PNG ≤ 10 MB): git push/pull로 동기화
- 큰 파일: rsync로 역방향

### git 동기화 (양방향)
```bash
git pull origin main   # 다른 환경에서 작업한 내용 받기
git push origin main   # 현재 환경 작업 push
```

---

## 업데이트 규칙

- 새 학습 ckpt 디렉토리 생성 시: "클러스터 - 학습된 모델 ckpt" 섹션에 1줄 추가
- 데이터셋 위치 변경/추가 시: 즉시 갱신
- 워크스테이션으로 전송 완료 시: "로컬 워크스테이션 - 학습 ckpt" 표에 행 추가
- 폐기된 ckpt는 제거하거나 한 줄로 압축
