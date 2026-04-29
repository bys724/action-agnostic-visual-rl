# Claude Development Guide

**세션 시작 시 필독**: 아래 핵심 문서를 먼저 읽고 작업 맥락을 파악하세요.

## 핵심 문서 (업데이트 우선)

1. **`docs/RESEARCH_PLAN.md`** - 전체 연구 계획 및 현재 phase (마스터 문서)
2. **`docs/PROBING_GUIDE.md`** - Action probing 실험 가이드 + 결과
3. **`docs/setup/LIBERO_TEST_GUIDE.md`** - LIBERO 평가
4. **`docs/cluster_sessions.md`** - IBS 클러스터 sbatch/salloc 세션 로그 (비용 청구 대조)
5. **`docs/artifacts.md`** - 클러스터/로컬 워크스테이션 산출물·데이터셋 경로 인덱스 (양쪽 작업 시 참조)

**문서 작성 원칙**:
- 새 문서를 만들기보다 기존 핵심 문서 **업데이트** 우선
- 핵심 계획은 `RESEARCH_PLAN.md`에 집중 관리
- 일회성 정보는 git commit message에 기록

## 프로젝트 개요

**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?

**핵심 아이디어**:
- EgoDex로 행동-독립적(action-agnostic) 시각 표현 사전학습
- LIBERO에서 로봇 조작 태스크로 평가

## 관련 Obsidian 노트

이 프로젝트의 핵심 아이디어·논문 작성·실험 설계는 Vault에 정리됨. 코드 작업 중 맥락 보충이 필요하면 참조:

| 카테고리 | 경로 |
|---------|------|
| 프로젝트 메인 인덱스 | `Projects/Action-Agnostic Paper/README.md` |
| 단계별 정리 | `Projects/Action-Agnostic Paper/{1. Core Idea, 2. Technical Details, 3. Experiments, 4. Paper Writing, 5. Project Management}.md` |
| 우려사항 | `Projects/Action-Agnostic Paper/우려사항 및 대응방안.md` |
| 핵심 개념 | `Concepts/{Action-Agnostic Pretraining Framework, Two Visual Pathways, Two-Stream Image Preprocessing, Pixel-wise Channel Fusion for Behavior Representation, Target LayerNorm (V-JEPA)}.md` |
| 논문 노트 | `Sources/papers/{EgoDex (2025), V-JEPA 2 (2025)}.md` |

(Vault 루트: `/Users/bys724/LocalVault/Obsidian Vault/`)

### Vault 편집 정책 (이 프로젝트에서 자주 발생)

이 프로젝트는 Vault 노트 ↔ 저장소 코드의 양방향 작업이 잦음. 가드레일:

- ✅ Vault 노트 **읽기**: 자유 (filesystem MCP)
- ✅ 가벼운 추가 (한두 줄, 링크 걸기): 허용
- ⚠️ **신규 노트 / 폴더 구조 변경 / 대규모 노트 리팩토링**: 여기서 하지 말 것 → Vault 작업공간에서 별도로 클로드 코드를 띄워 진행
  - 이유: Vault 컨벤션(`Projects/` 숫자.md 단계 구분, 백링크 스타일, 이모지 규약)을 자동 적용받지 못함
- 🔁 반대로 Vault 작업공간에서는 이 저장소의 코드/설정을 직접 편집하지 않음. 필요 변경은 사용자에게 알리고 이 저장소에서 별도로 처리

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

2개 모델 사전학습: **Two-Stream v10** (현재 활성), **VideoMAE** (baseline)

**v10 확정 설정** (v6 base + mask_p 0.75):
```bash
# 클러스터 (sbatch, 권장 — 8 GPU DDP)
MODEL=two-stream-v10 sbatch scripts/cluster/pretrain.sbatch

# 로컬 (Docker dev-env)
docker exec -it dev-env bash
python scripts/pretrain.py --model two-stream \
    --depth 12 --num-stages 2 --use-ape --rotation-aug \
    --mask-ratio 0.5 --mask-ratio-p 0.75 \
    --v9-p-target future --v9-loss-weight-p 1.0 \
    --max-gap 60 --sample-dist triangular --sample-center 30 \
    --egodex-splits part1,part2,part3,part4,part5 \
    --epochs 50 --batch-size 64
```

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

### 3. LIBERO Fine-tuning & Evaluation

학습된 인코더로 로봇 조작 태스크 수행

```bash
# Encoder + MLP decoder fine-tuning
python scripts/eval/finetune_libero.py \
    --encoder two-stream \
    --checkpoint <ckpt> \
    --task-suite libero_spatial --epochs 50

# 시뮬레이터 rollout (success rate)
docker compose up -d libero
docker exec libero-eval python src/eval_libero.py ...
```

자세한 내용은 `docs/setup/LIBERO_TEST_GUIDE.md` 참고

## 주요 파일

| 파일 | 용도 |
|------|------|
| `scripts/pretrain.py` | Pre-training 메인 스크립트 (env-agnostic) |
| `scripts/local/download_egodex.sh` | EgoDex CDN 다운로드+추출 (로컬, `/mnt/data`) |
| `scripts/local/pretrain.sh` | 로컬 학습 launcher (Docker dev-env) |
| `scripts/cluster/download_egodex.sh` | EgoDex CDN 다운로드 (클러스터, `/proj/external_group/mrg`) |
| `scripts/cluster/extract_egodex.sbatch` | 프레임 추출 sbatch (144 CPU 병렬) |
| `scripts/cluster/README.md` | 클러스터 사용법 quickstart |
| `scripts/data/extract_droid_frames.py` | DROID 프레임 추출 |
| `scripts/data/extract_droid_actions.py` | DROID action 추출 (cross-domain probing용) |
| `scripts/eval/probe_action.py` | EgoDex action probing (v6/v10) |
| `scripts/eval/probe_action_droid.py` | DROID cross-domain probing (v6/v10) |
| `scripts/eval/finetune_libero.py` | LIBERO fine-tuning (v6/v10) |
| `scripts/eval/probe_action_v11.py` | EgoDex action probing (v11) |
| `scripts/eval/probe_action_droid_v11.py` | DROID cross-domain probing (v11) |
| `scripts/eval/finetune_libero_v11.py` | LIBERO BC fine-tune (v11) |
| `scripts/eval/visualize_attn_v11.py` | v11 attention 시각화 |
| `scripts/eval/visualize_inference.py` | 단일/다중 모델 비교 시각화 |
| `scripts/eval/visualize_sample_detail.py` | 단일 샘플 상세 (M/P channel + attention) |
| `src/models/two_stream.py` | Two-Stream 모델 (v6/v10) + 2D RoPE + MAE masking |
| `src/models/two_stream_v11.py` | Two-Stream v11 (motion-routing + dual-target) |
| `src/models/videomae.py` | VideoMAE baseline |
| `src/datasets/egodex.py` | EgoDex 데이터셋 |
| `src/datasets/droid.py` | DROID 데이터셋 |
| `src/training/pretrain.py` | Pre-training 루프 |

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

## 현재 Phase (2026-04-29)

**Phase 3-1 진입 — LIBERO BC-T 5 encoder × `libero_spatial` × seed=0 학습 중. v11-VfromM ablation도 동시 시작.**

NeurIPS 투고 목표 (~2026-05). V-JEPA은 16-frame × 384² 입력 비용으로 BC main table 제외, paper에서 별도 처리.

**Phase 1.5 (사전학습) — 🏆 v11 학습 종료 (50ep 완주). ep44 A+B+D' = +0.288 final champion 확정. ep48/ep50은 plateau.**

**완료 (Two-Stream lineup)**:
- **v4** (RoPE, mask 0.3/0.5): 48ep. Probing R²=0.197
- **v6** (APE + mask 0.5/0.5 + rotation aug): ep23 scancel, ep8 peak R²=0.259
- **VideoMAE-ours**: 50ep, R²=0.326
- **V-JEPA-ours**: 3차 발산, negative result
- **폐기 라인 (v7-big / v8 / v9)**: 모두 P stream collapse. 상세는 `docs/RESEARCH_PLAN.md`

**Two-Stream v10** (v6 base + mask_p 0.75) — 종료, ep40 plateau (+0.221):
- 50ep 거의 완주 (JobID 33570871, AIP_long 2노드×4 H100)
- ep4-8: 1차 peak +0.206 → ep20 collapse +0.137 → ep24-40 W-shape 회복 → ep40 peak +0.221 → ep44/48 plateau
- **v6 챔피언 (+0.259) 추월 실패 확정**. P-stream 내부 강화 방식의 한계

**Two-Stream v11** (Motion-Guided Attention Routing + Dual-Target) — **🏆 학습 종료**:
- **구조**: M encoder (6-layer) + P encoder (12-layer) + M decoder (3-layer, loss 없음) + P decoder 3-phase (interpreter_1 → motion-routing × 2 → interpreter_2) + shared recon_head
- **Loss**: L_t (Phase 1) + L_tk (Phase 3), masked positions only
- **Total params**: 250.9M (downstream encoder만 ~204M)
- **시작**: 2026-04-25 01:11 (JobID 33594155, AIP 2노드×4 H100). 23:30 TIMEOUT → 33600621 resume (--time=3d)
- **종료**: 2026-04-28 07:19 (resume 2-05:39 elapsed). 50 epoch 완주. ckpt dir `20260426_014333/` (ep16/20/24/28/32/36/40/44/48 + latest=ep50)

**v11 학습 추이** (loss + 표현 진단):

| Epoch | L_total | L_t | L_tk | std_p | cos_intra_p |
|-------|---------|-----|------|-------|-------------|
| 1 | 0.0196 | 0.0109 | 0.0088 | 0.349 | 0.866 |
| 4 | 0.0057 | 0.0044 | 0.0014 | 0.210 | 0.897 |
| 8 | 0.0043 | 0.0038 | 0.00052 | 0.009 | 1.000 |
| 12 | 0.0024 | 0.00197 | 0.00039 | 0.004 | 1.000 |
| 50 | 0.00220 | — | — | — | — |

→ Loss 단조 감소. P encoder CLS는 collapse (cos_intra_p≈1.0)이지만 patches는 healthy — 75% MAE 복구가 작동

**v11 ep4-ep50 — Representation 비교 (12 mode)**

4 위치: A (M encoder), B (P encoder), D' (motion-routing 후), D (Phase 3 final)

| Mode | ep4 | ep8 | ep12 | ep16 | ep20 | ep24 | **ep44** | ep48 | ep50 |
|------|-----|-----|------|------|------|------|----------|------|------|
| `patch_mean_m_enc` (A) | +0.170 | +0.176 | +0.208 | +0.213 | +0.220 | +0.222 | **+0.267** ★ | +0.264 | +0.265 |
| `patch_mean_p_enc` (B) | -0.041 | -0.025 | 0.000 | -0.001 | -0.002 | -0.004 | -0.003 | -0.000 | -0.001 |
| `patch_mean_p_state_after_routing` (D') | +0.121 | +0.066 | +0.072 | +0.077 | +0.098 | +0.113 | +0.135 | +0.138 | +0.129 |
| `patch_mean_p_features_tk` (D) | +0.023 | +0.055 | +0.054 | +0.047 | +0.060 | +0.057 | +0.050 | +0.049 | +0.048 |
| `patch_mean_concat_enc_only` (A+B) | +0.160 | +0.168 | +0.200 | +0.211 | +0.213 | +0.224 | +0.259 | +0.263 | +0.263 |
| `patch_mean_concat_enc_phase3` (A+D) | +0.143 | +0.194 | +0.219 | +0.217 | +0.230 | +0.232 | +0.264 | +0.264 | **+0.267** |
| `patch_mean_concat_enc_d_prime` (A+D') | +0.149 | +0.166 | +0.153 | +0.205 | +0.196 | +0.232 | +0.284 | +0.283 | +0.282 |
| `patch_mean_concat_p_enc_d_prime` (B+D') | +0.135 | +0.011 | +0.076 | +0.079 | +0.087 | +0.107 | +0.137 | +0.139 | +0.139 |
| **`patch_mean_concat_all`** (A+B+D') | +0.114 | +0.094 | +0.178 | +0.223 | +0.185 | +0.234 | **+0.288** ★★ | +0.281 | +0.279 |
| `cls_m_enc` (A CLS) | +0.066 | +0.155 | +0.162 | +0.163 | +0.172 | +0.158 | +0.125 | +0.123 | +0.123 |
| `cls_p_enc` (B CLS) | -0.059 | -0.011 | -0.008 | -0.010 | -0.009 | -0.013 | -0.002 | -0.002 | -0.002 |
| `cls_concat_enc` (A+B CLS) | -0.048 | +0.092 | +0.148 | +0.139 | +0.162 | +0.140 | +0.114 | +0.118 | +0.113 |

**🏆 최종 결론 — ep44 A+B+D' final champion**:
- **ep44 A+B+D' = +0.288** — v6 ep8 챔피언 (+0.259) **추월 +0.029** ★★ (final champion)
- **VideoMAE +0.326까지 격차 -0.038** (ep24 -0.092 → 절반 이상 좁힘)
- **ep44~ep50 plateau 확정**:
  · A+B+D' ep44 +0.288 → ep48 +0.281 → ep50 +0.279 (-0.010, 미세 over-tightening)
  · A+D' ep44 +0.284 → ep50 +0.282 (stable plateau, 가장 robust)
  · A+D ep44 +0.264 → ep50 **+0.267** (미세 신피크, 0.003 향상)
- **W-shape 회복 패턴 확정**: ep12 +0.219 → ep20 +0.185 dip → ep24 +0.234 → ep44 +0.288 (LR cosine decay 후반 ep24→44 +0.054 큰 도약)
- ep44 이후는 LR≈0 정체 구간으로 새 학습 효과 없음 → ep44가 진짜 peak
- **사용자 v11 설계 가설 정량 확정**:
  · 3-way concat (A+B+D')이 best — M+P+motion-routed P 상보적
  · A+D' > A+D — interpreter_2는 decoder wrapper, motion-routing 직후가 더 좋은 representation
  · CLS는 모두 약화 추세, patch_mean이 정답

**v11 Cross-domain DROID probing** (사용자 직감 검증, ep12 기준)

| Gap (DROID 15Hz) | VideoMAE | v11 best | 격차 |
|------------------|----------|----------|------|
| 1 (0.07초) | -0.006 | -0.005 | +0.001 |
| 10 (0.67초) | -0.006 | +0.006 (A+B) | +0.012 |
| **15 (1초)** ★ | **-0.035** | **+0.005 (A+B)** | **+0.040** |
| 30 (2초) | -0.028 | -0.010 | +0.018 |

모든 gap에서 v11 우위. gap=15 (EgoDex 학습 분포 1초)에서 격차 가장 큼.

**Cross-domain LIBERO BC** (libero_spatial 30 ep, val MSE):

| Encoder | best val MSE |
|---------|-------------|
| VideoMAE-ours ep50 | **0.0286** |
| v11 ep12 A+D | 0.0290 |

- 거의 동등 (격차 +0.0004). v11 ep12는 학습 초반 ckpt — ep44/ep50으로 재측정 필요
- VideoMAE 우위는 in-domain (EgoDex) 동등 학습 기반 advantage. v11의 진짜 강점은 cross-domain (DROID gap=15 +0.040)

**시각화 산출물 추가**: `docs/architecture/attn_v11_ep{48,50}.png`

---

## Phase 3-1 LIBERO BC-T 진행 상황 (2026-04-29 시작)

### NeurIPS 투고 실험 매트릭스

**Main BC table** (5 encoder × 3 suite × 3 seed = 45 BC runs):
- Two-Stream v11 ep44 (ours, A+D' mode)
- VideoMAE-ours ep50 (controlled comparison)
- DINOv2-Base, SigLIP-Base, VC-1-Base (공개 가중치)
- ~~V-JEPA 2.1~~ — 16-frame × 384² 비용으로 BC 비현실적 (sanity 측정 50ep ≈ 30-100일). probing 결과만 paper에 포함, BC main table 제외.

**Suite**: `libero_spatial` (1차 진행 중) → `libero_object` → `libero_goal` (1차 결과 본 후 추가 큐)

### Ablation (paper Section: Architecture Analysis)

**A1: Motion-routing source** — paper title-level novelty 검증
- Control: v11 (Q,K←M, V←P) — 기존 ep50 ckpt
- Ablation: v11-VfromM (Q←P, K,V←M) — 표준 cross-attention. 50ep × part1-5 신규 학습
- 평가: EgoDex probing (12 mode) + DROID probing (gap 1/10/15/30, A+B mode)
- Param count 동일 (208.33M), 단일 변인

**A2: Feature extraction position** — 이미 완료된 12-mode 분석 그대로 paper에 포함
- A (M enc), B (P enc), D' (motion-routing 후), D (Phase 3 final), 조합 8개
- 추가 비용 없음, paper Table로 정리만

### 핵심 코드 추가

| 파일 | 변경 |
|------|------|
| [`src/models/two_stream_v11.py`](src/models/two_stream_v11.py) | `MotionRoutingBlock`에 `routing_mode` 파라미터 (`v_from_p` / `v_from_m`) |
| [`scripts/pretrain.py`](scripts/pretrain.py) | `--v11-routing-mode` CLI 인자 |
| [`scripts/cluster/pretrain.sbatch`](scripts/cluster/pretrain.sbatch) | `V11_ROUTING_MODE` 환경변수 |
| [`scripts/eval/finetune_libero_bct.py`](scripts/eval/finetune_libero_bct.py) | `_align_actions(dist, actions)` (V-JEPA T_out trim 대응) + 첫 batch dtype 디버그 로그 |
| [`src/encoders/adapters/single_frame.py`](src/encoders/adapters/single_frame.py) | SigLIP은 `SiglipVisionModel` 사용, VC-1은 `vc_models` 패키지 사용 (이전 loader 두 인코더 모두 fail) |
| [`scripts/cluster/finetune_libero_bct.sbatch`](scripts/cluster/finetune_libero_bct.sbatch) | `HF_HUB_OFFLINE=1` (CLIP 다운로드 race 회피) |

### 진행 중 잡 (2026-04-29 06:00 시점)

| JobID | 작업 | 자원 | 상태 |
|-------|------|------|------|
| 33615385 | BC-T two-stream-v11 ep44 × spatial | AIP 1×1 H100 | RUNNING (~43h 예상) |
| 33615386 | BC-T videomae-ours best × spatial | AIP 1×1 H100 | RUNNING (~25h) |
| 33615387 | BC-T dinov2 × spatial | AIP 1×1 H100 | RUNNING (~20h) |
| 33615391 | BC-T siglip × spatial | AIP 1×1 H100 | RUNNING (~20h) |
| 33615392 | BC-T vc1 × spatial | AIP 1×1 H100 | RUNNING (~20h) |
| 33615394 | v11-VfromM sanity | AIP 1×1 H100, 30min | PENDING |
| 33615395 | v11-VfromM full ablation 50ep × part1-5 | AIP_long 2×4 H100, 3.5d | PENDING |

### 다음 단계

1. **1차 BC-T spatial 결과 (~2일 후)** → `libero_object`/`goal` 추가 + seed=1, 2 병렬 큐
2. **v11-VfromM full (33615395) 진행 중 ep4/ep8 probing 가능 시점 확인** (~5-6시간 후 시작 후)
3. **LIBERO Rollout setup (로컬 워크스테이션)** — BC ckpt 로컬 전송 후 mujoco rollout 평가. main paper의 closed-loop 증거
4. **V-JEPA 별도 처리 결정** — 옵션 A (pre-extraction) vs 옵션 C (skip from main table). 1차 결과 본 후

자세한 내용은 [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md), probing 결과는 [`docs/PROBING_GUIDE.md`](docs/PROBING_GUIDE.md), 클러스터 자원·잡 추적은 [`docs/cluster_sessions.md`](docs/cluster_sessions.md) 참고

## 트러블슈팅 로그

### 재발 가능 — 숙지 필요

- **Slurm `--time` 부족 timeout (반복 사고)**: 2026-04-29 BC-T 5 encoder 학습에서 dinov2/siglip/vc1 모두 `--time=24h`로 제출 후 epoch당 시간 측정해보니 적자 -4.5h ~ -6.3h 확인. 8h 학습 후 cancel + 재제출 (~24 GPU·h 손실). **잡 제출 전 epoch당 시간 × 총 ep × 1.3 안전 마진 계산**, 추정 어려우면 partition max (`2-00:00:00`)로. multi-encoder 동시 제출 시 가장 느린 모델 기준 통일. 자세한 가이드는 "클러스터 세션 로깅 → `--time` 결정 가이드" 참고.
- **Sanity checkpoint → 본 학습 auto-resume 오염**: 서로 다른 학습 설정의 체크포인트가 같은 디렉토리에 있으면 scheduler/optimizer 상태 오염 (T_max, lr, batch_size). **sanity 잡은 반드시 `CHECKPOINT_SUFFIX` 사용**.
- **Slurm DDP 3대 함정**: (1) `--gpus-per-task=1` 대신 `--gres=gpu:N` 사용 (NCCL PCI 탐색 실패 방지), (2) `MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))` (포트 충돌 방지), (3) srun에서 `$CONDA_PREFIX/bin/python` 절대 경로 사용
- **로그인 노드 프로세스 제한**: 대용량 다운로드는 순차 실행, 여러 다운로드 동시 실행 금지. `gsutil -m`은 thread/process count 제한 필수 (`-o parallel_thread_count=8 -o parallel_process_count=4`)
- **Scratch stage-in 비현실적**: 소형 파일 수백만 개는 메타데이터 병목. GPFS 직접 읽기 사용

### 해결 완료 — 참고용

- **BF16 autocast**: `use_bf16` 플래그로 scaler와 분리 (GradScaler 불필요). 37.6→62.2 samples/sec
- **SSIM BF16 NaN**: SSIM 연산 FP32 강제 + sigma clamp(min=0)
- **V-JEPA 발산**: (1) EMA target 기반 학습은 LR warmup 필수, (2) 2-frame에서는 mask ratio 대폭 완화 필요 (16-frame temporal redundancy 없음). 결국 3차 모두 발산 → negative result
