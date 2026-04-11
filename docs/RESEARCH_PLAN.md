# Action-Agnostic Visual RL Research Plan

**마지막 업데이트**: 2026-04-11
**연구 질문**: 행동 정보 없이 학습한 시각 표현이 더 범용적인가?

---

## 실험 대상 모델 (Encoder 5종)

| 구분 | Encoder | 사전학습 데이터 | 파라미터 | 입력 | 학습 주체 |
|------|---------|---------------|---------|------|----------|
| **제안** | **Two-Stream v4 (ours)** | EgoDex (1.3 TB) | ~213M | (img_{t-1}, img_t) | **우리 학습** |
| **Same-data ablation** | **VideoMAE (ours)** | EgoDex (1.3 TB) | ~101M | (img_{t-1}, img_t) | **우리 학습** |
| 로봇 특화 baseline | VC-1-Base | Ego4D + Manipulation | 86M | img_t (2 frame concat) | 공개 가중치 |
| 범용 SSL baseline | DINOv2-Base | LVD-142M (웹) | 86M | img_t (2 frame concat) | 공개 가중치 |
| 범용 VL baseline | SigLIP-Base | WebLI (웹) | 86M | img_t (2 frame concat) | 공개 가중치 |

**공정성 원칙**:
- 모든 encoder에 **동일한 입력 형식** — 단일 프레임 encoder도 `(img_{t-1}, img_t)` 2개를 feature 레벨 concat
- 모든 encoder는 downstream 실험에서 **frozen** (학습 안 함)
- downstream에서 유일하게 학습되는 것은 **MLP action decoder** (Phase 3) 또는 **LoRA + projection** (Phase 3B)

---

## 실험 로드맵

### Phase 1: EgoDex 사전학습 → Hand Pose Probing ✅ 완료

**목표**: 제안한 Two-Stream 구조의 표현 품질 검증 (within-domain)

**작업**:
- Two-Stream v4, VideoMAE를 EgoDex part1~5 일부로 반복 ablation
- frozen encoder → EgoDex test에서 hand pose linear probe (R²)
- Architecture / MAE masking / gap 분포 / composition consistency 전체 ablation

**결과**: v4 설정 확정 (d=12, s=2, M=0.3/P=0.5, max_gap=60 triangular).
상세 ablation 결과는 하단 "Phase 1 Ablation 결과" 참고.

### Phase 1.5: 데이터 스케일업 (Full EgoDex) 🔄 진행 중

**목표**: 전체 EgoDex (part1~5, 314k videos)로 최종 체크포인트 확보

**작업**:
- Two-Stream v4 50 epoch 학습 (IBS 클러스터 8 H100 DDP) — **현재 진행 중**
- VideoMAE 50 epoch 학습 (동일 데이터, baseline 역할)
- 학습 후 hand pose probing으로 품질 확인 (part1 기준 대비 개선 여부)

**산출물**: `results/checkpoints/two_stream/` 및 `.../videomae/` 에 best_model.pt

### Phase 2: Cross-Domain Action Probing (DROID) ⏸️ 대기

**목표**: EgoDex 표현이 **로봇 도메인**으로 전이되는지 검증

**작업**:
1. frozen encoder로 DROID에서 로봇 행동 linear/MLP probe
2. 5개 encoder 모두 동일 probe 프로토콜로 비교
3. 지표: R², cosine similarity

**데이터**: DROID v1.0.1 (95k episodes Franka, gsutil rsync 진행 중)

**Go/No-Go**:
- 우리 encoder가 baseline 대비 우위 → Phase 3 순항
- 열세 → 원인 분석 (EgoDex와 DROID 도메인 갭 vs 표현 품질)

### Phase 3: LIBERO BC (메인 downstream 실험) ⏸️ 대기

**목표**: "표현에 인코딩된 action 정보가 **실제 제어**에 유용한가"를 검증

**실험 구조**:
```
frozen encoder → MLP action decoder → 7-DOF action
  (5종 비교)     (학습 대상)          (task-conditioned)
```

**학습 대상**: MLP action decoder만. encoder는 모두 frozen.
**학습 데이터**: LIBERO demonstrations (BC, supervised regression)
**태스크 구분**: per-task policy 또는 task ID embedding (자연어 사용 안 함)

**평가**:
- LIBERO 시뮬레이터에서 closed-loop rollout
- Task suite: `libero_spatial` (main), `libero_object`, `libero_goal`, `libero_10` (supp)
- Task당 최소 50 trials, 3 seed 평균
- 지표: success rate (%)

**공정 비교 체크리스트**:
- [ ] 모든 encoder에 동일 MLP 아키텍처 (hidden dim, depth)
- [ ] 단일 프레임 encoder는 (img_{t-1}, img_t) feature concat으로 입력 맞춤
- [ ] 동일 action 정규화, lr, epoch, batch size
- [ ] 동일 trial 수 + seed

**기대 결과**: Two-Stream v4 > VC-1 > VideoMAE > DINOv2 ≈ SigLIP

### Phase 3B: OpenVLA 통합 (supplementary) ⏸️ 대기

**목표**: "기존 SOTA VLA에 우리 encoder를 넣으면 성능이 오르는가"를 직접 검증

**실험 구조**:
```
Two-Stream encoder → projection → Llama 7B (OpenVLA backbone) → action tokens
  (frozen 또는      (신규 학습)   (LoRA fine-tune)
   partial fine-tune)
```

**학습 대상**:
- Projection layer (encoder dim → Llama embed dim): **신규 학습**
- Llama 7B: **LoRA 어댑터만 학습** (rank 32, OpenVLA 논문 setup)
- Two-Stream encoder: 실험 옵션 (frozen 권장, 필요 시 마지막 N layer fine-tune)

**학습 데이터**: LIBERO RLDS 포맷 (OpenVLA 기존 인프라 활용)

**실험 대상**:
| 모델 | Encoder | 방식 | 역할 |
|------|---------|------|------|
| OpenVLA (원본) | SigLIP (원본) | LoRA FT | Reference (기존 결과 libero_spatial 40%) |
| OpenVLA + Two-Stream | Two-Stream v4 | LoRA FT | 주된 비교 대상 |

**평가**: libero_spatial만 (scope 제한). 필요 시 다른 suite로 확장.

**기존 인프라** (대부분 준비됨):
- OpenVLA 코드: https://github.com/openvla/openvla (fine-tuning 스크립트 포함)
- LIBERO RLDS 데이터: `~/.cache/openvla/datasets/modified_libero_rlds/` (9.6 GB)
- OpenVLA fine-tuned 체크포인트: `data/checkpoints/openvla-libero/`
- Docker 환경: `docker/openvla-libero/`
- Rollout 서버: `src/eval_libero.py`

**신규 작업**:
- [ ] OpenVLA의 `vision_backbone` 교체 로직 작성 (~수십 줄)
- [ ] Two-Stream output → Llama embedding projection layer
- [ ] OpenVLA `vla-scripts/finetune.py` 에 encoder 교체 경로 삽입

**의도적 제외** (scope 관리):
- Pi0, RT-2, LAPA — 인프라 불완전 또는 비공개. Future work.
- 여러 IL 알고리즘 비교 (BC-RNN, Diffusion Policy 등) — 필요 시 rebuttal에서 추가 가능.
- Full fine-tuning (non-LoRA) — 비용 대비 효과 낮음.

---

## 논문 스토리 3단계

| 단계 | 주장 | 뒷받침 Phase | 지표 |
|------|------|-------------|------|
| 1 | 표현에 action 정보가 **인코딩되어 있다** | Phase 1 (EgoDex) + Phase 2 (DROID) | R², cosine similarity (open-loop) |
| 2 | 인코딩된 정보가 **실제 제어에 유용하다** | Phase 3 (LIBERO BC rollout) | success rate (closed-loop) |
| 3 | action-agnostic 사전학습이 **범용 비전 feature보다 낫다** | Phase 3 (5종 encoder 비교) + Phase 3B (SOTA 통합) | 순위 + success rate 차이 |

---

## 요약: 학습 작업 분류

### 🔥 우리가 학습해야 하는 모델 (from scratch)
1. **Two-Stream v4** on EgoDex full (50 epoch, 8 H100) — **현재 진행 중**
2. **VideoMAE** on EgoDex full (50 epoch, 8 H100) — 다음 차례

### 📦 공개 가중치 그대로 사용 (학습 없음)
3. **VC-1-Base** (Meta eai-vc 저장소)
4. **DINOv2-Base** (Facebook)
5. **SigLIP-Base** (Google)

### 🔧 Downstream에서 학습하는 것 (encoder는 frozen)
- **Phase 2**: MLP probing head (5 encoder × DROID)
- **Phase 3**: MLP action decoder (5 encoder × 4 task suite)
- **Phase 3B**: Projection layer + Llama LoRA 어댑터 (Two-Stream + OpenVLA backbone, libero_spatial)

---

## 모델

| 모델 | 학습 방식 | 목적 |
|-----|----------|------|
| Two-Stream | Future prediction (M/P 채널 분리 + CLS 교환) | 제안 모델 |
| Single-Stream | Future prediction (9ch 단일 ViT) | Baseline (구조 효과 검증) |
| VideoMAE | Masked reconstruction (75%, 공식 Base 설정) | Comparison |

### 비교 의미
- Single-Stream vs VideoMAE → 학습 방식 차이 (prediction vs reconstruction)
- Two-Stream vs Single-Stream → 같은 M/P 입력, 구조 차이 (interleaved vs single)

---

## 학습 설정

```
인스턴스: H100 워크스테이션 (x2, 81GB each) — 로컬 권장
epochs: 30
batch_size: 64 (single-GPU)
max_gap: 30 (~1초)
sample_decay: -1 (linear, 큰 gap 선호 — gap에 비례한 샘플링 확률)
데이터: EgoDex frames (로컬: /mnt/data/egodex_frames/)
```

**Gap 샘플링 전략 (v4에서 변경)**:
- 기존 (v1~v3): uniform (gap 1~30 동일 확률)
- v4: **linear weighting** (`sample_decay=-1`, gap에 비례)
  - gap<=5: 3.2%, gap>=20: 59.1%
  - 이유: 작은 gap에서 M/P가 너무 비슷 → cross-stream 정보 교환 압력 부족
    → 학습 데이터의 색상/배경 bias를 학습하게 됨 (DROID unseen에서 확인)
  - 큰 gap에서 프레임 차이가 커야 CLS exchange를 통한 정보 교환이 필수적

```bash
# 로컬 (H100 x2)
bash scripts/pretrain_local.sh --model two-stream        # 특정 모델만
bash scripts/pretrain_local.sh --sanity                  # Sanity test

# 8x H100 서버 (본 학습)
torchrun --nproc_per_node=8 scripts/pretrain.py \
    --model two-stream --depth 12 --num-stages 2 \
    --mask-ratio 0.3 --sample-decay -1 \
    --egodex-splits part1 part2 part3 part4 part5 \
    --epochs 30 --batch-size 64
```

### 8x H100 서버 체크리스트 (본 학습 전)

**환경 확인:**
- [ ] PyTorch 2.4+ (torch.compile 최적화)
- [ ] NCCL 버전 + GPU 간 연결 확인 (`nvidia-smi topo -m` — NVLink 여부)
- [ ] 스토리지 I/O 속도 (프레임 JPG 로딩 — NVMe 권장)

**코드 수정:**
- [ ] DataParallel → DistributedDataParallel 전환 (8 GPU 필수)
- [ ] lr scaling: `lr = base_lr * num_gpus` (batch 512 대응)
- [ ] warmup 스케줄 조정 (큰 batch에서 안정화)
- [ ] torch.compile() 적용 테스트

**데이터 전송:**
- [ ] EgoDex part1~5 프레임 전체 전송 (S3 또는 rsync)
- [ ] DataLoader: `num_workers=8~16`, `pin_memory=True`

**리소스 분배:**
| 작업 | 서버 | 예상 시간 |
|------|------|----------|
| Two-Stream v4 사전학습 | 8x H100 | 며칠 |
| VideoMAE 사전학습 | 8x H100 | 며칠 |
| Action probing (full) | 로컬 2x H100 | 수 시간 |
| LIBERO fine-tuning | 로컬 2x H100 | 수 시간 |
| LIBERO 시뮬레이터 평가 | 로컬 1 GPU | 수 시간 |

---

## 현재 상태 (2026-04-10)

### Phase 1 완료 (모델 설계 + ablation)
- [x] Two-Stream v4 아키텍처 확정: d=12, s=2, MAE M=0.3/P=0.5, max_gap=60 triangular
- [x] Architecture ablation, MAE masking, composition consistency, gap 분포 실험 전부 완료
- [x] VideoMAE 30 epoch baseline 완료
- [x] Action probing 초기 결과 확보 (EgoDex + DROID)

### IBS 클러스터 환경 구축 완료
- [x] EgoDex part1~5 + test CDN 다운로드 (1.6 TB, 5-way 병렬 ~2시간)
- [x] EgoDex 프레임 추출 완료 (314,839 train videos, ~1.25 TB frames)
  - part1: 46,234 / part2: 95,125 / part3: 53,779 / part4: 44,129 / part5: 75,572 / test: 3,243
- [x] DROID 다운로드 진행 중 (gsutil rsync)
- [x] scripts/{local,cluster}/ 환경별 launcher 분리
- [x] conda env 2개: aavrl-extract (cv2), aavrl-train (torch 2.6+cu124)
- [x] 클러스터 저장소 정책 확인: GPFS 16 GB/s, scratch 7 TB (GPU 노드 로컬 NVMe)

### DDP 전환 + 학습 가속
- [x] DataParallel → DistributedDataParallel 변환 (Slurm srun 직접, torchrun 불필요)
- [x] DDP sanity Stage 2 통과 (1 GPU, 37.6→BF16 검증중)
- [x] DDP sanity Stage 3 통과 (2 GPU 1 node, NCCL over NVLink, 90.8 samples/sec)
- [ ] DDP sanity Stage 4 (2 node multi-node NCCL) — 큐 대기
- [x] AMP BF16 autocast 버그 수정 (기존: scaler 기반 조건문 → 실제 FP32 실행. ~2배 성능 손실)
- [x] Fused AdamW 적용 (5-10% 가속)
- [x] torch.compile 선택적 지원 (--compile 플래그)
- [x] pretrain.sbatch scratch stage-in/out 구현 (USE_SCRATCH=1)

### 로컬 워크스테이션 (이전 완료)
- [x] Two-Stream v1~v4 iterative ablation (part1 기준, 30 epoch)
- [x] VideoMAE 30 epoch 완료
- [x] DROID 프레임 추출 완료 (ext1/ext2/wrist)
- [x] Action probing 코드 (EgoDex, DROID, Bridge)

### Action Probing 초기 결과 (2026-03-31, 500 videos, linear probe, 20 epochs)

**모델 간 공정 비교 (patch mean pool 기준)**:

| Encoder | Split | R² | Cosine Sim | 비고 |
|---------|-------|----|------------|------|
| VideoMAE | part1 (학습) | 0.416 | 0.494 | |
| Two-Stream (patch_mean) | part1 (학습) | 0.389 | 0.482 | gap 0.027 |
| Two-Stream (patch_mean_p) | part1 (학습) | 0.401 | 0.490 | P stream만 |
| Two-Stream (patch_mean) | part4 (미사용) | 0.237 | 0.369 | 일반화 |
| VideoMAE | part4 (미사용) | 0.137 | 0.330 | 일반화 |

**Two-Stream CLS 방식별 비교 (part1)**:

| 방식 | R² | 설명 |
|------|-----|------|
| CLS average | 0.326 | (m_cls + p_cls) / 2 |
| CLS concat | 0.157 | [m_cls; p_cls] |
| CLS m_only | 0.153 | M stream CLS |
| CLS p_only | 0.169 | P stream CLS |
| patch_mean (M+P) | 0.389 | 전체 패치 mean pool |
| patch_mean_m | 0.184 | M 패치만 |
| patch_mean_p | 0.401 | P 패치만 |

**해석**:
1. patch mean pool 기준 VideoMAE와 Two-Stream 격차가 작음 (0.416 vs 0.389)
2. **미사용 데이터(part4)에서 Two-Stream이 우세** (0.237 vs 0.137) → 일반화 성능
3. CLS average(768-dim 1개)로 patch mean(196개 평균)의 83% 달성 → CLS exchange 효과
4. P stream > M stream: 외형 정보가 hand pose probing에 더 유리
5. 모든 조건에서 R² < 0.7 — 데이터 규모 확대/MLP probe 등으로 개선 여지

### Gap별 Embedding 비교 (part4, 미사용 데이터, 2026-04-01)

| Embedding | dim | gap=1 | gap=5 | gap=10 |
|-----------|-----|-------|-------|--------|
| TS CLS average | 768 | 0.159 | 0.354 | 0.364 |
| TS CLS concat | 1536 | -0.150 | 0.330 | 0.353 |
| TS m_only (CLS) | 768 | 0.046 | 0.276 | 0.359 |
| TS p_only (CLS) | 768 | -0.266 | 0.311 | 0.325 |
| TS patch_mean | 768 | 0.225 | 0.466 | 0.532 |
| **TS patch_mean_concat** | **1536** | **0.117** | **0.489** | **0.585** |
| VM patch_mean | 768 | 0.138 | 0.474 | 0.571 |

**추가 해석**:
1. **gap 효과**: gap=1의 delta는 노이즈 수준. gap=10에서 전반적으로 R²가 2~4배 상승
2. **M stream의 temporal 특성**: gap=1 최하위(0.046) → gap=10 P를 추월(0.359 > 0.325). M/P 분리 설계가 의도대로 작동
3. **patch_mean_concat(0.585) > VideoMAE(0.571)**: M/P 분리 보존 시 Two-Stream이 역전
4. **CLS concat < CLS average**: CLS exchange로 이미 동질화되어 concat이 차원만 증가. CLS는 average 방식이 적합
5. **CLS average(0.364)는 patch_mean(0.532)의 68%**: CLS bottleneck에 개선 여지

### v4 Masking 비교 결과 (5 epoch, 2026-04-06)

| | pmc gap=1 | pmc gap=5 | pmc gap=10 | CLS avg gap=10 |
|---|---------|---------|----------|--------------|
| v4_mask30 | -2.501 | **0.277** | **0.437** | **0.397** |
| v4_nomask | -3.203 | 0.255 | 0.418 | 0.297 |

- mask30이 모든 조건에서 우세
- **CLS 정보 밀도 33% 개선** (0.397 vs 0.297)
- masking이 표현 품질 향상 + 연산 30% 절약 + CLS exchange 강제

**P stream self-sufficiency 문제**: P는 현재 프레임 외형을 직접 입력받아 CLS exchange 없이도
낮은 loss 달성 가능 → 빠른 수렴 + shallow 표현 + blurry 출력. 해결: P의 masking 비율을
M보다 높게 설정 (`--mask-ratio-p`). P의 자기 입력 의존성을 제한하여 cross-stream 교환 압력 강화.

### Composition Consistency 실험 결과 (30 epoch, 500 videos, 동일 조건 비교)

| | Final train | Final eval | pmc g=5 | pmc g=10 | CLS avg g=10 | M loss | P loss |
|---|-------------|-----------|---------|----------|--------------|--------|--------|
| **v4_base (gap30 linear)** | 0.0020 | 0.0387 | 0.246 | 0.290 | **0.239** | 0.0015 | 0.0006 |
| **v4_base_gap60_tri** | 0.0022 | **0.0326** | **0.260** | **0.344** | 0.218 | 0.0016 | 0.0006 |
| v4_comp_sg | 0.0052 | 0.0655 | 0.187 | 0.280 | 0.217 | 0.0016 | 0.0015 |
| v4_comp_grad | 0.0076 | 0.0559 | 0.219 | 0.369 | -0.009 | 0.0028 | 0.0026 |

**핵심 발견**:

1. **Composition loss는 불필요** — composition 없는 base가 pixel loss/probing 모두 우세.
   - sg는 pixel loss가 2.6배 높고 probing도 base 이하
   - grad는 CLS collapse(-0.009), patch만 살아남음
   - 학습 시간 5배 증가 + 메모리 3배 → 이점 없음

2. **max_gap 확장 (30 → 60, triangular)이 더 효과적**:
   - eval loss 16% 개선 (0.0387 → 0.0326)
   - probing pmc gap=10: 0.290 → 0.344 (+0.054)
   - composition 없이 같은 data로 더 나은 결과

3. **M/P loss 균형은 표현 품질과 무관**:
   - base (M=0.0015, P=0.0006): M>>P인데 probing 최고
   - comp_sg (M≈P=0.0016): 균형 맞췄지만 probing 낮음
   - **M/P 균형 자체가 목표가 아님**, 전체 표현 품질이 중요
   - sg에서 M≈P가 된 이유: composition의 multi-pair 학습 + compositor K/V gradient 상호작용 (P 마스킹 50% 때문이 아니고, gap 때문도 아님)

### 확정된 v4 설정

```
depth=12, num_stages=2 (6 blk/stage, CLS exchange 2회)
mask_ratio=0.3 (M), mask_ratio_p=0.5 (P)
max_gap=60, sample_dist=triangular, sample_center=30
2D RoPE
Composition 없음 (compositor 제거)
```

### 다음 단계 (2026-04-10)
- [ ] DDP sanity Stage 4 완료 (멀티노드 NCCL 검증)
- [ ] **Full training (IBS 8 H100)**: Two-Stream v4 50 epoch
- [ ] **Full training (IBS 8 H100)**: VideoMAE 50 epoch
- [ ] DROID action probing (DROID 다운로드 완료 후)
- [ ] LIBERO fine-tuning + rollout (encoder 비교)
- [ ] OpenVLA encoder 교체 실험

### 학습 전략 (2026-04-10, IBS 클러스터)

**환경**: IBS olaf 클러스터, AIP/AIP_long 파티션, 노드당 H100 × 4

**본 학습 계획**:
- 2 노드 × 4 GPU = 8 H100 DDP, AIP_long (14일 max)
- Per-GPU batch 64, global batch 512, lr=2e-4 (square-root scaling)
- Scratch stage-in (~10분) → 로컬 NVMe에서 학습 → 체크포인트 stage-out
- 예상 시간: ~3.5일 (50 epoch), 비용 ~170만원

**대안 (스케줄링 어려울 시)**:
- 1 노드 × 4 GPU, AIP (3일 max), 25 epoch씩 2번 `--resume`
- 비용 동일, wall-clock 2배

---

## Architecture Ablation (완료)

**목적**: depth(표현력)와 CLS exchange 빈도의 효과 분리

### 설정

| Config | depth | stages | blk/stage | exchange | Params |
|--------|-------|--------|-----------|----------|--------|
| 기존 | 12 | 3 | 4 | 3 | 193M |
| A | 6 | 3 | 2 | 3 | 135M |
| B | 6 | 2 | 3 | 2 | 128M |
| C | 4 | 2 | 2 | 2 | 100M |

### 결과 (3 epoch 간이 학습, part4 probing)

| Config | blk/stage | Epoch 3 Loss | pmc gap=5 | pmc gap=10 | cls avg gap=10 |
|--------|-----------|-------------|-----------|------------|----------------|
| 기존 (30ep) | 4 | 0.0009 | 0.489 | **0.585** | 0.364 |
| A | 2 | 0.0017 | 0.212 | 0.359 | 0.243 |
| **B** | **3** | **0.0017** | **0.256** | **0.405** | **0.255** |
| C | 2 | 0.0019 | 0.175 | 0.376 | 0.138 |

### 결론

1. **B > A** (같은 depth=6): stage당 block 수가 많을수록 좋음 (3 > 2). 충분히 처리 후 교환이 효과적
2. **C도 선전** (100M으로 A 135M에 근접): depth보다 blk/stage가 중요할 수 있음
3. **최적 설정: blk/stage를 최대화** → 기존 d=12에서 s=3→s=2로 변경 (4→6 blk/stage)

### 확정 아키텍처 (v4)

```
depth=12, num_stages=2 (6 blocks/stage, CLS exchange 2회)
→ ~186M params (기존 193M 대비 약간 감소)
```

---

## Composition Consistency 실험

### 핵심 아이디어

시간 간격(gap)은 temporal 다양성이 아니라 **변화의 크기에 대한 간접적 표현**.
(t1, t2, t3) triplet으로 **변화의 합성 구조**를 검증하고, encoder가 구조화된 change embedding을 만들도록 유도.

### 동기

- 현재 encoder는 (t1, t2) 쌍으로 단일 변화만 학습
- 실제 조작은 연속적 변화의 합성: "물체를 집고(Δ12) → 옮기는(Δ23)" 과정이 전체 변화(Δ13)
- Δ12 + Δ23 ≈ Δ13 이 embedding 공간에서 성립하면, 표현이 진정으로 변화의 구조를 포착하고 있다는 증거

### 학습 구조

```
데이터: (img_t1, img_t2, img_t3) — 기존 multi-gap 확장, 3장 샘플링

encoder(t1,t2) → cls_12   (구간 1: 변화 크기 small~medium)
encoder(t2,t3) → cls_23   (구간 2: 변화 크기 small~medium)
encoder(t1,t3) → cls_13   (전체 구간: 변화 크기 large)

[Loss 1] Both-Predict-Future × 3 (주된 학습 신호)
  구간 (1,2): M/P decoder → predict img_t2
  구간 (2,3): M/P decoder → predict img_t3
  구간 (1,3): M/P decoder → predict img_t3
  = 기존 loss를 3개 쌍에 적용

[Loss 2] Composition Consistency (auxiliary)
  compositor(cls_12, cls_23) → predicted_cls_13
  L_comp = ||predicted_cls_13 - sg(cls_13)||²

Total Loss = Loss_1 + α * Loss_2   (α=0.1로 시작)
```

### Compositor 구조: Learnable Query + Cross-Attention

입력 CLS를 변형하지 않고, learnable query가 4개 CLS를 참조하여 합성 결과 생성.
Encoder의 CLS Exchange와 동일한 철학 — 정보 훼손 없이 교환.

```
Compositor:
  learnable tokens: [query_m, query_p]  (학습 가능)

  Q = [query_m, query_p]               # 2 tokens
  K, V = [cls_m_12, cls_p_12, cls_m_23, cls_p_23]  # 4 tokens

  Cross-Attention (1~2 layers) → predicted_cls_m_13, predicted_cls_p_13
```

**sg(cls_13)** target: stop-gradient. Loss 1의 pixel target이 encoder를 잡아주므로 collapse 없음.

### PE Ablation (compositor 내)

| Variant | 설명 | 해석 |
|---------|------|------|
| A (PE 없음) | K에 순서 정보 없음 (set operation) | 교환법칙이 성립하는 합성만 가능 |
| B (순서 PE) | K에 first/second PE 추가 | 비가환적 합성(회전→이동 ≠ 이동→회전) 가능 |

- A ≈ B: change embedding 자체에 공간 구조 보존 → **encoder 품질의 강력한 증거**
- B > A: 비가환적 변화에서 순서가 필요 → **순서의 역할 증명**

### 구현 변경점

1. **데이터 로더**: 2장 → 3장 샘플링 (기존 EgoDexDataset 확장)
2. **Compositor 모듈**: learnable query + cross-attention (경량, ~수 MB)
3. **Loss 항 추가**: composition consistency loss
4. **학습 스크립트**: `--composition` 플래그로 on/off

### 실험 계획

| 실험 | 설명 | 비교 |
|------|------|------|
| v4 baseline | 기존 (t1,t2) 쌍 학습 | 기준 |
| v4 + comp (PE 없음) | triplet + composition loss (variant A) | 합성 효과 |
| v4 + comp (순서 PE) | triplet + composition loss (variant B) | 순서 효과 |

**평가**: 동일 probing 프로토콜 (EgoDex part4, gap sweep, DROID cross-domain)

---

## Ablation Study (Phase 1 ��후)

| 변형 | 설명 |
|-----|------|
| M-only | P 채널 제거 |
| P-only | M 채널 제거 |
| No Exchange | CLS 교환 없이 concat만 |

---

## MAE-style Masked Auxiliary Loss (구현 예정)

Architecture 확정(d=12, s=2)에 적용. 구현 후 full training에 포함.

### 구현 방향

```
인코더 (MAE 방식):
  M: 196 patches → 독립 random mask (30~50%) → visible만 처리
  P: 196 patches → 독립 random mask (30~50%) → visible만 처리
  ※ M/P 마스킹은 완전 독립 (종속성 없음, 겹침도 자연 발생)
  ※ CLS는 마스킹 안 함 (CLS exchange 품질 유지)
      ↓ CLS exchange (visible + CLS)
      ↓ (6 blocks/stage × 2 stages)
디코더:
  visible embeddings + learnable mask_token + positional embedding → future image 복원
      ↓
Loss:
  L_total = L_future_prediction + α * L_masked_reconstruction
  ※ L_masked: masked 위치의 복원 오차만 계산
```

### 동기
- CLS average가 patch_mean의 68% 수준 (0.364 vs 0.532) → CLS 정보 밀도 부족
- 인코더가 일부만 보고도 전체를 이해해야 → 표현력 향상 (MAE 핵심 발견)
- 부수 효과: 인코더 연산 절약 (visible만 처리)

### 3단계 랜덤성 (모델 고유 특성)
1. **Spatial**: RandomCrop(224) — 매번 다른 영역
2. **Temporal**: gap=1~30 — 매번 다른 시간 간격
3. **Structural**: stream별 독립 랜덤 마스킹 — 매번 다른 정보 가림
→ 동일 데이터도 매번 다른 문제. 단, 과도한 랜덤성은 수렴 저하 → 비율 튜닝 필요.

### 마스킹 비율 가이드
| 방법 | 비율 | 이유 |
|------|------|------|
| MAE | 75% | 이미지 redundancy 높음 |
| VideoMAE | 90~95% | 비디오 시간축 중복 |
| iBOT(DINOv2) | ~50% | discriminative, teacher 신호 보존 |
| **우리 모델** | **30~50%** | 보조 loss, 메인은 future prediction |
- 실험 조건: 0% (baseline), 30%, 50%

### 주의점
- future prediction이 메인 task, masking은 보조 → α 가중치 튜닝 필요
- 랜덤성 과다 시 학습 신호 noisy → 수렴 저하 가능
- cross-stream 정보는 CLS exchange만 통과 (patch-level cross-attention 없음)

### 참고
DINOv2(iBOT)도 masked patch prediction이지만 pixel이 아닌 feature-level distillation.
우리는 pixel reconstruction target이 있으므로 MAE 방식(MSE)이 직접적.

---

## 참고 문서

- [`docs/PROBING_GUIDE.md`](PROBING_GUIDE.md) — Action probing 가이드 + 결과
- [`docs/setup/LIBERO_TEST_GUIDE.md`](setup/LIBERO_TEST_GUIDE.md) — LIBERO 평가
