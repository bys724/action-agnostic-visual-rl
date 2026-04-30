# IBS Cluster Session Log

IBS olaf 클러스터에서 사용한 모든 자원 기록. **비용 청구 대조용**.

> **중요**: 새 잡 제출 시 즉시 "진행 중" 표에 기록, 종료 시 "완료" 표로 이동.
> 저장소 과금은 다운로드/생성 시점부터 시작 (지속적). Claude/사용자 모두
> 이 문서를 작업 흐름의 일부로 유지해야 함 (CLAUDE.md "클러스터 세션 로깅" 참조).

---

## 공식 단가 (2025년 초고성능컴퓨팅자원 이용료, VAT 별도)

출처: [`docs/초고성능컴퓨팅자원 이용료 안내.pdf`](초고성능컴퓨팅자원%20이용료%20안내.pdf)

| 자원 | 파티션 | 단위 | 단가 (원) |
|------|--------|------|----------|
| 데이터분석시스템 (H100) | `AIP`, `AIP_long` | **GPU·일** | **61,000** |
| AIP MIG 1g.10gb (1/7 성능) | `mig-1g.10gb` | GPU·일 | 8,700 |
| AIP MIG 3g.40gb (1/2 성능) | `mig-3g.40gb` | GPU·일 | 30,500 |
| 초고성능 클러스터 (CPU) | `normal_cpu`, `long_cpu`, `large_cpu`, `core_s` | **노드·일** | **7,000** |
| 고성능 GPU (V100) | `normal`, `long` | GPU·일 | 8,000 |
| 본원 공동활용 (CPU) | `HQcomp2`, `HQmem` | 50코어·일 | 14,000 |
| 병렬파일 (`/proj`, `/mnt/lustre`) | — | **10TB·월** | **13,000** |
| 장기파일 보관 (테이프) | — | 10TB·월 | 3,000 |

**할증**: 긴급 1.5배 / 전용 = 단가 × 점유자원 × 일수 × 1.5

**로그인 노드**: 미과금 추정. 단, 로그인 노드에서 `/proj`에 쓰는 즉시 저장소 과금 대상.

---

## 청구 단위: 월 누적 후 일 단위 올림 (CEIL)

**잡 단위 ceil이 아니라, 한 달 동안 사용한 모든 잡의 자원·시간을 누적해서 월말에 일 단위로 올림.**

```
월간_GPU·초_누적 = Σ (n_gpus_i × elapsed_seconds_i)
청구일수         = ceil(월간_GPU·초_누적 / 86400)
월_청구액        = 단가(원/GPU·일) × 청구일수
```

CPU도 동일: `청구일수 = ceil(월간 노드·초 누적 / 86400)` × 7,000원/노드·일.

### 시나리오 (8 H100 학습 위주)

| 월 사용 패턴 | GPU·시간 합 | 청구일수 | 월 청구액 (H100) |
|-------------|-----------|---------|------------------|
| 1 GPU × 1h sanity 30회 | 30 | 2일 | 122,000 |
| 8 GPU × 72h (정확히 3일) | 576 | 24일 | 1,464,000 |
| 8 GPU × 73h (3일 1시간) | 584 | **25일** | 1,525,000 |
| 8 GPU × 144h (6일) | 1152 | 48일 | 2,928,000 |

### 핵심 함의

1. **GPU sanity test 자유** — 짧은 잡 누적은 거의 무손실
2. **월말 ceil 손실은 1회만** — 최대 ≈ (사용 자원 수)일 분량
3. **잡 길이 24h 강박 불필요** — 월 누적 ceil이라 1시간 차이가 1일 청구 차이
4. **월 경계 주의** — 두 달에 걸치면 각각 ceil 손실 발생
5. **CPU 비용은 사실상 무시 가능** — 1 노드 × 24h = 7,000원

### 전략

- 빠른 디버그 사이클은 GPU sanity도 자유
- 본 학습은 한 달 안에 묶기
- `--time` 안전 마진만 (정확히 72:00:00 불필요)

### 불확실 사항 (필요 시 운영팀 확인)

- mrg 그룹 단위 vs 사용자 개별 합산
- GPU 종류별 개별 합산 여부 (단가 다르므로 분리 추정)
- 월 마감 기준

---

## 저장소 (mrg 그룹 할당)

**현재 할당: 50 TB (`/proj/external_group/mrg/`, 2026-04-14 증설).** 이 범위 내 사용은 추적 불필요.

| 신청일 | 증설량 (TB) | 누적 (TB) | 사유 |
|--------|------------|-----------|------|
| 초기 | +10 | 10 | 그룹 기본 할당 |
| 2026-04-14 | +40 | 50 | EgoDex/DROID + Ego4D 등 추가 |

참고 단가: 추가 10 TB·월 = 13,000원 (VAT 별도)

---

## 진행 중 세션 (sbatch / salloc)

### 2026-04-30 Phase 2.5 Trajectory-Level Value Alignment (VIP-inspired)

신규 [`scripts/eval/value_alignment.py`](../scripts/eval/value_alignment.py) + [`scripts/cluster/value_alignment.sbatch`](../scripts/cluster/value_alignment.sbatch) — frozen encoder × LIBERO trajectory frame-wise embedding → Spearman ρ(t, V(t)). 학습 없음, inference만.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33632473 | AIP 1×1 H100 | 01:30:00 | sanity videomae-ours × spatial × task 0 × 5 demos | ✅ COMPLETED 18s. ρ mean=+0.799 (5/5 valid, len cutoff P95=143) |
| 33632521 | AIP 1×1 H100 | 01:30:00 | sanity v11 ep44 (A+B+D' 경로) × spatial × task 0 × 5 demos | ✅ COMPLETED 22s. ρ mean=+0.527 |

**Full run** (5 encoder × 3 suite = 15 잡, AIP 1×1 H100 each, --time=01:30:00) — ✅ 모두 COMPLETED 13:04 (총 elapsed 2535s = **0.70 GPU·h**, 약 0.03 GPU·일):

| Encoder | spatial | object | goal | spatial ρ | object ρ | goal ρ |
|---------|---------|--------|------|----------:|---------:|-------:|
| two-stream-v11 ep44 (A+B+D') | 33632528 | 33632533 | 33632538 | +0.531 | +0.379 | +0.513 |
| videomae-ours best | 33632529 | 33632534 | 33632539 | +0.795 | +0.559 | +0.654 |
| dinov2 | 33632530 | 33632535 | 33632540 | +0.805 | +0.626 | +0.748 |
| siglip | 33632531 | 33632536 | 33632541 | +0.833 | +0.567 | +0.725 |
| vc1 | 33632532 | 33632537 | 33632542 | **+0.905** | **+0.727** | **+0.768** |

길이 cutoff(P95): spatial 165, object 179, goal 204. N=475-476/500 trajectories valid per cell. 산출물 → `paper_artifacts/value_alignment/<encoder>_<suite>_<ts>/{per_demo_rho.csv, per_demo_rho_summary.json}`.

⚠️ **결과 ≠ plan 가설 (Phase 2.5 No-Go 시나리오)**: plan은 "v11 ≥ VideoMAE ≈ VC-1 > DINOv2 ≈ SigLIP" 예상했으나 실측은 **VC-1 > SigLIP ≈ DINOv2 > VideoMAE > v11** (모든 suite 일관). v11이 baselines 대비 -0.27~-0.35.

**가설 확장 — Trailing fraction sweep** (사용자 제안): v11이 motion-specific representation이라면 e_T와 장면 격차가 작은 *골 근접* 구간에선 baselines 대비 격차가 좁아져야 함. 같은 사용자 제안에 따라 fractions={1.0, 0.5, 0.3, 0.15}로 sweep.

| 33632705 | AIP 1×1 H100 | 01:30:00 | sanity sweep videomae × spatial × task 0 × 5 demos × 4 fractions | ✅ COMPLETED 14s. frac=1.0 ρ=+0.799 (이전과 일치 — 코드 검증 OK) |

Full sweep (15 잡, AIP 1×1 H100 each, fractions={1.0,0.5,0.3,0.15}):

| Encoder | spatial | object | goal |
|---------|---------|--------|------|
| two-stream-v11 ep44 (A+B+D') | 33632714 | 33632719 | 33632724 |
| videomae-ours best | 33632715 | 33632720 | 33632725 |
| dinov2 | 33632716 | 33632721 | 33632726 |
| siglip | 33632717 | 33632722 | 33632727 |
| vc1 | 33632718 | 33632723 | 33632728 |

상태: ✅ COMPLETED 13:21 (~2535s = 0.70 GPU·h). 결과는 RESEARCH_PLAN.md §Phase 2.5 참조 — 모든 fraction에서 v11 꼴찌, 가설(short window에서 격차 좁아짐) 기각.

**v11 mode ablation** (사용자 의문 1 — motion 신호가 value alignment에 무관하다는 직감 검증, scripts에 `--v11-mode {abd_prime|b_only|d_prime_only}` 추가):

| Mode | spatial | object | goal |
|------|---------|--------|------|
| b_only (P encoder patch_mean) | 33632852 | 33632854 | 33632856 |
| d_prime_only (motion-routed P state) | 33632853 | 33632855 | 33632857 |

각 잡: AIP 1×1 H100, --time=01:30:00, fractions={1.0, 0.5, 0.3, 0.15}. baselines (videomae/dinov2/siglip/vc1)은 위 sweep 결과 재활용. 추정 ~0.3 GPU·h.

### 2026-04-29 LIBERO BC-T 6 encoder 본격 학습 (Phase 3-1)

**1차 제출** (05:36, 모두 AIP 1×1 H100):

| JobID | Encoder | --time | 결과 / 비고 |
|-------|---------|--------|------------|
| 33615385 | two-stream-v11 ep44 | 2-00:00:00 | **CANCELLED** 04-30 14:xx — broken cfg(`use_joint=False`)로 학습된 ep42/50 ckpt rollout 0%. ~33.5 GPU·h 손실. 33633419로 재제출 |
| 33615386 | videomae-ours best | 1-00:00:00 | **COMPLETED** ep50/50, 04-29 20:39 — 그러나 **broken cfg ckpt** 로컬 H100 rollout sanity = **0/50 SR** (libero_spatial 10×5). 진단 → cfg fix. 33633420으로 재학습 |
| 33615387 | dinov2 | 1-00:00:00 | **CANCELLED** 04-29 13:14 — ETA 30.3h, --time=24h로 timeout 위험. 2-00:00:00로 재제출 (33616659) |
| 33615388 | siglip | 1-00:00:00 | **FAILED** 40s — `SiglipModel(pixel_values=...)` text input 요구 (VisionModel 사용해야) |
| 33615389 | vc1 | 1-00:00:00 | **FAILED** 1m23s — CLIPModel HF download race (`HF_HUB_OFFLINE` 미설정), VC-1 loader도 잘못 (vc_models 사용해야) |
| 33615390 | vjepa2-1 sanity | 01:00:00 | **CANCELLED** — driver의 action loss alignment 버그 (V-JEPA T_out=10 vs actions T=25 mismatch) 발견하여 미실행 |

**Driver 패치 (코드 리뷰 후)**:
- [scripts/eval/finetune_libero_bct.py](scripts/eval/finetune_libero_bct.py) `_align_actions(dist, actions)` 추가 — V-JEPA causal trim 지원
- 첫 batch dtype/range/shape debug log 추가 (encoder native 분포 검증)
- [src/encoders/adapters/single_frame.py](src/encoders/adapters/single_frame.py) SigLIP은 `SiglipVisionModel`, VC-1은 `vc_models` 패키지 사용으로 수정
- [scripts/cluster/finetune_libero_bct.sbatch](scripts/cluster/finetune_libero_bct.sbatch) `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` (CLIP/DINOv2/SigLIP 캐시 사용으로 race 회피)

**2차 제출** (re-submission after patches):

| JobID | Encoder | --time | 결과 |
|-------|---------|--------|------|
| 33615391 | siglip | 1-00:00:00 | **CANCELLED** 04-29 13:14 — ETA 23.0h, --time=24h 한계 위험. 2-00:00:00로 재제출 (33616660) |
| 33615392 | vc1 | 1-00:00:00 | **CANCELLED** 04-29 13:30 — ETA 20.8h, --time=24h timeout 위험. 2-00:00:00로 재제출 (33616669) |
| 33615393 | vjepa2-1 sanity | 01:00:00 | **PASS** 15m38s. loss 5.27→2.58, `_align_actions` 정상 |

**3차 제출** (`--time=2-00:00:00` 일관 적용, 04-29 13:14~13:30):

| JobID | Encoder | --time | 결과 |
|-------|---------|--------|------|
| 33616659 | dinov2 | 2-00:00:00 | **CANCELLED** 04-30 14:xx — ep27/50 broken cfg. ~25 GPU·h 손실. 33633421로 재제출 |
| 33616660 | siglip | 2-00:00:00 | **CANCELLED** 04-30 14:xx — ep34/50 broken cfg. ~25 GPU·h 손실. 33633422로 재제출 |
| 33616669 | vc1 | 2-00:00:00 | **CANCELLED** 04-30 14:xx — ep37/50 broken cfg. ~25 GPU·h 손실. 33633423로 재제출 |

**3차 재제출 사유**: 1/2차의 `--time=1-00:00:00`로는 dinov2/siglip/vc1 모두 50 ep 학습 불가능 (epoch당 시간 측정 후 적자 -4.5h ~ -6.3h 확인). dinov2/siglip은 8h 진행 후 cancel 손실, vc1은 7h45m 진행 후 cancel 손실 — 누적 ~24 GPU·h ≈ 0.99 GPU·일 (~6.1k원). main BC table 5 encoder 동일 epoch 비교 우선.

### 2026-04-30 LIBERO Action Probing (Phase 2 보강)

신규 [`scripts/eval/probe_action_libero.py`](../scripts/eval/probe_action_libero.py) + [`scripts/cluster/probe_action_libero.sbatch`](../scripts/cluster/probe_action_libero.sbatch). Plan: [`docs/libero_action_probing_plan.md`](libero_action_probing_plan.md). DROID R²~0.005 한계 보완 목적, gap-matched protocol (LIBERO 20Hz {1,13,20,40} = DROID 15Hz {1,10,15,30}). Target = pose-derived 7-DoF (3 pos + 3 rotvec + 1 gripper), cumulative action sum 회피.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33633959 | AIP 1×1 H100 | 01:30:00 | sanity videomae × spatial × task 0 × 5 demos × gap=20 | ✅ COMPLETED 17s. R² agg=+0.053 (per-dim pos 모두 +0.27~+0.52, rotvec 약함) |

**Full sweep 1차** (한 잡 = encoder × suite × 4 gaps loop, 5 × 3 = 15잡, AIP 1×1 H100, --time=01:30:00):

| 33633961~33633975 | 15잡 | **OUT_OF_MEMORY** 38~55s. train demos 전체 frames preprocess해서 메모리에 누적 → OOM_KILL. 결과 없음. ~10 GPU·min 손실 (~0.7원). |

**원인 + 수정**: streaming refactor — demo 단위 forward + embeddings only 누적 (raw frames 즉시 폐기). 코드: [scripts/eval/probe_action_libero.py](../scripts/eval/probe_action_libero.py) `collect_embed()` 함수.

**Sanity (streaming 검증)**:
| 33633976 | AIP 1×1 H100 | sanity (videomae × spatial × task 0 × 5 demos × gap=20) | ✅ COMPLETED 12s. R²=+0.0527 (1차 sanity와 일치 — 코드 동치성 확인) |

**Full sweep 2차** (streaming, 15잡):

| Encoder | spatial | object | goal |
|---------|---------|--------|------|
| two-stream-v11 ep44 (A+B+D') | 33633977 | 33633982 | 33633987 |
| videomae-ours best | 33633978 | 33633983 | 33633988 |
| dinov2 | 33633979 | 33633984 | 33633989 |
| siglip | 33633980 | 33633985 | 33633990 |
| vc1 | 33633981 | 33633986 | 33633991 |

✅ **모두 COMPLETED 17:14, 누적 13010s = 3.61 GPU·h** ≈ 0.15 GPU·일 (~9k원).

**핵심 결과 (R² aggregate matrix)** — 자세한 분석은 [RESEARCH_PLAN.md §Phase 2 보강](RESEARCH_PLAN.md):

🏆 **Controlled comparison (v11 > VideoMAE-ours, 12/12 cells)**: +0.04~+0.25, 모든 (suite, gap) cell. Architectural contribution 입증.

🏆 **v11 gap=1 dominance**: spatial +0.660 (vs dinov2 +0.611), object +0.702 (vs +0.690). Internet-scale도 fast-motion에서는 추월.

**DINOv2가 internet-scale 강자**: 12 cells 중 9 best (gap≥13). v11의 약점은 long-gap (gap=40 −0.29 ramp).

**Phase 2.5 negative 반전 framing**: action-relevance metric (probing) → v11 win, state-similarity metric (value alignment) → DINOv2 win. 두 metric 두 능력으로 paper §4.5 main supplementary 통합.



### 2026-04-30 BC-T 4차 제출 (cfg 결함 수정 후) — `use_joint=True` + `joint_states`

**원인 진단** (commit f6a3b5c): VideoMAE-ours BC-T 33615386 ep50 ckpt 로컬 rollout sanity 0/50 SR. 학습 cfg (`scripts/eval/finetune_libero_bct.py`)가 LIBERO 공식 default와 불일치 — `use_joint=False`로 `robot0_joint_pos` (7-d) 누락 = robot kinematics state 부재로 spatial control 실패. ckpt/inference 코드는 정상.

**수정**:
- `scripts/eval/finetune_libero_bct.py:111` `use_joint: False → True`
- `scripts/eval/finetune_libero_bct.py:119,321` `low_dim: ["gripper_states"] → ["gripper_states", "joint_states"]`
- `scripts/cluster/finetune_libero_bct.sbatch:18` `--time=08:00:00 → 2-00:00:00` (영구. 트러블슈팅 "반복 사고" 재발 방지)

**Cancel + 재제출 결정 비용**:
- Cancel 4잡 손실: ~108 GPU·h (≈ 4.5 GPU·일, ~275k원). 완료해도 broken cfg ckpt는 0% SR 보장 → cancel이 정답.
- 추가 PENDING cancel (33633414~18): `--time=8h` 기본값 누락 발견 — PENDING 상태 cancel, 0 비용.

**4차 제출** (AIP 1×1 H100, --time=2-00:00:00):

| JobID | Encoder | 결과 |
|-------|---------|------|
| 33633419 | two-stream-v11 ep44 | PENDING |
| 33633420 | videomae-ours best | PENDING |
| 33633421 | dinov2 | PENDING |
| 33633422 | siglip | PENDING |
| 33633423 | vc1 | PENDING |

`SUFFIX=usejoint` 적용 → 출력 dir `<encoder>_libero_spatial_seed0_<ts>_usejoint/`로 broken ckpt와 구분.

**검증 결과** (2026-04-29):
- LIBERO obs RGB: float32 [0, 1] (robomimic 정규화 완료) → 어댑터 가정 일치 ✓
- shape_meta: `ac_dim=7`, RGB (3, 128, 128) → driver의 resize로 224/384 ✓
- V-JEPA bs=4 epoch당 460s → bs=32 환산 시 50ep ≈ 30-100일 → **본격 학습 불가, 별도 처리 필요**
  · 옵션: feature pre-extraction (1 pass, ~50 GB cache) → 정책 head만 학습 → 가장 가능성 높음
  · 옵션: skip V-JEPA from main BC table (probing 결과는 paper에 포함). decision 보류.

### 2026-04-29 v11 ablation A1: motion-routing source (V from P → V from M)

**Paper claim**: v11의 motion-routing (Q,K←M, V←P)이 표준 cross-attn (Q←P, K,V←M)보다 표현 품질 우월.

**구현**: `src/models/two_stream_v11.py` MotionRoutingBlock에 `routing_mode` 파라미터 (기본 `v_from_p`, ablation `v_from_m`). 두 mode 동일 param count (208.33M). CLI: `--v11-routing-mode v_from_m`. sbatch: `V11_ROUTING_MODE=v_from_m`.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33615394 | AIP 1×1 H100 | 30:00 | sanity (1 ep × part1 × 200 vid × bs=64) | PENDING |
| 33615395 | AIP_long 2×4 H100 | 3-12:00:00 | full ablation (50 ep × part1-5) | PENDING |

**Control은 기존 v11 ep50 ckpt** (`/proj/external_group/mrg/checkpoints/two_stream_v11/20260426_014333/`). 추가 학습 없음.

**예상 비용** (33615395만): v11과 동일 ≈ 617 GPU·h ≈ 1.59M 원 (월누적 ceil).

### 2026-04-29 v11 EgoDex probing 보충 (ep32, ep40 × 12 mode) ✅ 완료

**목적**: paper Figure F1 (representation evolution trajectory)의 ep24~ep44 사이 8 epoch gap 보충. W-shape 회복 구간 해상도 향상.

**ckpt**: `two_stream_v11/20260426_014333/checkpoint_epoch00{32,40}.pt`. 12 mode = patch_mean × {A, B, D, D', A+B, A+D, A+D', B+D', A+B+D'} + cls × {A, B, A+B}.

| JobID 범위 | 자원 | 잡수 | 결과 |
|-----------|------|-----|------|
| 33616548-33616559 | AIP 1×1 H100 | 12 (ep32 × 12 mode) | COMPLETED ~12:58-13:06 |
| 33616560-33616571 | AIP 1×1 H100 | 12 (ep40 × 12 mode) | COMPLETED ~13:05-13:13 |

**실제 비용**: 24 잡, 평균 11.1 min/잡, 누적 **4.45 GPU·h** ≈ 0.19 GPU·일 (월누적 ceil 영향 미미).

**핵심 결과** (`patch_mean_concat_all` = champion mode):
- ep24 +0.234 → **ep32 +0.263** → **ep40 +0.261** → ep44 +0.288 → ep50 +0.279
- ep24~ep44 ramp이 단순 monotonic이 아님: ep32에서 +0.263 도약 → ep40 미세 후퇴 → ep44 peak. LR 후반부 oscillation의 정량 증거. F1 figure에서 표현 진화 곡선 더 부드러워짐.
- CSV 갱신: 11 epoch × 12 mode = 132 rows in `paper_artifacts/probing/v11_egodex_summary.csv`.

---

## 2026-04-28 LIBERO BC-T sanity 시리즈 — 7회 디버깅 후 통과

`scripts/eval/finetune_libero_bct.py` 첫 동작 확인용. 1 task (libero_spatial task 0), 10 batch × bs=16 × 2 epoch. 1 GPU × ~1분 × 7회. 누적 ≈ 0.15 GPU·h.

| JobID | 결과 | 디버깅 내용 |
|-------|------|------------|
| 33615282 | FAIL | conda env libero pip install에 `lifelong/models/modules/` 누락 → external/LIBERO에서 복사 |
| 33615285 | FAIL | `cfg.data.max_word_len` 누락 → cfg_emb에 추가 |
| 33615286 | FAIL | `clip_model.get_text_features().detach()` AttributeError (transformers 버전 호환) → text_model.pooler_output 직접 사용 |
| 33615287 | FAIL | 위 동일 — pooler_output 추출 로직 수정 |
| 33615288 | FAIL | `omegaconf` OrderedDict 미지원 (shape_meta) → cfg에서 분리, 별도 인자로 전달 |
| 33615289 | FAIL | BasePolicy가 `cfg.policy.color_aug` + `translation_aug` + `cfg.train.use_augmentation` 요구 → IdentityAug로 추가 |
| 33615290 | FAIL | robomimic `np.bool` 사용 (numpy 1.20+ deprecated) → monkeypatch |
| 33615291 | FAIL | ExtraModalityTokens가 'ee_states' 키 요구 → use_ee=False로 비활성화 |
| **33615297** | ✅ **PASS** | train 4.12→0.80, ckpt 저장 OK. 211.3M params (3.0M trainable, encoder frozen) |

---

## 완료 세션 요약 (주요 잡만)

> 상세 기록은 git history 참조 (2026-04-23 정리 이전 버전). 전체 sacct 기록은 `sacct -u bys724` 로 복구 가능.
> 여기엔 **주요 full training + 최근 v10 probing**만 유지.

### 2026-04 주요 잡

| JobID | 종료 | Elapsed | 파티션 | 자원 | 자원·시간 | 목적 / 결과 |
|-------|------|---------|--------|------|----------|-------------|
| 32438409 | 04-09 12:35 | 00:52:14 | core_s | 96 CPU | 83.5 CPU·h | EgoDex part1 추출 (46,234 videos → 316 GB) |
| 32712324 | 04-14 04:06 | 3-12:00:06 | AIP_long | 2×4 H100 | 672.0 GPU·h | Two-Stream v4 full (GPFS). TIMEOUT ep48, R²=0.197 |
| 32950553 | 04-15 01:16 | 2-02:41:18 | AIP_long | 2×4 H100 | 405.5 GPU·h | V-JEPA-ours 3차 (mask 50/60%, warmup). ep30 CANCELLED, **negative result** |
| 33003926 | 04-17~ | ~50h | AIP_long | 2×4 H100 | ~400 GPU·h | VideoMAE-ours 50ep 완주. **R²=0.326** (VideoMAE 베이스라인) |
| 33222151 | 04-18~ | ~2d | AIP_long | 2×4 H100 | ~384 GPU·h | Two-Stream v6 ep23 scancel. **ep8 peak R²=0.259** (현 챔피언) |
| 33277774 | 04-19 21:x | ~19h | AIP_long | 2×4 H100 | ~152 GPU·h | v7-big Option 3 — probing R²~0, CLS 균질화 → 폐기 |
| 33282021 | 04-20 14:49 | 16:50:17 | AIP_long | 2×4 H100 | 134.7 GPU·h | v7-big +attention isolation. cos=+0.9997 collapse → **폐기** |
| 33346962 | 04-21 05:19 | 14:55 | AIP_long | 2×4 H100 | ~119.3 GPU·h | v8 full 1차 (λ=0.2). ep8 R²=-0.22, L_P scale mismatch → 재설계 |
| 33451989 | 04-21 14:58 | 09:11:48 | AIP_long | 2×4 H100 | 73.6 GPU·h | v8 full 2차 (BYOL, λ=0.05). ep12 P=-0.468 static salience → **v8 전면 폐기** |
| 33492965 | 04-22 (scancel) | 07:14:48 | AIP_long | 2×4 H100 | 57.9 GPU·h | v9 full (P=current MAE). ep4 probing concat=+0.154, P=-0.102. 재설계 |
| 33555333 | 04-22 13:04 (scancel) | 13:24:52 | AIP_long | 2×4 H100 | 107.3 GPU·h | v9 residual+patch-norm full. ep4/ep8 P +0.100 → -0.006 degrade → **v10 전환** |

### 2026-04 Two-Stream v10 — 종료 + Probing 통합 요약

**Full run** (33570871, AIP_long 2노드×4 H100, 50ep, ~3일): ep40 plateau **+0.221** (`patch_mean_concat`), ep44/ep48 +0.221/+0.222. **v6 챔피언 (+0.259) 추월 실패 확정**.

**Probing 추세** (`patch_mean_concat / M / P`):

| Epoch | concat | M | P | 비고 |
|-------|--------|---|---|------|
| ep4   | +0.195 | +0.176 | +0.126 | v9 lineup 추월 |
| ep8   | +0.206 | +0.150 | +0.152 | 1차 peak |
| ep12  | +0.148 | +0.129 | +0.083 | collapse 시작 |
| ep16  | +0.144 | +0.125 | +0.038 | sparse pinpoint viz |
| ep20  | +0.137 | +0.135 | +0.022 | 저점 |
| ep24  | +0.202 | +0.138 | +0.092 | W-shape 회복 |
| ep36  | +0.214 | +0.129 | +0.141 | new peak |
| **ep40** | **+0.221** | — | — | **plateau** |
| ep44  | +0.221 | — | — | plateau |
| ep48  | +0.222 | — | — | plateau |

각 probing 잡 1 노드 × 1 H100, ~15min. v10 probing 시리즈 (ep4~ep48 × {concat, M, P} ≈ 30+ 잡, 누적 ≈ 7.5 GPU·h).

### 2026-04 Two-Stream v11 — Sanity + Full + Probing 시리즈 (학습 종료, 🏆 final champion)

**Sanity** (33591381, AIP 1×1 H100, 10:26): forward/backward OK, L 단조 감소. M collapse는 200vid×5ep 소규모 한정 — full scale에서 healthy.

**Full 1차** (33594155, AIP 2노드×4 H100, 23:30:22 elapsed, TIMEOUT — `--time` default 23:30 함정): ep1~ep12까지 학습. **188.1 GPU·h** (8 × 23.51h).

**Full Resume** (33600621, AIP 2노드×4 H100, --time=3d, 2-05:39:14 elapsed, **COMPLETED** 2026-04-28 07:19): ep13부터 latest.pt 자동 detect → ep50까지 완주. ckpt dir `20260426_014333/` (ep16/20/24/28/32/36/40/44/48 + latest=ep50). **429.2 GPU·h** (8 × 53.65h).

**Probing — ep4~ep50 × 12 mode 통합** (1 노드 × 1 H100 × ~14min × 84 잡, 누적 ≈ 19.6 GPU·h):

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
| `cls_m_enc` | +0.066 | +0.155 | +0.162 | +0.163 | +0.172 | +0.158 | +0.125 | +0.123 | +0.123 |
| `cls_p_enc` | -0.059 | -0.011 | -0.008 | -0.010 | -0.009 | -0.013 | -0.002 | -0.002 | -0.002 |
| `cls_concat_enc` | -0.048 | +0.092 | +0.148 | +0.139 | +0.162 | +0.140 | +0.114 | +0.118 | +0.113 |

**🏆 ep44 final champion**: A+B+D' = +0.288 → v6 ep8 (+0.259) **추월 +0.029**. VideoMAE +0.326까지 격차 -0.038. ep44~ep50은 plateau 확정 (A+D'/A+D 안정, A+B+D' -0.010 미세 over-tightening).

**가시화** (33609996-33609997, ep48/ep50 attention map): `docs/architecture/attn_v11_ep{48,50}.png`. 각 ~3min, 누적 ≈ 0.1 GPU·h.

**LIBERO BC fine-tune** (libero_spatial 30 epoch, 1 노드 × 1 H100):

| JobID | Encoder | Elapsed | best val MSE |
|-------|---------|---------|--------------|
| 33600616 | VideoMAE-ours ep50 | 31:00 | **0.0286** |
| 33600617 | v11 ep12 (A+D) | 1:07:27 | 0.0290 |

거의 동등 (격차 +0.0004). v11 ep44/ep50 ckpt로 재측정 필요.

### 2026-04 DROID Cross-domain Probing 시리즈

VideoMAE-ours ep50 vs Two-Stream v11 ep12 (3 mode), gap 1/10/15/30. 각 1 노드 × 1 H100 × ~15min, 누적 ≈ 4 GPU·h.

| Gap (DROID 15Hz) | VideoMAE | v11 best (mode) | 격차 |
|------------------|----------|-----------------|------|
| 1 | -0.006 | -0.005 | +0.001 |
| 10 | -0.006 | +0.006 (A+B) | +0.012 |
| **15** ★ | **-0.035** | **+0.005 (A+B)** | **+0.040** |
| 30 | -0.028 | -0.010 | +0.018 |

모든 gap에서 v11 우위. gap=15(EgoDex 학습 분포 1초)에서 격차 가장 큼.

---

## 기록 절차

### 1. 잡 제출 시
즉시 "진행 중" 표에 행 추가:
```
| <JobID> | YYYY-MM-DD HH:MM | <파티션> | <노드>N × <자원> | <한 줄 목적> |
```

### 2. 잡 종료 확인 시
sacct로 정확한 시각 조회 → "진행 중"에서 "완료" 표로 이동:
```bash
sacct -j <JobID> --format=JobID,JobName,Partition,AllocTRES,Start,End,Elapsed,State
```
**자원·시간** 계산 후 기입 (GPU: `n_gpus × elapsed_h`, CPU: `n_nodes × elapsed_h`).
비용은 월말에 누적 합산 후 한 번에 계산.

### 3. 저장소 증설 시에만
mrg 그룹 할당 범위(50 TB) 내 사용은 추적 불필요. **추가 증설한 경우에만** 저장소 표에 1줄 추가.

### 4. 다중 작업 묶음
sbatch array는 부모 JobID 1줄 + 비고에 array 범위.

### 5. 비정상 종료
`TIMEOUT`, `FAILED`, `CANCELLED`, `OUT_OF_MEMORY` 명시 + 짧은 원인 기입.

### 6. 문서 비대화 방지
- 반복되는 variant probing 등 취소선 누적되면 과감히 삭제 (git history가 archive 역할)
- 월말에 "완료 세션 요약"만 유지, 개별 probe/viz 잡 로그는 월 단위로 통합

---

## 누적 사용량 (월별 요약)

매월 말 "완료 세션"의 자원·시간을 합산 → 일수 환산 → ceil.

```
H100·일수 = ceil(월 H100·hours 합 / 24)
CPU·일수  = ceil(월 노드·hours 합 / 24)
H100 비용 = H100·일수 × 61,000원
CPU 비용  = CPU·일수 × 7,000원
```

| 월 | H100·hours 합 | H100·일수 (ceil) | 노드·hours 합 | CPU·일수 (ceil) | 비용 추정 (원, VAT 별도) | 비고 |
|----|--------------|-----------------|--------------|-----------------|--------------------------|------|
| 2026-04 | ~3,200 (예상, v10 종료 + v11 진행 중) | 134 (예상) | ~1.5 | 1 | ~8,181,000 (예상) | v4/v6/v7big/v8/v9/v10/v11 + 50TB 저장소 증설 |

(월말 확정 숫자로 갱신 필요 — v10 full 종료 시점에 합산)