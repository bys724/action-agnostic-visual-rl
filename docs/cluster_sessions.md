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

| JobID | 제출 시각 | 파티션 | 자원 | 목적 |
|-------|----------|--------|------|------|
| **33594155** | **2026-04-25 01:11** | **AIP_long** | **2 노드 × 4 H100** | **Two-Stream v11 full run — motion-routing + dual-target, p_depth=12, m_depth=6. 50 epoch, `--time=3d`. RUNNING (ep12 도달)** |
| **33600616** | **2026-04-25 ~** | **AIP** | **1 노드 × 1 H100** | **LIBERO BC fine-tune (VideoMAE-ours ep50, libero_spatial 30ep). RUNNING** |
| **33600617** | **2026-04-25 ~** | **AIP** | **1 노드 × 1 H100** | **LIBERO BC fine-tune (v11 ep12 A+D, libero_spatial 30ep). RUNNING** |

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

### 2026-04 Two-Stream v11 — Sanity + Full + Probing 시리즈

**Sanity** (33591381, AIP 1×1 H100, 10:26): forward/backward OK, L 단조 감소. M collapse는 200vid×5ep 소규모 한정 — full scale에서 healthy.

**Full run** (33594155, AIP_long 2노드×4 H100, 50ep, RUNNING, 2026-04-25 01:11~): ep12 도달, A+D **+0.219** (v10 ep40 plateau 도달, 12 epoch만에).

**Probing — ep4/ep8/ep12 × 12 mode 통합** (1 노드 × 1 H100 × ~15min, 누적 ≈ 9 GPU·h):

| Mode | ep4 | ep8 | ep12 |
|------|-----|-----|------|
| `patch_mean_m_enc` (A) | +0.170 | +0.176 | **+0.208** |
| `patch_mean_p_enc` (B) | -0.041 | -0.025 | 0.000 |
| `patch_mean_p_state_after_routing` (D') | +0.121 | +0.066 | +0.072 |
| `patch_mean_p_features_tk` (D) | +0.023 | +0.055 | +0.054 |
| `patch_mean_concat_enc_only` (A+B) | +0.160 | +0.168 | +0.200 |
| `patch_mean_concat_enc_phase3` (A+D) | +0.143 | +0.194 | **+0.219** ★ |
| `patch_mean_concat_enc_d_prime` (A+D') | +0.149 | +0.166 | +0.153 |
| `patch_mean_concat_p_enc_d_prime` (B+D') | +0.135 | +0.011 | +0.076 |
| `patch_mean_concat_all` | +0.114 | +0.094 | +0.178 |
| `cls_m_enc` | +0.066 | +0.155 | +0.162 |
| `cls_p_enc` | -0.059 | -0.011 | -0.008 |
| `cls_concat_enc` | -0.048 | +0.092 | +0.148 |

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