# IBS Cluster Session Log

IBS olaf 클러스터에서 사용한 모든 자원 기록. **비용 청구 대조용**.

> **중요**: 새 잡 제출 시 즉시 "진행 중" 표에 기록, 종료 시 "완료" 표로 이동.
> 저장소 과금은 다운로드/생성 시점부터 시작 (지속적). Claude/사용자 모두
> 이 문서를 작업 흐름의 일부로 유지해야 함 (CLAUDE.md "클러스터 세션 로깅" 참조).

> **월별 archive 정책**: 매월 말 직전 월 entries는 `docs/archive/cluster_sessions_YYYY-MM.md`로 분리. 누적 사용량 표만 main에 유지.

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

### 2026-05-03 BC-T 2차 ckpt 로컬 sanity (cluster cost 0)

**활동**: 4차(use_joint fix) 학습 종료된 5 encoder ckpt를 로컬 H100 워크스테이션으로 일괄 전송 (`bct_usejoint_best.tar` 2.2 GB). LIBERO closed-loop rollout sanity (1 task × 5 trial × 5 encoder).

**결과**: vc1 80% / siglip 60% / dinov2 40% / videomae-ours 0% / two-stream-v11 0%

**추가 발견 (rollout 코드)**: 5개 어댑터 모두 `self.prev_obs` cache cross-camera 오염 — `agentview_rgb` + `eye_in_hand_rgb`가 동일 어댑터 인스턴스 공유. fix: `src/eval_libero.py` `BCTransformerClient` 재설계 (latent_queue 폐기 + raw obs history 누적). 추가 fix 4건 (env_resolution 256→128, joint_states shape_meta, v11 adapter checkpoint=None 허용, dummy wait obs 누적).

**ours 0% 진단**: encoder representation 정상 (centered cos / PCA r1 baseline과 동급). BC fit이 가장 정확 (`‖p−r‖` 최소) → overfit 의심. test init state는 학습 demo init과 다름 (검증 완료) → BC generalization 격차.

**클러스터 후속 작업** (V3 본 학습): augmentation + multi-seed 동시 적용. 2-frame pair adapter augmentation 일관성 시각 검증 필수. 자세한 cfg 변경 + 검증 절차: [`docs/refactor_plan_2026-05-03.md`](refactor_plan_2026-05-03.md) §3, [`docs/RESEARCH_PLAN.md`](RESEARCH_PLAN.md) Phase 3-1 V3 § + git commit `4d4f89c` (rollout fix) 참조. 진단 narrative 원본은 [`docs/archive/PHASE3_BCT_DEBUG_2026-05-03.md`](archive/PHASE3_BCT_DEBUG_2026-05-03.md).

---

## 이전 월 archive

| 월 | Archive 위치 | 핵심 산출물 |
|----|------------|------------|
| 2026-04 | [`docs/archive/cluster_sessions_2026-04.md`](archive/cluster_sessions_2026-04.md) | v4/v6/v10/v11 사전학습 + BC-T 1~4차 + Phase 2.5 value alignment + Phase 2 보강 LIBERO probing |

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
- 매월 말 직전 월 entries는 `docs/archive/cluster_sessions_YYYY-MM.md`로 분리

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
| 2026-04 | ~3,200 (예상, v10 종료 + v11 진행 중) | 134 (예상) | ~1.5 | 1 | ~8,181,000 (예상) | v4/v6/v7big/v8/v9/v10/v11 + 50TB 저장소 증설 + BC-T 1~4차 (cancel 손실 ~108 GPU·h 포함) |

(월말 확정 숫자로 갱신 필요)
