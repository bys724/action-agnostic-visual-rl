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

**할증 옵션**:
- 긴급 활용 (express service): 단가 × 1.5
- 전용 활용 (exclusive service): 단가 × 점유 노드(GPU) × 점유일 × 1.5

**로그인 노드**: PDF에 항목 없음 → 미과금으로 추정. 다만 로그인 노드에서 `/proj`에 데이터를 쓰는 즉시 저장소 과금이 시작될 수 있음.

---

## 청구 단위: 월 누적 후 일 단위 올림 (CEIL)

**잡 단위 ceil이 아니라, 한 달 동안 사용한 모든 잡의 자원·시간을 누적해서 월말에 일 단위로 올림한다.**

```
월간_GPU·초_누적 = Σ (n_gpus_i × elapsed_seconds_i)   # 한 달 동안 모든 잡 합산
청구일수         = ceil(월간_GPU·초_누적 / 86400)
월_청구액        = 단가(원/GPU·일) × 청구일수
```

CPU도 같은 방식: `청구일수 = ceil(월간 노드·초 누적 / 86400)` × 7,000원/노드·일.

### 시나리오 (8 H100 학습, GPU·시간 누적)

| 월 사용 패턴 | GPU·시간 합 | 환산 일수 | 청구일수 | 월 청구액 (H100) |
|-------------|-----------|----------|---------|------------------|
| 1 GPU × 1h sanity 1회 | 1 | 0.042 | 1일 | 61,000 |
| 1 GPU × 1h sanity 30회 | 30 | 1.25 | 2일 | 122,000 |
| 8 GPU × 23h (1일 미만 학습) | 184 | 7.67 | 8일 | 488,000 |
| 8 GPU × 71h (~3일 학습) | 568 | 23.67 | 24일 | 1,464,000 |
| 8 GPU × 72h (정확히 3일) | 576 | 24.0 | 24일 | 1,464,000 |
| 8 GPU × 73h (3일 1시간) | 584 | 24.33 | **25일** | 1,525,000 |
| 위 73h 학습 + sanity 30회 | 614 | 25.58 | 26일 | 1,586,000 |
| 8 GPU × 144h (~6일 학습) | 1152 | 48.0 | 48일 | 2,928,000 |

### 핵심 함의 (잡 단위 ceil 대비)

1. **GPU sanity test 자유롭게 가능** — 짧은 잡 누적은 거의 무손실. 1시간 sanity 30회 = 122,000원에 불과
2. **월말 ceil 손실은 1회만** — 최대 ≈ (사용 자원 수)일 분량. 8 H100 사용 시 최대 488,000원 손실
3. **잡 길이 24h 직전 강박 불필요** — 73h 학습은 25일 청구 (정확히 72h 24일과 비교 시 1일 = 61,000원 차이)
4. **월 경계 주의** — 한 달 안에 작업 끝내는 게 유리. 두 달에 걸치면 각각 ceil 손실 발생
5. **CPU 비용은 사실상 무시 가능** — 1 노드 × 24h = 7,000원. 30번 추출 잡 돌려도 ~10만원

### 전략

- **빠른 디버그 사이클은 GPU에서도 가능** — 짧은 sanity test 부담 없음
- **본 학습은 한 달 안에 묶기** — 월 ceil 손실 회피
- **여러 단기 실험을 같은 달에 묶기** — 어차피 누적
- **--time 설정은 안전 마진만** — 73h 잡이라면 `--time=80:00:00` 정도. 정확히 72:00:00으로 둘 필요 없음

### 불확실 사항 (필요 시 운영팀 확인)

- mrg 그룹 단위 합산 vs 사용자 개별 합산 — 본인 단독이면 무관, 여러 사용자면 분담 필요
- GPU 종류별 (H100, V100, MIG) 개별 합산일 가능성이 큼 (단가가 다르므로)
- 월 마감 기준 (말일 자정? 다음달 1일?)

---

## 환산 비용 (자주 쓰는 시나리오, 월 누적 ceil 가정)

| 작업 (월 누적) | 자원 | 시간 | GPU·일 환산 | 비용 (원, VAT 별도) |
|---------------|------|------|------------|---------------------|
| H100 1장 × 1시간 sanity (단독) | 1 GPU | 1h | 0.042 → ceil 1일 | 61,000 |
| H100 1장 × 1시간 × 30회 (단독) | 1 GPU | 30h | 1.25 → ceil 2일 | 122,000 |
| 8 H100 × 23h × 1회 (단독) | 8 GPU | 23h | 7.67 → ceil 8일 | 488,000 |
| 8 H100 × 71h × 1회 (단독) | 8 GPU | 71h | 23.67 → ceil 24일 | 1,464,000 |
| **8 H100 × 72h (full pretrain)** | 8 GPU | 72h | 24.0 → 24일 | **1,464,000** |
| 8 H100 × 144h (긴 학습) | 8 GPU | 144h | 48.0 → 48일 | 2,928,000 |
| CPU 1 노드 × 1시간 (sanity) | 1 노드 | 1h | 0.042 → ceil 1일 | 7,000 |
| CPU 1 노드 × 12시간 × 5 parts | 1 노드 | 60h | 2.5 → ceil 3일 | 21,000 |
| CPU 1 노드 × 24시간 (1 part 풀배치) | 1 노드 | 24h | 1.0 → 1일 | 7,000 |

**관찰**:
- GPU 비용이 압도적. CPU는 사실상 무시 가능
- 짧은 sanity 잡은 누적이라 부담 없음 (단, 첫 1일분은 항상 발생)
- 저장소(/proj)는 mrg 그룹이 10 TB 이미 할당받음 → 추적 불필요 (증설 시에만 기록)
- VAT 10% 추가 시 위 금액 ×1.1

---

## 저장소 (mrg 그룹 기본 할당)

**기본 할당량: 10 TB (`/proj/external_group/mrg/`)** — 이미 할당받은 상태로, 그룹 단위 고정 비용으로 처리됨.
이 범위 내 사용은 별도 추적/계산 불필요.

**증설 시에만 기록** — 10 TB를 초과해 추가 용량을 신청한 경우 아래 표에 기록:

| 신청일 | 증설량 (TB) | 누적 (TB) | 사유 | 비고 |
|--------|------------|-----------|------|------|
| 2026-04-14 | +40 | 50 | EgoDex/DROID + 추가 Ego 시리즈 다운로드 예정 | 확인일 2026-04-14 (`mmlsquota -j mrg proj` → 50T). 신청 시각 미상 |

참고 단가: 추가 10 TB·월 = 13,000원 (VAT 별도)

---

## 진행 중 세션 (sbatch / salloc)

| JobID | 제출 시각 | 파티션 | 자원 | 목적 |
|-------|----------|--------|------|------|
| 32710540 | 2026-04-10 13:08 | AIP | 2 노드 × 1 H100 | DDP sanity Stage 4 (multi-node NCCL 검증, port 29501) |
| ~~32712324~~ | ~~2026-04-10 16:15~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~Two-Stream v4 full training — TIMEOUT at epoch 48/50 (3-12h 한도)~~ |
| ~~32983533~~ | ~~2026-04-14 10:4x~~ | ~~AIP~~ | ~~2 노드 × 4 H100~~ | ~~Two-Stream resume 제출했으나 epoch 33부터 plateau 확인되어 resume 불필요로 판단, scancel (PENDING 상태에서 취소 → 과금 없음)~~ |
| ~~32867433~~ | ~~2026-04-12 06:20~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~V-JEPA-ours 1차 — loss 상승(warmup 부재)으로 epoch 4에서 CANCELLED~~ |
| ~~32867645~~ | ~~2026-04-12 12:40~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~V-JEPA-ours 2차 — warmup 추가했으나 mask ratio 과다(80/85%)로 loss 계속 상승, epoch 7에서 CANCELLED~~ |
| **33003822** | **2026-04-15 01:08** | **AIP** | **1 노드 × 1 H100** | **Two-Stream best_model(ep48) EgoDex action probing (test split, gap=10, patch_mean_concat)** |
| **33003926** | **2026-04-15 01:17** | **AIP_long** | **2 노드 × 4 H100** | **VideoMAE-ours 2-frame full training (mask 0.5, WD param group 분리, 50 epoch)** |
| 33012187 | 2026-04-15 10:02 | AIP | 1 노드 × 1 H100 (salloc, ~3분) | V4 ep4 RoPE rotation diagnostic 생성 (docs/architecture/rotation_V4_mask30_RoPE_s2_ep4_4.png) |
| ~~33012271~~ | ~~2026-04-15 18:30~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~Two-Stream APE 진단 학습 — ep4까지 저장 확인 후 2026-04-16 03:30 수동 cancel (회전 diag content-driven 확인 + probing peak R²=0.22로 mask 지배 확정)~~ |
| 33179509 | 2026-04-16 02:51 | AIP | 1 노드 × 1 H100 (3분) | APE 진단 ep4 rotation diagnostic 생성 (rotation_APE_diag_ep4_4.png) |
| 33179673 | 2026-04-16 03:08 | AIP | 1 노드 × 1 H100 (~16분) | APE 진단 ep4 action probing — peak R²=0.2191, ep48 RoPE(0.197)와 거의 동일 → mask 지배 확정 |
| ~~33179788~~ | ~~2026-04-16 03:30~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~Two-Stream v5 — APE + mask 0.5/0.5. ep8에서 position prior overfit 확인 후 CANCELLED, v6(rotation aug)으로 대체~~ |
| ~~33222151~~ | ~~2026-04-16 20:xx~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~Two-Stream v6 full training — ep23/30에서 scancel. ep8 peak (R²=0.259) → ep12 0.160 → ep20 0.146로 단조 악화, 추가 학습 실익 없음 판단 (약 2일 elapsed)~~ |
| **33183045** | **2026-04-16 10:20** | **AIP** | **1 노드 × 1 H100** | **VideoMAE-ours ep28 action probing (test split, gap=10, patch_mean)** |
| **33213615** | **2026-04-16 14:0x** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v4 ep48 probing — cls_mode=average** |
| **33213616** | **2026-04-16 14:0x** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v4 ep48 probing — cls_mode=concat** |
| **33213617** | **2026-04-16 14:0x** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v4 ep48 probing — cls_mode=patch_mean** |
| **33213640** | **2026-04-16 14:xx** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v5 ep4 probing — cls_mode=patch_mean_concat** |
| **33213646** | **2026-04-16 15:xx** | **AIP** | **1 노드 × 1 H100** | **VideoMAE-ours ep4 probing — cls_mode=patch_mean (동일 epoch 비교)** |
| **33222107** | **2026-04-16 19:4x** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v5 ep8 action probing — patch_mean_concat** |
| **33222108** | **2026-04-16 19:4x** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v5 ep8 rotation diagnostic** |
| **33257496** | **2026-04-17 09:xx** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v6 ep4 rotation diagnostic** |
| **33257497** | **2026-04-17 09:xx** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v6 ep8 rotation diagnostic** |
| **33257510** | **2026-04-17 09:xx** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v6 ep8 action probing — patch_mean_concat** |
| **33257511** | **2026-04-17 09:xx** | **AIP** | **1 노드 × 1 H100** | **VideoMAE-ours best_model(ep50) action probing — patch_mean** |
| **33257672** | **2026-04-17 09:xx** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v6 ep8 rotation diagnostic — DROID 프레임** |
| **33276335** | **2026-04-18 17:4x** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v6 ep20 rotation diagnostic — EgoDex(338) + DROID(ep_000002)** |
| **33276338** | **2026-04-18 17:5x** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v6 ep20 action probing — EgoDex test, patch_mean_concat, gap=10** |
| **33276351** | **2026-04-18 18:4x** | **AIP** | **1 노드 × 1 H100** | **Two-Stream v6 ep12 action probing — 중간 추세 확인용** |
| **33276769** | **2026-04-18 19:2x** | **AIP** | **1 노드 × 1 H100** | **Random-init Two-Stream action probing — probing 프로토콜 floor 확인** |
| **33276770** | **2026-04-18 19:2x** | **AIP** | **1 노드 × 1 H100** | **DINOv2 action probing — probing 프로토콜 ceiling 참조** |
| ~~33277747~~ | ~~2026-04-19 01:2x~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v7-big sanity — dataset path 오설정(`raw` vs `frames`)으로 0 videos, 실패. CPU sanity는 이미 통과했으므로 full training 진행~~ |
| **33277748** | **2026-04-19 01:3x** | **AIP** | **1 노드 × 1 H100** | **DINOv2 action probing 재시도 (HF_HUB_OFFLINE=1, login node에서 사전 캐싱 완료)** |
| ~~33277749~~ | ~~2026-04-19 01:3x~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~v7-big Option 1 (CLS_M in decoder) — DDP FAILED 6분 후. `mask_token_p` 고아 파라미터 + 설계 의도 Option 3로 전환 필요~~ |
| ~~33277774~~ | ~~2026-04-19 01:4x~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~v7-big Option 3 — ep8 loss 0.0007까지 수렴했으나 probing R² ~0 + attention 시각화에서 CLS_P_bg≈CLS_P_motion 균질화 확인. scancel after 19h~~ |
| ~~33282021~~ | ~~2026-04-19 21:58~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~v7-big +attention isolation — 2026-04-20 14:49 scancel. ep4 진단(cos=+0.9997, swap 무반응) + ep5~8 loss 0.0007 plateau로 sigma003과 동일 collapse 확정. Elapsed 16:50:17 → 134.7 GPU·h. ep8 진단(attn viz 33346987, CLS diag 33346988) 제출됨.~~ |
| ~~33345568~~ | ~~2026-04-20 08:47~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v7-big isolated ep4 attention viz (완료 1:17, attn_v7big_isolated_ep4.png)~~ |
| ~~33345591~~ | ~~2026-04-20 08:58~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v7-big isolated ep4 CLS 진단 (완료 1:06, cos/swap ablation)~~ |
| ~~33346565~~ | ~~2026-04-20 12:2x~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 sanity 1차 — 학습 완료되었으나 scheduler T_max=0 (num_epochs ≤ warmup) 버그로 end-of-epoch crash. 수정 후 재제출.~~ |
| ~~33346639~~ | ~~2026-04-20 13:16~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 sanity 2차 (완료 1:11, scheduler 가드 검증 통과)~~ |
| ~~33346956~~ | ~~2026-04-20 14:14~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 sanity 3차 (완료 0:44, A+B+D collapse 방어 검증 통과)~~ |
| ~~33346957~~ | ~~2026-04-20 14:2x~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~v8 full — 실수로 30ep 제출. PENDING에서 scancel 후 재제출 (과금 없음)~~ |
| ~~33346962~~ | ~~2026-04-20 14:24~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~v8 full 1차 — scancel 2026-04-21 05:19 (elapsed ~14:55). ep8 probing R²=-0.22 → L_P scale mismatch로 collapse 확정 (λ·L_P가 L_M 35배 압도). 수정 방향: L_P BYOL form + λ_max 0.05로 재설계.~~ |
| ~~33451082~~ | ~~2026-04-20 21:28~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 ep4 attention viz (완료 1:48, attn_v8_ep4.png)~~ |
| ~~33451969~~ | ~~2026-04-21 04:26~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 ep8 attention viz (완료 1:26, attn_v8_ep8.png — M/P attention focused화, v7-big 붕괴 패턴 유사)~~ |
| ~~33451974~~ | ~~2026-04-21 04:34~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 ep8 action probing (EgoDex test, gap=10, patch_mean_concat). R² = **-0.2214** (FAIL, random 이하) → 1차 run 폐기~~ |
| ~~33451976~~ | ~~2026-04-21 05:28~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 sanity 4차 (L_P BYOL form + λ=0.05 scale 검증) — 완료 1:35, 예상치 정확히 일치 (L_P_init ≈ 2.0, L_M 변화 없음)~~ |
| ~~33451982~~ | ~~2026-04-21 05:40~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 sanity 5차 (cls_m_grad_ratio=0.3 + mask_ratio_m 0.5 추가 검증) — 완료 0:50, forward/backward OK~~ |
| ~~33451989~~ | ~~2026-04-21 05:43~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~Two-Stream v8 full 2차 — ep13 도달 후 2026-04-21 14:58 scancel. ep12 probing 종합 FAIL (patch_mean_concat R²=-0.147, patch_mean_m=+0.160, patch_mean_p=-0.468). Static salience collapse 확정. L_P + EMA teacher 접근 전면 폐기. Elapsed 9:11:48 → ~73.6 GPU·h.~~ |
| ~~33453342~~ | ~~2026-04-21 11:11~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 ep12 attention viz (완료 0:48, attn_v8_ep12_run2.png — P가 극도로 sparse한 핀포인트, M이 배경 전역 분산. Static salience 패턴 관찰)~~ |
| ~~33453613~~ | ~~2026-04-21 11:22~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 ep12 rotation diagnostic (완료 1:23, docs/architecture/rotation_v8/ 4개 샘플. EgoDex content-driven, DROID cos_st 0.98+ trivial alignment 관찰)~~ |
| ~~33454356~~ | ~~2026-04-21 11:42~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 ep12 probing patch_mean_concat (완료 15:29, R²=-0.147 FAIL)~~ |
| ~~33466588~~ | ~~2026-04-21 13:51~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 ep12 probing patch_mean_m (완료 15:10, R²=+0.160 — M stream에 의미있는 motion signal)~~ |
| ~~33466589~~ | ~~2026-04-21 13:51~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 ep12 probing m_only CLS (완료 15:10, R²=+0.011 — CLS 수준엔 정보 희석)~~ |
| ~~33466590~~ | ~~2026-04-21 13:51~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v8 ep12 probing patch_mean_p (완료 15:10, R²=-0.468 — P가 action-harmful static feature로 수렴)~~ |
| ~~33477204~~ | ~~2026-04-21 14:49~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v9 sanity (완료 1:07, forward/backward OK. loss_m=0.021, loss_p_raw=0.026 → ratio p/m=1.2 (loss_weight_p=1.0 적정). cos_intra_p=1.000 early-collapse 경고 — sanity 규모 한계 가능성)~~ |
| ~~33479577~~ | ~~2026-04-21 15:07~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v9 residual 중간 sanity (완료 4:18, 200 videos × 5 epoch). P encoder collapse 확정 (cos_intra_p=1.000 5ep 내내 고정, std_p=0.028). Residual target의 "0 출력 trivial minimum" 함정 확인 → MAE 방식(target=frame_t)으로 설계 전환~~ |
| ~~33490873~~ | ~~2026-04-21 15:35~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v9 MAE 중간 sanity (완료 5:01). ep5 std_p=0.327 (residual 대비 12배), cos_intra_p=0.857 (분화 진행), ratio p/m=1.25. MAE 방식이 healthy 학습 궤도 확인 → full run 제출~~ |
| ~~33492965~~ | ~~2026-04-21 16:04~~ | ~~AIP_long~~ | ~~2 노드 × 4 H100~~ | ~~v9 full run (P=current, MAE frame_t). ep4 probing 결과 M=+0.188, concat=+0.154, P=-0.102. P=current의 trivial성 + residual collapse 재설계 위해 ep6 진행 중 scancel (07:14:48). 체크포인트 보존 (resume 가능)~~ |
| ~~33553317~~ | ~~2026-04-21 23:16~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~P=future 비교 sanity — 0:03:07에 scancel (설계 전면 재검토로 불필요해짐)~~ |
| ~~33555312~~ | ~~2026-04-21 23:33~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v9 residual+patch-norm sanity (완료 4:02). L_m=0.0147, L_p_raw=0.999, ratio p/m=68 → loss_weight_p=0.02 결정. std_p=0.047 (collapse 경향은 LR=0 말미 상태의 artifact 추정)~~ |
| **33555333** | **2026-04-21 23:41** | **AIP_long** | **2 노드 × 4 H100** | **v9 residual+patch-norm full run — 50 epoch, loss_weight_p=0.02, --time=3d. CHECKPOINT_SUFFIX=residual_norm (별도 디렉토리). ep4에서 사용자 진단 예정** |
| ~~33547220~~ | ~~2026-04-21 22:31~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v9 ep4 probing patch_mean_concat (완료 14:30, **R²=+0.154** — v8 ep12 -0.147 대비 positive 전환)~~ |
| ~~33547221~~ | ~~2026-04-21 22:31~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v9 ep4 probing patch_mean_m (완료 14:26, **R²=+0.188** — M stream motion 신호, v8 +0.160과 유사)~~ |
| ~~33547222~~ | ~~2026-04-21 22:31~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v9 ep4 probing patch_mean_p (완료 14:28, **R²=-0.102** — v8 ep12 -0.468 대비 크게 개선, static salience collapse 해소)~~ |
| ~~33547223~~ | ~~2026-04-21 22:41~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~v9 ep4 attention viz (완료 1:32, attn_v9_ep4.png — M/P 분화된 attention, v8 sparse pinpoint 패턴 없음)~~ |
| ~~33277748~~ | ~~2026-04-19 01:2x~~ | ~~AIP~~ | ~~1 노드 × 1 H100~~ | ~~DINOv2 probing — HF cache 위치 오류로 또 실패. 22초만에 FAILED~~ |
| **33277820** | **2026-04-19 04:0x** | **AIP** | **1 노드 × 1 H100** | **DINOv2 probing 3차 재제출 — 캐시 경로 확정 후 정상 동작 기대** |
| **33282006** | **2026-04-19 20:xx** | **AIP** | **1 노드 × 1 H100** | **v7-big ep4 probing — CLS_P_bg only** |
| **33282007** | **2026-04-19 20:xx** | **AIP** | **1 노드 × 1 H100** | **v7-big ep4 probing — CLS_P_motion only** |
| **33282008** | **2026-04-19 20:xx** | **AIP** | **1 노드 × 1 H100** | **v7-big ep4 probing — all_cls_concat (M+bg+motion)** |
| **33282009** | **2026-04-19 20:xx** | **AIP** | **1 노드 × 1 H100** | **v7-big ep4 probing — patch_mean_concat (v4/v5/v6과 동일 mode)** |
| **33282010** | **2026-04-19 20:xx** | **AIP** | **1 노드 × 1 H100** | **v7-big ep4 attention viz (3 CLS 비교, EgoDex+DROID)** |
| 32712320 | 2026-04-10 16:05 | core_s | 72 CPU | 프레임 손상 검증 (validate_frames.py) |
| 32866942 | 2026-04-11 | mig-1g.10gb | 1 MIG GPU | V-JEPA 모델 sanity check (38초, COMPLETED) |
| 32867620 | 2026-04-12 12:17 | normal_cpu | 1 노드 8 CPU | DROID 10ep 프레임 추출 (OOM 5ep에서 중단) |

---

## 완료 세션 (sbatch / salloc)

비용은 잡 단위가 아닌 월 누적 ceil이므로, 여기엔 **자원·시간**만 기록한다 (GPU·hours 또는 노드·hours).
월말에 누적 합산해서 "누적 사용량" 표에 반영.

| JobID | 시작 | 종료 | Elapsed | 파티션 | 자원 | 자원·시간 | 목적 | State |
|-------|------|------|---------|--------|------|----------|------|-------|
| 32438207 | 2026-04-09 10:23:18 | 10:23:26 | 00:00:08 | normal_cpu | 1 노드 전체 (144 CPU) | 0.32 CPU·h | EgoDex part1 추출 첫 시도 | FAILED (unzip 없음) |
| 32438237 | 2026-04-09 10:25:27 | 10:44:44 | 00:19:17 | normal_cpu | 1 노드 전체 (144 CPU) | 46.3 CPU·h | EgoDex part1 추출 (단일 unzip baseline) | CANCELLED (병렬 unzip 발견 후 취소, 22% unzip 진행) |
| 32438378 | — | — | 00:00:00 | core_s,normal_cpu | --exclusive | 0 | part1 (병렬 unzip 적용) | CANCELLED (큐 4h 대기, 부분 점유로 전환) |
| 32438384 | — | — | 00:00:00 | core_s,normal_cpu | 64 CPU | 0 | part1 (64 CPU) | CANCELLED (normal_cpu OverSubscribe=EXCLUSIVE 충돌, core_s 단독 시도) |
| 32438390 | 2026-04-09 10:53:09 | 11:42:30 | 00:49:21 | core_s | 32 CPU (olaf-c006) | 26.3 CPU·h | EgoDex part1 (병렬 unzip OK, 추출 32 CPU 느림) | CANCELLED (32→96 CPU 확장) |
| 32438409 | 2026-04-09 11:43:22 | 12:35:36 | 00:52:14 | core_s | 96 CPU (olaf-c034) | 83.5 CPU·h | EgoDex part1 추출 (최종) | COMPLETED (46,234 videos → 316 GB) |
| 32438442_4 | 2026-04-09 13:02:24 | 13:03:32 | 00:01:08 | core_s | 72 CPU (olaf-c041) | 1.4 CPU·h | EgoDex part4 추출 첫 시도 | FAILED (parallel_unzip race condition, dir 동시 생성) |
| 32438443 | 2026-04-09 13:02:24 | 13:02:33 | 00:00:09 | core_s | 48 CPU (node?) | 0.12 CPU·h | EgoDex test 추출 첫 시도 | FAILED (동일 race condition) |
| 32453576 | 2026-04-09 20:12:58 | 20:13:03 | 00:00:05 | AIP | 1 노드 × 1 H100 | 0.0 GPU·h | DDP sanity 첫 시도 (scratch 확인 결과 7 TB 확인됨) | FAILED (srun이 `python` 절대경로 못 찾음) |
| 32685867 | 2026-04-10 11:10:19 | 11:12:10 | 00:01:51 | AIP | 1 노드 × 1 H100 | 0.03 GPU·h | Stage 2: 1 GPU DDP sanity | COMPLETED (37.6 samples/sec, loss 0.035) |
| 32685895 | — | — | — | AIP | 1 노드 × 4 H100 | 0 | Stage 3: 4 GPU sanity | CANCELLED (자원 부족, 2 GPU로 축소) |
| 32685913 | — | — | — | AIP | 2 노드 × 4 H100 | 0 | Stage 4: 8 GPU sanity | CANCELLED (자원 부족, 2×1 GPU로 축소) |
| 32686381 | 2026-04-10 11:16:32 | 11:17:03 | 00:00:31 | AIP | 1 노드 × 2 H100 (--gpus-per-task=1) | 0.01 GPU·h | Stage 3 재시도 | FAILED (CUDA invalid device ordinal: gpus-per-task CUDA_VISIBLE_DEVICES 제한) |
| 32686403 | — | — | — | AIP | 2 노드 × 1 H100 (--gpus-per-task=1) | 0 | Stage 4 재시도 | CANCELLED (동일 문제 예상하여 gres로 전환) |
| 32686648 | 2026-04-10 11:20:27 | 11:20:39 | 00:00:12 | AIP | 1 노드 × 2 H100 (device_idx fix) | 0.007 GPU·h | Stage 3 재시도 2 | FAILED (NCCL nvmlDeviceGetHandleByPciBusId — 여전히 gpus-per-task 제한) |
| 32686803 | 2026-04-10 11:23:04 | 11:23:15 | 00:00:11 | AIP | 1 노드 × 2 H100 (--gres=gpu:2) | 0.006 GPU·h | Stage 3 재시도 3 | COMPLETED (but no training: auto-resume이 Stage 2 checkpoint에서 epoch 2로 복귀, num_epochs=1라 루프 0회) |
| 32686826 | — | — | — | AIP | 2 노드 × 1 H100 | 0 | Stage 4 재시도 | CANCELLED (checkpoint collision 수정 후 재제출) |
| 32687205 | 2026-04-10 11:25:09 | 11:27:04 | 00:01:55 | AIP | 1 노드 × 2 H100 | 0.06 GPU·h | Stage 3: 2 GPU 1 node DDP sanity | ✅ COMPLETED (90.8 samples/sec, 2.4× vs Stage 2, NCCL over NVLink OK) |
| 32687214 | 2026-04-10 12:24:05 | 12:24:15 | 00:00:10 | AIP | 2 노드 × 1 H100 | 0.006 GPU·h | Stage 4 (port 29500) | FAILED (EADDRINUSE — BF16 검증 잡과 포트 충돌) |
| 32708103 | 2026-04-10 12:24:05 | 12:25:45 | 00:01:40 | AIP | 1 노드 × 1 H100 | 0.03 GPU·h | BF16+FusedAdamW 검증 | ✅ COMPLETED (62.2 s/s, FP32 대비 +65%) |
| 32710540 | 2026-04-10 13:08:23 | 13:12:19 | 00:03:56 | AIP | 2 노드 × 1 H100 | 0.13 GPU·h | Stage 4: multi-node NCCL (port 29501) | ✅ COMPLETED (102.4 s/s, NCCL Init COMPLETE) |
| 32710541 | 2026-04-10 13:38:36 | 13:38:37 | 00:00:01 | AIP_long | 2 노드 × 4 H100 | 0.002 GPU·h | Full training 첫 시도 | FAILED (/scratch Permission denied) |
| 32710577 | 2026-04-10 14:09:41 | 15:47:00 | 01:40:28 | AIP_long | 2 노드 × 4 H100 | 13.4 GPU·h | Full training (scratch stage-in) | CANCELLED (cp -a 수천만 소형 파일 메타데이터 병목, stage-in만 20h+ 예상 → GPFS 직접으로 전환) |
| 32712324 | 2026-04-10 16:06:38 | 2026-04-14 04:06:44 | 3-12:00:06 | AIP_long | 2 노드 × 4 H100 | 672.0 GPU·h | Two-Stream v4 full training (GPFS, BF16) | TIMEOUT (epoch 48/50 도달, ep33부터 loss plateau라 resume 불필요로 판단) |
| 32993295 | 2026-04-14 17:05:34 | 17:05:41 | 00:00:07 | AIP | 1 노드 × 1 H100 | 0.002 GPU·h | Probe 첫 시도 | FAILED (probe_action.py에 --mask-ratio 인자 없음) |
| 32993335 | 2026-04-14 17:09:18 | 17:09:20 | 00:00:02 | AIP | 1 노드 × 1 H100 | 0.001 GPU·h | Probe 재시도 | FAILED (동일 원인) |
| 32993348 | 2026-04-14 17:10:01 | 17:10:17 | 00:00:16 | AIP | 1 노드 × 1 H100 | 0.004 GPU·h | Probe 재시도 2 | FAILED (동일 원인, sbatch 수정 전 중단) |
| 32950553 | 2026-04-12 22:35:29 | 2026-04-15 01:16:47 | 2-02:41:18 | AIP_long | 2 노드 × 4 H100 | 405.5 GPU·h | V-JEPA-ours 3차 (mask 50/60%, warmup) | CANCELLED (epoch 30 도달 후 계획 중단, negative result로 기록; loss collapse→recovery→재발산 패턴) |
| 33345568 | 2026-04-20 08:47:22 | 08:48:39 | 00:01:17 | AIP | 1 노드 × 1 H100 | 0.02 GPU·h | v7-big isolated ep4 attention viz (3-CLS 비교) | COMPLETED (attn_v7big_isolated_ep4.png) |
| 33345591 | 2026-04-20 08:58:23 | 08:59:29 | 00:01:06 | AIP | 1 노드 × 1 H100 | 0.02 GPU·h | v7-big isolated ep4 CLS 진단 (cos + decoder swap) | COMPLETED (cos=+0.9997 → collapse 확정) |
| 33346639 | 2026-04-20 13:16:00 | 13:17:11 | 00:01:11 | AIP | 1 노드 × 1 H100 | 0.02 GPU·h | v8 sanity 2차 (scheduler 가드 검증) | COMPLETED |
| 33346956 | 2026-04-20 14:14:44 | 14:15:28 | 00:00:44 | AIP | 1 노드 × 1 H100 | 0.01 GPU·h | v8 sanity 3차 (α_var collapse 방어 검증) | COMPLETED |
| 33451082 | 2026-04-20 21:28:29 | 21:30:17 | 00:01:48 | AIP | 1 노드 × 1 H100 | 0.03 GPU·h | v8 ep4 attention viz (student+teacher+Pred M) | COMPLETED (attn_v8_ep4.png — 샘플별 패턴 상이, collapse 징후 없음) |
| 33346962 | 2026-04-20 14:24:24 | 2026-04-21 05:19:xx | 14:55:xx | AIP_long | 2 노드 × 4 H100 | ~119.3 GPU·h | v8 full 1차 (λ_max=0.2) | CANCELLED — ep8 probing R²=-0.22 (collapse). L_P scale mismatch 진단 → λ=0.05 + BYOL normalize로 재설계 |
| 33451969 | 2026-04-21 04:26:xx | 04:27:xx | 00:01:26 | AIP | 1 노드 × 1 H100 | 0.02 GPU·h | v8 ep8 attention viz | COMPLETED (attn_v8_ep8.png) |
| 33451974 | 2026-04-21 04:34:xx | 04:50:xx | 00:15:41 | AIP | 1 노드 × 1 H100 | 0.26 GPU·h | v8 ep8 action probing | COMPLETED (R²=-0.2214 FAIL) |
| 33451976 | 2026-04-21 05:28:xx | 05:30:xx | 00:01:35 | AIP | 1 노드 × 1 H100 | 0.03 GPU·h | v8 sanity 4차 (L_P BYOL form + λ=0.05) | COMPLETED (scale 검증 통과) |
| 33451982 | 2026-04-21 05:40:xx | 05:41:xx | 00:00:50 | AIP | 1 노드 × 1 H100 | 0.01 GPU·h | v8 sanity 5차 (cls_m_grad_ratio=0.3 + mask_m=0.5) | COMPLETED (forward/backward OK) |
| 33451989 | 2026-04-21 05:46:32 | 2026-04-21 14:58:20 | 09:11:48 | AIP_long | 2 노드 × 4 H100 | ~73.6 GPU·h | v8 full 2차 (BYOL form, λ=0.05, grad 0.3) | CANCELLED — ep12 probing patch_mean_concat R²=-0.147. M/P 분리 probing(patch_mean_m=+0.160, patch_mean_p=-0.468)으로 P가 static salience로 수렴 확정. v8 접근 전면 폐기, v9로 전환 |
| 33453342 | 2026-04-21 11:11:51 | 11:12:39 | 00:00:48 | AIP | 1 노드 × 1 H100 | 0.013 GPU·h | v8 ep12 attention viz | COMPLETED (attn_v8_ep12_run2.png — P 극단 sparse 핀포인트, M 배경 분산) |
| 33453613 | 2026-04-21 11:22:xx | 11:23:xx | 00:01:23 | AIP | 1 노드 × 1 H100 | 0.023 GPU·h | v8 ep12 rotation diagnostic 4샘플 | COMPLETED (EgoDex content-driven 유지, DROID cos_st 0.98+) |
| 33454356 | 2026-04-21 11:42:03 | 11:57:32 | 00:15:29 | AIP | 1 노드 × 1 H100 | 0.26 GPU·h | v8 ep12 probing patch_mean_concat | COMPLETED (R²=-0.1467 FAIL) |
| 33466588 | 2026-04-21 13:51:09 | 14:06:19 | 00:15:10 | AIP | 1 노드 × 1 H100 | 0.25 GPU·h | v8 ep12 probing patch_mean_m | COMPLETED (R²=+0.1595) |
| 33466589 | 2026-04-21 13:51:09 | 14:06:19 | 00:15:10 | AIP | 1 노드 × 1 H100 | 0.25 GPU·h | v8 ep12 probing m_only CLS | COMPLETED (R²=+0.0108) |
| 33466590 | 2026-04-21 13:51:09 | 14:06:19 | 00:15:10 | AIP | 1 노드 × 1 H100 | 0.25 GPU·h | v8 ep12 probing patch_mean_p | COMPLETED (R²=-0.4676 — P가 action-harmful) |
| 33477204 | 2026-04-21 14:49:xx | 14:50:xx | 00:01:07 | AIP | 1 노드 × 1 H100 | 0.019 GPU·h | v9 sanity (residual P + mask 0.75) | COMPLETED (loss_m=0.021/loss_p_raw=0.026, ratio p/m=1.2. cos_intra_p=1.000 early-collapse 경고) |

**자원·시간** 계산:
- GPU 잡: `n_gpus × elapsed_hours` (단위: GPU·h)
- CPU 잡 전체 노드: `n_nodes × elapsed_hours` (단위: 노드·h)
- CPU 잡 부분 점유: `n_cpus × elapsed_hours` (단위: CPU·h). 월말에 노드·h = CPU·h / 144로 환산 (노드당 144 CPU 가정, 운영팀 확인 후 반영)

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
**자원·시간** 계산해서 기입:
- GPU: `n_gpus × elapsed_hours` (단위 GPU·h)
- CPU: `n_nodes × elapsed_hours` (단위 노드·h)

비용은 월말에 누적 합산 후 한 번에 계산 (잡 단위 ceil 아님).

### 3. 저장소 증설 시에만
mrg 그룹은 이미 10 TB를 할당받았으므로 그 범위 내 사용은 추적 불필요.
**10 TB 초과 증설을 신청한 경우에만** "저장소" 표에 1줄 추가.

### 4. 다중 작업 묶음
sbatch array는 부모 JobID로 1줄 + 비고에 array 범위 (예: `array=1-5`).
잡 단계가 많은 워크플로우는 단계별로 분리 기록.

### 5. 비정상 종료
`TIMEOUT`, `FAILED`, `CANCELLED`, `OUT_OF_MEMORY` 등은 State에 명시 + 비고에 짧은 원인.
다음 잡에서 같은 실수 반복 방지.

---

## 누적 사용량 (월별 요약)

매월 말에 "완료 세션"의 자원·시간을 합산해서 일수로 환산하고 ceil 적용.

```
H100·일수 = ceil(월 H100·hours 합 / 24)
CPU·일수  = ceil(월 노드·hours 합 / 24)
H100 비용 = H100·일수 × 61,000원
CPU 비용  = CPU·일수 × 7,000원
```

| 월 | H100·hours 합 | H100·일수 (ceil) | 노드·hours 합 | CPU·일수 (ceil) | 비용 추정 (원, VAT 별도) | 비고 |
|----|--------------|-----------------|--------------|-----------------|--------------------------|------|
| 2026-04 | 0 | 0 | 0 | 0 | 0 | 클러스터 환경 셋업 시작 |
