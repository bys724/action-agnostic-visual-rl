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

### 2026-06-15 Run A — target 정규화 anti-collapse (붕괴 정공법 전환)

**배경**: SSIM/LR/warmup 5연패(35757465·35760276·35760399·35760680) 종결. 붕괴 근인 재진단 = V-JEPA P **자기참조 constant collapse**(target=EMA student 거울 → student 저분산화하면 target도 따라 무너져 loss→0). recon 노브는 증상만 건드림. → **target 정규화**(① target LayerNorm ② VICReg variance reg)로 전환. masked anchor(scaffold용)는 별개 영역이라 **Run B로 분리**(단일변수 격리). 상세 = [v15b_retraining_status.md](v15b_retraining_status.md) §8.

**코드**: `lambda_var`(VICReg, P enc 출력 per-dim std<1 hinge) + `target_ln`(I-JEPA식 target LayerNorm) 신규 플래그 (기본 off=기존 동작). [two_stream_v15.py](../src/models/two_stream_v15.py) `_variance_loss`/`_vjepa_p_one_segment`/`_forward_pair`, [pretrain.py](../scripts/pretrain.py) `--v15-lambda-var`/`--v15-target-ln`, sanity sbatch env wiring. CPU 소형 smoke 통과(forward/backward·loss finite·variance grad→P enc norm 5.45).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 35763563 | AIP 1×1 H100 | 02:00:00 | **Run A sanity** (parvo pair, gate=0, V15_LAMBDA_VAR=1.0 + target_ln, 50vid×3ep, SUFFIX=runA). 체크: DDP/GPU 무크래시·NaN 없음·loss 스케일·std 회복 | ✅ COMPLETED 3m13s (~0.05 GPU·h). **붕괴 궤적 역전 확인**: std_p 0.58→**1.0**(옛 붕괴는 →0, variance reg가 target=1로 끌어올림), cos_intra_p 0.60→**0.165**(P 다양화), cos(pred,tgt) 0.57~0.84(0.99 미saturate), NaN 없음. ⚠️ std_m 0.05(M엔 reg 없음, 옛 transient 패턴). 3ep라 증명 아님 but 궤적 명백 역전 → 통과 |
| 35764680 | AIP_long 2×4 H100 | 18:00:00 | **Run A 본학습 (cap 20ep)** (parvo pair scratch, V15_LAMBDA_VAR=1.0+target_ln, gate=0+warmup5(V-JEPA 일찍 활성→판정 빠름), batch 64, lr 2e-4, part1-5, SUFFIX=runA → ckpt `parvo_runA/`). **50ep 안 감**(진단+baseline용, 라우팅순서 fix는 Run B). | ⏳ RUNNING. **판정 ep12**(cos(pred,tgt)<0.97·std_p 유지·viz GT급이면 통과), **조기경보 ep4~5**(L_t 역행 없어야). 붕괴 시 즉시 abort. 건강 시 ep16~20 멈춤(resume로 50 확장 가능=sunk cost 아님). 예상 ~12h/~96 GPU·h |

### 2026-06-12 Parvo 2-frame pair 재설계 (compose 제거 + ep8 init)

**배경**: 3-frame Parvo(35560151) ep12+ V-JEPA P trivial collapse (cos(pred,tgt)→1.0, M 기여=0) → 19h45m(158 GPU·h)에 중단. 진단: collapse 근인 = V-JEPA P 자체 trivial(정지영역 지배), compose와 독립. 사용자 결정: **compose(미입증) 배제 + 2-frame pair**로 motion routing만 격리 검증. ep8(gate 구간 pure-MAE, collapse 전 최신 ckpt)에서 가중치만 init.

**코드 (커밋 7b47a7f)**: `pair_mode`/`use_compose` 플래그 + `_forward_pair`(P MAE×2 + V-JEPA P×1 + V-JEPA M×1, L_compose 제거, composition_head 미생성). `--init-from`(strict=False 가중치만). training loop pair unpacking/forward/eval. 독립 크롭 유지(사용자 의도=global-shift 불변 correspondence 학습).

**sanity (35755046)**: ep8 init 정상(L_t 0.0015), L_compose=0, ~37% 빠름(51s vs 81s/50vid). cos(pred,tgt) ep3 0.94 (3-frame 0.999 대비 미붕괴 — 단 3ep라 비결정적).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 35560151 | AIP_long 2×4 H100 | — | 3-frame Parvo 본학습 | ❌ CANCELLED 19h45m (158 GPU·h). ep12+ V-JEPA collapse |
| 35755051 | AIP_long 2×4 H100 | 2-12:00:00 | **2-frame pair 본학습** (compose 제거, ep8 init, batch 64, gate=0 warmup 5ep, 40ep) | ❌ FAILED 4h20m (35 GPU·h). ep10에서 NCCL/IB 네트워크 장애(olaf-g001↔g002, vendor err 249)로 죽음 — 코드 버그 아님. ckpt latest.pt/ep9 보존. **지표 판단 주의**(사용자 지적): cls_p에는 학습 압력 없음 → cos_intra_p/std_p(CLS)로 성패 판단 무효. 유효 지표 = P 복구 L_t/L_tk(patch MAE, ep10 ~0.003) + V-JEPA P cos(pred,tgt)=0.993·L_pred 0.005. cos(pred,tgt)>0.99는 V-JEPA P trivial 의심 신호이나 target informativeness 미확인 → **가시화로 확정 필요**(35757146) |
| 35757146 | AIP 1×1 H100 | 00:20:00 | **parvo_pair 복구+라우팅 가시화** (latest.pt=ep10, NO_SOBEL=1). P MAE nomask 복구 + V-JEPA P motion routing 픽셀화 | ✅ COMPLETED 1m43s (0.03 GPU·h). 산출물 `paper_artifacts/parvo_pair_recon_samples/epoch_010_nomask.png`. **판독**: ep8 init 대비 ep10이 형상/고주파 정보 더 반영(사용자 확인) — collapse 아님. warmup(λ_pred·λ_m_jepa 0.01→1.0, 5ep)으로 loss 복잡화 = 학습 진행. cos(pred,tgt) 단독으론 trivial 판정 불가(target informativeness·routing-off 대조 필요). 주의: viz가 모델을 pair_mode 없이 생성→has_compose=True(ckpt에 ep8 stale composition_head 잔존, pair loss 미사용), col10 무시 |
| 35757158 | AIP_long 2×4 H100 | 2-00:00:00 | **parvo_pair 본학습 resume** (35755051 NCCL 장애 이어받기, ep11~40). 새 run_dir `parvo_pair/20260613_072456` | ❌ CANCELLED 5h26m (~43 GPU·h, ep23까지). resume 정상. **중지 사유**: ep23 가시화(35757428) 결과 P 복구가 ep10 대비 균일 갈색으로 흐려짐(고주파/형상 손실). 원인 진단 = **v15 복구 loss가 MSE-only** → MSE는 평균색만 맞춰도 낮아 형상 collapse에 둔감(L_t 0.003 유지하나 실제 형상 붕괴). 과거 two-stream(v4~v10)엔 있던 `mse+0.1·ssim`의 **SSIM term이 v15 `_mae_one_frame`에서 누락**됨. → SSIM 보강 후 재학습 결정 |
| 35757428 | AIP 1×1 H100 | 00:20:00 | **parvo_pair ep23 가시화** (resume run latest.pt, NO_SOBEL=1) | ✅ COMPLETED (0.03 GPU·h). 산출물 `paper_artifacts/parvo_pair_recon_samples/epoch_023_nomask.png`. ep10 대비 복구 균일 갈색으로 흐려짐(형상 손실) → MSE-only 복구 loss 진단 근거 |
| 35757463 | AIP 1×1 H100 | 02:00:00 | **SSIM 보강 sanity** (scratch, MODEL=parvo PAIR_MODE no-sobel, lambda_ssim=0.1, gate=0 V-JEPA 활성, 50vid×3ep). 체크: L_t(MSE+SSIM) 정상·BF16 NaN 없음·loss 스케일 | ✅ COMPLETED. **NaN 없음**, L_t ep1 0.122→ep3 0.048 단조감소(MSE-only ~0.0015 대비 상승 = SSIM 0.1 가중 작동 확인). scratch forward 정상 |
| 35757465 | AIP_long 2×4 H100 | 2-00:00:00 | **SSIM 보강 parvo_pair 본학습** (scratch, lambda_ssim=0.1, gate=5+warmup=10). ckpt `parvo_pair_ssim/` | ❌ CANCELLED 5h26m (~44 GPU·h, ep11+). **복구 퇴화 재발**: L_t ep3 0.034(복구됨)→ep5 0.053 역행→평탄. 가시화(`parvo_pair_ssim_recon_samples/epoch_{003,011}_pair.png`) ep3 장면 복원 vs ep11 균일 갈색. **진단(diagnose_ssim_scale.py)**: L_t=MSE+0.1·SSIM에서 **SSIM term이 MSE 압도**(ep3 3.4×). 퇴화=MSE 0.0066→0.0228 폭증인데 SSIM 거의 불변 → SSIM이 MSE 픽셀압력 가려 균일화 방어 실패. **lambda_ssim 0.1 과도가 원인**(사용자 가설=loss 스케일 적중). v15(MSE-only) 뭉갬 적던 것과 일관 |
| 35760276 | AIP_long 2×4 H100 | 2-00:00:00 | **SSIM 재보강 본학습** (scratch, lambda_ssim=0.02, gate5+warmup10, LR 2e-4). ckpt `parvo_pair_ssim002/` | ❌ CANCELLED 4h18m (~34 GPU·h, ep10+). **lambda_ssim 효과 확인 but 퇴화 별개**: 가시화(`parvo_pair_ssim002_recon_samples/epoch_{003,009}_pair.png`) **ep3 GT급 선명 복구**(0.1 run 흐릿 대비 개선=MSE 주도 효과 입증) vs **ep9 균일 갈색 재퇴화**. L_t ep3 0.0115→ep5 0.0155 역행 패턴 동일. → SSIM 스케일은 복구 품질엔 기여, **퇴화 원인은 LR/scratch 초기 불안정**(ep4 LR peak 도달 타이밍, 두 run 공통)으로 분리 확정 |
| 35760399 | AIP_long 2×4 H100 | 2-00:00:00 | **LR 인하 본학습** (scratch, LR 2e-4→1e-4, warmup 4ep, lambda_ssim=0.02·gate5). ckpt `parvo_pair_lr1e4/` | ❌ CANCELLED 2h15m (~18 GPU·h, ep5+). **LR peak 주범 아님 확정**: ep3 L_t 0.0081(역대 최저, 가시화 GT급 선명) → ep4 0.0195 역행. peak 절반으로 낮췄는데 ep4 역행 **오히려 더 큼**(2.4× vs 2e-4의 1.25×). 두 run 모두 ep4(LR warmup 종료)에 역행 → 원인=peak 크기 아닌 **warmup 종료 시 급가속 충격** 가설. 가시화 `parvo_pair_lr1e4_recon_samples/epoch_{003,004}_pair.png` |
| 35760680 | AIP_long 2×4 H100 | 2-00:00:00 | **🔵 LR warmup 연장 본학습** (scratch, **LR warmup 4→10ep**(ep10까지 천천히 증가, 급가속 충격 완화), LR peak 1e-4·lambda_ssim 0.02·gate5 동일=변수 격리). 코드: `--lr-warmup-epochs` 인자 신설([pretrain.py](src/training/pretrain.py#L1186)). ckpt `parvo_pair_wu10/` | ❌ CANCELLED 06-15 (~20h13m, ~161 GPU·h, ep35+). warmup 연장도 collapse 미방지(35763305 viz 확정). **SSIM/LR/warmup 5연패 종결** → 진단 전환: 붕괴 근인 = V-JEPA P 자기참조(teacher=EMA student) constant collapse. recon 노브가 아니라 **target 정규화(variance reg + target LayerNorm)**가 정공법 → Run A로 전환 |
| 35762739 | AIP 1×1 H100 | 00:20:00 | parvo_pair_wu10 가시화 (AIP 풀 H100) | ❌ CANCELLED (PENDING-Resources). 가시화=추론이라 풀 H100 불필요 → 최소사양으로 재제출 |
| 35762741 | mig-1g.10gb 1×1 | 00:20:00 | parvo_pair_wu10 가시화 (MIG 시도) | ❌ CANCELLED. MIG-1g 슬라이스 28개 전부 점유 → 예상 시작 **~2일 뒤**. V100으로 전환 |
| 35763304 | normal(V100) 1×1 | 00:20:00 | parvo_pair_wu10 가시화 (V100, sbatch module load) | ❌ FAILED 1s. **V100 노드엔 `module load miniforge3` 모듈 없음** → conda env GPFS python 직접 호출로 회피 |
| 35763305 | normal(V100) 1×1 | 00:20:00 | **parvo_pair_wu10 본학습 가시화** (latest.pt=ep35, NO_SOBEL=1, pair_mode, env python 직접호출 --wrap). 핵심 판독=warmup 연장(4→10ep)이 collapse 막는지 | ✅ COMPLETED 1m37s (~0.03 GPU·h, V100). bf16 V100 정상. **판독 = collapse 재현**: nomask 복구(col5-6)·motion routing(col7) **전부 균일 남보라**(완전 붕괴, V-JEPA P trivial=M기여0). 75%-masked 복구만 보이는 패치 부분복원. ep3 GT급→ep35 균일 = **warmup 연장도 collapse 미방지**. 산출물 `paper_artifacts/parvo_pair_wu10_recon_samples/epoch_035_pair.png`. **이후 viz/추론 = V100(normal) + env python 직접호출이 기본**(viz_v15.sbatch 변경, 단가 최저+최다가용) |

**코드 변경 (커밋)**: ① cleanup(b6d912e): 루트 tar 69GB + deprecated v12-14 스크립트 12개 삭제. ② rename(9ce778d): `MODEL=parvo`→내부 v15b 정규화 + ckpt `parvo/` (최소 rename, 코드 식별자 별도). ③ no-Sobel(9ce778d): `use_sobel` 플래그 — Paper 2(Parvo)는 P=RGB(3ch)/M=ΔL(1ch), Sobel은 Paper 1 주제라 격리. M depth 6 유지(1ch≠작은모델). ④ 진단계측(cd23ee8): `PRETRAIN_DIAG=1`로 첫 batch 입력통계+data/compute 병목분리, NaN guard 상시.

검증: CPU forward 양 모드 통과(sobel True 5/3ch, False 3/1ch).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 35550604 | AIP 1×1 H100 | 02:00:00 | **Parvo no-Sobel 진단 sanity** (gate=0, 50vid×3ep, PRETRAIN_DIAG=1) | ✅ COMPLETED 5m15s (0.09 GPU·h). **입력 parity OK**(P=RGB 3ch [0,1], M=ΔL 1ch [-0.7,0.7]). loss 스케일 Sobel판과 동등(L_t 0.047→0.019). **병목 = COMPUTE-bound 99%, data-load ≤1%** (3-frame 추출 병목 아님 확정). NaN 없음 |
| 35560151 | AIP_long 2×4 H100 | 5-00:00:00 | **Parvo 본학습** (MODEL=parvo = v15b+no-Sobel, gate=10, EMA 0.996, 50ep × part1-5, lr 2e-4). Paper 2 scaffold 검증. ckpt → checkpoints/parvo/ | PENDING. abort: ep12-18 trivial collapse(cos>0.99&L_pred<0.01)/발산 |

### 2026-06-10 v15b student-anchor 재학습 (main 브랜치, catalyst 최종 검증)

**배경**: 논문 v15(teacher-anchor)는 V-JEPA P anchor=`teacher_p(frame_t).detach()`라 P encoder가 V-JEPA gradient를 못 받음 → **M stream이 P 학습에 영향 0**. `0fb74c8`에서 anchor를 student P encoder로 변경(표준 V-JEPA 복원) → P·M 모두 motion routing gradient 수신. `b41b177` v15b = 동일 아키텍처 + collapse 방지 레시피(① recon-first hard-gate ② EMA 0.996 ⑤ lr scaling). 이번 재학습으로 M→P catalyst 가설 최종 검증.

**코드 변경 (2026-06-10)**:
- `scripts/cluster/pretrain.sbatch`: v15 분기가 `two-stream-v15b`도 매칭. `V15_GATE_EPOCHS`(default 0) + 모델별 EMA init default(v15b=0.996) 노출. CKPT_DIR `${MODEL//-/_}`로 v15/v15b 분리.
- `scripts/cluster/sanity_v15.sbatch`: `MODEL` override + `V15_GATE_EPOCHS` + 모델별 EMA/CKPT 분리. (v15 기존 동작 불변.)

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 35478663 | AIP 1×1 H100 | 02:00:00 | v15b sanity (student-anchor, gate=0, 50vid × 5ep, batch=8, nw=4). loss 규모·per-ep 시간·collapse 진단 → 본학습 파라미터 조정용 | ✅ COMPLETED 9m11s (0.15 GPU·h). ~93s/ep, 53.7 samp/s. **구조 안정**(NaN/발산 없음), **P MAE 건강**(L_t 0.049→0.0186, std_p 0.06→0.45, cos_intra_p 0.996→0.66). ⚠️ **L_pred trivial collapse**(0.044→0.0014@ep2, cos(pred,tgt) 0.955→0.998) — catalyst 채널 신호 ~0. ⚠️ std_m ep2-3 near-collapse(0.010)→ep5 0.118 회복. L_compose ~0.42 stuck(total의 87%). **단, gate=0이라 V-JEPA가 미성숙 P에서 켜진 preview** — gate=10 본학습과 timing 다름 |

성공 기준: ep2+에서 L_pred/L_m_jepa/L_compose nonzero·하강, total loss 발산/NaN 없음, cls_p·cls_m std uniform collapse 없음. **판정**: 구조/P MAE는 통과, 그러나 catalyst 핵심인 L_pred가 trivial → 본학습 직행 전 추가 진단/파라미터 검토 필요 (save-interval 999라 ckpt 미저장 → diagnose 불가).

| 35481545 | AIP 1×1 H100 | 02:00:00 | v15b sanity #2 진단 (gate=3 본학습 timing 모사, 200vid × 8ep, batch=8, nw=4, SAVE_INTERVAL=8 ckpt 저장). L_pred trivial이 P 미성숙 탓인지 과제 자체 trivial인지 분리 + diagnose_vjepa_p_trivial(M=0 vs M-on ablation) 용 | PENDING |

**코드 변경 (sanity #2)**: `sanity_v15.sbatch` `--save-interval` env 노출(`SAVE_INTERVAL`, default 999).

**sanity #2 결과 (35481545 COMPLETED 50m38s, 0.84 GPU·h)**: gate=3가 trivial collapse 방어 성공. gate 중(λ_pred=0) baseline cos(pred,tgt)≈0.77 = **과제 intrinsic trivial 아님** (sanity #1 0.998 collapse는 미성숙 P에서 V-JEPA 조기 점화 탓). gate 후 L_pred ~0.08 / cos ~0.94 plateau (sanity #1 0.002/0.998과 질적으로 다름). P 최종 건강(std_p 0.53, cos_intra_p 0.52). L_compose 0.43→0.32 학습됨. ⚠️ gate가 L_pred·L_mj·L_compose 셋 다 막아 M이 gate 동안 gradient 0(std_m flat 0.185) → 본학습 gate 설계 재고 필요(L_pred만 gate, M은 조기 성숙).

| 35493291 | AIP 1×1 H100 | 00:20:00 | diagnose_vjepa_p_trivial (sanity#2 ep8 ckpt). baseline cos + **M=0 vs M-on ablation** = M routing 실제 기여 측정 → 본학습 최종 분기 | ✅ COMPLETED 1m6s (0.02 GPU·h). **baseline cos 0.9015** (프레임 유사), predictor 0.9279, M=0 0.9147 → **Δ(M routing 기여)=+0.0132**. trivial collapse 아님(0.928<1.0). M 기여 작으나 ep8 미성숙(M 5ep만) — teacher-anchor ep50은 +0.31였음. gap별 baseline: gap0-9 0.92 → gap30 0.81 (큰 gap=motion 많음). **판정: collapse 없음 + M 양수(미성숙 한정) → 본학습 진행** |

### 2026-06-10 v15b 8-GPU 본학습 (student-anchor, gate=10)

sanity 2회 + diagnose 통과 후 본학습. 파라미터: 8 GPU(2×4) global batch 256, lr 2e-4(원래 v15 동일), V15_GATE_EPOCHS=10(sanity#2 검증), V15_EMA_INIT=0.996, warmup-start=0/epochs=10(gate→ramp), nw=8, MAX_GAP=30(원래 v15 유지). shared gate(pred/mj/compose 공통) 유지. ckpt → `/proj/external_group/mrg/checkpoints/two_stream_v15b/`.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 35493293 | AIP_long 2×4 H100 | 5-00:00:00 | **v15b 본학습 1차** (50ep, student-anchor, gate=10). 63min/ep 실측 | ❌ CANCELLED 34m39s = 8 GPU × 0.58h = **4.6 GPU·h**. forward 최적화 적용 위해 조기 중단 → 캐시 검증 후 재제출 |

### 2026-06-11 v15b forward 무손실 최적화 (중복 unmasked P-encoder 캐시)

**병목 진단**: 실행 중 잡 GPU util 88-98% = **compute-bound** (데이터 로딩/3-frame 추출 아님, 프레임은 사전 추출 JPEG). 모델 forward가 step당 full unmasked depth-12 P-encoder를 **6회** (V-JEPA P 3 segment × {student anchor, teacher target}) 호출하는데, 그중 2회가 중복(student P(t) ×2, teacher P(t+m) ×2). 인코더에 dropout/droppath 없음(`drop_path_rate=0.0` 미사용) → unmasked forward deterministic → **gradient 합산 동치로 무손실 캐시 가능**.

**수정** (`src/models/two_stream_v15.py`): forward에서 unique frame당 1회만 인코딩(student {t,t+n}, teacher {t+n,t+m})해 `_vjepa_p_one_segment`에 전달. full P forward 6→4 (~30% P 연산↓, wall-clock ~15-20% 예상). batch는 32 유지(원래 v15 비교 보존 — batch 키우면 update수·LR 변해 comparability 깨짐 → 미적용).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 35493303 | AIP 1×1 H100 | 02:00:00 | 캐시 forward smoke sanity (gate=0, 50vid×3ep — sanity#1 동일조건). loss 규모가 sanity#1과 일치하는지 검증 | ✅ COMPLETED 4m37s (0.08 GPU·h). **무손실 확인**: well-conditioned 항 L_t 0.0497/L_cp 0.408/std_p 0.058 ≈ sanity#1(0.0493/0.418/0.059) 1-3% 이내, total 궤적 동일, NaN 없음. L_pred/L_mj 미세차는 trivial-collapse chaotic+bf16 비결정성(버그 아님) |

**⏸ 본학습 2차 보류 (2026-06-11)**: forward 최적화 검증 통과했으나, 사용자가 **추가 리팩토링을 다른 워크스테이션에서 진행 후** 재제출 예정. 현재 작업 commit+push 후 대기. (재제출 명령은 `docs/v15b_retraining_status.md` §6 참조 — 캐시 코드 반영본.)

### 2026-05-15 paper_experiments_plan §C6 + §C7 (신규 코드 작성 + 잡 제출)

§C6 (recon quality v11 vs v15) + §C7 (VideoMAE-ours P_t+P_tk catalyst evidence) 는 기존 probing 인터페이스로 안 됐던 작업. 코드 추가 후 잡 제출.

**코드 변경**:
- `scripts/eval/probe_action.py`: videomae branch에 `patch_mean_concat_p_t_p_tk` mode 추가 (same-frame replica forward로 단일 frame representation 추출). `--cls-mode` choices/load_encoder embed_dim 동기화
- `src/encoders/adapters/videomae.py`: VideoMAEOursAdapter에 `mode={paired,p_t_p_tk}` 인자 추가
- `scripts/eval/probe_action_libero.py` + `scripts/cluster/probe_action_libero.sbatch`: `--videomae-mode` / `VIDEOMAE_MODE` env var
- `scripts/eval/recon_quality_v11_vs_v15.py` + `scripts/cluster/recon_quality.sbatch`: 신규 (§C6 spike). EgoDex test split N samples × {v11 (img_t, img_tk), v15 (img_t, img_n, img_tk)} forward → pred_t/pred_tk per-sample MSE 비교

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34467116 | AIP 1×1 H100 | 03:00:00 | §C7 EgoDex VideoMAE × `patch_mean_concat_p_t_p_tk` × gap=10 | ❌ FAILED 7s (argparse choices 누락, fix됨) |
| 34467123 | AIP 1×1 H100 | 01:30:00 | §C7 LIBERO spatial × videomae-ours × `p_t_p_tk` × gap=20 | ✅ COMPLETED 4m27s |
| 34467129 | AIP 1×1 H100 | 03:00:00 | §C7 EgoDex 재제출 (cls-mode choices fix) | RUNNING |
| 34467135 | AIP 1×1 H100 | 02:00:00 | §C6 recon quality v11 ep44 vs v15 ep50 × 200 sample (EgoDex test, triple) | PENDING |

총 ~4 GPU·h. 결과는 paper_artifacts/{libero_action_probing, probing, recon_quality}/ 각각 등록 예정.

### 2026-05-15 v15 ep50 paper_experiments_plan §C4 + §C5 (의존성 없는 probing 일괄 제출)

paper_experiments_plan §C4 (v15 ep50 LIBERO object/goal 완료) + §C5 (v15 ep50 DROID gap 1/10/30 × 2 mode 완료). 5/13 cancel/누락된 cell 일괄 채움. v15 ckpt path = `/proj/external_group/mrg/checkpoints/two_stream_v15/20260511_045319/latest.pt` (ep50). encoder는 `two-stream-v11`로 load (`load_v11_model(..., strict=False)`로 v15 base 구조만 load, v15 specific 모듈은 probing forward에 영향 없음 — 5/13 v15 ep32 fair pair 사례 동일).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34466999 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO **object** × `p_t_p_tk` × 4 gap (1/13/20/40) | PENDING |
| 34467000 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO **goal** × `p_t_p_tk` × 4 gap | PENDING |
| 34467001 | AIP 1×1 H100 | 03:00:00 | v15 ep50 DROID `patch_mean_concat_p_t_p_tk` × gap=1 | PENDING |
| 34467002 | AIP 1×1 H100 | 03:00:00 | v15 ep50 DROID `patch_mean_concat_enc_only` × gap=1 | PENDING |
| 34467003 | AIP 1×1 H100 | 03:00:00 | v15 ep50 DROID `patch_mean_concat_p_t_p_tk` × gap=10 | PENDING |
| 34467004 | AIP 1×1 H100 | 03:00:00 | v15 ep50 DROID `patch_mean_concat_enc_only` × gap=10 | PENDING |
| 34467005 | AIP 1×1 H100 | 03:00:00 | v15 ep50 DROID `patch_mean_concat_p_t_p_tk` × gap=30 | PENDING |
| 34467006 | AIP 1×1 H100 | 03:00:00 | v15 ep50 DROID `patch_mean_concat_enc_only` × gap=30 | PENDING |

총 ~3.5 GPU·h (LIBERO 2잡 × ~17min + DROID 6잡 × ~1min). DROID gap=15 + LIBERO spatial은 이미 5/13 완료, 본 entry로 §C4/§C5 모든 cell 보충 완료.

§C6 (frame_t vs frame_tk recon quality v11 vs v15) + §C7 (VideoMAE-ours P_t+P_tk catalyst) 는 새 코드 작성 필요 (probe_action.py 단일 frame 인터페이스 부재, recon quality 별도 forward script) → 별도 dev session에서 처리.

### 2026-05-15 v15 V-from-M ablation (paper §5.1 main ablation, paper_experiments_plan §C1)

v15 본 학습(34288968)이 motion-routing `softmax(Q_M K_M^T) @ V_P` = **V from P** (M self-similarity graph로 P value re-route, ours novelty). 본 ablation은 같은 hyperparameter 하에 **V from M** (standard cross-attention, Q from P, K/V from M) 비교. paper §5.1 main ablation. 5/14 paper_experiments_plan §C1 사용자 결정.

**fair pair 조건 (v15 본 학습과 동일)**: 50ep, EgoDex part1-5, batch 32/GPU global 256, num_workers=8, λ_pred=λ_m_jepa=λ_compose=1.0 with warmup 10ep 0.01→1.0, composition_mode=linear_residual, EMA 0.999→0.9999, mask_p=0.75, mask_m_jepa=0.5, max_gap=30, sample_center=15. 단일 변인: `V11_ROUTING_MODE=v_from_m`.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34464714 | AIP 1×1 H100 | 02:00:00 | v15-vfromm sanity (50vid × 3ep × num_workers=4, MAX_VIDEOS/EPOCHS/SUFFIX/V11_ROUTING_MODE override) | ✅ COMPLETED 2026-05-15 15:32:32 (5m02s). 4 loss path 정상 통과 → afterok 트리거 |
| 34464715 | AIP_long 2×4 H100 | 10-00:00:00 | **v15-vfromm 본 학습** (50ep × part1-5, V11_ROUTING_MODE=v_from_m, CHECKPOINT_SUFFIX=vfromm). dependency=afterok:34464714. 예상 ~43h. ckpt → `/proj/external_group/mrg/checkpoints/two_stream_v15_vfromm/20260515_153323/checkpoint_epoch00*.pt` (13 epoch + best/latest) | ✅ COMPLETED 2026-05-17 20:05:50, Elapsed **2-04:33:15 = 52.55h × 8 GPU = 420.4 GPU·h**. 학습 자체 NaN/crash 없음. eval_loss history.json NaN은 5ep 간격 측정으로 빈칸이 NaN으로 채워진 거짓 알람. **단, ep45-50 train loss 0.37→0.77 + eval 0.43→0.92 단조 급등 → late-stage divergence**. ep3 best_model.pt는 underfit (target encoder 초기에 너무 쉬움) → 사용 금지. plateau 구간(ep30-44) ckpt 중 fair pair 후보 확정 필요 |

### 2026-05-18 v15-vfromm 진단 — plateau ckpt sweep (paper §5.1 C1 fair pair ep 확정)

v15-vfromm 학습 후 trajectory 분석: ep3 best/latest 미사용 + ep48 이상 발산 영역 → plateau 안정 구간(ep30-44)에서 paper main과 fair pair용 ckpt 확정 필요. v15 main이 ep32 champion이었으므로 동일 ep 기본 후보. `probe_action_v11.py` + `--cls-mode patch_mean_concat_p_t_p_tk` (v15 main과 동일 mode) × 5 ckpt sweep.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34579946 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep28** EgoDex probing (`patch_mean_concat_p_t_p_tk`, gap=10, test split) | ✅ COMPLETED 14m25s. **R²=+0.4131 / Cos 0.309** |
| 34579948 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep32** EgoDex probing (동일 — v15 main champion ep과 직접 비교) | ✅ COMPLETED 14m25s. **R²=+0.3804 / Cos 0.295** (v15 main +0.3898 대비 −0.010, fair pair 동률) |
| 34579949 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep36** EgoDex probing (동일) | ✅ COMPLETED 14m25s. **R²=+0.4016 / Cos 0.297** |
| 34579950 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep40** EgoDex probing (동일 — plateau 최저 loss ep) | ✅ COMPLETED 14m25s. **R²=+0.4047 / Cos 0.301** |
| 34579951 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep44** EgoDex probing (동일 — plateau 끝, 발산 직전) | ✅ COMPLETED 14m25s. **🏆 R²=+0.4145 / Cos 0.301** (vfromm 자체 champion ep) |

**5 ckpt 종합**: EgoDex within-domain에서 vfromm R²=0.38~0.41 안정, v15 main(+0.390)과 ep32 fair pair 사실상 동률. **paper claim은 cross-embodiment generalizability이므로 within-domain EgoDex는 main evidence 부적합** → cross-domain (LIBERO/DROID) probing이 fair test. EgoDex 5잡 cost: 5 × 14.4분 = **1.2 GPU·h**.

### 2026-05-18 v15-vfromm cross-embodiment probing (LIBERO spatial + DROID, paper main fair test)

EgoDex within-domain 비교는 paper claim(action-agnostic representation 범용성)에 부합 X. **Cross-embodiment (human → robot arm)** 평가가 fair: LIBERO simulation arm + DROID real arm. v15 main ep32 baseline: LIBERO spatial p_t_p_tk gap=20 **+0.584** ★ (34367612), DROID p_t_p_tk gap=15 **−0.006** (34367577).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34592050 | AIP 1×1 H100 | 01:30:00 | v15-vfromm **ep28** LIBERO spatial probing (`v11-mode=p_t_p_tk`, 4 gaps 1/13/20/40) | ✅ COMPLETED 17m01s. gap1 **+0.437**, gap13 **+0.603**, gap20 **+0.600**, gap40 **+0.364** (vfromm **LIBERO champion** ★) |
| 34592051 | AIP 1×1 H100 | 01:30:00 | v15-vfromm **ep32** LIBERO spatial probing (동일 — v15 main 직접 비교) | ✅ COMPLETED 17m01s. gap1 +0.383, gap13 +0.563, gap20 **+0.565** (v15 main +0.584 대비 −0.019, fair pair 동률), gap40 +0.329 |
| 34592052 | AIP 1×1 H100 | 01:30:00 | v15-vfromm **ep36** LIBERO spatial probing (동일) | ✅ COMPLETED 16m48s. gap1 +0.378, gap13 +0.564, gap20 +0.569, gap40 +0.337 |
| 34592053 | AIP 1×1 H100 | 01:30:00 | v15-vfromm **ep40** LIBERO spatial probing (동일) | ✅ COMPLETED 16m48s. gap1 +0.368, gap13 +0.549, gap20 +0.569, gap40 +0.338 |
| 34592054 | AIP 1×1 H100 | 01:30:00 | v15-vfromm **ep44** LIBERO spatial probing (동일) | ✅ COMPLETED 16m52s. gap1 +0.387, gap13 +0.564, gap20 +0.576, gap40 +0.331 |
| 34592055 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep28** DROID probing (`patch_mean_concat_p_t_p_tk`, gap=15, max_episodes=200) | ✅ COMPLETED 1m11s. **R²=−0.014 / Cos 0.020** |
| 34592057 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep32** DROID probing (동일 — v15 main 직접 비교) | ✅ COMPLETED 1m45s. **R²=−0.016** (v15 main −0.006 대비 −0.010, noise level 동률) |
| 34592058 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep36** DROID probing (동일) | ✅ COMPLETED 1m45s. **R²=−0.005** |
| 34592059 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep40** DROID probing (동일) | ✅ COMPLETED 1m45s. **R²=−0.003** (vfromm DROID champion, 그래도 noise) |
| 34592060 | AIP 1×1 H100 | 03:00:00 | v15-vfromm **ep44** DROID probing (동일) | ✅ COMPLETED 1m45s. **R²=−0.009** |

10잡 cost: LIBERO 5 × 16.9분 + DROID 5 × 1.4분 = **1.6 GPU·h**.

**핵심 결론 (paper §5.1 C1 ablation)**:
1. **fair pair (ep32 동일): LIBERO gap20 vfromm +0.565 vs main +0.584 = −0.019** (probing 변동성 범위). DROID, EgoDex 모두 동률.
2. **routing 차이 negligible across 3 domains (EgoDex / LIBERO / DROID)** → V-from-P (ours) "우위" 강한 claim 불가. 사용자 확정 framing (5/18): "**Motion-routing이 핵심, value source(P vs M)는 부차적**" — robustness evidence로 활용.
3. **부가 관찰**: vfromm는 ep28이 LIBERO champion (+0.600), 학습 진행될수록 cross-embodiment 소폭 하락. main(ep50 사용)과 vs vfromm 비교는 같은 ep32 단위가 fair.

### 2026-05-18 v15-vfromm LIBERO BC-T fine-tuning (paper §5.1 C1 main matrix)

v15-vfromm ep32 (fair pair ep) × v15 main 동일 매트릭스로 BC-T fine-tuning. paper §5.1 C1 ablation의 정량 evidence — closed-loop success rate에서도 routing 차이 negligible 확인 또는 vfromm 약점 발견. v15 main BC-T (ep50 ckpt 사용)와 SR 비교.

**매트릭스 (v15 main 18잡과 동일)**: vfromm ep32 × {`two-stream-v15-ptptk` (Option B P_t+P_tk), `two-stream-v15-mp` (C-variant M+P)} × 3 suite × 3 seed = **18잡**. V3 cfg (use_joint=True + augmentation). epochs=50, batch=32, seq_len=10, lr=1e-4.

| JobID 범위 | 자원 | --time | 목적 | 결과 |
|-----------|------|--------|------|------|
| 34595206~227 (18잡) | AIP 1×1 H100 each | 2-00:00:00 | vfromm ep32 LIBERO BC-T (ptptk × 9 + mp × 9). 출력 → `/proj/external_group/mrg/checkpoints/libero_bct/two-stream-v15-{ptptk,mp}_{suite}_seed{0,1,2}_*_vfromm_ep32/` | 2026-05-19 진행: 3잡 (`mp_spatial` × seed 0/1/2) COMPLETED 22h02m~22h18m. **15잡 RUNNING (22h 19m 경과, 1-2h 내 순차 완료 예상)**. 총 cost ≈ 500 GPU·h |

종료 후 → 로컬 워크스테이션 전송 + `bash scripts/local/run_libero_rollouts.sh two-stream-v15-{ptptk,mp} 50` rollout → `paper_artifacts/libero_rollout/` SR 등록.

### 2026-05-18 C12 LIBERO view-sensitivity probing (eye_in_hand, 3 enc 축소)

paper §5 ¶6 sub-analysis (Tab 7 appendix). 비교군: **v15-ptptk + LIBERO BC SR 상위 2 baseline** (siglip 0.855, vc1 0.821) — 사용자 결정 5/18. probing only, BC 제외. probe_action_libero.py에 `--view eye_in_hand_rgb` flag 이미 구현되어 있어 코드 변경 불필요.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34599785 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO spatial **eye_in_hand** probing (`p_t_p_tk`, 4 gaps) | ✅ COMPLETED 16m38s. gap1 +0.777, gap13 +0.764, gap20 **+0.735**, gap40 +0.595 |
| 34599786 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO object eye_in_hand probing (동일) | ✅ COMPLETED 19m59s. gap1 +0.727, gap13 +0.748, gap20 **+0.710**, gap40 +0.650 |
| 34599788 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO goal eye_in_hand probing (동일) | ✅ COMPLETED 16m59s. gap1 +0.733, gap13 +0.801, gap20 **+0.784**, gap40 +0.765 |
| 34599613 | AIP 1×1 H100 | 01:30:00 | siglip LIBERO spatial eye_in_hand probing | ✅ COMPLETED 16m12s. gap1 +0.818, gap13 +0.802, gap20 **+0.775**, gap40 +0.653 |
| 34599617 | AIP 1×1 H100 | 01:30:00 | siglip LIBERO object eye_in_hand probing | ✅ COMPLETED 19m04s. gap1 +0.760, gap13 +0.799, gap20 **+0.772**, gap40 +0.640 |
| 34599621 | AIP 1×1 H100 | 01:30:00 | siglip LIBERO goal eye_in_hand probing | ✅ COMPLETED 15m57s. gap1 +0.778, gap13 +0.794, gap20 **+0.794**, gap40 +0.781 |
| 34599614 | AIP 1×1 H100 | 01:30:00 | vc1 LIBERO spatial eye_in_hand probing | ✅ COMPLETED 14m28s. gap1 +0.804, gap13 +0.800, gap20 **+0.769**, gap40 +0.664 |
| 34599618 | AIP 1×1 H100 | 01:30:00 | vc1 LIBERO object eye_in_hand probing | ✅ COMPLETED 17m44s. gap1 +0.751, gap13 +0.797, gap20 **+0.764**, gap40 +0.673 |
| 34599622 | AIP 1×1 H100 | 01:30:00 | vc1 LIBERO goal eye_in_hand probing | ✅ COMPLETED 14m40s. gap1 +0.748, gap13 +0.792, gap20 **+0.795**, gap40 +0.781 |

9잡 COMPLETED, 실 cost = **2.5 GPU·h**. **최초 제출 시 v15 ckpt 경로 오타 (`checkpoint_epoch0050.pt`)로 34599612/616/619 3잡 즉시 cancel + `latest.pt` 재제출 (34599785~788). 비용 무시 가능 (PENDING 단계에서 cancel).**

### 2026-05-19 C12 agentview baseline 보강 (Δ 계산용)

C12 eye_in_hand 결과 확보 후 비교 metric Δ(eih − av) 산출이 필요. paper_artifacts에 v15ep50 agentview는 object/goal만 있고 spatial + siglip 전체 + vc1 전체 부재 → 7잡 추가 제출.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34619063 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO spatial **agentview** probing (`p_t_p_tk`, 4 gaps) | PENDING |
| 34619064 | AIP 1×1 H100 | 01:30:00 | siglip LIBERO spatial agentview probing | PENDING |
| 34619065 | AIP 1×1 H100 | 01:30:00 | vc1 LIBERO spatial agentview probing | PENDING |
| 34619066 | AIP 1×1 H100 | 01:30:00 | siglip LIBERO object agentview probing | PENDING |
| 34619067 | AIP 1×1 H100 | 01:30:00 | vc1 LIBERO object agentview probing | PENDING |
| 34619068 | AIP 1×1 H100 | 01:30:00 | siglip LIBERO goal agentview probing | PENDING |
| 34619069 | AIP 1×1 H100 | 01:30:00 | vc1 LIBERO goal agentview probing | PENDING |

v15ep50 object/goal agentview는 기존 `paper_artifacts/libero_action_probing/two-stream-v11_libero_{object,goal}_20260515_161112_v15ep50/` 재활용. 예상 cost: 7 × 16.9분 = **1.97 GPU·h**.

### 2026-05-19 C12 framing 재설계 — av+eih combined view 9잡 (paper §5.1 main)

사용자 비판 (5/19): eye_in_hand 단독은 unrealistic robot setting, single-view 비교는 의미 해석 제한적. **실제 paper main framing = "agentview에 wrist view를 추가했을 때 우리 모델이 가장 큰 격차"** — monotonic, practical. `probe_action_libero.py:251`에 `--view both` 옵션 추가 (av encode ⊕ eih encode → 2× embed_dim concat → linear probe). 기존 eye_in_hand 단독 9잡 = sunk cost (supplementary).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34619135 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO spatial **av+eih combined** probing (`p_t_p_tk`, 4 gaps, view=both) | ✅ COMPLETED 31m57s. gap1 +0.779, gap13 +0.779, gap20 **+0.755**, gap40 +0.625 |
| 34619136 | AIP 1×1 H100 | 01:30:00 | siglip LIBERO spatial av+eih combined | ✅ COMPLETED 30m09s. gap20 +0.767 |
| 34619137 | AIP 1×1 H100 | 01:30:00 | vc1 LIBERO spatial av+eih combined | ✅ COMPLETED 28m15s. gap20 +0.776 |
| 34619138 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO object av+eih combined | ✅ COMPLETED 39m59s. gap20 +0.756 |
| 34619139 | AIP 1×1 H100 | 01:30:00 | siglip LIBERO object av+eih combined | ✅ COMPLETED 37m08s. gap20 +0.782 |
| 34619140 | AIP 1×1 H100 | 01:30:00 | vc1 LIBERO object av+eih combined | ✅ COMPLETED 34m31s. gap20 +0.783 |
| 34619141 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO goal av+eih combined | ✅ COMPLETED 32m57s. gap20 +0.804 |
| 34619142 | AIP 1×1 H100 | 01:30:00 | siglip LIBERO goal av+eih combined | ✅ COMPLETED 30m43s. gap20 +0.801 |
| 34619143 | AIP 1×1 H100 | 01:30:00 | vc1 LIBERO goal av+eih combined | ✅ COMPLETED 28m43s. gap20 +0.806 |

9잡 모두 COMPLETED, 실 cost = **5.0 GPU·h** (each ~17~40분, 2 view encode로 av_only 대비 ~2배).

**Δ 분석 결과** → [paper_artifacts/tables/tab7_view_sensitivity/README.md](../paper_artifacts/tables/tab7_view_sensitivity/README.md):
- Encoder Δ avg overall: **vc1 +0.221 ≈ v15 +0.211 ≫ siglip +0.114**
- v15는 **goal suite 1위** (Δ +0.238), spatial/object는 vc1 1위
- "motion-routing unique advantage" 강한 claim 불가, "action-relevant pretrain > VL-SSL" framing 가능

### 2026-05-21 C10 CALVIN unzip + loader 검증 + validation 5잡

CALVIN 다운로드 완료 (2026-05-19 22:19, 704 GB) 후 5/20-5/21 압축 해제 + loader 검증.

**unzip 잡 진행**:
- 34653934 long_cpu PENDING 4h (35h 예약 됨) → cancel
- 34666478 core_l RUNNING → FAILED 1m (compute node에 `unzip` 명령 부재). conda env에 unzip 설치 + sbatch script 수정 후 재제출
- **34673903 core_l × 8 CPU × 12h** ✅ COMPLETED elapsed **4h 40m** (commit `bd4c675`). 2,406,171 파일 추출 (99.9%). `task_ABCD_D/{training/178ep, validation/4ep}/episode_XXXXXXX.npz` 구조 확인 — `src/datasets/calvin.py` loader 가정 일치 ✅

**Episode 길이 진단**:
- training: 178 episodes 평균 12,961 frames (max 44,087)
- validation: 4 episodes 평균 24,756 frames (max 37,683)
- 전체 frame을 stack 시 episode 1개 = 4.5 GB RGB uint8 → 메모리 폭증 발견 (sanity 잡 36 GB RSS stuck)

**loader 수정 (commit `21f907d`)**: `--frame-stride 10` default 추가 (30Hz → effective 3Hz), gaps default = [1, 3, 5, 10] (stride space, gap=3 = 1.0s key).

| JobID | 자원 | 결과 |
|-------|------|------|
| 34674929 (sanity v1) | AIP 1 GPU | CANCELLED — 17m stuck (메모리 36 GB, stride 없음) |
| 34676122 (sanity v2 stride=10) | AIP 1 GPU | ✅ COMPLETED 6m58s. validation 4ep, gap=1/3/5/10. **v15 main ep50 R²: gap=3 +0.239 / gap=10 +0.399** — loader OK |

### 2026-05-21~22 C10 본 매트릭스 (5 enc × 2 splits = 10잡)

| JobID 범위 | 자원 | 결과 |
|-----------|------|------|
| 34676187/189/191/193/195 (validation 5잡) | AIP 1×1 H100 each | ✅ COMPLETED 3~5분/잡, total **0.4 GPU·h**. paper_artifacts/calvin_action_probing/*_validation_*/ |
| 34676186/188/190/192/194 (training 5잡, time=2h) | AIP 1×1 H100 each | ❌ **TIMEOUT @ 2h** (gap1만 partial). training=178 ep 6h+ 필요 |
| **34721946~998 (training 재제출, time=8h)** | AIP 1×1 H100 each | RUNNING 시작 2026-05-22 11:39, 4-6h 예상. total ~25 GPU·h |

**Validation 결과 (CALVIN D split OOD, 4 ep, gap=3 = 1.0s @ effective 3 Hz)**:
| Encoder | R² gap=3 | R² gap=10 | Rank |
|---------|----------|-----------|------|
| **dinov2** | **+0.469** ★ | **+0.588** | 1위 |
| **siglip** | **+0.411** | +0.535 | 2위 |
| videomae-ours | +0.246 | +0.352 | 3위 |
| v15 (ep50, two-stream-v11 mode) | +0.239 | +0.399 | 4위 |
| vc1 | +0.210 | +0.369 | 5위 |

**Paper §4 ¶2 (iii) narrative 영향** (비판적):
- CALVIN cleaner setting (cf. DROID 모두 R²≈0)에서 **v15가 single-frame SSL baseline 대비 명확히 낮음**
- "action-agnostic 단점" evidence 가능성 — cross-embodiment gap (EgoDex human hand → CALVIN tabletop arm)에서 우리 방식이 일반 SSL보다 약함
- training (in-distribution) 결과 확보 후 OOD vs in-distribution 격차 확인 필요. paper framing 신중: e.g., "v15 advantage is on natural human videos; for specialized tabletop robotics generic vision SSL retains an edge."

### 2026-05-22 C10 training 재측정 (cumulative delta → pose-derived target)

target metric 변경: rel_actions cumulative sum → robot_obs[:6] pose delta (LIBERO/EgoDex 일관). commit `e195021`/`b90ce37`. 34755884~899로 10잡 재측정. validation 5잡 6m, training 5잡 4h41m.

**Pose-derived training 결과 @ gap=3 (1s, in-folder 80:20 split)**: dinov2 +0.399 > siglip +0.318 > videomae-ours +0.306 > vc1 +0.265 > **v15 +0.187** (5위 유지). 모든 encoder R² 감소 (pose 단위 더 엄격), 단 rank 패턴 동일.

### 2026-05-23~26 C10 setting 진화 (사용자 통찰 기반 단계적 정제)

사용자 비판 단계:
1. (5/23) target 정확 = pose-derived (LIBERO와 일관). ✅ 적용
2. (5/26) stride=10 sampling은 다른 dataset과 protocol 불일치, 의미 약화. random pair에 task boundary/idle pair 35% 포함 → unfair
3. (5/26) **segment-based sampling**: lang annotations로 같은 task 안 frame pair만 (commit `21f907d` 후속). 잡 34782482/483, 34796897~908
4. (5/26) sub-folder 80:20 self-contained split도 paper §C10의 ABC→D OOD 의도 아님. **cross-folder OOD**: training/ probe 학습 + validation/ R² 평가가 진짜 generalization test

| JobID 범위 | 자원 | 결과 |
|-----------|------|------|
| 34782482/483 (v15만 sanity, segment-based self-contained) | AIP 1×1 H100 | ✅ COMPLETED. validation +0.310 (random pair +0.072 대비 4배 회복) |
| 34796897~908 (5 enc × 2 split self-contained, time=2h training fail) | AIP | training 5잡 cancel — sub-folder self-contained 의미 약함 |
| **34798739~744 (5 enc × cross-folder OOD)** | AIP 1×1 H100 × ~2h | ✅ COMPLETED 2h07m each, total **10.6 GPU·h** |

**Cross-folder OOD (gap=30 = 1.0s) — ⚠️ episode-based sampling, deprecated**:
| Encoder | R² aggregate | 비고 |
|---------|------------|------|
| dinov2 | +0.535 | gripper R² dominated (per-dim 미보고) |
| siglip | +0.345 | |
| vc1 | +0.242 | |
| videomae-ours | +0.180 | |
| v15 | +0.105 | |

**🔴 이 결과는 paper §C10 main 아님**: 본 잡 시점(2026-05-26 14시)은 `b90ce37` commit (episode-based + stride=10, idle/transition pair 35% 포함). 같은 날 22시 `7eb0c48` commit(=segment-based fair)에서 GAP sweep 재측정. paper §C10 main은 5/26 22시 결과(아래 entry) 사용. n_train_pairs 비교: 본 잡 59,904 (episode) vs 22시 잡 6,114 (segment, ~10×차이).

### 2026-05-26 CALVIN cross-folder 원인 진단 (Case 1 per-task + GAP sweep)

**배경**: §C10 cross-folder OOD에서 v15가 다른 벤치 (EgoDex/LIBERO/CortexBench MetaWorld 모두 1위)와 정반대로 5위. 이유 규명을 위한 진단 series.

**Case 1 (완료) — per-task R² breakdown**:
| JobID | 자원 | 결과 |
|-------|------|------|
| 34815090 (`diagnose_calvin_per_task.py`, 5 enc × 1 gap=30) | AIP 1×1 H100 × 2h49m = **2.81 GPU·h** | ✅ COMPLETED. v15가 34 task 중 33개에서 dinov2에 lag (median Δ=−0.52). worst 8개 중 5개가 `rotate_*_block_*` — rotation/fine-motion task에 집중. **단 v15만 약함이 아님**: siglip/vc1/videomae도 rotate task에 약함, dinov2만 일관 우위. → "motion/temporal SSL의 CALVIN-specific 약점" 가설로 재정의 |

**Case 2 (완료, 18:11) — motion magnitude 분포**: CALVIN bi-modal (peak 0.03-0.05m + 0.1m), LIBERO uni-modal (0.18-0.22m). CALVIN idle pair 2배, 작은 motion 2.4배. **단 baseline에도 동등 영향이라 v15-specific 원인은 설명 못 함**.

**Case 3 (완료) — GAP sweep + per-dim 분석 = paper §C10 main fair**:

코드 점검 발견 (commit `7eb0c48` segment-based 적용 후 재측정):
- [src/datasets/calvin.py:17](../src/datasets/calvin.py#L17) segment 길이 **min 34, max 65 frame**. eval 32183 pairs / 1087 segs ≈ 30 pairs/seg → 평균 segment ≈ 60 frame
- gap=30 (1.0s) → P_tk가 segment의 50-88% 지점 = **task 종료 직전 frame cluster**

| JobID | 자원 | 결과 |
|-------|------|------|
| 34869586~590 (5 enc × cross-folder × gap 10/15/20/30, segment-based) | AIP 1×1 H100 × 27분 each, total **2.3 GPU·h** | ✅ COMPLETED 22:04 |

**R² aggregate**:
| Encoder | gap=10 | gap=15 | gap=20 | gap=30 |
|---------|-------|-------|-------|-------|
| dinov2 | +0.508 | +0.471 | +0.412 | +0.307 |
| siglip | +0.293 | +0.272 | +0.212 | +0.162 |
| vc1 | +0.093 | +0.072 | +0.060 | +0.035 |
| videomae-ours | +0.077 | +0.069 | +0.072 | +0.056 |
| **v15** | **+0.059** | +0.045 | +0.032 | **−0.012** |

**Per-dim R² @ gap=30** (사용자 직관 검증 — gripper가 aggregate dominate):
| Encoder | pos_x | pos_y | pos_z | rot_x | rot_y | rot_z | **gripper** | agg |
|---------|------|------|------|------|------|------|------|------|
| **v15** | +0.355 | +0.199 | +0.234 | +0.050 | −0.001 | −0.063 | **−0.005** | −0.012 |
| videomae-ours | **+0.617** ★ | +0.510 | +0.533 | +0.049 | +0.421 | −0.015 | +0.059 | +0.056 |
| vc1 | +0.544 | +0.453 | +0.610 | +0.158 | +0.434 | +0.016 | +0.022 | +0.035 |
| dinov2 | +0.413 | +0.169 | +0.088 | +0.110 | +0.271 | +0.139 | **+0.359** ★ | +0.307 |
| siglip | +0.062 | −0.071 | −0.934 | −0.218 | −0.005 | +0.136 | +0.183 | +0.162 |

**🔥 핵심 발견 (paper §C10 narrative 결정)**:

1. **dinov2 aggregate 우위는 motion 인코딩 아니라 binary gripper R² (+0.359)에서 옴**. v15/videomae는 pos delta에서 dinov2 추월(pos avg v15 +0.262 / videomae **+0.553** / dinov2 +0.223)
2. **LIBERO에서도 같은 패턴** (gap=20 per-dim): v15 pos avg **+0.896 1위**, dinov2 +0.766, gripper만 dinov2 우위. 단 LIBERO는 pos delta scale 큼 → aggregate에서 격차 작음 (v15 +0.581 vs dinov2 +0.666, Δ=−0.085)
3. **EgoDex (gripper 없음, 18-dim joint pose)**: v15 +0.390 1위 — gripper bias 없는 motion-only metric으로 v15 main claim 직접 입증
4. **CALVIN-specific 격차 큰 이유**: (a) pos delta scale 매우 작음 (fine manipulation) → pos R² 절댓값 모두 낮아 gripper dim이 aggregate 더 dominate. (b) gripper R² 격차가 LIBERO 대비 4배 (−0.36 vs −0.09)
5. **H2 (segment-끝 cluster) 가설 부분 입증**: v15 R² drop gap=10→gap=30 −120%, dinov2 −40%, videomae −27%. v15가 gap 증가에 가장 민감. 단 base 격차(gap=10에서도 −0.45)는 gripper에서 옴

**paper §C10 narrative 재정의**: "v15가 CALVIN OOD에서 약함"(오해) → **"R² aggregate는 binary gripper에 dominated metric. v15는 continuous motion (pos delta)에서 image-SSL과 동급/우위, gripper binary 식별에서 약함. 모든 벤치(EgoDex/LIBERO/CALVIN)에서 per-dim 분석 시 v15가 motion 인코딩 1위 또는 공동 1위로 일관"**

산출물:
- [paper_artifacts/calvin_action_probing/_diagnostic/per_dim_r2.csv](../paper_artifacts/calvin_action_probing/_diagnostic/per_dim_r2.csv) — CALVIN + LIBERO per-dim 통합 table
- [paper_artifacts/calvin_action_probing/_diagnostic/per_dim_r2.png](../paper_artifacts/calvin_action_probing/_diagnostic/per_dim_r2.png) — bar chart (2 panel)
- [paper_artifacts/calvin_action_probing/_diagnostic/aggregate_per_dim.py](../paper_artifacts/calvin_action_probing/_diagnostic/aggregate_per_dim.py) — regeneration script

### 2026-05-27 PCA overlay 가시화 prototype (representation_visualization_plan #3)

Paper accept 후 project page용 가시화 자료 1단계 — DINOv2-style RGB overlay (3 encoder 비교). LIBERO demo HDF5로 prototype 검증 후 로컬에서 rollout video 적용 예정.

| JobID | 자원 | 결과 |
|-------|------|------|
| 34990706 (1차) | AIP 1×1 H100 | ❌ FAILED 14s — `ModuleNotFoundError: sklearn`. → numpy SVD로 PCA 구현 변경 (sklearn 의존성 제거) |
| 34990711 (alpha=0.55) | AIP 1×1 H100 × 57s | ✅ COMPLETED. 3 encoder × 98 frame patch feature 추출 + PCA(numpy SVD top-3) → RGB overlay. side_by_side.gif (9 MB) + static_grid.png. alpha=0.55가 다소 강해 원본 묻힘 |
| **34990721 (alpha=0.4 + explained variance)** | AIP 1×1 H100 × 36s = **0.01 GPU·h** | ✅ COMPLETED. 원본 detail 보존 + PC1-3 % 라벨 표시. **VideoMAE PC1=24%** vs v15/dinov2 PC1=19% → VideoMAE 단조 패턴이 PCA artifact 아닌 encoder 자체 특성 정량 확정 |

**PCA explained variance**:
- v15: 19/12/7 % (cum 37.5)
- dinov2: 19/11/7 % (cum 37.3)
- **videomae-ours: 24/10/8 % (cum 42.0)** — PC1 dominance = paper "no scaffolding baseline" 약점 정량 evidence

**검증 통과**:
- frame 간 색 일관성 OK (전체 video patches로 한 번만 PCA fit)
- v15 ↔ DINOv2 ↔ VideoMAE 명확히 다른 patch clustering
- v15: 그리퍼/물체/배경 spatial separation
- DINOv2: 강한 instance discrimination contrast
- VideoMAE: 단조 (PC1 dominate)

산출물: [paper_artifacts/visualizations/pca_overlay/libero_spatial_task0_demo0/](../paper_artifacts/visualizations/pca_overlay/libero_spatial_task0_demo0/)

### 2026-05-27 Grad-CAM arrow viz + axis-alignment 분석 (representation_visualization_plan #1)

Paper accept 후 project page용 두 번째 가시화 prototype. concat probe (P_t patches mean ⊕ P_tk patches mean → 7-DoF) 위 Grad-CAM contribution 계산 + motion arrow overlay + ground-truth motion 정합 검증 + probe weight 직접 분석.

| 잡 범위 | 자원 | 시간 |
|---------|------|------|
| 34990729 (1차, single-demo R²=0.318) | AIP 1×1 H100 | 12s |
| 34990805 (2차, **multi-demo 47k pair, R²=0.685**) | AIP 1×1 H100 | 2m22s |
| 34990886~34997017 (v3~v9 iteration: V 부호 fix → tail pivot → paired t/tk layout → red/cyan sum + pixel scale → GT overlay → V flip → axis-alignment) | AIP 1×1 H100 each, ~2m | 각 ~2-3분 |
| **34997500~502 (v15/dinov2/videomae-ours 3 encoder × 70 frame pair full analysis)** | AIP 1×1 H100 each | 각 ~3분 |

총 잡: ~12개. 약 **0.5 GPU·h**.

**Iteration 핵심 (사용자 review 기반)**:
1. multi-demo probe (47k pair) → arrow magnitude 4배 + frame별 다른 패턴
2. paired t/tk panel (frame_t arrow | frame_tk arrow) — frame transition 보임
3. sum vector pixel-scale (red P_t / cyan P_tk, 상대 magnitude 유지)
4. V 부호 flip (사용자 시각: 그리퍼 down=image up)
5. Ground-truth motion overlay (green dashed) — 좌표축 정합 직접 검증 도구
6. Per-axis sign-match rate (전체 70 frame pair) — 사용자 관찰 "P_t는 vertical, P_tk는 horizontal" 검증
7. Probe weight 직접 분석 (cos(W_t, −W_tk)) — implicit subtraction 가설 직접 증명

**🔥 핵심 발견 (paper-relevant insight)**:

**A. Linear probe가 implicit (P_tk − P_t) subtraction 학습** — 3 encoder 모두 공통:
| Encoder | cos(W_Δx) | cos(W_Δy) | cos(W_Δz) | cos(W_Δrx) | cos(W_Δrz) | grip |
|---------|-----------|-----------|-----------|------------|------------|------|
| v15 | +0.61 | +0.67 | **+0.79** | +0.59 | +0.75 | −0.10 |
| DINOv2 | +0.60 | +0.61 | +0.76 | +0.62 | +0.73 | +0.15 |
| VideoMAE | +0.60 | +0.68 | +0.78 | +0.61 | +0.76 | +0.02 |

→ 모든 encoder에서 W_t와 −W_tk가 정합 (cos +0.6~0.8 for motion dims). **motion = frame embedding difference로 추론 가능한 metric-aligned representation** = ViT general property.

**B. 그러나 patch contribution sum 분배는 encoder-specific** (sign-match rate, |GT|≥10cm):
| Encoder | P_t u | P_t v | P_tk u | P_tk v |
|---------|-------|-------|--------|--------|
| v15 | 0.37 | **1.00** | 0.63 | **0.00** (정반대) |
| DINOv2 | **1.00** | **1.00** | 0.63 | 0.03 |
| VideoMAE | 0.63 | 0.71 | 0.37 | 0.97 (P_t와 같이) |

→ DINOv2: P_t single frame이 가장 informative (당연 — single frame instance discrimination 강함). v15: axis-specific role (P_t vertical, P_tk horizontal). VideoMAE: subtraction 약함.

**paper narrative 위치**: viz는 **mechanism interpretability** (어떻게 motion 정보 인코딩하나)에 들어감. v15의 quantitative advantage는 main BC/probing table (LIBERO/CortexBench/EgoDex)에서 보임.

산출물: [paper_artifacts/visualizations/grad_cam_arrow/](../paper_artifacts/visualizations/grad_cam_arrow/) — v1~v9 + 3 encoder 비교 디렉토리. 각자 `arrow_pair_*.png` + `axis_alignment.csv` + `probe_weight_summary.csv` + `static_arrow_grid.png`.

### 2026-05-27 viz prototype 확장 — 5 encoder × 4 dataset (PCA) + Grad-CAM align 검증

**1. PCA overlay 5 encoder × 4 dataset**: `scripts/viz/pca_overlay.py` v3 (단일 파일 update 정책):
- 5 encoder (ours / dinov2 / siglip / vc1 / videomae-ours) — VC-1는 ViT forward 재현 (forward_features는 CLS만)
- 4 dataset source (libero HDF5 / egodex frames / droid frames / calvin npz)
- "v15" → "ours" 라벨 변경

| Job | Source | OUT_TAG | Elapsed |
|-----|--------|---------|---------|
| 35003236 | LIBERO spatial task_0 demo_0 | libero_spatial_task0_demo0_5enc | ~1m |
| 35003237 | EgoDex part1/add_remove_lid/0 | egodex_add_remove_lid_0 | ~1m |
| 35003238 | DROID ext1/ep_000000 | droid_ep000000 | ~1m |
| 35003239 | CALVIN validation seg 37683-37803 | calvin_val_37683 | ~1m |

**PC1 explained variance (단조성 지표, 높을수록 정보 한 축에 쏠림)**:
| Dataset | ours | dinov2 | siglip | **vc1** | videomae |
|---------|------|--------|--------|---------|----------|
| LIBERO  | 19% | 19% | 9% | **39%** | 24% |
| EgoDex  | 19% | 18% | 13% | **47%** | 14% |
| DROID   | **30%** | 17% | 9% | **52%** | 38% |
| CALVIN  | 23% | 15% | 11% | **39%** | 22% |

→ VC-1이 모든 dataset에서 PC1 dominate (39-52%) = ImageNet single-frame discriminative. DROID는 image content가 가장 1차원적으로 인코딩됨.

**2. Grad-CAM yellow vs red 정합 검증** (사용자 우려: yellow patches 평균이 red sum과 반대로 보임):
- Job 35003240 → `grad_cam_align_check/`
- 모든 10 케이스 (5 frame pair × 2 source): **yellow_kept_mean vs red_sum cos = +0.996 ~ +1.000**
- `all_mean vs sum cos = +1.000` (수학적 trivial 정합)
- → **bug 아님, 시각적 illusion** (분산된 작은 화살표에서 mean direction 가시화 한계)

총 5잡 × ~1분 = **~0.1 GPU·h**.

**3. Cleanup (사용자 정책: 최신 버전만 commit/push)**:
- 삭제: grad_cam_arrow v1~v9 iteration 디렉토리, pca_overlay 첫 3-encoder prototype
- 유지: pca v3 (5 encoder × 4 dataset), grad_cam v15/dinov2/videomae 3 encoder + align_check

### 2026-05-27 v15 architecture fix — V-JEPA P anchor를 student P encoder로 (catalyst 경로 복원)

**배경**: viz 작업 중 사용자 질문 "P stream이 motion gradient 안 받는데 왜 P_t+P_tk가 motion 예측?" → 코드 점검에서 발견:
- 기존 v15 (`4230551`부터): V-JEPA P의 anchor가 `teacher_p(frame_t).detach()` → **student P encoder가 V-JEPA gradient 못 받음** (predictor-only). P encoder = MAE only.
- 반증 데이터: VideoMAE-ours (motion stream 0, MAE only) EgoDex P_t+P_tk **+0.470** > v15 +0.39 → 기존 catalyst 가설("M→P transfer") 코드/데이터 모두 반증.
- 사용자 확인: 원래 의도는 **student P(frame_t) anchor → teacher P(frame_tk) target** 표준 V-JEPA. P encoder가 motion routing gradient를 받아야 함.

**수정** (`src/models/two_stream_v15.py` `_vjepa_p_one_segment`): `anchor_repr_T = teacher_p(...)` → `anchor_repr_S = self._encode_p_unmasked(frame_t)` (student, grad ON). target은 teacher_p(frame_tk) 유지.
- → P encoder가 V-JEPA P gradient 수신 (catalyst 성립). MAE + V-JEPA P 동시 학습.

**collapse 위험**: student anchor + EMA teacher target은 trivial collapse 가능 (MAE가 보조 방지). sanity 필수.

| JobID | 자원 | 결과 |
|-------|------|------|
| 35023407 (`sanity_v15.sbatch`, SUFFIX=student_anchor, 50vid×3ep) | AIP 1×1 H100 × 6m | ✅ COMPLETED. **cos(pred,tgt)=1.0, L_pred=0.00055** → trivial로 보임. 단 std_p=0.479/cos_intra_p=0.635 healthy (representation collapse 아님) |

**trivial 원인 진단 (`diagnose_vjepa_p_trivial.py`, 35027249/883/884)**:

| 측정 | sanity 3ep (student) | **ep50 (기존, teacher anchor)** |
|------|---------------------|--------------------------------|
| baseline cos(teacher_t, teacher_tn) | 0.9996 (frame 구분 X) | **0.8066** (frame 구분 O) |
| M=0 ablation cos | 0.9992 | 0.6600 |
| **Δ(predictor − M=0) = M routing 기여** | +0.0001 (무의미) | **+0.3073** (큼) |
| gap별 baseline cos | 모두 0.9996 | gap[0-4] 0.855 → gap[15-19] 0.762 |

**🔥 핵심 발견 (사용자 가설 입증)**:
- sanity 3ep의 trivial(cos 0.9996)은 **P encoder 미성숙** 때문. 구조적 문제 아님.
- **ep50 학습 후 P encoder가 frame을 명확히 구분** (baseline cos 0.81), gap 클수록 cos 낮아짐 (motion이 repr에 드러남).
- **M routing이 실제 motion 정보 제공** (M=0이면 0.66, M 있으면 0.97). ΔL 누출 아님 (`v_from_p`: V=P, M은 Q,K만) — M attention routing이 진짜 작동.
- 즉 student anchor 수정 + 50ep 학습 → P encoder가 V-JEPA P gradient를 받아 motion 압력 획득 (catalyst 의도 실현 기대).

**본 학습 제출 → 즉시 취소**:
| JobID | 자원 | 결과 |
|-------|------|------|
| 35028337 (`pretrain.sbatch`, MODEL=two-stream-v15, CHECKPOINT_SUFFIX=student_anchor) | AIP_long 2×4 H100 × 2m07s | ❌ CANCELLED (사용자 결정). 8 GPU × 2m07s ≈ **0.28 GPU·h** |

본 학습은 사용자 판단으로 보류. student-anchor가 기존 teacher-anchor(predictor-only) 대비 P_t+P_tk probing 향상되는지(catalyst 가설 최종 검증)는 추후 재논의 후 제출 예정.

### 2026-05-18 데이터셋 확보 작업 (로그인 노드, 미과금)

paper §C10 + §C11 진행을 위한 데이터셋 확보. CLAUDE.md 명시대로 로그인 노드 활동이라 cluster_sessions에는 별도 cost entry 없음, artifacts.md 인덱스에만 등록.

- **CALVIN task_ABCD_D** (실제 ~700GB 압축, paper plan 추정 166GB는 부정확): `/proj/external_group/mrg/datasets/calvin/` 다운로드 진행. 2026-05-19 15:12 시점 588 GB / ~700 GB (84%), 속도 3-10 MB/s 진동. ETA ~3-6h 남음
- **CortexBench Adroit (1.6 GB) + Meta-World (8.9 GB)**: ✅ 모두 완료 (`/proj/external_group/mrg/datasets/cortexbench/*.zip`)
- **eai-vc codebase**: `external/eai-vc/` host clone 완료 (`.gitignore`로 untrack — 로컬 워크스테이션에서도 별도 clone, commit `95faa12` Docker 환경에 통합)
- **로컬 워크스테이션 전송용 tar**: 저장소 최상위 `cortexbench_local_setup_*.tar` (v15 ckpt + VideoMAE-ours ckpt + Adroit/MW zip 14.6 GB) — 2026-05-19 생성

### 2026-05-12 dinov2 + v11 BC-T LIBERO rollout (로컬 H100, cluster cost 0)

V3 BC-T main table 마지막 두 행. siglip/vc1는 commit `1d572b` 시점에 paper_artifacts/libero_rollout/summary.csv 등록 완료. 남은 dinov2 + v11 9 ckpt × 2 enc를 로컬 워크스테이션 H100 × 2 GPU 병렬로 진행.

| 활동 | 시작 | 진행 | 비고 |
|------|------|------|------|
| `dinov2` rollout (`scripts/local/run_libero_rollouts.sh dinov2 50`, master pid 2583933) | 2026-05-12 11:38 | RUNNING. ckpt: `/mnt/data/checkpoints/libero_bct/dinov2_libero_{spatial,object,goal}_seed{0,1,2}_*_v3/` 9개. 1차 batch `goal seed0/1` 진행 중 (seed0 5/10 task 완료, task당 ~16분) | 종료 예상 22:00~23:00 KST. 끝나면 자동 `aggregate_libero_rollouts.py` 호출 → summary.csv 등록 |
| `two-stream-v11` rollout (큐, chain pid 2663465) | DINO 종료 직후 | QUEUED. `kill -0 2583933` polling. 종료 즉시 `bash scripts/local/run_libero_rollouts.sh two-stream-v11 50` 실행 | ckpt 9개 (`/mnt/data/checkpoints/libero_bct/two-stream-v11_libero_*_v3/`). 로그: `logs/v11_bct_v3_rollout.log` |

**ckpt 출처**: v11 9 ckpt는 cluster 33866077~33866085 (또는 34004021~34004029) 학습 산출물. 2026-05-12에 `v11_bct_v3_rollout.tar` (7.1 GB)로 로컬 전송 → `/mnt/data/checkpoints/libero_bct/` 추출 (tar 삭제).

**결과 기록 흐름**:
- 각 rollout 종료 시 `aggregate_libero_rollouts.py`가 `data/libero/results/{dinov2_v3_t50,two-stream-v11_v3_t50}/*.json` 스캔 → `paper_artifacts/libero_rollout/{summary,per_task,episodes}.csv` 갱신
- v11 rollout 종료 후 본 entry 하단에 dinov2/v11 SR (3 suite × 3 seed 평균) 표 추가 예정

### 2026-05-06 v14 sanity (Phase A — stream-wise paradigm specialization)

v13 paradigm conflict (P encoder가 reconstruction + DINO 동시 만족 못 해 ep10+ uniform collapse) 진단 → paradigm을 stream-wise로 분리. P=MAE+V-JEPA, M=DINO. commit `35cad9e`.

**Phase A 검증 포인트** (sbatch §체크): L_t/L_tk_recon/L_pred 단조 감소, L_dino < log(K)≈6.93, cos_intra_p<0.95, cos_intra_dino<0.95, std_student_dino>0.1, cos(pred,target)<0.99 (V-JEPA predictor identity 아님), ‖dino_center‖ 안정.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34011203 | AIP 1×1 H100 | 02:00:00 | v14 Phase A sanity (200vid × 5ep × N=1, default hyperparam, ep당 시간 측정 → 본 학습 --time 결정) | ❌ FAILED 05-06 20:32 (2분 44초). 첫 batch loss 7.29 후 두번째 batch에서 DDP `Parameter indices which did not receive grad for rank 0: 4` (= `mask_token_m`). v14는 M stream 항상 unmasked라 v11 base의 `mask_token_m`이 forward에 안 쓰임. **Fix**: `__init__` 끝에 `self.mask_token_m.requires_grad_(False)` 추가 (login 노드 forward+backward로 unused=0 검증). |
| 34014747 | AIP 1×1 H100 | 02:00:00 | v14 sanity 재제출 (mask_token_m fix 후) | ✅ COMPLETED 05-07 09:54 (17m21s, 5ep × 200vid). L_t/L_tk=0.0187, L_pred=0.0388, L_dino=6.44(<log K=6.93 평형 아래로 내려옴), cos_intra_p=0.589 ✅, std_sdino=0.210 ✅, cos(pred,tgt)=0.978 ✅, ‖dino_center‖=0.38. ⚠️ **cos_intra_dino=0.909** — sanity 한계 근처(dino_n_crop=1 영향), 본 학습 ep1-3 추이 관찰 필요. **🔴 Loss imbalance 발견**: 합산 시 DINO가 total의 98.7% 차지 (λ_dino=1.0 × 6.44 vs L_t+L_tk=0.037) → 본 학습은 v13 1차 선례 따라 **`V14_LAMBDA_DINO=0.01`** 로 축소 결정. |
| 34015672 | AIP_long 2×4 H100 | 4-00:00:00 | **v14 본 학습** (50ep, EgoDex part1-5, λ_dino=0.01, λ_pred=1.0, dino_n_crop=2, K=1024, T_temp=0.04, EMA 0.996→0.9999) | ❌ CANCELLED 2026-05-10 20:47 (ep20 + ep21 8.6% 시점, elapsed 3d 00:23). 사용 자원: 8 GPU × 72.4h = **579 GPU·hours**. ep20 ckpt를 로컬 BC-T fine-tune + rollout으로 v14 평가 분기. 추가 학습은 ep15-18 oscillation pattern + ep26~27 timeout 위험으로 stop. |
| 34118629 | AIP 1×1 H100 | 00:20:00 | v14 ep4 nomask reconstruction viz (5 cols: frame_t/frame_tk/pred_t MAE/pred_tk MAE/pred_tk V-JEPA motion-routed) | ❌ FAILED 20s `ModuleNotFoundError: No module named 'src'`. sys.path.insert 추가 후 재제출 |
| 34120195 | AIP 1×1 H100 | 00:20:00 | v14 ep4 nomask viz 재제출 (sys.path fix) | ✅ COMPLETED 37s. `paper_artifacts/v14_main_train_samples/epoch_004_nomask.png` 산출 (4 sample × 5 col). pred_t/pred_tk/pred_tk_motion 3개 모두 patch grid 패턴만 보일 뿐 frame 구조 없음 — ep4 단계에서 recon decoder 학습 매우 미숙 (train loss 0.0156이지만 픽셀 시각화는 unrecognizable) |
| 34171143 | AIP 1×1 H100 | 00:20:00 | v14 ep4 attention viz (nomask recon + motion-routing 2 iter attention map, 4 sample × 9 col) | ❌ FAILED 1m19s — nomask viz는 ✅ 산출, attention viz 단계에서 `RuntimeError: Can't call numpy() on Tensor that requires grad`. V-JEPA path 디코드가 try 블록 밖에서 실행돼 no_grad context 벗어남. fix: `@torch.no_grad()` 데코레이터 추가 |
| 34177451 | AIP 1×1 H100 | 00:20:00 | v14 ep4 attention viz 재제출 (no_grad fix) | ❌ CANCELLED (사용자 결정 — ep8로 재타게팅) |
| 34177493 | AIP 1×1 H100 | 00:20:00 | v14 ep8 viz (nomask + attention) | ✅ COMPLETED 1m51s. `epoch_008_nomask.png` + `attn_v14_ep8.png` 산출 |
| 34177494 | AIP 1×1 H100 | 03:00:00 | v14 ep8 EgoDex action probing (`patch_mean_concat_all`, gap=10, test) | ✅ COMPLETED 14m41s. **R²=-0.203** (FAIL). v11 ep8 baseline +0.094 대비 -0.297 악화. 모든 train epoch에서 R² 음수, per-joint R² 모두 -0.15~-0.43 |
| 34203856 | AIP 1×1 H100 | 00:20:00 | v14 ep12 viz (nomask + attention) | PENDING |
| 34203857 | AIP 1×1 H100 | 03:00:00 | v14 ep12 EgoDex action probing (`patch_mean_concat_all`, gap=10, test) | ✅ COMPLETED 15m6s. **R²=-0.071** (ep8 -0.203 → +0.132 향상). per-joint R² 모두 ep8 대비 향상 (-0.005~-0.15) |
| 34211118 | AIP 1×1 H100 | 00:20:00 | v14 ep16 viz (nomask + attention) | PENDING |
| 34211119 | AIP 1×1 H100 | 03:00:00 | v14 ep16 EgoDex action probing (`patch_mean_concat_all`, gap=10, test) | ✅ COMPLETED. **R²=-0.065** (ep12 -0.071과 거의 동일). collapse 진행에도 representation 변화 없음 — stream 분리 측정 필요 |
| 34211412 | AIP 1×1 H100 | 03:00:00 | v14 ep16 probing — M encoder only (`patch_mean_m_enc`, A) | PENDING |
| 34211413 | AIP 1×1 H100 | 03:00:00 | v14 ep16 probing — motion-routing 후 P state (`patch_mean_p_state_after_routing`, D') | PENDING |
| 34211414 | AIP 1×1 H100 | 03:00:00 | v14 ep16 probing — P encoder only (`patch_mean_p_enc`, B), collapse 검증 | ✅ COMPLETED. **R²(A)=+0.100, R²(D')=-0.229, R²(B)=-0.032**. M stream 살아있음 ★, motion-routing이 destructive 학습 (V-JEPA target 따라잡기 시도). 사용자 통찰: v14 D' = V-JEPA predictor 중간 단계, v11과 다른 의미 — D' 측정 자체가 잘못된 질문 |
| 34211491 | AIP 1×1 H100 | 01:30:00 | v14 ep16 LIBERO action probing (`abd_prime`, libero_spatial, gaps 1/13/20/40) | ✅ COMPLETED 11m21s. **gap=1 +0.483, gap=13 +0.550, gap=20 +0.467, gap=40 +0.245**. v11 ep44 baseline 대비 -0.13~-0.18 (v14 16ep만 학습됨, fair X). 그러나 EgoDex within-domain (-0.065)에선 망가졌는데 LIBERO에서 양수 → cross-domain transfer 가능성 |
| 34246462 | AIP 1×1 H100 | 00:20:00 | v14 ep20 viz (nomask + attention) | PENDING |
| 34246463 | AIP 1×1 H100 | 03:00:00 | v14 ep20 EgoDex action probing (`patch_mean_concat_all`, gap=10, test) | PENDING |
| 34246464 | AIP 1×1 H100 | 02:00:00 | v15 초안 sanity (DINO 포함) — DINO collapse 확정 (L_dino plateau at log K, cos_intra_dino=1.0) → v15 final design 도입 (DINO 제거 + L_compose) | ✅ COMPLETED 19m26s |
| 34270603 | AIP 1×1 H100 | 02:00:00 | v15 sanity warmup_v1 (warmup + λ_dino 0.01 + EMA 0.999) | ❌ FAILED 5m59s (TB scalar `actual_model` scope NameError, fix됨) |
| 34271010 | AIP 1×1 H100 | 02:00:00 | v15 sanity warmup_v2 (재제출) | ✅ COMPLETED 36m08s. DINO 더 빠르게 collapse (cos_intra_dino=1.0 ep2부터). v15 final design 정당화 강화 |
| 34279264 | AIP 1×1 H100 | 02:00:00 | v15 final sanity v1 (DINO 제거 + L_compose + 3-frame triple + V-JEPA-M Option B) | ❌ FAILED 1m51s (`dino_center` AttributeError, train_epoch에 hasattr 분기 fix됨) |
| 34279476 | AIP 1×1 H100 | 02:00:00 | v15 final sanity v2 (10ep, num_workers=16) | 🔴 47분 hang → CANCELLED. Dataset triple 단독 정상, num_workers=16 multiprocessing issue 추정 |
| 34285612 | AIP 1×1 H100 | 02:00:00 | v15 final mini sanity (3ep × 50vid, num_workers=4) | ✅ COMPLETED 4m28s. ep당 77s. L_t/L_tk 정상 감소, L_pred → identity collapse(cos=1.0), std_m collapse, L_compose plateau (3ep만이라 trajectory 부족). Hang 원인 확정 — num_workers=16 → 4-8로 회피. v15 forward 정상 |
| 34288968 | AIP_long 2×4 H100 | 10-00:00:00 | **v15 본 학습** (50ep, EgoDex part1-5, batch 32/GPU = global 256, num_workers=8, λ all 1.0 with warmup 10ep 0.01→1.0, composition_mode=linear_residual, EMA 0.999→0.9999, mask_p=0.75, mask_m_jepa=0.5, max_gap=30 sample_center=15) | ✅ COMPLETED 2026-05-12 23:46 (Elapsed 1d 18h 54m24s, **8 GPU × 42.91h = 343.3 GPU·h**, ExitCode 0:0). 50ep 완주 — ckpt ep4/8/12/16/20/24/28/32/36/40/44/48 + latest.pt(=ep50) + best_model.pt(=ep1, eval_loss min). eval_loss는 ep1 0.050 → ep10 0.20 단조 증가 (P encoder collapse 진행), 그러나 downstream representation은 ep32 P_t+P_tk concat이 v11 champion 추월 — best_model.pt는 paper용 ckpt 아님 |
| 34343947 | AIP 1×1 H100 | 00:20:00 | v15 ep4 viz (nomask P MAE × 3 frame + motion routing × 3 segment + L_compose path, 10-column 4-sample figure) | ✅ COMPLETED 58s. `paper_artifacts/v15_main_train_samples/epoch_004_nomask.png`. GT(1-3) + P MAE recon(4-6) + motion routing short/step/long(7-9) + composition(10). ep4 단계라 col 4-10 모두 patch grid speckle만 보임 — v14 ep4와 동일 (recon decoder 미숙). ep8+ 재실행 권장 |
| 34344317/319/326 | AIP 1×1 H100 | 00:20:00 ×3 | v15 ep4 viz EgoDex+DROID 확장 디버그 (3 회) | ❌ 1차 sbatch `--num-samples` 폐기 인자, 2차 `DROIDDataset.__init__` sample_dist 미지원, 3차는 잡 30s 완료지만 DROID row 검정 fallback (ext1 95658 ep 중 단 10개만 frame 추출됨, 빈 ep retry 5회 fallback) |
| 34344358 | AIP 1×1 H100 | 00:20:00 | v15 ep4 viz EgoDex+DROID 확장 4차 (max_videos=10) | ✅ COMPLETED 41s. `paper_artifacts/v15_main_train_samples/epoch_004_nomask.png` 4 row (EgoDex 2 + DROID 2) × 10 col. ep4 단계라 col 4-10 모두 patch grid speckle, v14 ep4 viz와 동일 양상. ep8+ 재실행 권장 |
| 34349393 | AIP 1×1 H100 | 00:20:00 | v15 ep8 viz (동일 EgoDex+DROID 4 sample) | ✅ COMPLETED. `paper_artifacts/v15_main_train_samples/epoch_008_nomask.png` ep4와 거의 동일한 양상 — col 4-10 모두 patch grid speckle. TB 진단으로 P encoder collapse (feat_std_p=0.09, cos_intra_p=0.987 @ ep8) 확인 → recon_head가 sample-independent 평균 패턴만 출력. ep10 warmup 완료 후 회복 신호 있으므로 ep12+ 재시도 권장 |
| 34357733 | AIP 1×1 H100 | 00:20:00 | v15 ep28 viz (동일 EgoDex+DROID 4 sample) | ✅ COMPLETED. `paper_artifacts/v15_main_train_samples/epoch_028_nomask.png` 산출. TB 분석: ep10 warmup 종료 후에도 **P encoder 회복 실패** (cos_intra_p 0.997, feat_std_p 0.020 @ ep28). **M encoder healthy** (cos_intra_m 0.27, feat_std_m 1.23). L_t/L_tk 0.0151→0.00105 수렴, L_pred 0.025→0.0072 수렴, L_compose 0.286→0.036. 🔴 **L_m_jepa ep7 0.0009 → ep28 0.086 폭증 후 plateau**. Eval loss ep5 0.050 → ep25 0.154 단조 증가 |
| 34357799 | AIP 1×1 H100 | 00:15:00 | v15 ep28 collapse 진단 (CLS / patch token / recon image pairwise MSE, EgoDex+DROID 각 16 sample) | ✅ COMPLETED. **시나리오 (A) 확정 — CLS만 collapse, patch는 healthy**: EgoDex `cls_p_cos=0.998 / patch_cos=0.414`, DROID `0.999 / 0.413` (patch token cross-domain 거의 동일). Recon image pairwise MSE ratio (pred/GT) EgoDex 0.428 / DROID 0.416 — **GT 다양성의 43%만 recon이 표현** (mean pattern으로 부분 수렴, 완전 collapse 아님). M encoder 양쪽 모두 healthy (cls_m_cos 0.15~0.19). 사용자 가설 부분 검증: CLS loss 추가는 CLS 살리지만 downstream cosmetic 가능성. **다음: ep28 patch_mean probing 측정 → CLS loss 추가 여부 결정** |
| 34358483 | AIP 1×1 H100 | 00:20:00 | v15 ep32 viz (동일 EgoDex+DROID 4 sample, 10-col nomask reconstruction + motion routing) | ✅ COMPLETED 1m48s |
| 34363191 | AIP 1×1 H100 | 03:00:00 | v15 ep32 EgoDex action probing (`patch_mean_concat_all`, gap=10, test) — v11 호환 mode | ✅ COMPLETED 16m6s. **R²=-0.1293 / Cos 0.129** (FAIL). v11 ep44 +0.288 / v14 ep16 -0.065 / VideoMAE +0.326 대비 전면 후퇴. Train MSE 0.0019 정상 감소 but Eval R² -1.47~-0.13 oscillation — P encoder collapse + L_m_jepa 폭증 영향 확정 |
| 34363192 | AIP 1×1 H100 | 03:00:00 | v15 ep32 DROID cross-domain probing (`patch_mean_concat_enc_only` A+B, gap=15) | ✅ COMPLETED 1m26s. **R²=-0.0485** (gap=15, A+B). v11 ep12 +0.005 / VideoMAE -0.035 대비 후퇴. P encoder collapse가 cross-domain에서도 나타남 |
| 34363193 | AIP 1×1 H100 | 01:30:00 | v15 ep32 LIBERO action probing × spatial (`abd_prime`, gaps 1/13/20/40) | RUNNING |
| 34363194 | AIP 1×1 H100 | 01:30:00 | v15 ep32 LIBERO action probing × object (`abd_prime`, gaps 1/13/20/40) | ❌ CANCELLED (사용자 결정 — 중간 결과만 보기에 1 suite 충분) |
| 34363195 | AIP 1×1 H100 | 01:30:00 | v15 ep32 LIBERO action probing × goal (`abd_prime`, gaps 1/13/20/40) | ❌ CANCELLED (동일) |
| 34367235 | AIP 1×1 H100 | 03:00:00 | v15 ep32 EgoDex probing — P_t only (`patch_mean_p_enc`, gap=10) | ✅ COMPLETED 13m44s. **R²=−0.0532** |
| 34367236 | AIP 1×1 H100 | 03:00:00 | v15 ep32 EgoDex probing — P_tk only (`patch_mean_p_enc_tk`, gap=10) — **신규 mode** | ✅ COMPLETED 13m37s. **R²=−0.0135** |
| 34367237 | AIP 1×1 H100 | 03:00:00 | v15 ep32 EgoDex probing — M only (`patch_mean_m_enc`, gap=10) | ✅ COMPLETED 13m37s. **R²=−0.0831** (M encoder 가장 망가짐, L_m_jepa 폭증과 일치) |
| 34367238 | AIP 1×1 H100 | 03:00:00 | v15 ep32 EgoDex probing — P_t + P_tk (`patch_mean_concat_p_t_p_tk`, gap=10) — **신규 mode** | ✅ COMPLETED 13m37s. **🔥 R²=+0.3898 / Cos 0.306** — v11 ep44 champion(+0.288) 및 VideoMAE(+0.326) 추월. 단독 음수인데 concat이 큰 양수 → 두 frame feature가 서로 다른 정보 인코딩, linear probe가 implicit difference 학습 |
| 34367239 | AIP 1×1 H100 | 03:00:00 | v15 ep32 EgoDex probing — M + P_t (`patch_mean_concat_enc_only`, gap=10) | ✅ COMPLETED 13m37s. **R²=−0.0807** (M 포함 시 음수, M 망가진 영향) |
| 34367240 | AIP 1×1 H100 | 03:00:00 | v15 ep32 EgoDex probing — P_tk + M (`patch_mean_concat_p_tk_m`, gap=10) — **신규 mode** | ✅ COMPLETED 13m40s. **R²=−0.1033** (M 포함 시 음수) |

**v15 ep32 종합 — 핵심 결론**:
- P encoder가 살아있음 (patch-level instance discrimination 유지, CLS만 collapse)
- M encoder가 가장 망가짐 (L_m_jepa 폭증이 representation 파괴 확정)
- **P_t + P_tk concat이 v11 champion + VideoMAE 모두 추월 ★** — v15 가치 재평가 필요
- A+B+D' 측정값 −0.129는 M+P_t+D' = M 망가짐이 끌어내린 결과 (D'은 motion-routing 거치며 M dependency)

### P_t + P_tk fair 비교 — v15 vs v11 cross-domain
| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34367515 | AIP 1×1 H100 | 03:00:00 | v11 ep44 EgoDex `patch_mean_concat_p_t_p_tk` gap=10 | ✅ COMPLETED 11m18s. **R²=+0.0097** (거의 0). v11 champion mode A+B+D' +0.288 대비 큰 차이 — v11에서는 P_t+P_tk가 motion 정보 캐리어 아님 |
| 34367577 | AIP 1×1 H100 | 03:00:00 | v15 ep32 DROID `patch_mean_concat_p_t_p_tk` gap=15 | ✅ COMPLETED 52s. R²=−0.006 (cross-domain 약함) |
| 34367579 | AIP 1×1 H100 | 03:00:00 | v11 ep44 DROID `patch_mean_concat_p_t_p_tk` gap=15 | ✅ COMPLETED 52s. R²=−0.001 (cross-domain 약함, DROID 한계 일관) |
| 34367612 | AIP 1×1 H100 | 01:30:00 | v15 ep32 LIBERO spatial `p_t_p_tk` (4 gaps) — **probe_action_libero.py 신규 mode 추가** | ✅ COMPLETED 16m47s. gap=1 **+0.401**, gap=13 **+0.576**, gap=20 **+0.584** ★, gap=40 **+0.379**. **v11 champion abd_prime gap=20 (+0.555) 추월** ★★★ |
| 34367614 | AIP 1×1 H100 | 01:30:00 | v11 ep44 LIBERO spatial `p_t_p_tk` (4 gaps) | ✅ COMPLETED 16m40s. gap=1 +0.058, gap=13 +0.038, gap=20 **+0.041**, gap=40 +0.061. **v15 ep32 동일 mode 대비 −0.54 격차** — P_t+P_tk가 v15-specific 우위 확정 |
| 34370933 | AIP 1×1 H100 | 03:00:00 | DINOv2 EgoDex `patch_mean` (P_t+P_tk equivalent, single-frame SSL baseline) — 가설 인과 검증 마지막 증거 | ✅ COMPLETED 16m13s. **R²=+0.0062 / Cos 0.211**. v15 ep32 +0.390 대비 격차 +0.384 — 단순 frame-discriminative SSL은 P_t+P_tk pattern 생성 X. **motion routing이 P encoder에 motion-friendly 압력 transfer 인과 확정** |
| 34373722 | AIP 1×1 H100 | 00:20:00 | v15 ep40 viz (동일 EgoDex+DROID 4 sample, 10-col nomask reconstruction + motion routing) | ✅ COMPLETED |

### 2026-05-13 v15 ep50 (latest.pt) paper main 분석 + V3 BC-T 9잡

v15 본 학습 (34288968) 종료 → ep50 = `latest.pt` 사용. paper main result로 ep32 mode sweep 동일 cfg fair 비교 + V3 BC-T full 매트릭스 (3 suite × 3 seed) 시작.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34375951 | AIP 1×1 H100 | 00:20:00 | v15 ep50 viz (EgoDex+DROID 4 sample, 10-col nomask + motion routing) | ✅ COMPLETED 1m17s |
| 34375952 | AIP 1×1 H100 | 03:00:00 | v15 ep50 EgoDex probing — `patch_mean_concat_p_t_p_tk` (ep32 champion, gap=10) | RUNNING |
| 34375953 | AIP 1×1 H100 | 03:00:00 | v15 ep50 EgoDex probing — `patch_mean_p_enc` (P_t) | RUNNING |
| 34375954 | AIP 1×1 H100 | 03:00:00 | v15 ep50 EgoDex probing — `patch_mean_p_enc_tk` (P_tk) | RUNNING |
| 34375955 | AIP 1×1 H100 | 03:00:00 | v15 ep50 EgoDex probing — `patch_mean_m_enc` (M alone) | RUNNING |
| 34375956 | AIP 1×1 H100 | 03:00:00 | v15 ep50 EgoDex probing — `patch_mean_concat_enc_only` (M+P_t) | RUNNING |
| 34375957 | AIP 1×1 H100 | 03:00:00 | v15 ep50 EgoDex probing — `patch_mean_concat_p_tk_m` (P_tk+M) | RUNNING |
| 34375958 | AIP 1×1 H100 | 03:00:00 | v15 ep50 EgoDex probing — `patch_mean_concat_all` (A+B+D') | RUNNING |
| 34375959 | AIP 1×1 H100 | 03:00:00 | v15 ep50 DROID cross-domain probing — `patch_mean_concat_p_t_p_tk` (gap=15) | ✅ COMPLETED 50s |
| 34375960 | AIP 1×1 H100 | 03:00:00 | v15 ep50 DROID cross-domain probing — `patch_mean_concat_enc_only` (A+B, gap=15) | ✅ COMPLETED 41s |
| 34375961 | AIP 1×1 H100 | 01:30:00 | v15 ep50 LIBERO spatial probing — `p_t_p_tk` (4 gaps 1/13/20/40) | ❌ CANCELLED 1m29s (사용자 결정 — BC-T adapter mismatch 재검토 필요) |
| 34375964~34375972 | AIP 1×1 H100 ×9 | 2-00:00:00 | v15 ep50 V3 BC-T (3 suite × 3 seed, A+D' adapter) | ❌ CANCELLED 미시작 (사용자 결정 — v15 학습 분포(3-frame)와 v11 adapter(2-frame pair) mismatch 검토 후 재제출 결정) |

**Cancel 이유**: v15는 학습 시 **3-frame triple** (frame_t, frame_t+n, frame_t+m), LIBERO probing champion mode가 `patch_mean_concat_p_t_p_tk` (P_t + P_tk encoder concat). 그러나 BC-T 어댑터 `TwoStreamV11Adapter`는 **2-frame pair (prev, curr) + A+D' (M encoder + motion-routing 후 P state)** 고정. 학습/inference 분포 mismatch + champion mode 미사용 → 재설계 필요.

### 2026-05-13 v15 ep50 V3 BC-T 재제출 — 신규 2 어댑터 × 9잡 = 18잡

기존 v11 어댑터(A+D') 대신 v15-적합 2 어댑터로 다시 매트릭스:
- **`two-stream-v15-ptptk`** (옵션 B): (prev, curr) 2-frame pair → P encoder 각자 forward → P_t patches mean ⊕ P_tk patches mean = 1536-d. LIBERO probing champion mode (`patch_mean_concat_p_t_p_tk`)와 동일 출력 형태
- **`two-stream-v15-mp`** (C-variant): (prev, curr) → M encoder × 1 (m_channel=Δ) + P encoder × 1 (curr만, p_channel) → M patches mean ⊕ P_curr patches mean = 1536-d. Motion-routing 단계 제거, v11 A+D'에서 D' → D으로 단순화

신규 어댑터 파일: [src/encoders/adapters/two_stream_v15_pt_ptk.py](../src/encoders/adapters/two_stream_v15_pt_ptk.py), [two_stream_v15_mp.py](../src/encoders/adapters/two_stream_v15_mp.py). [base.py](../src/encoders/adapters/base.py) + [finetune_libero_bct.py](../scripts/eval/finetune_libero_bct.py) choices 등록. login 노드 forward sanity 통과 (out shape (2,3,1536)).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34376001~34376009 | AIP 1×1 H100 ×9 | 2-00:00:00 | v15 ep50 V3 BC-T **ptptk** × 3 suite × 3 seed | PENDING |
| 34376010~34376018 | AIP 1×1 H100 ×9 | 2-00:00:00 | v15 ep50 V3 BC-T **mp** × 3 suite × 3 seed | PENDING |

### Causal Future Prediction probing (TARGET_MODE=future, target = pose[t+2gap]−pose[t+gap])

probe_action.py에 `target_mode` 옵션 추가 — 기존 "same" (변화 인지) 외 "future" (미래 prediction, features 본 frame 너머 변화) mode. v15 ep32 + v11 ep44 × 7 mode = 14잡 fair pair 비교.

| JobID | ckpt | mode | R² | Cos |
|-------|------|------|------|------|
| 34374136 | v15 ep32 | patch_mean_p_enc | −0.020 | 0.111 |
| 34374137 | v15 ep32 | patch_mean_p_enc_tk | −0.077 | 0.062 |
| 34374138 | v15 ep32 | patch_mean_m_enc | **−0.083** | 0.055 |
| 34374139 | v15 ep32 | patch_mean_concat_p_t_p_tk | −0.020 | 0.136 |
| 34374140 | v15 ep32 | patch_mean_concat_enc_only (P_t+M) | −0.142 | 0.115 |
| 34374141 | v15 ep32 | patch_mean_concat_p_tk_m | −0.155 | 0.085 |
| 34374149 | v15 ep32 | patch_mean_concat_p_t_p_tk_m (3-concat) ★ | −0.145 | 0.106 |
| 34374142 | v11 ep44 | patch_mean_p_enc | −0.002 | -0.000 |
| 34374143 | v11 ep44 | patch_mean_p_enc_tk | −0.002 | 0.058 |
| 34374144 | v11 ep44 | patch_mean_m_enc | **+0.092** ★★ | 0.158 |
| 34374145 | v11 ep44 | patch_mean_concat_p_t_p_tk | +0.002 | 0.049 |
| 34374146 | v11 ep44 | patch_mean_concat_enc_only (P_t+M) | **+0.092** | 0.148 |
| 34374147 | v11 ep44 | patch_mean_concat_p_tk_m | **+0.094** | 0.151 |
| 34374150 | v11 ep44 | patch_mean_concat_p_t_p_tk_m | **+0.093** | 0.153 |

**🔥 핵심 발견 — 사용자 가설 검증 완료**:
- **v11 ep44 M alone = +0.092** (유일 양수 mode 그룹 캐리어). M 포함 mode 모두 +0.092~0.094로 유사 → **M이 미래 action 정보의 캐리어**
- **v15 ep32 M alone = −0.083** (망가진 상태) → 미래 prediction 자체 무너짐. M 포함하면 noise로 R² 더 악화
- **v11 ep44 P_t+P_tk = +0.002** vs **v15 ep32 P_t+P_tk = −0.020** — 변화 인지에서 압도적이던 P_t+P_tk가 미래 prediction에는 무력
- **두 task는 완전히 다른 신호 측정**: P_t+P_tk = post-hoc reconstruction (frame 차이 inverse), M alone = 진짜 causal future inference
- ⚠️ **paper 전략 재검토 시사**: BC-T = robot deployment ≈ future inference 가까움. v15 only main 결정의 validity는 BC-T 결과로 최종 검증 필요

**🏆 P_t + P_tk fair 비교 핵심 결론** (사용자 가설 검증 완료):
- **v15 ep32 P_t+P_tk가 v11 ep44 champion mode (abd_prime) 추월** (LIBERO spatial gap=20 +0.584 vs +0.555)
- v11에서는 P_t+P_tk가 +0.010 (거의 0) → motion 정보가 D' (motion-routing 후 P state)에 인코딩
- v15에서는 P_t+P_tk가 +0.390 (EgoDex) / +0.584 (LIBERO) → **P encoder 자체가 motion-friendly representation 학습**
- 사용자 가설 정확히 성립: "motion routing이 가능한 이미지 복구 형태" → P encoder에 motion-friendly 압력 transfer
- v15 구조 (3-frame triple + composition + V-JEPA-M)가 P encoder 학습 압력을 v11 대비 더 강하게 전달
- **v15 BC-T 가치 매우 높음 (이전 결론 전면 재검토)** — 50ep 완주 후 BC-T adapter 작업 가치 매우 큰 후보
- Cross-domain DROID는 두 모델 모두 약함 (DROID 자체 한계, [[feedback-droid-image-only-limitation]] 참고)

### 2026-05-04 v13 DINO redesign (1차 학습 33833830 ep10+ uniform collapse 후속)

1차 학습 33833830 (ep1~12, ckpt 4/8/12) ep10부터 student CLS uniform collapse (cos_intra_pred_cls 0.99, std_pred_cls 0.016, L_pc=log(K)=6.93에 갇힘) → 구조 변경 후 재학습.

**변경 요지** (사용자 토론 후):
- DINO source: `predicted_cls_tk` (motion-routed) → **student P encoder CLS 직접** (P encoder가 학습 신호 직접 받음)
- DINO target: `image_future_global` 단일 → **frame_t/tk 둘 다 256² global** (multi-crop teacher, 표준 DINOv2)
- DINO mask: recon mask 0.75와 분리해 **mask_p_dino=0.4** (DINOv2 30~50% 표준)
- DINOHead: 단순 normalize+Linear → **MLP 보강** (D→2048→256→K, 표준 DINO/DINOv2)
- λ_cls: 0.01 → **0.3** (P encoder direct distill로 신호 강화 가능)
- center momentum: 0.95 → **0.9** (DINOv2 default)
- forward 4 → 7개 → ep당 ~4.5~5h 추정 (1차 2.78h 대비 1.6~1.8배)

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33988629 | AIP 1×1 H100 | 02:00:00 | sanity v13 redesign (200 vid × 5 ep, suffix=dinov2redesign) | ✅ COMPLETED 05-06 08:08 (21분, ExitCode 0) |
| 33988630 | AIP_long 2×4 H100 | 10-00:00:00 | v13 본 학습 (50 ep, suffix=dinov2redesign, **dependency=afterok:33988629**) | ❌ CANCELLED 05-06 20:24 (사용자 결정, 미시작 — Reason=Resources 12h 대기) |

### 2026-05-04 V3 BC-T full 매트릭스 1차 (4 enc × 3 suite × 3 seed = 36 잡)

V3 BC-T sanity (33834914) 통과 + aug-check.png 저장 → full 매트릭스 제출. **v11 제외** (사용자 결정: baseline + MAE 우선, v11은 후순위 비교군).

| Encoder | Checkpoint | JobID 범위 |
|---------|-----------|------------|
| videomae-ours | `videomae/20260415_012017/best_model.pt` | 33866041~33866049 |
| dinov2 (HF) | — | 33866050~33866058 |
| siglip (HF) | — | 33866059~33866067 |
| vc1 (HF) | — | 33866068~33866076 |

각 9 잡 (3 suite × 3 seed). AIP 1×1 H100, --time=2-00:00:00 (4월 BC-T elapsed 24~25h 기준 안전 마진 ×2). 36 잡 PENDING/RUNNING.

### 2026-05-06 v11 V3 BC-T 추가 제출 (main table 행 확보) + LIBERO probing 재확인

**배경**: v11 0% rollout 원인 진단 P0 — 4월 보류한 v11 V3 BC-T를 baseline 36잡과 같은 cfg로 추가 제출 + LIBERO action probing 재실행해 representation 도메인 적합도 확인.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 34004014 | AIP 1×1 H100 | 01:30:00 | v11 ep44 LIBERO probing × spatial (4 gaps) | ✅ COMPLETED 10:45 — gap=1 R²=0.660, gap=20 R²=0.555, gap=40 R²=0.374 (PROBING_GUIDE 4월 값과 동일) |
| 34004015 | AIP 1×1 H100 | 01:30:00 | v11 ep44 LIBERO probing × object | ✅ COMPLETED 13:05 — gap=1 0.702, gap=20 0.681, gap=40 0.614 |
| 34004016 | AIP 1×1 H100 | 01:30:00 | v11 ep44 LIBERO probing × goal | ✅ COMPLETED 11:08 — gap=1 0.546, gap=20 0.613, gap=40 0.631 |
| 34004021~34004029 | AIP 1×1 H100 | 2-00:00:00 | v11 ep44 V3 BC-T × {spatial,object,goal} × seed{0,1,2} = 9 잡 | RUNNING |

**probing 진단**: v11 representation은 LIBERO action을 baseline (videomae 0.408, dinov2 0.611) 동급 이상으로 인코딩. 0% rollout은 representation 문제 아님. → **v13 redesign은 잘못된 문제를 풀고 있을 가능성 농후** (ep1~3 진단 통과해도 BC 0% 그대로일 수 있음).

**중복 비용 주의**: probing 3잡은 PROBING_GUIDE에 이미 기록된 값과 동일. 결과 재확인 목적이라 sunk cost 수용 (~3 GPU·h).

### 2026-05-04 Ego4D extract resume (33834913 TIMEOUT 후속)

원인 분석: sanity 첫 10 vid 평균 353 MB, 전체 9821 vid 평균 765 MB (2.17배). vid 평균 길이 ~31 min × 30 fps = 93k frame → 144 worker × ~18 min/vid 디코드 한계 = 0.0753 vid/s. Sanity 1.2 vid/s 추정은 단일 sample (12k frame짜리 짧은 영상) 기반 빗나간 추정. **수정 없이 단순 resume**으로 충분 (skip 로직 검증).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33866094 | normal_cpu 1×144 CPU | 2-00:00:00 | Ego4D resume (남은 ~6566 vid, 추정 24h) | PENDING |

### 2026-05-04 v13 ep8 ckpt viz + action probing (33833830 학습 중간 진단 #2)

ep4 (sanity ep4 R²-0.004 → 본 학습 ep4 R²-0.363, 악화) 후속. ep5-9 진단에서 ep7-8 anti-collapse 강력 작동 (cos_intra_p 0.28/0.31, L_pc 0.35) 직후 ckpt → representation 회복 측정.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33865946 | AIP 1×1 H100 | 00:20:00 | v13 ep8 nomask viz | RUNNING |
| 33865947 | AIP 1×1 H100 | 03:00:00 | EgoDex action probing (`patch_mean_concat_all`, gap=10) | PENDING |
| 33865948 | AIP 1×1 H100 | 03:00:00 | DROID cross-domain probing (`patch_mean_concat_enc_only`, gap=15) | PENDING |

### 2026-05-03 v13 ep4 ckpt viz + action probing (33833830 학습 중간 진단)

v13 본 학습 33833830 ep4 완료 → ckpt `two_stream_v13_encroute/20260503_072606/checkpoint_epoch0004.pt`. 이전 v13 sanity ep4 probing baseline (R² ≈ -0.004 EgoDex / -0.001 DROID gap=15) 대비 본 학습 (Option β, K=1024, λ_cls=0.01, λ_patch=1.5)에서 회복 여부 측정.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33834909 | AIP 1×1 H100 | 00:20:00 | v13 ep4 nomask reconstruction viz | ✅ 21s, `paper_artifacts/v13_encroute_train_samples/epoch_004_nomask.png` |
| 33834911 | AIP 1×1 H100 | 03:00:00 | EgoDex action probing (`patch_mean_concat_all`, gap=10, test split) | RUNNING |
| 33834912 | AIP 1×1 H100 | 03:00:00 | DROID cross-domain probing (`patch_mean_concat_enc_only`, gap=15) | ✅ 56s, **R²=+0.0089** (sanity ep4 -0.0006 → 본 학습으로 +0.0095 개선, 미약하지만 회복 신호) |

### 2026-05-03 Ego4D full frame extraction

Sanity (33834238, 10 vid 6.5 min, 11 GB) 통과 → 전체 9,823 영상 sbatch.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33834913 | normal_cpu 1×144 CPU | 12:00:00 | Ego4D 9,823 mp4 → `frames/<uuid>/frame_*.jpg` 256² 30 fps (subsample 없음) | PENDING |

### 2026-05-03 V3 BC-T 본 학습 (cfg fix + augmentation, refactor_plan §3)

신규 cfg: `use_augmentation=True`, `ImgColorJitterAug` (brightness/contrast/saturation/hue=0.3, ε=0.05) + `TranslationAug` (translation=4, LIBERO 공식 default). LIBERO `DataAugGroup`이 dim=1 시점 concat 후 단일 random 적용 → **시점/카메라 일관성 자동 보장** (코드 분석 결과). 그래도 PNG 시각 검증 추가 안전장치.

신규 코드: [`scripts/eval/finetune_libero_bct.py`](../scripts/eval/finetune_libero_bct.py) — `save_aug_check_png()` + `--aug-check-png`/`--no-augmentation`/`--img-size` CLI. [`scripts/cluster/finetune_libero_bct.sbatch`](../scripts/cluster/finetune_libero_bct.sbatch) — `IMG_SIZE`/`NO_AUGMENTATION`/`AUG_CHECK_PNG` env vars.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33834242 | AIP 1×1 H100 | 00:30:00 | sanity V3 (videomae-ours × spatial × seed=0 × task=0 × bs 32 × 2ep × 20batch + aug-check-png) | ❌ FAILED 56s. `policy.img_aug` reshape mismatch (224 input vs 128 native). 학습 루프가 augmentation 호출조차 안 하던 추가 결함도 발견 (BasePolicy.preprocess_input은 `image_encoders` 참조, 우리는 `image_projections`라 자동 흐름 작동 X). 근본 fix: `apply_augmentation` 헬퍼 + train_one_epoch에 명시 호출 + aug-check-png는 resize 전에 호출 (LIBERO native 128 기준) |
| 33834914 | AIP 1×1 H100 | 00:30:00 | sanity V3 재제출 (코드 fix 후) | PENDING |

**검증 포인트 (sanity)**:
1. `v3_sanity_aug_check.png` 시각 확인: 같은 row(시점 4개) 동일 augmentation, 같은 sample의 두 카메라(agent/wrist) 함께 augmented
2. 학습 normal 진행 (loss decrease + ckpt 저장)
3. Epoch당 시간 측정 → V3 full 매트릭스 시간 추정

**Full 매트릭스 (sanity 통과 시)**: 5 encoder × 3 suite × 3 seed = **45 잡** AIP 1×1 H100 --time=2-00:00:00. 추정 ~45 GPU·일 ≈ 2.75M원 (월누적 ceil).

### 2026-05-03 Ego4D frame extraction (학습 데이터 추가)

신규 [`scripts/data/extract_ego4d_frames.py`](../scripts/data/extract_ego4d_frames.py) + [`scripts/cluster/extract_ego4d.sbatch`](../scripts/cluster/extract_ego4d.sbatch) — `/proj/external_group/mrg/datasets/ego4d/v2/full_scale/` (9,823 mp4 / 7.2 TB) → 256² JPEG frames 30 fps 그대로(subsample 없음). EgoDex와 동일 spec (센터 크롭 + 256² + JPEG q=95).

**login 노드 사전 검증** (5 영상 metadata + 1 영상 전체 추출):
- 5 영상 모두 30 fps 디코드 OK (1920×1440 / 2560×1920, 영상별 dur 21~958s)
- 1 영상 (12,466 frame) single-thread 119.8s = 104 fps decode → 144 worker 병렬 시 영상당 ~120s, 약 1.2 영상/s throughput → **9,823 영상 / 1.2 / 3600 ≈ 2.3시간** 추정
- 출력 31 KB/frame × 78.6 M frames ≈ **2.4 TB 디스크**

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33834238 | normal_cpu 1×144 CPU | 00:30:00 | sanity (10 영상, multi-process 검증) | PENDING |

**Full submit 조건**: sanity ✅ → 9,823 영상 normal_cpu 1×144 --time=12h 1잡 제출 예정.

### 2026-05-01 ~ 05-03 v13 Sanity + 본 학습 (Dual-Frame Recon + Motion-Routed Latent + DINO Global CLS)

신규 [`src/models/two_stream_v13.py`](../src/models/two_stream_v13.py) — post-v12 sanity 진단 (cls_p collapse 본질) 후 architectural redesign:
- frame_t / frame_{t+k} 모두 student P encoder 통과 → 각자 reconstruction (L_t, L_tk)
- motion-routing은 frame_t에서 시작 → predicted_p_tk (V-JEPA-style latent prediction)
- teacher (EMA) 두 input: cropped frame_{t+k} (per-patch target) + raw 256² (DINO-style global CLS). pos_embed bicubic interpolate (224↔256)
- L_total = L_t + L_tk + λ_patch · SmoothL1(predicted_patches, teacher_patches) + λ_cls · DINO_cosine_centered(predicted_CLS, teacher_global_CLS)

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 33830230 | AIP 1×1 H100 | 02:00:00 | sanity v13 (cosine + center only, λ_patch=1.0, λ_cls=0.1) | ✅ COMPLETED. **anti-collapse 1차 작동 확인** (ep1-2 cos_intra_p 0.51/0.49 vs v12 1.000), ep3 single-mode collapse → ep4-5 회복. Sharpening 부재 → full DINO 도입 결정 |
| 33830233 | AIP 1×1 H100 | 02:00:00 | sanity v13 full DINO (K=4096, λ_cls=0.005) | ✅ COMPLETED. **L_pc=8.32(=log K)에 갇힘** (prototype 학습 부족). ep5 회복 신호 폭발 (cos_intra_p 0.840→0.266) → ramp-up 미완. K=1024로 축소 결정 |
| 33830234 | AIP_long 2×4 H100 | 3-12:00:00 | v13 본 학습 1차 (Option β: K=1024, λ_cls=0.01, λ_patch=1.5, EgoDex part1-5, 50ep, global bs 512) | **CANCELLED** 05-03 06:01 (20:29h elapsed) — 33833830으로 재제출 |
| **33833830** | AIP_long 2×4 H100 | 3-12:00:00 | **v13 본 학습 2차** (재제출) | **RUNNING** 05-03 07:22 시작. ep1 진단 정상 (cos_intra_p 0.584, std_pred_cls 0.826, L_pc=3.69), ep2 cos_intra_p **0.917** 급등 ⚠️ |
| 33830237/239/406-409 | AIP 1×1 H100 | 00:15~03:00 | v13 sanity/ep4 ckpt viz + EgoDex/DROID action probing | viz: epoch_001/003 nomask 산출. col 5(motion-routed→recon_head, off-distribution)는 학습 초기 speckle |

**Option β 보정 근거**: sanity 2의 L_pc=log(K)≈8.32 평형은 prototype space 학습 미발현 신호. K=4096→1024 + λ_cls=0.005→0.01. ep1-3 cos_intra_p 추이로 조기 검증 → 이상 시 cancel.

**검증 포인트**: L_t/L_tk/L_pred_patch/L_pred_cls 단조 감소, cos_intra_p < 0.95, cos_intra_pred_cls < 0.95, std_pred_cls > 0.1 (DINO centering 작동), ‖pred_cls‖ ~ ‖target_cls‖ magnitude 정합, ‖dino_center‖ 안정 update.

**⚠️ --time 부족 위험**: epoch당 ~10,036s × 50ep = 139.4h vs --time 84h → 적자 -55h. ep1-3 진단 + ep4 ckpt 수령 후 cancel/연장 결정.

**사전 코드 점검** (login 노드): import / standard config (p_depth=12, m_depth=6) total 294M / trainable 208M (v11 base 동일) / teacher 86M / Forward(b=8) → 4 loss 모두 정상 / EMA + DINO center update 풀스텝 검증.

### 2026-05-03 BC-T 2차 ckpt 로컬 sanity (cluster cost 0)

**활동**: 4차(use_joint fix) 학습 종료된 5 encoder ckpt를 로컬 H100 워크스테이션으로 일괄 전송 (`bct_usejoint_best.tar` 2.2 GB). LIBERO closed-loop rollout sanity (1 task × 5 trial × 5 encoder).

**결과**: vc1 80% / siglip 60% / dinov2 40% / videomae-ours 0% / two-stream-v11 0%

**추가 발견 (rollout 코드)**: 5개 어댑터 모두 `self.prev_obs` cache cross-camera 오염 — `agentview_rgb` + `eye_in_hand_rgb`가 동일 어댑터 인스턴스 공유. fix: `src/eval_libero.py` `BCTransformerClient` 재설계 (latent_queue 폐기 + raw obs history 누적). 추가 fix 4건 (env_resolution 256→128, joint_states shape_meta, v11 adapter checkpoint=None 허용, dummy wait obs 누적).

**ours 0% 진단**: encoder representation 정상 (centered cos / PCA r1 baseline과 동급). BC fit이 가장 정확 (`‖p−r‖` 최소) → overfit 의심. test init state는 학습 demo init과 다름 (검증 완료) → BC generalization 격차.

**클러스터 후속 작업** (V3 본 학습): augmentation + multi-seed 동시 적용. 2-frame pair adapter augmentation 일관성 시각 검증 필수. 자세한 cfg 변경 + 검증 절차: [`docs/archive/refactor_plan_2026-05-03.md`](archive/refactor_plan_2026-05-03.md) §3, [`docs/RESEARCH_PLAN.md`](RESEARCH_PLAN.md) Phase 3-1 V3 § + git commit `4d4f89c` (rollout fix) 참조. 진단 narrative 원본은 [`docs/archive/PHASE3_BCT_DEBUG_2026-05-03.md`](archive/PHASE3_BCT_DEBUG_2026-05-03.md).

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
