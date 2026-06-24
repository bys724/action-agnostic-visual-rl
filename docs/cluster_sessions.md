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

### 2026-06-22 no-M ablation (§11) — motion routing 기여 격리

**배경**: Parvo BC-T 0.785 ≈ v15-ptptk(현상유지). §11.4 "motion routing이 LIBERO control 표현에 load-bearing이냐" 판정 = **같은 코드에서 routing만 끈 인코더 학습 후 동일 P-only BC-T 비교**. 효율형 `--v15-no-motion` 플래그 구현(`_forward_pair`에서 M/JEPA 분기 skip + M stream·p_motion_decoder 동결 → dead branch ~절반 제거, teacher_p full forward 등 제거. DDP-safe smoke 통과: 192 trainable 전부 grad 수신). encoder A(Run B-2)와 **환경 완전 동일** + no_motion만 추가.

⚠️ **part1 단독 확정**: A scratch(35788399)·continuation(36045098) 로그 확인 = **part1만**(46234 vid) 학습. 기존 doc "part1-5"는 오류(Run A·Parvo main 포함 최근 Parvo 계열 전부 part1 개발 런). B도 part1로 맞춤 = valid control. **논문용 part1-5 full 학습(full Parvo + no-M)은 별도 후속 결정**(현재는 개발 단계 비교).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36126892 | AIP_long 2×4 H100 | 18:00:00 | **no-M scratch (20ep, part1)** — A scratch(35788399) env 완전 동일 + `V15_NO_MOTION=1`. ep20 후 continuation 예정 | ✅ COMPLETED 20ep, 5h27m (~43.7 GPU·h), 06-22 15:59. ckpt `two_stream_v15b/20260622_103242/checkpoint_epoch0020.pt` |
| 36128321 | AIP_long 2×4 H100 | 18:00:00 | **no-M continuation (+30ep, part1)** — Parvo runB2cont(36045098) 정확 미러 + `V15_NO_MOTION=1`. init_from no-M ep20, EPOCHS=30, LR=1e-4, masked_anchor+λ_m_jepa=0+gate0+warmup5, pair, no-sobel, `CHECKPOINT_SUFFIX=noM_cont`. → Parvo와 동일 누적 50ep 매칭. 완료 후 BC-T(libero_object) 제출 예정 | ✅ COMPLETED 30ep, 7h55m (~63.3 GPU·h), 06-23 01:22. ckpt `two_stream_v15b_noM_cont/20260622_172841/checkpoint_epoch0030.pt`. ep30 recon 건강(L_t/tk≈0.028), P CLS std_p=0.090(Parvo 0.008보다 덜 붕괴=순수 MAE 정상). → BC-T(libero_object) 제출 대기 |
| 36130509~511 | AIP 1×1 H100 ×3 | 2-00:00:00 | **no-M BC-T (libero_object × seed 0/1/2)** — Parvo BC-T(36053625~627) 정확 미러 + ckpt만 no-M ep30(`two_stream_v15b_noM_cont/.../epoch0030.pt`)으로 교체. ENCODER=parvo-ptptk(P_t⊕P_tk, strict=False), V3(use_joint+aug)·50ep, SUFFIX=noM. **목적**: §11 M 기여 격리 — no-M vs Parvo(object 0.885) epoch-matched 비교. Parvo 최고 suite만(연산 절감) | 🔵 RUNNING (509=s0, 510=s1, 511=s2) |
| 36130526~529 | AIP 1×2 H100 ×4 | 00:30:00 | **신규 baseline sanity** (1ep, 64vid, 2-GPU=DDP unused-param 검출). 526=SiamMAE-base, 527=SiamMAE-small, 528=§9 pixel-pred base, 529=§9 pixel-pred small(ViT-S P=384/6) | ✅ COMPLETED 전부 통과(각 ~1m). **DDP-safe**(unused-param 0). params(trainable): SiamMAE 120M/56M, §9 158M/40M(base/small). §9 L_t/tk/pred 전부 계산·L_mj=0. ⚠️ §9 frozen 207M(teacher EMA+interpreter) 메모리 낭비 — 본학습 전 del 검토 **(✅ 2026-06-23 완료, commit 42db84c: teacher/interpreter_1/M-decoder del)** |
| 36137148 | AIP 1×1 H100 | 02:00:00 | **MCP-MAE-S sanity** (§9 pixel-pred, ViT-S 384/6·m_depth6, no-Sobel, pair, part1 MV200, 5ep, batch32, SUFFIX=sanity_mcp_s). STEP 1 본학습 전 검증 | ✅ COMPLETED 5m54s (~0.1 GPU·h), 06-23 19:03. **healthy**: L_t/L_tk/L_pred≈0.025 균형(λ=1 적정)·recon no-M(0.028)급. CLS collapse(cos_intra_p=1.0)=정상(cls_p 학습압력0→patch로 판단). **254.9 samp/s/GPU @batch32, COMPUTE-bound** → full part1 50ep ≈ **252 GPU·h**(3×predict 오버헤드, no-M 대비 2.3×). ⚠️ batch32=ViT-S GPU 저활용 → **batch↑(128+) 권장**(throughput·비용). launcher 4GPU·36h 부족(≈63h wall) → 8GPU/2노드 또는 batch↑ 필요 |
| 36138625/628 | mig-3g.40gb→AIP | 00:20:00 | **MCP-MAE-S recon 가시화** (sanity ep5) | ❌ 둘 다 CANCELLED(MIG·AIP 큐 혼잡) → **로그인 노드 CPU로 실행**(viz는 소규모). 결과: nomask는 평균색(OOD 아티팩트)이나 **masked 복원 std=0.156(vs target 0.20)=학습 정상**. PNG `paper_artifacts/mcp_mae_sanity_recon/` |
| 36138706→36138720 | normal V100 1×1 | 00:30:00 | **MCP-MAE-S viz v2 (V100)** — 8열(frame_t/t+k + recover_t·recover_t+k·predict_t+k 각 **mask/nomask**) × **seen(EgoDex part1)/unseen(DROID cross-domain)** 2샘플씩. 706=part4(취소)→720=DROID(잡이라 95k 스캔 감당). `sanity_ep5_droid.png` | 🔵 RUNNING(720) |
| 36138668 | AIP_long 2×4 H100 | 1-12:00:00 | **MCP-MAE-S 풀 학습 (STEP 1, ours/gate)** — §9 pixel-pred·ViT-S(384/6·m6)·no-Sobel·pair·**part1·50ep**. batch **64**/GPU(eff 512; sanity서 batch32 GPU 저활용 관찰→2×)·LR **4e-4**(linear 2×). SUFFIX=step1_mcp_mae_s. 8GPU로 ~16h 예상. gate=첫 run health(collapse 없이 patch+CLS R²)+slope vs VideoMAE(−0.083) | 🔵 RUNNING |

### 2026-06-18 Parvo LIBERO BC-T (현재 모델 control 평가)

**배경**: EgoDex 선형 probe(변화 인지)에선 VideoMAE +0.47 > Parvo +0.29지만, **control(LIBERO BC-T)에서 two-stream이 VideoMAE 압도**(legacy v15-ptptk 0.63/0.86/0.84 vs VideoMAE 0.22/0.24/0.42) = readout↔control dissociation. → 현재 Parvo(Run B-2 cont ep30)로 BC-T 직접 측정. **CLS std/cos로 붕괴 판단 안 함**(cls_p 학습압력 0, patch-level healthy = 유효 지표).

**신규 코드**: Parvo 전용 BC-T 어댑터 [src/encoders/adapters/parvo_pt_ptk.py](../src/encoders/adapters/parvo_pt_ptk.py) (`parvo-ptptk`). 기존 v15-ptptk 어댑터는 v11(Sobel 5ch) instantiate라 Parvo(no-Sobel 3ch) 부적합 → `TwoStreamV15Model(use_sobel=False, pair_mode=True)` + `_encode_p_unmasked`. base.py dispatch + finetune choices 등록.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36053580/603/621 | mig-1g.10gb | 00:10:00 | **어댑터 parity sanity** (3차 시도: /tmp→GPFS, PYTHONPATH fix) | ✅ 36053621 COMPLETED 22s. **PARITY OK**: compute_p_channel [0,1] RGB 3ch passthrough(no-Sobel, 학습 입력 일치), forward (B,T,1536) 정상 |
| 36053622~630 | AIP 1×1 H100 ×9 | 2-00:00:00 | **Parvo BC-T 매트릭스** (parvo-ptptk × {spatial,object,goal} × seed{0,1,2}, ckpt=parvo_runB2cont ep30, V3 cfg=use_joint+aug, 50ep). vs baseline siglip .80/.91/.86 / VideoMAE .22/.24/.42 | 🔵 RUNNING. 622-624 spatial, 625-627 object, 628-630 goal (seed 0/1/2) |
| 36055271 | normal(V100) | 00:20:00 | **VideoMAE 도메인 의존성 viz** (신규 [visualize_videomae_recon.py](../scripts/eval/visualize_videomae_recon.py)): SEEN(EgoDex) vs UNSEEN(DROID) 2-frame 복원. masked recon(학습 mask 0.5) + full recon, GT 통계 de-normalize(normalize_target=True). epoch_030_pair 구도 | ✅ COMPLETED 1m28s. `videomae_recon_seen_vs_unseen_ep49.png`. (full recon은 디코더 OOD=mask-token만 학습 → 깨짐 정상) |
| 36115626 | normal(V100) | 00:20:00 | **Parvo viz에 self-pair 라우팅 열 추가** (runB2cont ep30, no-sobel+masked-anchor). `Motion x→x (self-pair, M(x,x)≈0)` = 단일프레임 inference 모드. ⚠️ M엔 OOD(학습은 실제 motion). "정지 관찰을 라우팅이 어떻게 변환하나" 진단 | ✅ COMPLETED 1m30s. `epoch_030_pair.png` 갱신(9열, 우측 끝 self-pair). 전체 키 매칭 |

### 2026-06-16 Run B-2 continuation — baseline 성숙 (init_from ep20 +30ep)

**배경**: B-2 ep20 probing P R²=0.30/M 0.28(양수, patch healthy) → 가설 A("masked MAE concat baseline 쓸만") 지지. 더 학습 시 R² 오르는지 검증. **resume_from은 ckpt scheduler(T_max=15 이미 끝=LR≈0) 복원해 ep21+ 학습 안 됨** → `init_from`(가중치만+fresh LR)로 우회.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| ~~35956105~~ | AIP_long (요청 gtx1080ti×8) | 18:00:00 | **B-2 continuation** 1차 제출 | ❌ CANCELLED. **GRES 오지정** — AIP_long엔 H100만 있는데 `gres/gpu:gtx1080ti=8` 요청(클러스터엔 jepyc 파티션 2장뿐) → 영원히 매칭 불가(Reason=None). 36045098로 재제출 |
| 36045098 | AIP_long 2×4 H100 | 18:00:00 | **B-2 continuation 재제출** (init_from ep20=`parvo_runB2/20260615_212918/checkpoint_epoch0020.pt`, EPOCHS=30, LR=1e-4 gentle, masked_anchor+L_m_jepa=0 동일, gate=0+warmup5, pair-mode, no-sobel, SUFFIX=runB2cont) | ✅ COMPLETED 11h12m (**~89.6 GPU·h** = 8 GPU × 11.2h), 06-18 07:25 종료. **P 붕괴 지속**: ep30 std_p≈0.008, cos_intra_p→1.000, cos(pred,tgt)→1.000 (M·DINO는 std≈0.19 정상). anti-collapse 0이라 예상 범위 — continuation이 ep20 붕괴 basin 못 벗어남(CLS 레벨; patch는 별개, probing으로 확정). 누적 50ep(20+30, 단 fresh LR 2구간) |
| 36052423 | mig-1g.10gb | 00:20:00 | **ep30(누적 50ep) viz** (dependency=afterok:36045098, no-sobel + masked-anchor, OUT_DIR=parvo_runB2cont_recon_samples). 최종 모델 reconstruction 가시화 | 🔵 RUNNING (학습 완주 즉시 자동 실행됨, olaf-g009) |
| ~~36052433~~ | AIP 1×1 H100 | 06:00:00 | ~~ep30 EgoDex probing (part4 full)~~ | ❌ CANCELLED (06-18). **split parity 오류 발견** — baseline(v15 +0.39, VideoMAE, v11 등) 전부 `test` split(3243 vid, 180921/40914 샘플)에서 측정됐는데 이 잡은 `part4`(44129 vid, **3.34M/827K 샘플** ~18배)로 제출됨. `test`≠`part4`(별도 공식 split). 라인 "full part4라 직접 비교 가능"은 **오류** → 36052612로 재제출 |
| 36052612 | AIP 1×1 H100 | 03:00:00 | **ep30 EgoDex probing 재제출** (SPLIT=`test` = baseline 정규 프로토콜 정확 일치: 180921/40914, gap=10, 20ep, batch 256, p_t_p_tk, target_mode=same) | ✅ COMPLETED 15m24s (~0.26 GPU·h). **R²=+0.2884** (best-ep). **input-only baseline VideoMAE +0.4705에 못 미침** — Run B-2 붕괴와 일관(ep20 MIG +0.30 → continuation 미개선). scaffold 미지지, "Parvo(붕괴본) < multi-frame MAE concat baseline" |
| 36052991 | AIP 1×1 H100 | 03:00:00 | **ep30 EgoDex probing — P_t⊕M mode** (신규 cls_mode `patch_mean_concat_p_t_m` = P(frame_t) appearance ⊕ M(t,tk) motion, 1536-d. p_t_p_tk가 우회한 **M stream 직접 평가**. test split 동일 프로토콜) | ✅ COMPLETED 15m9s (~0.25 GPU·h). **R²=+0.1051** (best-ep). **동일 full-test에서 P_t⊕P_tk(+0.2884)보다 낮음** → 2번째 component로 M(motion 인코딩)이 P(tk)(2번째 appearance)보다 action 정보 적음. M stream이 raw 2-frame appearance보다 약함(붕괴본 한정). MIG M-only +0.28은 small-sample(MV300) 낙관 의심 |
| 35956537/540/541 | mig-1g.10gb ×3 | 00:20:00 | **B-2 viz 재생성** (ep2/11/20, `--masked-anchor`: decode_first 일치 + col5에 "masked t→라우팅→t+k 예측" 컬럼 신설) | ⏳ RUNNING |
| 35956542 | mig-1g.10gb | 02:00:00 | **routed_tk probe** (ep20 part4 gap10 MV300). frame_t→라우팅→추론 t+k 임베딩 patch_mean(mean only). vs P_t⊕P_tk(0.30) — 라우팅이 action 정보 담나 | ⏳ RUNNING |

### 2026-06-15 Run B-2 — masked anchor (마스킹만으로 붕괴 해결 검증)

**배경**: Run A(full anchor + variance reg λ=1.0) 붕괴(ep2 cos_intra_p→1.0). penalty가 강한 attractor에 짐 → **task-structural 해법 직행**. 가설(사용자) = anti-collapse 알고리즘 *없이* masking만으로 상수 read-off를 약화시킬 수 있나. 설계 = [v15b_retraining_status.md](v15b_retraining_status.md) §8.

**설계 (커밋 예정)**: V-JEPA P를 MAE 포맷으로 통일 — ① anchor = student VISIBLE frame_t(MAE p_t_visible 재사용) + mask token 주입 → ② **interp(self-attn=완성)→routing(모션)** 순서(decode_first) → ③ **masked 위치에서만** teacher frame_tk 예측 loss. ④ M은 conditioning oracle이라 full routing 유지하되 **L_m_jepa 폐기**(λ_m_jepa=0, M 일관성). anti-collapse 알고리즘(variance reg/target_ln) **0**. 코드: `_vjepa_p_masked`, `RoutingInterpreterStep(decode_first)`, `--v15-masked-anchor`. CPU smoke 통과.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 35788399 | AIP_long 2×4 H100 | 18:00:00 | **Run B-2 본학습 (20ep)** (parvo pair scratch, masked_anchor + L_m_jepa=0, reg 없음, gate=0+warmup5, batch 64, **part1**(로그 확인, 기존 "part1-5"는 오류), SUFFIX=runB2) | ❌ COMPLETED 20ep 7h29m (**~60 GPU·h**) but **붕괴 — 마스킹만으론 불충분**. 궤적: **ep1-2 Run A보다 건강**(ep2 std_p=0.261/cos_intra_p=0.877, Run A는 ep2 0.039/0.999) → ep3 붕괴 → **ep4-7 부분 회복**(std_p 0.05-0.075, 진동) → ep12+ 완전 붕괴(std_p 0.006, cos_intra_p=1.0, cos(pred,tgt)=0.999). **마스킹이 basin을 *완화·지연*했으나 상수 attractor가 결국 승**(예측대로 "minimum 미제거"). M(std_m 0.18)·DDP(λ_mj=0)는 정상. **사고: ep4 abort 미설정**(21:28 야간 시작, 무감시 완주) → sbatch는 Monitor/wakeup 능동 폴링 필요. 다음 = B-1(masked + *타게팅 수정*된 reg) |

| 35926031/032 | mig-1g.10gb ×2 | — | Run B-2 probing 1차 | ❌ CANCELLED. parvo probe 경로 작동 확인(embed_dim 1536, 가중치 로드 OK) but **max_videos 미설정 → part4 전체 4.2M 샘플 폭증**(MIG 2h 내 불가). probe sbatch에 MAX_VIDEOS 인자 추가 후 재제출 |
| 35948313/314 | mig-1g.10gb ×2 | 02:00:00 | **Run B-2 probing 재제출** (MAX_VIDEOS=300, ep20 part4 gap10). 313=P, 314=M | ✅ COMPLETED 2m9s/1m28s. **P R²=0.30 / M R²=0.28 (둘 다 양수)** → **patch 표현 healthy**(CLS만 붕괴, scaffold trivial). unmatched 맥락: v11 +0.29 / v15 +0.39 / VideoMAE +0.47 → v11급. 가설 A(masked MAE concat baseline) 지지 → continuation(35956105) |
| 35924392/554/555 | normal(V100) 1×1 ×3 | 00:20:00 | **Run B-2 가시화 3시점** (collapse 진행 시각화, NO_SOBEL=1, OUT_DIR=parvo_runB2_recon_samples). 392=ep20(붕괴 std_p 0.006), 554=ep2(건강 std_p 0.26=peak, best_model.pt은 eval loss 함정이라 제외), 555=ep11(중간) | ⏳ 제출 (전부 PENDING, V100 자원 대기) |

### 2026-06-15 Run A — target 정규화 anti-collapse (붕괴 정공법 전환)

**배경**: SSIM/LR/warmup 5연패(35757465·35760276·35760399·35760680) 종결. 붕괴 근인 재진단 = V-JEPA P **자기참조 constant collapse**(target=EMA student 거울 → student 저분산화하면 target도 따라 무너져 loss→0). recon 노브는 증상만 건드림. → **target 정규화**(① target LayerNorm ② VICReg variance reg)로 전환. masked anchor(scaffold용)는 별개 영역이라 **Run B로 분리**(단일변수 격리). 상세 = [v15b_retraining_status.md](v15b_retraining_status.md) §8.

**코드**: `lambda_var`(VICReg, P enc 출력 per-dim std<1 hinge) + `target_ln`(I-JEPA식 target LayerNorm) 신규 플래그 (기본 off=기존 동작). [two_stream_v15.py](../src/models/two_stream_v15.py) `_variance_loss`/`_vjepa_p_one_segment`/`_forward_pair`, [pretrain.py](../scripts/pretrain.py) `--v15-lambda-var`/`--v15-target-ln`, sanity sbatch env wiring. CPU 소형 smoke 통과(forward/backward·loss finite·variance grad→P enc norm 5.45).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 35763563 | AIP 1×1 H100 | 02:00:00 | **Run A sanity** (parvo pair, gate=0, V15_LAMBDA_VAR=1.0 + target_ln, 50vid×3ep, SUFFIX=runA). 체크: DDP/GPU 무크래시·NaN 없음·loss 스케일·std 회복 | ✅ COMPLETED 3m13s (~0.05 GPU·h). **붕괴 궤적 역전 확인**: std_p 0.58→**1.0**(옛 붕괴는 →0, variance reg가 target=1로 끌어올림), cos_intra_p 0.60→**0.165**(P 다양화), cos(pred,tgt) 0.57~0.84(0.99 미saturate), NaN 없음. ⚠️ std_m 0.05(M엔 reg 없음, 옛 transient 패턴). 3ep라 증명 아님 but 궤적 명백 역전 → 통과 |
| 35764680 | AIP_long 2×4 H100 | 18:00:00 | **Run A 본학습 (cap 20ep)** (parvo pair scratch, V15_LAMBDA_VAR=1.0+target_ln, gate=0+warmup5, batch 64, lr 2e-4, **part1**(로그 확인, 기존 "part1-5"는 오류), SUFFIX=runA) | ❌ CANCELLED 2h19m (~18.6 GPU·h, ep5). **붕괴 재현 — variance reg(λ=1.0) 실패**: ep2부터 cos_intra_p→1.0, std_p→0.02, cos(pred,tgt)→0.995. **sanity(35763563)의 std_p 1.0은 오판** — 18 step뿐이라 붕괴 발현 전. 본학습은 ep1에 9030 step → 강한 collapse attractor가 λ_var=1.0 압도. 진단: λ_var 과소 or directional collapse(cos_intra→1, magnitude varies라 per-dim variance reg 미발화) → λ_var↑(VICReg=25) 또는 normalize-then-var 타게팅 수정, 혹은 Run B(task-structural)로 직행 |

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
| 35760680 | AIP_long 2×4 H100 | 2-00:00:00 | **🔵 LR warmup 연장 본학습** (scratch, **LR warmup 4→10ep**(ep10까지 천천히 증가, 급가속 충격 완화), LR peak 1e-4·lambda_ssim 0.02·gate5 동일=변수 격리). 코드: `--lr-warmup-epochs` 인자 신설([pretrain.py](../src/training/pretrain.py#L1186)). ckpt `parvo_pair_wu10/` | ❌ CANCELLED 06-15 (~20h13m, ~161 GPU·h, ep35+). warmup 연장도 collapse 미방지(35763305 viz 확정). **SSIM/LR/warmup 5연패 종결** → 진단 전환: 붕괴 근인 = V-JEPA P 자기참조(teacher=EMA student) constant collapse. recon 노브가 아니라 **target 정규화(variance reg + target LayerNorm)**가 정공법 → Run A로 전환 |
| 35762739 | AIP 1×1 H100 | 00:20:00 | parvo_pair_wu10 가시화 (AIP 풀 H100) | ❌ CANCELLED (PENDING-Resources). 가시화=추론이라 풀 H100 불필요 → 최소사양으로 재제출 |
| 35762741 | mig-1g.10gb 1×1 | 00:20:00 | parvo_pair_wu10 가시화 (MIG 시도) | ❌ CANCELLED. MIG-1g 슬라이스 28개 전부 점유 → 예상 시작 **~2일 뒤**. V100으로 전환 |
| 35763304 | normal(V100) 1×1 | 00:20:00 | parvo_pair_wu10 가시화 (V100, sbatch module load) | ❌ FAILED 1s. **V100 노드엔 `module load miniforge3` 모듈 없음** → conda env GPFS python 직접 호출로 회피 |
| 35763305 | normal(V100) 1×1 | 00:20:00 | **parvo_pair_wu10 본학습 가시화** (latest.pt=ep35, NO_SOBEL=1, pair_mode, env python 직접호출 --wrap). 핵심 판독=warmup 연장(4→10ep)이 collapse 막는지 | ✅ COMPLETED 1m37s (~0.03 GPU·h, V100). bf16 V100 정상. **판독 = collapse 재현**: nomask 복구(col5-6)·motion routing(col7) **전부 균일 남보라**(완전 붕괴, V-JEPA P trivial=M기여0). 75%-masked 복구만 보이는 패치 부분복원. ep3 GT급→ep35 균일 = **warmup 연장도 collapse 미방지**. 산출물 `paper_artifacts/parvo_pair_wu10_recon_samples/epoch_035_pair.png`. **이후 viz/추론 = V100(normal) + env python 직접호출이 기본**(viz_v15.sbatch 변경, 단가 최저+최다가용) |

**코드 변경 (커밋)**: ① cleanup(b6d912e): 루트 tar 69GB + deprecated v12-14 스크립트 12개 삭제. ② rename(9ce778d): `MODEL=parvo`→내부 v15b 정규화 + ckpt `parvo/` (최소 rename, 코드 식별자 별도). ③ no-Sobel(9ce778d): `use_sobel` 플래그 — Paper 2(Parvo)는 P=RGB(3ch)/M=ΔL(1ch), Sobel은 Paper 1 주제라 격리. M depth 6 유지(1ch≠작은모델). ④ 진단계측(cd23ee8): `PRETRAIN_DIAG=1`로 첫 batch 입력통계+data/compute 병목분리, NaN guard 상시.

검증: CPU forward 양 모드 통과(sobel True 5/3ch, False 3/1ch).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 35550604 | AIP 1×1 H100 | 02:00:00 | **Parvo no-Sobel 진단 sanity** (gate=0, 50vid×3ep, PRETRAIN_DIAG=1) | ✅ COMPLETED 5m15s (0.09 GPU·h). **입력 parity OK**(P=RGB 3ch [0,1], M=ΔL 1ch [-0.7,0.7]). loss 스케일 Sobel판과 동등(L_t 0.047→0.019). **병목 = COMPUTE-bound 99%, data-load ≤1%** (3-frame 추출 병목 아님 확정). NaN 없음 |
| 35560151 | AIP_long 2×4 H100 | 5-00:00:00 | **Parvo 본학습** (MODEL=parvo = v15b+no-Sobel, gate=10, EMA 0.996, 50ep × **part1**(로그 확인 35560151, 기존 "part1-5"는 오류), lr 2e-4). Paper 2 scaffold 검증. ckpt → checkpoints/parvo/ | PENDING. abort: ep12-18 trivial collapse(cos>0.99&L_pred<0.01)/발산 |

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

---

## 이전 월 archive

| 월 | Archive 위치 | 핵심 산출물 |
|----|------------|------------|
| 2026-04 | [`docs/archive/cluster_sessions_2026-04.md`](archive/cluster_sessions_2026-04.md) | v4/v6/v10/v11 사전학습 + BC-T 1~4차 + Phase 2.5 value alignment + Phase 2 보강 LIBERO probing |
| 2026-05 | [`docs/archive/cluster_sessions_2026-05.md`](archive/cluster_sessions_2026-05.md) | v15 V-from-M ablation(C1) + C10 CALVIN + C12 view-sensitivity + v13/v14 + V3 BC-T 매트릭스 + representation viz prototype |

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
