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
| **34288968** | AIP_long 2×4 H100 | 10-00:00:00 | **v15 본 학습** (50ep, EgoDex part1-5, batch 32/GPU = global 256, num_workers=8, λ all 1.0 with warmup 10ep 0.01→1.0, composition_mode=linear_residual, EMA 0.999→0.9999, mask_p=0.75, mask_m_jepa=0.5, max_gap=30 sample_center=15) | **RUNNING** 2026-05-11 04:52 시작. 6h25m 시점에 ep4 ckpt 저장 (~85min/ep, 50ep 추정 ~71h 안전) |
| 34343947 | AIP 1×1 H100 | 00:20:00 | v15 ep4 viz (nomask P MAE × 3 frame + motion routing × 3 segment + L_compose path, 10-column 4-sample figure) | ✅ COMPLETED 58s. `paper_artifacts/v15_main_train_samples/epoch_004_nomask.png`. GT(1-3) + P MAE recon(4-6) + motion routing short/step/long(7-9) + composition(10). ep4 단계라 col 4-10 모두 patch grid speckle만 보임 — v14 ep4와 동일 (recon decoder 미숙). ep8+ 재실행 권장 |
| 34344317/319/326 | AIP 1×1 H100 | 00:20:00 ×3 | v15 ep4 viz EgoDex+DROID 확장 디버그 (3 회) | ❌ 1차 sbatch `--num-samples` 폐기 인자, 2차 `DROIDDataset.__init__` sample_dist 미지원, 3차는 잡 30s 완료지만 DROID row 검정 fallback (ext1 95658 ep 중 단 10개만 frame 추출됨, 빈 ep retry 5회 fallback) |
| 34344358 | AIP 1×1 H100 | 00:20:00 | v15 ep4 viz EgoDex+DROID 확장 4차 (max_videos=10) | ✅ COMPLETED 41s. `paper_artifacts/v15_main_train_samples/epoch_004_nomask.png` 4 row (EgoDex 2 + DROID 2) × 10 col. ep4 단계라 col 4-10 모두 patch grid speckle, v14 ep4 viz와 동일 양상. ep8+ 재실행 권장 |

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
