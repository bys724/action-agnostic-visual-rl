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

**Cross-folder OOD (paper §C10 main, gap=30 = 1.0s)**:
| Encoder | training-folder train + validation-folder eval R² |
|---------|------------|
| **dinov2** | **+0.535** ★ |
| siglip | +0.345 |
| vc1 | +0.242 |
| videomae-ours | +0.180 |
| **v15** | **+0.105** (5위, drop −66% vs in-dist self-contained 0.310) |

**핵심 발견**: cross-folder OOD에서도 v15가 baseline 대비 명확히 약함. dinov2가 OOD에서도 가장 robust. siglip은 in-dist 0.769 → OOD 0.345 (drop −0.424) — vision-language SSL의 generalization 약점. **paper §4 ¶2 (iii) limitation 보고**:
- v15 weakness on CALVIN cross-environment OOD probing
- 대조적으로 **CortexBench MetaWorld BC 1위** (commit `24eda1a`, v15 89.87 vs siglip 88.8) — v15 main claim 직접 지지

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
