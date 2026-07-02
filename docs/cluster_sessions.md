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

### 2026-07-01 STEP 0 value 게이트 — CoMP-MAE-S/VideoMAE OOD probing (CALVIN·LIBERO)

**목적**(restart_plan §3.1): CoMP-MAE-S를 STEP 1(대규모 학습) 전에 OOD probe → same-corpus slope(ours vs VideoMAE −0.083) + OOD-motion 절대값으로 value 현상 판정. **신규 배선**: `probe_action_libero.py`(공유 base)에 parvo(CoMP-MAE) 로더 + VideoMAE-VLA 토큰 로더 + `AttentivePoolProbe` + `--readout {mean,attentive}`·`--parvo-mode {p_t_p_tk,p_t_m}`·`--videomae-encoder {adapter,vla}` (CALVIN/LIBERO 상속). CPU smoke 통과(mean 768·attentive (392,384) fp16). readout parity: EgoDex in-domain = patch_mean concat P_t⊕P_tk **+0.236**/M patch_mean **+0.094**. attentive는 comp+videomae 2개만 배선(frozen baseline 후속). CALVIN eval=32,183 pairs(ViT-S 384d attentive ~10GB fp16 → 노드 안전, EgoDex 180k OOM과 무관).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36222791 | AIP 1×1 H100 | 00:40:00 | **CALVIN sanity** (comp attentive `p_t_m`, cross-folder, MAX_EPISODES=10·gap30) — 데이터 로더·preprocess parity·attentive 메모리 검증 | ✅ COMPLETED 4m04s, MaxRSS 23.3GB. R²=−0.10(10ep train 언더핏, 파이프라인 PASS). readout=attentive n_streams=2 정확. |
| 36222792 | AIP 1×1 H100 | 00:40:00 | **LIBERO sanity** (comp attentive `p_t_m`, libero_spatial·task0·5demos·gap20) | ✅ COMPLETED 38s. R²=+0.12(5demo). 파이프라인 PASS. |

**본 매트릭스 12잡** (sanity PASS 후 제출) — comp{mean,attentive}×{p_t_p_tk,p_t_m} + videomae-vla{mean,attentive}, GAPS default(CALVIN 10/20/30/45·MAX_EPISODES=200 xfolder, LIBERO_spatial 1/13/20/40). 게이트 판독 = slope_comp − slope_VideoMAE(mean, gap30/20) + mean→attentive crossover.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36224980/981 | AIP 1×1 H100 ×2 | 02:00:00 | **CALVIN comp mean** — 980=`p_t_p_tk` / 981=`p_t_m` | ✅ 13m. CALVIN pos R²(gap30): ptptk **0.257** / ptm **0.405** |
| 36224982/983 | AIP 1×1 H100 ×2 | 02:00:00 | **CALVIN comp attentive** — 982=`p_t_p_tk` / 983=`p_t_m` | ✅ 14m. CALVIN pos: ptptk **0.175** / ptm **0.486**(attn>mean=M국소) |
| 36224984/985 | AIP 1×1 H100 ×2 | 02:00:00 | **CALVIN videomae-vla** — 984=mean / 985=attentive (self-consistent) | 984 ✅ pos **0.529**. 985 ❌ **OOM**(768d 토큰 `torch.cat` 스파이크>63GB) → mem=120G 재제출 **36225602** |
| 36224986/987 | AIP 1×1 H100 ×2 | 01:30:00 | **LIBERO_spatial comp mean** — 986=`p_t_p_tk` / 987=`p_t_m` | ✅ 5m. LIBERO pos R²(gap20): ptptk **0.741** / ptm **0.809** |
| 36224988/989 | AIP 1×1 H100 ×2 | 01:30:00 | **LIBERO_spatial comp attentive** — 988=`p_t_p_tk` / 989=`p_t_m` | ✅ 7m. LIBERO pos: ptptk **0.766** / ptm **0.814** |
| 36224990/991 | AIP 1×1 H100 ×2 | 01:30:00 | **LIBERO_spatial videomae-vla** — 990=mean / 991=attentive | ✅ 11/21m. LIBERO pos: mean **0.853** / attn **0.879** |
| 36225602 | AIP 1×1 H100 (mem120G) | 02:00:00 | **CALVIN vmae attentive 재제출**(985 OOM 수정) | ✅ COMPLETED 21m35s. CALVIN pos **0.610**(mean 0.529). mem120G로 해결. |
| 36225603 | AIP 1×1 H100 | 03:00:00 | **VideoMAE attentive in-domain EgoDex**(`attentive_concat_p_t_p_tk`) — attentive slope-diff에 필요 | ❌ **OUT_OF_MEMORY** MaxRSS **545GB**(768d × full EgoDex 180k 토큰). = 문서화된 "B(768d) disk-backed 필요" 케이스. → **attentive slope-diff 유보**(comp 384d는 통과, VideoMAE in-domain attentive만 blocked). 후속 = pre-alloc/disk-backed 캐시. |

### 2026-07-02 STEP 0 efficiency 표 확장 — LIBERO object·goal probing

**목적**: 3b efficiency 표를 libero_spatial 외 **object·goal** suite로 확장. spatial 매트릭스(07-01 `36224986–991`)를 2개 suite로 복제 — CoMP-MAE-S{mean,attn}×{p_t_p_tk,p_t_m} + VideoMAE-vla{mean,attn}, 이 두 모델만 mean/attn 둘 다. ckpt·readout·gaps(`1 13 20 40`)·view 전부 spatial과 동일(parity, TASK_SUITE만 차이). baseline(VC-1/DINOv2/SigLIP object·goal)은 `tab2_probing/libero_all_gaps_summary.csv`에 기존재 → 재제출 불필요. ⚠️ build 스크립트 확장 시 parity 앵커는 spatial(9690) 전용이라 suite별 n_eval로 분리 필요.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36300651~654 | AIP 1×1 H100 ×4 | 01:30:00 | **LIBERO_object comp** — 651 mean_ptptk / 652 mean_ptm / 653 attn_ptptk / 654 attn_ptm | ✅ 5~9m. object pos R²(gap20, n_eval=12710): mean ptptk **0.811**·ptm **0.838** / attn ptptk **0.829**·ptm **0.851** |
| 36300655/656 | AIP 1×1 H100 ×2 | 01:30:00 | **LIBERO_object videomae-vla** — 655 mean / 656 attn | 655 ✅ 12m pos **0.867**. 656 ❌ **OOM**(768d attentive 스파이크, CALVIN 985와 동일) → mem120G 재제출 **36303903** |
| 36300657~660 | AIP 1×1 H100 ×4 | 01:30:00 | **LIBERO_goal comp** — 657 mean_ptptk / 658 mean_ptm / 659 attn_ptptk / 660 attn_ptm | ✅ 5~9m. goal pos R²(gap20, n_eval=11100): mean ptptk **0.702**·ptm **0.708** / attn ptptk **0.709**·ptm **0.751** |
| 36300661/662 | AIP 1×1 H100 ×2 | 01:30:00 | **LIBERO_goal videomae-vla** — 661 mean / 662 attn | ✅ 10/16m. goal pos: mean **0.791** / attn **0.830** |
| 36303903 | AIP 1×1 H100 (mem120G) | 01:30:00 | **LIBERO_object vmae attentive 재제출**(656 OOM 수정) | ✅ COMPLETED 19m43s. object attn pos **0.903**(mean 0.867). mem120G로 해결(CALVIN 985→602와 동일 패턴). |

**STEP 0 게이트 판독 (mean slope-diff 완결 / attentive slope-diff 유보=vmae in-domain OOM)**:
- attentive OOD 절대값(참고): CALVIN pos comp ptm **0.486** vs vmae **0.610** / LIBERO comp ptm **0.814** vs vmae **0.879** (comp 40M < vmae 86M, 효율 신호 유지). attentive slope-diff는 VideoMAE in-domain EgoDex attentive(545GB OOM)가 막혀 미완 — comp crossover(아래)만 확정.
- **slope-diff = slope_comp − slope_VideoMAE** (in=EgoDex 18d − OOD=CALVIN pos; diff-in-diff로 target-space confound 상쇄): **p_t_m −0.247**(comp가 M 통해 OOD서 gap 좁힘=factorization 신호) vs **p_t_p_tk +0.038**(appearance 단독=이득 없음, 정상 null). → **M stream이 OOD robustness 원천** = Paper 2 가설 정합.
- **B(OOD-motion 효율)**: comp p_t_m ≈ VideoMAE 양 벤치(LIBERO attn 0.814 vs 0.879 / CALVIN attn 0.486 vs 0.529), comp ViT-S 40M < ViT-B 86M → 효율 신호 有.
- **attentive crossover**: p_t_m서 attn>mean(CALVIN +0.081) = M 공간국소(mean under-read) 확증 / p_t_p_tk는 attn<mean = appearance holistic. 사용자 attentive 직관 검증.
- ⚠️ caveat(추정): comp p_t_m in-domain 낮음(0.099)→OOD서 회복 = regression-to-mean 일부 가능. slope-diff가 부분 통제하나 완전치 않음. attentive slope-diff(pending)로 재확인.
- **잠정 판정(2026-07-01, 후속 철회)**: value 신호(A·B) 존재로 STEP 1 지지 판단했으나 ↓서 정정.

**🔴 최종 판정 (STEP 0.5 de-confound, 2026-07-02)**:
- **slope(3a) 폐기**: ① dim-match(rightHand≈18d, −0.248 유지) + ② LIBERO(p_t_p_tk null 깨짐) + **로그 스캔 결정타**(EgoDex in-domain 300 probe 전부 ~0.47 천장) → slope는 in(0.47 어려움)/OOD(0.85 쉬움) **난이도 비대칭이 지배 = regression-to-ceiling**, motion robustness 아님. diff-in-diff도 이 스케일링 못 잡음.
- **살아남은 주장 = 3b efficiency(절대값)**: 검증·parity 통과 표 → [`paper_artifacts/tables/step0_ood_efficiency/`](../paper_artifacts/tables/step0_ood_efficiency/README.md). comp `P_t⊕M`(~32M P+M) > DINOv2/SigLIP(86M), VC-1/VideoMAE(86M)에 근접. ⚠️ **LIBERO baseline은 spatial-only여야**(3-suite 평균 금지 — 초안 오류 정정: dinov2 0.766≠0.644, siglip 0.787≠0.694).
- **다음 = ③ controlled-shift corrupt-in-place**(selectivity mechanism, 유일한 클린 테스트) → 그 후 STEP 1.

### 2026-06-30 CoMP-MAE-S STEP 1 gate — EgoDex action probing (조합 sweep)

**목적**: CoMP-MAE-S(step1 ep50)가 action 정보를 인코딩하는지 + 어떤 표현 조합이 best인지. 게이트 R² readout. **EgoDex `test` split 정규 프로토콜**(gap=10, 20ep, batch256, 180921/40914 샘플 = 과거 baseline parity). 비교: VideoMAE input-only **+0.4705** / 과거 Parvo(붕괴본) P_t⊕P_tk **+0.2884**·P_t⊕M **+0.1051**.

**코드**: `probe_action.py` parvo 로더 = ckpt에서 arch(dim/head/m_depth/comp_mae) **추론**(ViT-S 384 자동 대응, 기존 768 하드코딩 제거) + 신규 `patch_mean_p_t` 모드(P(t) 단독 = Paper1 single-frame readout). CPU smoke 통과(p_t 384·p_t_p_tk 768·m 384, finite).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36186234→36186237 | AIP 1×1 H100 | 03:00:00 | **probe P(t) only** (`patch_mean_p_t`, 384d) — 단일프레임 appearance baseline | ❌ 36186234 FAILED 11s (argparse `choices` 누락). 수정 후 **36186237** ✅ COMPLETED 8m33s. **R²=−0.009 (≈0)** — 단일프레임 appearance는 action 정보 거의 0. |
| 36186235 | AIP 1×1 H100 | 03:00:00 | **probe P(t)⊕P(t+k)** (`patch_mean_concat_p_t_p_tk`, 768d) — 2프레임 appearance | ✅ COMPLETED 14m34s. **R²=+0.236 ★ best** — 2프레임 페어가 action 정보 최다. 단 VideoMAE input-only +0.4705 미달, 과거 붕괴본 ViT-B +0.2884에도 못 미침(⚠️ size confound: 본건 ViT-S 384 vs 비교군 ViT-B 768). |
| 36186236 | AIP 1×1 H100 | 03:00:00 | **probe M(t,t+k)** (`patch_mean_m`, 384d) — motion(ΔL) 단독 | ✅ COMPLETED 14m33s. **R²=+0.094** — motion 단독은 양수(>0=motion 인코딩 有)이나 2프레임 appearance(+0.236)보다 낮음. 3잡 합 ~0.63 GPU·h. |
| 36186442 | AIP 1×1 H100 | 03:00:00 | **probe P(t)⊕M** (`patch_mean_concat_p_t_m`, 768d=384×2) — appearance(t)+motion(t,tk) 통합 readout | ✅ COMPLETED 15m44s. **R²=+0.099** ≈ M 단독(+0.094) → **M이 combo 지배**(P(t)≈0이라 가산 효과 ~0). 과거 붕괴본 +0.1051과 동급. 결론: ΔL(M)에 raw 2번째 프레임보다 action 정보 적음(P(t)⊕P(t+k) +0.236 ≫ P(t)⊕M +0.099). |

### 2026-06-30 LIBERO BC-T: CoMP-MAE-S attentive + M-stream + AMP 배선

**구현**(어댑터 `parvo_pt_ptk.py` + `finetune_libero_bct.py` + sbatch): ① **arch 추론**(ckpt→384/768, CoMP-MAE-S 대응) ② **attentive pooling**(`--pooling attentive`, stream별 learnable query, encoder frozen·query만 학습) ③ **M-stream**(`--use-m`, ΔL 현재−직전 motion → P_t⊕P_tk⊕M, M도 frozen·별도 query) ④ **속도개선 즉시2개**: AMP bf16(`--amp`, autocast)·frozen encoder no_grad. baseline 공정성 기확보(모든 image enc가 prev+curr 2프레임). ⚠️ M rollout gap=1 vs 학습 gap~15 분포차(성능 미지수, parity는 BC train/rollout 일관).

| JobID | 자원 | 목적 | 결과 |
|-------|------|------|------|
| 36189071 | AIP 1×1 H100 | smoke: CoMP-MAE-S **attentive** P-only (TASK0·10batch·1ep) | ✅ COMPLETED 1m15s. trainable 2.9M(pool_q 포함), loss 5.94→4.11/eval0.57. **AMP 없이 11s/ep** |
| 36190397 | AIP 1×1 H100 | smoke: **full-stack**(attentive+use_m+AMP) | ✅ COMPLETED 22s. trainable 3.0M, loss 5.56→3.65/eval0.65, NaN無. **6s/ep = AMP로 ~1.8×↑**(M 추가에도). 전 기능 검증 통과 |
| 36190933/935/937 | AIP 1×1 H100 ×3 | **CoMP-MAE-S BC-T P-only attentive** (libero_object **task0 단일**·50ep·AMP·aug off). seed 0/1/2. SUFFIX=ponly_attn | ✅ COMPLETED ~18분/seed. ep50 eval: s0 −19.95·s1 −19.03·s2 −20.09 → **mean −19.69** |
| 36190934/936/938 | AIP 1×1 H100 ×3 | **CoMP-MAE-S BC-T P+M attentive** (`use_m`, 동일 조건). seed 0/1/2. SUFFIX=pm_attn | ✅ COMPLETED ~23분/seed. ep50 eval: s0 −20.81·s1 −21.83·s2 −20.94 → **mean −21.19** |

**결과 (eval NLL)**: **P+M −21.19 < P-only −19.69 (−1.50, 전 seed 완전 분리** — 최악 P+M −20.81 > 최선 P-only −20.09, 겹침 0). ⚠️ **eval NLL ≠ SR** — 확정은 로컬 rollout. 단일 task·aug off = 탐색. 최종 reportable = aug-on full-suite 별도. ~~M stream robust 기여~~ → **아래 SR로 철회**.

**✅ 로컬 rollout SR (2026-07-01, task0 × 50 trials, seed=7)**: **P-only 68.7%**(s0 80·s1 62·s2 64) vs **P+M 2.0%**(s0 6·s1 0·s2 0) — **eval NLL 완전 역전**. NLL 우위 P+M이 SR **붕괴**. 원인: M=ΔL(직전프레임 변화)=직전행동 효과 → open-loop teacher-forced NLL은 컨닝으로 낮추나, closed-loop선 자기 오차 강화 피드백(**causal confusion/copycat**) + M-encoder gap OOD로 발산. baseline task0 대조: VC-1 94.0·Parvo(붕괴본) 90.0·SigLIP 86.0·DINOv2 85.3 ≫ **CoMP P-only 68.7** > VideoMAE 13.3 ≫ **CoMP P+M 2.0**. ⚠️ CoMP=aug-off·단일task vs baseline=aug-on·full-suite → 경향 지표. 상세·원인·문헌 = [comp_mae_plan.md](comp_mae_plan.md) §6.
> 🔧 rollout 배선 수정(commit): `ParvoPtPtkAdapter`가 arch를 별도 pretrain ckpt에서만 추론 → rollout(cfg.encoder.checkpoint=None)서 assert 실패. **policy_state_dict(encoder self-contained)에서 arch 추론**하도록 수정 → pretrain ckpt 없이 rollout 가능. + docker `libero-eval` 드라이버 desync(NVML) 재시작 복구.

**로컬 rollout 절차 (참고, ✅ 완료됨)** — 클러스터는 학습만, SR은 로컬 docker `libero-eval`.
1. 반출: 6× best.pt(각 226MB) → `/mnt/data/checkpoints/libero_bct/<run_dir>/best.pt` (3-hop = `docs/artifacts.md` §2a).
2. rollout: ⚠️ **단일 task 학습이라 `--task-ids 0` 필수**(suite 전체 rollout 시 미학습 task 실패). `run_libero_rollouts.sh`는 task-ids 미지원 → `src/eval_libero.py` 직접:
   ```
   for d in /mnt/data/checkpoints/libero_bct/parvo-ptptk_libero_object_seed*_*attn; do
     docker exec -e CUDA_VISIBLE_DEVICES=0 libero-eval python src/eval_libero.py \
       --checkpoint "$d/best.pt" --task-suite libero_object --task-ids 0 --num-trials 50 \
       --output-dir data/libero/results/comp_mae_s_attn --video-dir data/libero/videos/comp_mae_s_attn/$(basename $d)
   done
   ```
3. 집계: `scripts/eval/aggregate_libero_rollouts.py` → P-only vs P+M SR(seed3 평균) 비교 = eval NLL 우위가 SR로 이어지는지 판정.

**다음**: reportable = **P-only** full-suite·aug-on 매트릭스(P+M은 rollout 붕괴로 배포 제외 — comp_mae_plan §6). pretrain ckpt 로컬 전송 선결. LIBERO finetune+rollout 모두 **로컬**(역할 분담 정정 — 탐색만 클러스터였음).

### 2026-06-30 Attentive-pooling probing (readout 축 — mean vs attentive)

**배경**: mean-pool은 중립 readout 아님(holistic 유리·분산 MAE 불리). attentive pooling을 **모든 비교군 uniform** 적용해 readout 강도를 *축*으로 보고, mean→attentive **crossover**로 "정보가 분산 공간구조에 있다"를 입증. capacity는 linear와 맞춤(`AttentivePoolProbe` = stream별 query 1개, **attentive/linear param 1.055×**)해 structure vs capacity 혼동 차단. M/P 별도 query(구조적 분리). input-only(P_t 단독)=null anchor. 구현: `probe_action.py` `encode_batch_tokens`+`AttentivePoolProbe`(단위검증 통과). 비교 대상 mean baseline(split=test·gap=10): P_t **−0.009** / P_t⊕P_tk **+0.236**★ / M **+0.094** / P_t⊕M **+0.099**.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36188581 | AIP 1×1 H100 | 03:00:00 | **attentive probe sanity** (CoMP-MAE-S, `attentive_concat_p_t_p_tk`, MAX_VIDEOS=40·5ep) | ❌ FAILED 11s — argparse `choices`에 attentive 모드 누락. choices 추가 후 재제출(36188590). |
| 36188590 | AIP 1×1 H100 | 03:00:00 | **attentive sanity 재실행** (동일 config) | ✅ COMPLETED — **파이프라인 PASS**. 토큰 `[2765,392,384]`(2stream×196×384) 정확, AttentivePoolProbe(D384/n_streams2/params14.6k), R² −0.59→−0.10 단조상승(학습 정상). R²<0은 MAX_VIDEOS=40·5ep underfit(threshold 0.7 무관). 토큰 추출 20.5s. |
| 36188593~596 | AIP 1×1 H100 ×4 | 03:00:00 | **attentive gate 매트릭스** (CoMP-MAE-S, split=test·gap=10·full·**40ep**). 593=`attentive_p_t`(null) / 594=`attentive_concat_p_t_p_tk`(배포P★) / 595=`attentive_m` / 596=`attentive_concat_p_m`. **mean→attentive crossover 판독**(mean baseline: −0.009/+0.236/+0.094/+0.099). | ⚠️ single-stream만 완료: **593 R²=0.0007**(null, mean −0.009 → flat=capacity 아님 ✅) · **595 R²=0.2393**(M, mean +0.094 → **+0.145 점프**=M motion이 mean에 under-read). **594·596 concat(392토큰) OUT_OF_MEMORY** — 토큰캐시 fp32 ~134GB>노드 126GB. fp16 수정 후 재제출(33/34). |
| 36188633/634 | AIP 1×1 H100 ×2 | 03:00:00 | **concat 모드 재제출** (fp16 토큰캐시 + eval 배치화). 633=`attentive_concat_p_t_p_tk`(배포P★ vs mean +0.236) / 634=`attentive_concat_p_m`(P⊕M vs mean +0.099) | ✅ COMPLETED. **633 R²=0.2924**(배포P, mean +0.236 → +0.056) · **634 R²=0.2418**(P⊕M, mean +0.099 → +0.143). ⚠️ **MaxRSS 279GB**(fp16인데도 — `torch.cat` 리스트+결과 2× 스파이크). B(768d, 캐시 2×)·SiamMAE attentive 전 캐시 누적 pre-alloc/disk-backed 수정 필수. |

### 2026-06-30 size 격리 — CoMP-MAE-S vs B (둘 다 attentive, 동일 데이터)

**배경**: VideoMAE(ViT-B)를 ViT-S와 비교는 size confound(readout 맞춰도 무의미) → 깨끗한 size 테스트 = **같은 모델 S vs B, 같은 attentive readout**. B는 학습 중(ep28 ckpt, ep50 아님) → epoch-mismatch. "B-now ≥ S-ep50"이면 강한 신호. 메모리: pre-alloc(`torch.cat` 2× 제거)+**MAX_VIDEOS=1500 캡**(B 768d concat full이면 134GB>126GB). S도 같은 캡 재측정(데이터량 confound 차단).

| JobID | 자원 | 목적 | 결과 |
|-------|------|------|------|
| 36188929/930 | AIP 1×1 H100 ×2 | **deployed-P** (`attentive_concat_p_t_p_tk`) — 929=S / 930=B(ep28). MAX_VIDEOS=1500·40ep | **929 S R²=0.329** ✅(full 0.292보다↑=캡 효과). **930 B R²=−0.028 ❌ 무효**: arch 추론 정상(768d/h12) but **probe overfit/발산**(Train MSE↓ but Eval R² ep1 −0.028→ep40 **−1.92**). best=ep1(거의 untrained). "size 나쁨" 아니라 B feature에서 일반화 해 못 찾음. 의심: fp16×B outlier dim / attentive 불안정. |
| 36188931/932 | AIP 1×1 H100 ×2 | **M** (`attentive_m`) — 931=S / 932=B(ep28). 동일 캡 | **931 S R²=0.293** ✅ · **932 B R²=0.320** ✅. → **B-M > S-M = M stream이 size로 개선**(ep28 미완인데 S-ep50 상회). **attentive-on-B 정상**(fp16/attentive 일반 버그 아님) → 930 B-deployed-P 발산은 **B의 P-stream 특정 문제**(ep28 미수렴 P feature outlier 추정). |
| 36188970 | AIP 1×1 H100 | **진단1: B mean-pool(fp32)** `patch_mean_concat_p_t_p_tk` — B P feature가 informative한가 격리(안정 readout) | ✅ COMPLETED. **best −0.013 → ep40 −0.73 발산**(attentive와 동일 overfit) → attentive/fp16 아니라 **B-P ep28 feature 자체가 미수렴**(appearance 풍부, action linear-decodable 미형성). ep50 재측정 필요. |
| 36197899/900/901 | AIP 1×1 H100 ×3 | **B-ep50 재측정** (수렴본 `latest.pt`, S/B 1500과 동일조건: split=test·gap=10·MAX_VIDEOS=1500·40ep). 899=`attentive_concat_p_t_p_tk`(deployed-P★)·900=`attentive_m`·901=`attentive_p_t`(null). | ✅ COMPLETED. **deployed-P −0.026→ep40 −0.49 여전히 발산**(ep28과 동일 — 수렴 문제 아님) · **M best 0.352**(안정) · null −0.020(≈0). |
| 36197909/910 | AIP 1×1 H100 ×2 | **concat_p_m 보강** (`attentive_concat_p_m`=P_t⊕M, 1500·40ep). 909=B-ep50·910=S-ep50(matched) | ✅ COMPLETED. **B-ep50 0.236(안정)** · S 0.286. |

**S vs B size matrix (attentive, 1500·40ep, ep50 수렴본)**:
| mode | S | B(ep28) | **B(ep50)** |
|---|---|---|---|
| null(p_t) | 0.001 | — | −0.02 |
| deployed-P(P_t⊕P_tk) | 0.329 | −0.03 발산 | **−0.03 발산** |
| **M** | 0.293 | 0.320 | **0.352** |
| concat_p_m(P_t⊕M) | 0.286 | — | 0.236 |

**결론**: ① **M stream이 size·학습으로 단조 증가**(S 0.293→B28 0.320→B50 0.352) — 깨끗한 size 효과. **B-M(0.352) > S deployed-P(0.329)** = 작은 motion 인코더가 큰 2프레임 P readout을 능가(efficiency 가설 강화). ② **deployed-P는 B에서 overfit**(ep28·ep50 둘 다) — 원인=**P_t⊕P_tk redundancy**(concat_p_m 1536은 안정 → 차원 아님; ep50도 발산 → 수렴 아님). B의 풍부한 appearance 두 장을 attentive probe가 memorize. ③ concat_p_m에서 B(0.236)<S(0.286): B의 P_t가 M readout을 끌어내림(부분 overfit 오염). ⚠️ **P+M(0.236) < M단독(0.352)은 red flag** — 건강하면 무용 스트림은 무시되어야 하는데 P_t가 *적극적으로 해를 끼침*(overfit-유발 appearance 주입). → **deployed-P size 비교는 weight decay로 P-appearance overfit 억제 후 재측정 필요.**

| 36198128/130 (B), 36198129/131 (S) | AIP 1×1 H100 ×4 | **WD probe disambiguate** — deployed-P(`attentive_concat_p_t_p_tk`, 1500·40ep) + weight decay. {S,B}×WD{0.1,1.0} matched. | ✅ COMPLETED. **S**: WD0.1 **0.318**(healthy)·WD1.0 0.235(약간↓, 안정). **B**: WD0.1 **−0.79**·WD1.0 **−1.67** (둘 다 발산). **판정: WD 정상작동(S 예상대로), but B는 WD-저항성 발산** → 표준 overfit 아님 = **feature-geometry 병리**(B의 P_t⊕P_tk에 outlier/massive-activation dim → attentive single-query latch). **probe-design 아티팩트지 표현 결함 아님.** ⚠️ **"데이터부족 overfit" 가설 약화** — 데이터발이면 WD가 도왔어야(S엔 도움). 데이터量 아니라 feature 기하 문제. **deployed-P 발산은 배포(LIBERO BC P+M 작동)와 직교 → 추격 실익 낮음.** 확정 결론=M scale(size↑) + BC P+M로 충분. |

**EgoDex 데이터**: part1~5 + test 전부 추출됨(`/proj/external_group/mrg/datasets/egodex/frames/`) — 현재 part1만 학습. full-data 본학습은 데이터측 준비 완료.

**가설(다음 방향)**: B(ViT-B)의 P-appearance overfit-proneness = **데이터(EgoDex part1) 대비 모델 규모 과대** 정황. part1은 appearance 다양성 제한 → 큰 모델이 part1-특정 appearance를 고밀도 memorize → probe overfit. **EgoDex 전체 학습**이 appearance 일반화로 완화 기대(M은 이미 part1로도 clean·scale). 검증: WD probe로 readout-overfit vs data-부족 먼저 disambiguate 후 full-data 본학습 결정.

**Gate 결론 (CoMP-MAE-S, mean→attentive crossover)**: null anchor flat(−0.009→0.0007)=capacity 아님 ✅ · M 대폭(+0.094→0.239, +0.145)=mean이 M motion under-read · 배포P modest(+0.236→0.292) · P⊕M(0.242)≈M(0.239)=P_t 기여 0. **배포P 0.292 > 과거붕괴 ViT-B(+0.288) 넘었으나 VideoMAE +0.47(mean)엔 미달** → readout이냐 size냐는 **baseline도 attentive 재측정 필요**(uniform readout). Gate=통과 → SiamMAE-S 학습+baseline attentive+LIBERO BC attentive GO. 추천 순서: 메모리픽스→VideoMAE attentive(즉시)→SiamMAE-S 결정.

### 2026-06-30 CoMP-MAE-**B** scale-up (size confound 해소) — ViT-B 768

**배경**: CoMP-MAE-S(ViT-S 384) in-domain probing이 VideoMAE +0.47·과거 Parvo +0.29(둘 다 **ViT-B 768**)에 미달 → size confound. **인코더 ViT-B(768/d12)로 키워 동일 학습·테스트 재측정**. M-stream은 비대칭 강화(**m_depth 6→4**) + 마스킹 0.5→**0.6**(사용자 결정: M-recon이 OOD도 잘 복구=추론 쉬움). caseA_prob 1.0→**0.25**(연산 점검: Case A 정지 calibration이 매 step 돌지만 L_mA→0 trivial → skip 효율). floor 0.02 유지.

**연산 점검 결과**(forward 추적): init은 이미 최적(comp_mae가 teacher_p/m·interpreter_1·M-jepa decoder del, null_motion_token이 M full-forward 1회 대체). 유일 절감 = caseA_prob↓. 3× predict·P-helper full-pass는 중복 아님(각자 다른 routing/target).

**비용 추정**: ViT-S 110 GPU·h(50ep) → 폭 2× → ~4× → **50ep ≈ 440 GPU·h**(40ep ~350 + 나머지 10ep ~88). 8 GPU max(2노드×4) → 40ep ~44h(48h 캡 내). 월누적 ceil ~1.1M원 상당.

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36186567/568 | AIP 1×1 H100 ×2 | 00:20:00 | **CoMP-MAE-B batch sweep** (embed768/h12/m_depth4/mask_m0.6/caseA0.25, 1ep MV200 part1). 567=batch128·568=batch64. ViT-B OOM 확인 + throughput → 본학습 batch·LR 확정 | ✅ COMPLETED ~2분 each. **OOM 없음**. throughput/GPU: **batch128=418.9 > batch64=371** → **batch128 채택**(eff1024). ⚠️ **비용 추정 정정**: ViT-S 578→ViT-B 418.9 = **1.38×만 느림**(추정 4× 오류 — 이 규모는 compute가 d²-bound 아님 + m_depth4·caseA0.25 절감). **50ep ≈ 19h ≈ ~154 GPU·h**(추정 440 대폭 하향), 48h 캡 내 단일잡 충분. |
| 36186569 | AIP_long 2×4 H100 | 1-06:00:00 | **CoMP-MAE-B 본학습 (size-matched, 50ep)** — embed**768**/h12/**m_depth4**(비대칭↑)·no-Sobel·pair·part1·**mask_m0.6**·**caseA_prob0.25**·floor0.02·batch128(eff1024)·LR2.8e-4. SUFFIX=step1_comp_mae_b. **목적**: ViT-S size confound 해소 → VideoMAE/Parvo(둘 다 ViT-B)와 epoch-matched 비교 | ✅ COMPLETED 50ep, **18h59m** (8 GPU = **~152 GPU·h**). 최종 loss 0.0165, L_t/tk~0.006·L_pred~0.0024·L_mB~0.0034·L_mA→0. ckpt `latest.pt`(=ep50)·epoch0004~0048. **다음: attentive probing으로 S vs B size 확정**(B-P는 ep28 overfit이었으니 ep50 수렴본으로 재측정). |
| 36188199 | normal V100 1×1 | 00:20:00 | **CoMP-MAE-B ep8 중간 가시화** (`checkpoint_epoch0008.pt`, 학습 중 progress 점검). `visualize_comp_mae.py` B arch(embed768/h12/m_depth4/mask_m0.6) + DROID 행. OUT=`scratch/viz/comp_mae_b/recon_ep0008_droid.png` | ✅ COMPLETED ~1분(~0.02 GPU·h), 06-30 11:00. **붕괴 없음**(중간 ep): P-recon 형상 coherent(rec_t/pred_tk std≈0.12), ΔL_recon(B)가 target motion edge 추종(DROID x-domain 포함). 단 **미성숙**: M-recon(B) std 0.027~0.062 ≪ target 0.048~0.232(undershoots), Case-A static 0.026(ep50 S=0.0001 대비 calibration 미수렴). ep50 향해 sharpening 예상. |
| 36197884 | normal V100 1×1 | 00:20:00 | **CoMP-MAE-B ep50 최종 가시화** (`latest.pt`=ep50). 동일 B arch + DROID 행. OUT=`scratch/viz/comp_mae_b/recon_ep0050_droid.png` | ✅ COMPLETED. **수렴 완료**(ep8→16→28→50 궤적 마감): Case-A static **0.0002**(S ep50 0.0001급, →0 도달)·M-recon in-domain target의 ~85-92%(SEEN 0.068/0.080·UNSEEN 0.119/0.130)·DROID x-domain 67-75%(cross-domain 한계)·P rec_t std 0.17~0.22. 붕괴 無, S(384)와 동급 복원 품질(size 열화 없음). |
| 36188942 | normal V100 1×1 | 00:20:00 | **CoMP-MAE-B ep28 중간 가시화** (`checkpoint_epoch0028.pt`). 동일 B arch + DROID 행. OUT=`scratch/viz/comp_mae_b/recon_ep0028_droid.png` | ✅ COMPLETED ~1분. **순항 지속**: Case-A static **0.0016(ep16)→0.0005**(ep50 S=0.0001 향해 수렴) · M-recon(B) in-domain target의 ~90-95%(SEEN 0.075/0.079·0.059/0.065)·**DROID gap8 0.032/0.034=94%**(ep16 undershoot 해소). P rec_t std 0.11~0.23 성숙. 붕괴 無. |
| 36188300 | normal V100 1×1 | 00:20:00 | **CoMP-MAE-B ep16 중간 가시화** (`checkpoint_epoch0016.pt`, ep8 대비 sharpening 점검). 동일 B arch + DROID 행. OUT=`scratch/viz/comp_mae_b/recon_ep0016_droid.png` | ✅ COMPLETED ~1분, 06-30 ~11:30. **ep8 대비 뚜렷한 sharpening 확인**(중간 ep 미성숙 = 단순 학습부족 재확인). ① **Case-A static 0.026→0.0016**(16× 감소, ep50 S=0.0001 향해 수렴). ② M-recon(B)가 target 근접: SEEN 0.086/0.093·UNSEEN 0.119/0.133·0.135/0.153(ep8 undershoot ~40% → ~90%). ③ P rec_t std 0.11~0.226(ep50 S 0.11~0.23 범위 도달). DROID x-domain은 아직 undershoot 잔존(0.052/0.092). 붕괴 無, 순항. |

### 2026-06-30 CoMP-MAE-S OOD probing (DROID cross-domain, gap=15)

**목적**: in-domain 조합 sweep(위) → cross-domain 일반화. DROID 15Hz **gap=15(1초=EgoDex 학습 분포 일치, 변별 최대)**, max_ep200, 20ep. 비교: VideoMAE DROID gap=15 **−0.035**(전 gap 음수), 과거 v11 best +0.005. ⚠️ DROID 절대 R²~0.005 noise 수준([[feedback_droid_image_only_limitation]]).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36186541 | AIP 1×1 H100 | 03:00:00 | **DROID probe P(t) only** (`patch_mean_p_t`) | ✅ COMPLETED 1m11s. **R²=−0.0035** |
| 36186542 | AIP 1×1 H100 | 03:00:00 | **DROID probe P(t)⊕P(t+k)** | ✅ COMPLETED 1m11s. **R²=−0.0039** |
| 36186543 | AIP 1×1 H100 | 03:00:00 | **DROID probe M(t,t+k)** | ✅ COMPLETED 1m11s. **R²=+0.0099** (4개 중 유일 양수, VideoMAE −0.035 상회) |
| 36186544 | AIP 1×1 H100 | 03:00:00 | **DROID probe P(t)⊕M** | ✅ COMPLETED 1m34s. **R²=+0.0012**. 합 ~0.08 GPU·h. **종합: 4개 모두 ±0.01 noise 수준(DROID 한계)** — in-domain 순위(P_t⊕P_tk 우위)와 달리 OOD에선 M이 미세 우위지만 변별 불가. cross-domain은 LIBERO BC-T로 판정 권장. |

### 2026-06-22 no-M ablation (§11) — motion routing 기여 격리

**배경**: Parvo BC-T 0.785 ≈ v15-ptptk(현상유지). §11.4 "motion routing이 LIBERO control 표현에 load-bearing이냐" 판정 = **같은 코드에서 routing만 끈 인코더 학습 후 동일 P-only BC-T 비교**. 효율형 `--v15-no-motion` 플래그 구현(`_forward_pair`에서 M/JEPA 분기 skip + M stream·p_motion_decoder 동결 → dead branch ~절반 제거, teacher_p full forward 등 제거. DDP-safe smoke 통과: 192 trainable 전부 grad 수신). encoder A(Run B-2)와 **환경 완전 동일** + no_motion만 추가.

⚠️ **part1 단독 확정**: A scratch(35788399)·continuation(36045098) 로그 확인 = **part1만**(46234 vid) 학습. 기존 doc "part1-5"는 오류(Run A·Parvo main 포함 최근 Parvo 계열 전부 part1 개발 런). B도 part1로 맞춤 = valid control. **논문용 part1-5 full 학습(full Parvo + no-M)은 별도 후속 결정**(현재는 개발 단계 비교).

| JobID | 자원 | --time | 목적 | 결과 |
|-------|------|--------|------|------|
| 36126892 | AIP_long 2×4 H100 | 18:00:00 | **no-M scratch (20ep, part1)** — A scratch(35788399) env 완전 동일 + `V15_NO_MOTION=1`. ep20 후 continuation 예정 | ✅ COMPLETED 20ep, 5h27m (~43.7 GPU·h), 06-22 15:59. ckpt `two_stream_v15b/20260622_103242/checkpoint_epoch0020.pt` |
| 36128321 | AIP_long 2×4 H100 | 18:00:00 | **no-M continuation (+30ep, part1)** — Parvo runB2cont(36045098) 정확 미러 + `V15_NO_MOTION=1`. init_from no-M ep20, EPOCHS=30, LR=1e-4, masked_anchor+λ_m_jepa=0+gate0+warmup5, pair, no-sobel, `CHECKPOINT_SUFFIX=noM_cont`. → Parvo와 동일 누적 50ep 매칭. 완료 후 BC-T(libero_object) 제출 예정 | ✅ COMPLETED 30ep, 7h55m (~63.3 GPU·h), 06-23 01:22. ckpt `two_stream_v15b_noM_cont/20260622_172841/checkpoint_epoch0030.pt`. ep30 recon 건강(L_t/tk≈0.028), P CLS std_p=0.090(Parvo 0.008보다 덜 붕괴=순수 MAE 정상). → BC-T(libero_object) 제출 대기 |
| 36130509~511 | AIP 1×1 H100 ×3 | 2-00:00:00 | **no-M BC-T (libero_object × seed 0/1/2)** — Parvo BC-T(36053625~627) 정확 미러 + ckpt만 no-M ep30(`two_stream_v15b_noM_cont/.../epoch0030.pt`)으로 교체. ENCODER=parvo-ptptk(P_t⊕P_tk, strict=False), V3(use_joint+aug)·50ep, SUFFIX=noM. **목적**: §11 M 기여 격리 — no-M vs Parvo(object 0.885) epoch-matched 비교. Parvo 최고 suite만(연산 절감) | ✅ COMPLETED 06-24 (s0 1d10h12m / s1 1d09h50m / s2 1d10h05m, 각 1 GPU = **~102.1 GPU·h**). ep50 best.pt 3개: `libero_bct/parvo-ptptk_libero_object_seed{0,1,2}_20260623_084816_noM/best.pt`. ⚠️ **클러스터는 학습만 — rollout(SR) 미실행**(eval loss s0 −24.86까지). **SR은 로컬 rollout 후 산출** → v15b_status §11 미기입. |
| 36130526~529 | AIP 1×2 H100 ×4 | 00:30:00 | **신규 baseline sanity** (1ep, 64vid, 2-GPU=DDP unused-param 검출). 526=SiamMAE-base, 527=SiamMAE-small, 528=§9 pixel-pred base, 529=§9 pixel-pred small(ViT-S P=384/6) | ✅ COMPLETED 전부 통과(각 ~1m). **DDP-safe**(unused-param 0). params(trainable): SiamMAE 120M/56M, §9 158M/40M(base/small). §9 L_t/tk/pred 전부 계산·L_mj=0. ⚠️ §9 frozen 207M(teacher EMA+interpreter) 메모리 낭비 — 본학습 전 del 검토 **(✅ 2026-06-23 완료, commit 42db84c: teacher/interpreter_1/M-decoder del)** |
| 36137148 | AIP 1×1 H100 | 02:00:00 | **MCP-MAE-S sanity** (§9 pixel-pred, ViT-S 384/6·m_depth6, no-Sobel, pair, part1 MV200, 5ep, batch32, SUFFIX=sanity_mcp_s). STEP 1 본학습 전 검증 | ✅ COMPLETED 5m54s (~0.1 GPU·h), 06-23 19:03. **healthy**: L_t/L_tk/L_pred≈0.025 균형(λ=1 적정)·recon no-M(0.028)급. CLS collapse(cos_intra_p=1.0)=정상(cls_p 학습압력0→patch로 판단). **254.9 samp/s/GPU @batch32, COMPUTE-bound** → full part1 50ep ≈ **252 GPU·h**(3×predict 오버헤드, no-M 대비 2.3×). ⚠️ batch32=ViT-S GPU 저활용 → **batch↑(128+) 권장**(throughput·비용). launcher 4GPU·36h 부족(≈63h wall) → 8GPU/2노드 또는 batch↑ 필요 |
| 36138625/628 | mig-3g.40gb→AIP | 00:20:00 | **MCP-MAE-S recon 가시화** (sanity ep5) | ❌ 둘 다 CANCELLED(MIG·AIP 큐 혼잡) → **로그인 노드 CPU로 실행**(viz는 소규모). 결과: nomask는 평균색(OOD 아티팩트)이나 **masked 복원 std=0.156(vs target 0.20)=학습 정상**. PNG `paper_artifacts/mcp_mae_sanity_recon/` |
| 36138706→36138720 | normal V100 1×1 | 00:30:00 | **MCP-MAE-S viz v2 (V100)** — 8열(frame_t/t+k + recover_t·recover_t+k·predict_t+k 각 **mask/nomask**) × **seen(EgoDex part1)/unseen(DROID cross-domain)** 2샘플씩. 706=part4(취소)→720=DROID(잡이라 95k 스캔 감당). `sanity_ep5_droid.png` | 🔵 RUNNING(720) |
| 36138668 | AIP_long 2×4 H100 | 1-12:00:00 | **MCP-MAE-S 풀 학습 (STEP 1, ours/gate)** — §9 pixel-pred·ViT-S(384/6·m6)·no-Sobel·pair·**part1·50ep**. batch **64**/GPU(eff 512; sanity서 batch32 GPU 저활용 관찰→2×)·LR **4e-4**(linear 2×). SUFFIX=step1_mcp_mae_s. 8GPU로 ~16h 예상. gate=첫 run health(collapse 없이 patch+CLS R²)+slope vs VideoMAE(−0.083) | ❌ CANCELLED 3m27s (~0.46 GPU·h), 06-26 17:13. 시작 직후 사용자 요청으로 취소(재제출 대기) |
| 36177294/295 | AIP 1×1 H100 ×2 | 00:20:00 | **CoMP-MAE-S batch sweep** (본학습 GPU 활용 최적 batch 결정). 1ep·MV200·part1·floor0.02, 294=batch128·295=batch256 | ✅ COMPLETED ~1m each (~0.04 GPU·h). throughput: **batch32=249 → 128=578 → 256=628 samp/s/GPU**(둘 다 OOM 없음). 128→256은 +9%뿐(GPU 포화)·eff 2배 → **batch128 선택**(포화+eff1024 수렴 안전). LR sqrt→2.8e-4 |
| 36177296 | AIP_long 2×4 H100 | 1-12:00:00 | **CoMP-MAE-S 본학습 (STEP 1, ours/gate)** — v16 대칭 cross-recon·ViT-S(384/6·m6)·no-Sobel·pair·**part1 full·50ep**. **batch128/GPU(eff 1024)·LR 2.8e-4**(sweep 최적). floor**0.02**(|ΔL| 가중 유효화)·caseA_prob1.0·λ_M1.0·mask_m0.5. SUFFIX=step1_comp_mae_s. 추정 ~111 GPU·h(8GPU ~15-20h wall). gate=health(collapse 없이 patch+CLS R²)+OOD slope vs VideoMAE(−0.083). 모니터: L_t/tk/pred + L_mA(정지)/L_mB(동적) | ✅ COMPLETED 50ep, 13h42m (8 GPU = **~109.7 GPU·h**), 06-29 23:58. **patch healthy**: L_t/L_tk 0.048→**0.0048**(10× 단조감소)·L_pred 0.0023·**L_mB 0.0024(동적 active)·L_mA→0(정지 calibration 정상)**. throughput 4694 samp/s(8GPU, batch128 sweep 일치). CLS는 collapse(std_p 0.076→0.013, cos_intra_p→1.0, cos(pred,tgt)=1.0) = **cls_p 학습압력0 → 비판단**([[feedback_no_cls_collapse_judgement]] 메모리·CLAUDE.md). ckpt `two_stream_v15b_step1_comp_mae_s/20260629_101634/{best_model,latest,epoch0004~0048}.pt`. ⚠️ **gate 미완**: patch health는 통과했으나 probing R²(patch)+OOD slope vs VideoMAE 미측정 → 다음 단계. |
| 36186233 | normal V100 1×1 | 00:20:00 | **CoMP-MAE-S ep50 가시화** (step1 본학습 latest.pt). `visualize_comp_mae.py`: P-recon(rec_t/rec_t+k/pred_t+k masked) + M-recon(ΔL_target/recon_B/static_A) × seen(part1)/unseen(part4)/DROID x-domain. ⚠️ 로그인 노드 CPU 先시도 30분 미완(forward 과다) → V100 전환(선례 viz=normal 1~2분). OUT=`scratch/viz/comp_mae_step1_ep50/recon_ep50.png` | ✅ COMPLETED 1m32s (~0.03 GPU·h), 06-30 01:08. **§6 검증 충족**: M-recon(B) std≈target(SEEN 0.083/0.092·0.074/0.086, UNSEEN 0.063~0.095, **DROID x-domain 0.040/0.043**=cross-domain motion 복원)·**Case A static_t/tk=0.0001~0.0002≈0**(정지 calibration 도메인 불변). P-recon coherent(rec_t/pred_tk std≈0.11~0.23, motion routing 무손실). PNG 6행(seen2/part4·2/DROID2)×9열. |
| 36177262 | AIP 1×1 H100 | 02:00:00 | **CoMP-MAE-S sanity** (v16 대칭 cross-recon, ours 축 = MCP-MAE 대체). ViT-S 384/6·m6, no-Sobel, pair, part1 MV200, 5ep, batch32, SUFFIX=sanity_comp_s. 신규 M-recon 분기 첫 검증 + **각 loss 분리 모니터링**(L_t/L_tk/L_pred + L_mA(정지)/L_mB(동적)) → λ·caseA 균형 결정. 기본값 caseA_prob=1.0·λ_M=1.0·mask_m=0.5·floor=0.1 | ✅ COMPLETED 6m32s (~0.11 GPU·h), 06-29. **healthy**: ep5 L_t/tk/pred≈0.025(MCP-MAE 동급)·**L_mA→0(정지 calibration 정상)**·L_mB≈0.020. collapse 없음. **249 samp/s/GPU**(MCP-MAE 255와 동급 — M-recon 오버헤드 batch32 저활용분 흡수). ⚠️ 본학습 전 튜닝: **floor=0.1 과대**(측정상 patch mean\|ΔL\| median 0.012, 90%<0.1 → floor가 90% patch 지배해 \|ΔL\| 가중 무의미) → **floor↓ 0.01~0.02 권장**. caseA 빠르게 0수렴→caseA_prob↓ 여지 |

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
