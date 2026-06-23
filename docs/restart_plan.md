# Restart Plan — subset-matched small 재시작 (2026-06-23)

> **상태**: 계획 확정, 실행 전. **문서 전용**(계획·주의·체크리스트·pseudocode). 실제 코드(slope 집계 스크립트·analog routing 분기·run config)는 이 저장소 dev 세션에서.
> **결정 출처**: Obsidian Vault `Projects/Action-Agnostic Paper/2. Experiments.md` §"결정: subset-matched small 재시작".
> **명명**: 코드 식별자 우선(rename deferred). 기능명↔코드 매핑 = `REFACTOR_PLAN.md` §1 / `CLAUDE.md` "명명 · 2논문 구조".

---

## 1. 배경 — 왜 재시작인가

1. **데이터 예산 사고**: `MS-JEPA`(=Parvo, code `v15b`)와 `Image MAE`(`--v15-no-motion`)는 실측 로그상 **EgoDex part1 서브셋(46,234 vid)만** 학습됨 — full part1-5(~314k) 아님(`cluster_sessions.md` L96, "기존 doc part1-5는 오류"). `VideoMAE-ours`는 full part1-5. → 기존 in-house 비교는 **cross-budget confound**.
2. **붕괴(recipe) 사고**: 서브셋 `MS-JEPA`는 clean checkpoint도 없음 — Run A ep5 cancel(collapse), Run B-2 CLS collapse(patch R²=0.2884 < VideoMAE 0.4705). LIBERO BC 78.5는 붕괴 잔재 위 값 → trend·development 참고용, 절대·앵커 주장 불가.
3. **비용 구조**: 지배 레버는 **데이터**(part1→part1-5 ~6.8×) ≫ 모델 크기(ViT-S↔B ~2×). 실측: subset single-stream 50ep ≈ 107 GPU·h(no-M JobID 36126892+36128321), full ViT-B 50ep = 343~420 GPU·h(v15 343.3 / VideoMAE ~400 / vfromm 420.4). → **"small+full"은 "small+subset"보다 ~6× 비쌈.**

→ 죽일 confound은 *cross-budget*이지 subset 자체가 아님. **subset에서 matched로 맞추면 같은 목적을 1/6 비용에 달성.** 현 narrative는 절대 LIBERO BC SR을 떠나 OOD/dissociation으로 이동했으므로 full 불필요.

---

## 2. 결정 (묶였던 4개를 분리)

| # | 결정 | 근거 |
|---|---|---|
| ① | matched 재시작 = **subset(part1)에서** | confound = cross-budget. subset matched면 1/6 비용 해소. 서브셋 MS-JEPA는 데이터+collapse 이중 오염이라 어차피 재시작 필요 |
| ② | ours 축 `MS-JEPA` → **`MCP-MAE`(`--v15-pixel-pred`)** 교체, **gate** | JEPA self-ref constant collapse 원천 제거. 단 미학습·미검증 → 첫 런이 곧 건강성 첫 증명 |
| ③ | **`VideoMAE-ours` 유지 (드롭 X, 재학습 X)** | dissociation 앵커(in-domain probe +0.47 ↔ control 0.293). 유지 비용 0. ViT-S 변종 없음 → 재학습 시 full-data monolithic 역할 파괴(zero gain) |
| ④ | 대조군 = **SiamMAE-analog** (claim-bearing) | routing Q/K를 M(ΔL) 대신 P, V는 P 유지 → "ΔL-where vs RGB-where" 단일변수 격리. vanilla literature SiamMAE는 objective·decoder·masking·routing 동시 변동이라 mechanism 입증 불가 → **외부 reference로만** |

---

## 3. 실행 순서 (비용 오름차순)

- **STEP 0 — (거의 무료, 게이트 아님) baseline slope 사전계산**: `in-domain R² − OOD R²`를 **probing CSV가 이미 있는 인코더만** 집계 = **`VideoMAE-ours` + frozen baselines(`DINOv2`·`SigLIP`·`VC-1`)**. → 비교군의 도메인 취약성(dissociation headroom) 공짜 확인. ⚠️ **thesis(ours) slope는 여기서 안 나옴**: ours 계열(`MS-JEPA`(v15b)·`no-M`)은 probing 산출물 없음(`paper_artifacts/`에 BC rollout만), `v15`/`v11`(full Sobel)은 논문 제외 + motion routing no-op이라 ours 프록시 부적합(slope는 내부 sanity로만 선택). ⟹ thesis OOD slope는 STEP 1로 이동.
- **STEP 1 — subset·small·matched 3런 → probe → OOD slope(ours 포함)** (~100 GPU·h each): `MCP-MAE`-S + `SiamMAE-analog`-S + `no-M`(Image MAE)-S, **part1 동일·epoch 동일·size 동일**. 학습+probing까지 = thesis slope **첫 진짜 신호**. **게이트 = `MCP-MAE`-S 첫 run health(collapse 없이 patch+CLS R² 정상) + slope가 `VideoMAE` 대비 우위** → 아니면 매트릭스 확장 보류.
- **STEP 2 — 핵심 비교만 2-scale**: `MCP-MAE`-S vs `SiamMAE-analog`-S 를 part1의 10%/30%에서 재실행 → rank 안정성(scale-interaction 방어).
- **STEP 3 — (보류·조건부) 승자만 ViT-B full 1회**: 2-size 확인 + dissociation de-confound(full `VideoMAE`-B vs ours-B)를 **한 런으로 동시 충족**. 절대 경쟁력 주장 재추구 시에만.
- **KEEP**: `VideoMAE-ours`-B(full) + frozen baselines(VC-1/DINOv2/SigLIP) = 그대로. cross-size external reference로 표기.

---

## 4. Critical guards (구현 시 실수 방지)

- 🔴 **`MCP-MAE` ~207M FROZEN param 삭제**: §9 경로가 teacher EMA + interpreter(미사용)를 들고 있어 메모리 낭비 → **본 학습 전 `del`** (sanity서 확인됨). 안 지우면 "small"이 메모리상 small 아님.
- 🔴 **analog ≠ vanilla SiamMAE**: `MotionRoutingBlock`(`src/models/common/blocks.py`, v11/v15 공유)에 routing-source 옵션 추가(M→Q/K vs P→Q/K). **V는 항상 P 유지**, param-symmetric. 별도 Siamese ViT 만들지 말 것.
- 🔴 **3런 matched 보장**: 동일 part1(MAX_VIDEOS 동일), 동일 epoch, 동일 ViT-S. subset fraction을 로그에 명시(향후 calibration).
- 🟠 **`VideoMAE` 재학습 금지**: cross-size external reference로만. cross-size mechanism attribution 금지(작은 ours를 큰 VideoMAE와 직접 메커니즘 비교 X).
- 🟠 **small-only 종착 금지**: ΔL inductive-bias 이점은 데이터·크기↑서 약화 가능(bitter-lesson) → 핵심 비교는 STEP 2(2-scale) 또는 STEP 3(ViT-B)로 재확인해야 paper claim.
- 🟢 **gate 위치 (정정)**: STEP 0는 게이트 아님(baseline 특성화만). 실제 게이트 = **STEP 1의 MCP-MAE-S 첫 run** — collapse 없이 healthy patch+CLS R² + OOD slope가 VideoMAE 대비 우위인지 확인 후에야 매트릭스(2-scale·ViT-B) 확장. ⚠️ thesis slope는 ours 학습 전엔 산출 불가(ours probing 부재, v15 no-op 프록시 부적합).

---

## 5. 구현 TODO 체크리스트 (dev 세션)

- [x] **STEP 0 집계 스크립트** = [`scripts/eval/aggregate_dissociation_slope.py`](../scripts/eval/aggregate_dissociation_slope.py) (학습 없음, 즉시 실행). 결과·실측 정정 ↓.
  - **in-domain(EgoDex) 데이터 현실**: canonical은 **VideoMAE-ours(+0.4705)·MS-JEPA(=Parvo v15b, +0.2884) 뿐**(eval_protocols §1). frozen(VC-1/SigLIP=0개, DINOv2=stale)은 EgoDex 학습 자체가 없어 in-domain 앵커 부재 → **slope 불가, OOD floor 참조만**. ⟹ slope는 EgoDex-trained만(VideoMAE; ours-S는 STEP 1 후 `ENCODERS`에 append).
  - **CALVIN OOD 정규 소스**(parity 가드가 오source 적발): `<enc>_training_20260526_213639_gapsweep`(§4 cross-folder, gap30) — 4인코더 §4값 정확 일치. ⚠️ `_validation_*_seg`는 within-validation(in-dist)이라 **OOD 아님**. LIBERO는 within-suite라 보조 표기만(slope 미사용).
  - **slope 절대값 해석 불가**(EgoDex 18-dim hand vs CALVIN 3-dim pos = target-space confound). **신호 = `slope_ours − slope_VideoMAE`**(동일 target → confound 상쇄). 현 baseline: VideoMAE slope **−0.083**, frozen OOD-pos floor = VC-1 +0.536 / DINOv2 +0.223 / SigLIP −0.314.
- [x] **SiamMAE-analog routing 분기**: `MotionRoutingBlock`(blocks.py)에 `routing_source ∈ {m,p}`(default m). `v_from_p`에서 `p`면 Q/K 입력을 `m_completed`→`p_state`로 교체(V는 항상 P, qk_m·norm_m·v_p 재사용). `RoutingInterpreterStep`→`TwoStreamV15Model`(`routing_source`)→CLI `--v15-routing-source` threading. analog = `--v15-pixel-pred --v15-routing-source p`. **smoke PASS**: param/state_dict 키 동일(ckpt 무손상)·flag 동작변경 확인(loss 2.060≠2.069).
- [x] **MCP-MAE frozen-param del**: pixel_pred에서 미사용 모듈 **freeze→del** (teacher_p/teacher_m=full P/M encoder copy + interpreter_1 + M decoder/pos/mask). smoke: 삭제 attr·state_dict 키 제거·param 감소·forward+backward·grad 정상. **no_motion(Image MAE)도 동일 teacher del 확장**(no-M-S = STEP 1 런). 비-pixel/비-no_motion(MS-JEPA)은 teacher 유지(scope 격리).
  - **🔴 점검 중 발견·수정한 버그**: SiamMAE-analog(routing_source="p")는 routing이 M을 무시(Q/K=P)하는데 M encoder가 **trainable+forward**돼 ① DDP `find_unused_parameters=False` hang ② 불필요 M forward ×2/step. → analog일 때 M encoder **freeze + forward skip**(param-symmetric 유지). smoke: analog trainable-no-grad=0(DDP-safe), MCP-MAE는 M 여전히 학습.
  - ⏳ 후속(미적용): no_motion의 M encoder/decoder·p_motion_decoder도 forward 미사용 → freeze 상태(DDP-safe). 완전 P-only로 추가 del 가능(고위험 회피로 이번 보류).
- [x] **run config** = [`scripts/cluster/run_step1_matched.sh`](../scripts/cluster/run_step1_matched.sh) + `pretrain.sbatch`에 `V15_ROUTING_SOURCE` env 추가. matched 공통(part1·ViT-S `EMBED_DIM=384 NUM_HEADS=6`·`V11_M_DEPTH=6`·50ep·`NO_SOBEL=1`·`PAIR_MODE=1`·동일 batch/LR) + per-run flag만 차이: mcp_mae_s(`V15_PIXEL_PRED=1`)·analog_s(`+V15_ROUTING_SOURCE=p`)·no_m_s(`V15_NO_MOTION=1`). 3런 모두 `two-stream-v15b`(vanilla siammae 모델 아님). no-arg=usage(대량제출 가드), gate=mcp_mae_s 먼저. `bash -n` 통과. **제출은 별도**(cluster_sessions 로그 필수).
- [ ] **STEP 2 subset**: part1의 10%/30% 2런(핵심 비교만).

### STEP 0 pseudocode (집계만 — 신규 학습 없음)

```
# 입력: probe_action 출력의 per-domain R² CSV (이미 존재)
# STEP 0 (무료): probing CSV 보유 인코더만 = baseline + VideoMAE
#   (MS-JEPA·no-M = probing 없음 / v15·v11 = 논문 제외·no-op → 제외)
for enc in [VideoMAE, VC-1, DINOv2, SigLIP]:
    r_in  = mean(R2[enc, EgoDex in-domain dims])
    r_ood = mean(R2[enc, CALVIN/LIBERO OOD dims])
    slope[enc] = r_in - r_ood          # 작을수록 도메인-robust
# STEP 1 이후: MCP-MAE-S / SiamMAE-analog-S / no-M-S 를 probe 후 같은 식으로 append
rank(enc by slope)                      # 지표 = 절대 R²가 아니라 slope
# 해석: ours(STEP 1)의 slope가 monolithic(VideoMAE)보다 작으면 factorization 이득 신호
```

---

## 6. 검증 체크리스트 (hand-off 전)

- [ ] 3런이 동일 part1·epoch·ViT-S로 학습됐는지 로그 확인.
- [ ] `MCP-MAE` frozen-param del 반영 + 메모리 감소 확인.
- [ ] analog가 V=P 유지(M은 Q/K만 생성)인지 weight shape/forward로 확인.
- [ ] STEP 0 slope 부호·순위를 예측 시그니처(semantic-shift서 ours robust, photometric서 ours 약세)와 대조.
- [ ] STEP 1 결과를 paper claim화하기 전 STEP 2(2-scale) rank 유지 확인.

---

## 7. Cross-refs

- **Vault 결정 출처**: Obsidian `Projects/Action-Agnostic Paper/2. Experiments.md` §"결정: subset-matched small 재시작" / `README.md` "현재 상태".
- **관련 dev docs**: [`v15b_retraining_status.md`](v15b_retraining_status.md) §9(MCP-MAE 채택)·§10(BC 78.5)·§11(no-M), [`siammae_baseline_plan.md`](siammae_baseline_plan.md), [`cluster_sessions.md`](cluster_sessions.md)(part1 확정·GPU·h), [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md), [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) §1(명명표).
