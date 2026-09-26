# Refinement-Floor Plan — CoMP M은 raw ΔL을 얼마나 정제하는가

> **상태**: 라운드 1(§3–§7) **완료 2026-09-26** — C1 학습·바닥선·판정 측정 끝, §6 중지 신호(M 단독 열) → C2–C4 보류. **현재 = 라운드 2(§9, 2026-09-26 지시): 학습 없음, probe 3건 + 진단.** 라운드 1 결과 = `paper_artifacts/tables/refinement_floor/README.md`.
> **읽는 법**: 이 문서는 **자족적**이다. dev/클러스터 세션은 Vault를 볼 수 없으므로 목적·논리·금지 사항을 전부 여기에 담았다. §1을 읽고 나서 §2를 **행동 규칙**으로 삼고, §7 순서대로 집행한다. 결과는 이 문서에 쓰지 않는다(결과 = `cluster_sessions.md`·`STATUS.md`·`paper_artifacts/`).
> **이 문서와 충돌하는 판단이 서면**: 코드를 고치지 말고 `docs/STATUS.md` "열린 것"에 질문을 적고 **멈춘다**. 사용자가 답한다.
> **결정 출처**: AAAI-27 리뷰(Reject 2026-09-25) · Vault `Projects/Action-Agnostic Visual Representation (AAAI)/{AAAI-27 Reviews, 2. Experiments}.md` · 사용자 결정(09-25).
> **관련 dev docs**: [`comp_mae_plan.md`](comp_mae_plan.md)(CoMP 설계·guard 7/8) · [`factorization_crossover_plan.md`](factorization_crossover_plan.md) §4.1(STEP 1 ablation 배선) · [`eval_protocols.md`](eval_protocols.md)(probe 정규 조건·parity 체크리스트) · [`cluster_sessions.md`](cluster_sessions.md)(비용·**정확한 submit line**).

---

## 1. 목적과 논리 — 잊으면 안 되는 그림

### 1.1 리젹 사유
AAAI-27 리뷰어 3명이 같은 구멍을 봤다. CoMP의 M 스트림 입력은 ΔL(두 프레임 휘도 차)이고, motion probe는 같은 구간의 EE 변위를 예측한다. **입력이 타깃의 hand-computed 버전**인데, 논문은 가장 당연한 바닥선을 안 냈다 — raw ΔL 패치를 probe에 바로 넣으면? 학습 안 한 random-init M은? DINOv2에 같은 ΔL을 주면? 이 숫자가 없어 "구조가 규모를 대신한다"(효율 주장)와 "frame differencing은 강한 feature다"(알려진 사실)가 구분되지 않았다. 부수 요구 = 거울 ablation(routing 유지·M-recon 제거, 미학습) · ΔL≠물리 motion(조명·카메라) · routing 단독 기여 · 두 프레임 concat vs ΔL 비교의 입력 불공정.

### 1.2 우리가 주장할 수 있는 것과 없는 것
- raw ΔL은 **정보량 최대**다(I(f(x);y) ≤ I(x;y)). "정보가 더 있다"는 주장은 불가능. 주장은 **접근성(refinement)** — 같은 용량의 판독기가 얼마나 쉽게, 얼마나 적은 라벨로, 얼마나 흔들림 없이 읽어내는가.
- **구조적 사실**: 판독기가 표현력 있고 타깃 라벨이 충분하면 raw + 판독기는 인코더∘판독기를 통째로 표현할 수 있어 **항상 같거나 낫다**(He et al. 2018 "Rethinking ImageNet Pre-training"과 같은 현상). 따라서 비교가 의미 있는 조건은 정확히 셋 — **판독기 용량 고정 · 라벨 제한 · probe 학습 분포에 없는 조건**. 논문은 이 안에서만 주장한다.
- **공정성**: 모든 팔은 [고정 인코더] + [타깃 도메인 probe] 구조가 같다. raw ΔL의 인코더는 항등함수일 뿐, probe는 CoMP와 똑같이 도메인마다 새로 맞춘다. "raw는 다른 도메인에 적용 안 됨"이라는 주장은 **하지 않는다**. 인코더의 가치는 그 probe에게 사전학습이 무엇을 미리 해줬는가로만 — 라벨 절감·못 본 nuisance 강건·probe 전이. 셋 다 안 나오면 "이 타깃엔 인코더가 필요 없다"가 결론이고 **그것도 보고한다**.
- **로봇 특수성**: EE 변위 라벨은 proprioception에서 공짜다. 라벨 효율은 **실용 동기가 아니라 표상 품질 진단**으로 서술한다. 실용 동기 = 데모에 없는 조건(조명·그림자·물체 배치)에서의 강건성과 전이.
- **타깃 선정 원칙**: "R3M류가 못 하는 것"은 반드시 **입력을 맞춘 상태**에서 정의한다(모두 ΔL 또는 모두 프레임 쌍). 입력 차이로 이기는 타깃은 리뷰어 지적의 재판이다. 고를 것은 같은 입력에서 **판독 함수가 복잡한** 타깃. ΔL→EE 변위는 "변한 픽셀 = 움직인 자리"라 거의 선형이고, 여기서 raw와 parity는 **예상 결과**다. 밝기 변화는 타깃이 아니라 입력이다.

### 1.3 사용자 결정 (09-25)
1. **새 base = CoMP-S + 프레임 쌍 독립 밝기 증강**(§3). 이후 연구는 이 모델 위에서.
2. **Small(ViT-S)만.** B 없음.
3. 스칼펠(V_P) 재학습 없음. novelty 대조군 = **표준 cross-attention**(기존 플래그).
4. 거울 ablation·M-only ΔL MAE 신규 학습. 바닥선 팔·nuisance 시험·probe 전이·다중 타깃은 학습 없이 probe만.
5. 판정 기준은 §6에 **사전 등록**. 결과 본 뒤 수정 금지.

---

## 2. 🔴 행동 규칙 (load-bearing)

### 2.1 변경 금지 — 제출본 CoMP-S(잡 36177296)와 동일 유지, 전 셀 공통
"개선"이 보여도 건드리지 않는다. 바꾸면 모든 비교가 무효다.

| 항목 | 고정값 | 근거 |
|---|---|---|
| 데이터 | EgoDex **part1**(46k vids), split·프레임 추출 그대로 | cluster_sessions 36177296 |
| 크기 | ViT-S `EMBED_DIM=384 NUM_HEADS=6 V11_M_DEPTH=6`, P depth 12 | 같은 곳 |
| 학습 | **50ep · batch128/GPU(eff 1024) · LR 2.8e-4**, 나머지 sbatch 기본 | 같은 곳. ⚠️ `run_step1_matched.sh` COMMON(batch32·LR2e-4)은 **옛 값** — 쓰지 말 것 |
| 입력 전처리 | P = RGB(no-Sobel, 이미지별 luminance z-score) · **M = raw ΔL, 정규화 없음** | `src/models/common/preprocessing.py:7,87`. ΔL에 정규화를 넣으면 실험 전제가 무너진다 |
| 마스킹 | `MASK_P=0.75` · `V15_MASK_RATIO_M_RECON=0.5` | comp_mae_plan guard 8 |
| M-recon 가중 | `V15_M_RECON_FLOOR=0.02` · scale 1.0 · `V15_LAMBDA_M_RECON=1.0` | comp_mae_plan guard 7 |
| Case A | `V15_CASEA_PROB=1.0` | 제출본과 동일(§3.3 ②) |
| 기하 증강 | 프레임별 **독립 RandomCrop 256→224**(항상 on, 옵션 없음) + 회전(공유 90° + `V15_INDEP_ROT_PROB=0.1`) | `src/datasets/base.py:169-178` · `two_stream_v15.py:911-919` |
| 페어 샘플링 | `MAX_GAP=30 SAMPLE_CENTER=15` triangular · `PAIR_MODE=1` | sbatch |
| qk-norm | **off**(제출본은 없음; S-full 안정화용이었다) | fulldata_scaling_plan §4-b |
| 코드 경로 | `two-stream-v15b` + `V15_COMP_MAE=1` → `_forward_pair_comp` | `two_stream_v15.py:905` |

**정본 submit line = `cluster_sessions.md` 36177296 항목 + 그 잡의 sbatch 로그 `Model args:` 줄.** 새 base는 그 line에 §3 증강 플래그만 더한다. 재구성하지 말고 복사한다.

### 2.2 코드 변경 허용 범위 (이 밖은 읽기만)
- `src/models/two_stream_v15.py` — `_forward_pair_comp`·`_recon_dL`에 밝기 증강 분기(§3). **기본 off**, off면 기존과 bit-identical.
- `scripts/pretrain.py` · `scripts/cluster/pretrain.sbatch` — 새 플래그 노출만.
- `scripts/eval/probe_action.py` · `probe_action_{calvin,libero}.sbatch` — 새 팔(§5.2)·시험(§5.3) 추가만. 기존 cls_mode 동작 불변.
- C4 M-only MAE(§4) — **별도 최소 클래스**. 기존 클래스 수정 금지.
- 모든 새 플래그·팔은 **명시적으로 켜야만** 동작. ckpt 로드 실패로 우연히 random이 되는 경로 차단(strict load 실패 = 즉시 에러).

### 2.3 금지 행동
새 loss 항 · 가중치 스케줄 변경 · 정규화 추가 · 마스크 비율 조정 · "나을 것 같은" 아키텍처 변경 · §6 기준 사후 수정 · 결과 본 뒤 팔 추가/삭제 · 회전·translation을 nuisance 시험에 사용(§5.3-e) · STEP 1 plain/스칼펠 수치를 새 표에 혼입(설정 불일치, §3.3 ②). · **커밋 메시지에 `Co-Authored-By` 라인 추가 금지**(사용자 전역 규칙; 라운드 1 커밋에 붙어 있었음) · 기존 결과 행 삭제·덮어쓰기 금지(구현이 바뀌면 새 행으로 추가하고 옛 행에 라벨).

### 2.4 멈추고 물을 것 (STATUS "열린 것"에 적고 대기)
- §3.4 sanity에서 DC 오프셋 자릿수가 motion |ΔL|와 다를 때 — gain 범위를 임의 조정하지 않는다.
- LIBERO hdf5에 물체 pose 필드가 없을 때(§5.3-c).
- LIBERO 저장 상태 재생 + robosuite segmentation 렌더가 불가할 때(§5.3-c).
- §6 kill 조건 해당 시 — C2–C4 제출 전.
- 그 밖에 이 문서가 답하지 않는 설계 선택이 필요할 때.

### 2.5 매 단계 기록
잡 제출 즉시 `cluster_sessions.md` 표(비용 대조) · 단계마다 `STATUS.md` "돌아가는 잡"·"열린 것" · 결과 CSV = `paper_artifacts/tables/refinement_floor/`(seed·ckpt 경로·probe 설정을 헤더 주석에).

---

## 3. 새 base C1 — 프레임 쌍 독립 밝기 증강 (BRIGHT_AUG)

### 3.1 목적
M이 "휘도 변화 중 motion인 것"만 남기고 photometric을 버리도록 학습 신호를 준다. 현재 사전학습엔 photometric 증강이 전혀 없다(ColorJitter는 LIBERO BC 파인튠 전용 — 인코더 무관).

### 3.2 스펙
위치: `_forward_pair_comp` 안, 회전 증강 직후·channel 계산 직전(`two_stream_v15.py:911-923`). `self.training`일 때만.

```
img_t_aug, img_tk_aug = photometric_aug(image_current, image_future)   # 프레임별 독립 샘플
p_channel_t  = compute_p_channel(img_t_aug)                    # P 입력·타깃 = 증강 이미지 (P 목적함수 불변)
p_channel_tk = compute_p_channel(img_tk_aug)
m_in_real    = compute_m_channel(img_t_aug, img_tk_aug)        # M 인코더 입력 = 증강 ΔL (nuisance 포함)
m_tgt_real   = compute_m_channel(image_current, image_future)  # M-recon 타깃·guard-7 가중 = 깨끗한 ΔL
m_in_null    = compute_m_channel(img_t_aug, img_t_aug2)        # Case A 입력 = 같은 프레임의 다른 증강 두 벌 → 순수 photometric 차
m_tgt_null   = 0                                               # Case A 타깃 = 0
m_real(P-side routing helper) = encode_m(m_in_real)            # M 인코더는 학습 내내 증강 입력만 본다
```

증강(픽셀 [0,1] 공간, 프레임마다 독립, 결과 clamp [0,1]):
- **전역 gain** g ~ U[0.8, 1.2], 항상.
- **공간 ramp** prob 0.5: 임의 방향 d, 진폭 a ~ U[0, 0.15], field = 1 + a·(2·proj_d(x,y) − 1), 곱셈.
- 그 외(색·대비·노이즈·blur·국소 그림자) **넣지 않는다** — §5.3-e에서 "학습 때 못 본 nuisance"로 남겨야 한다.

플래그: `--bright-aug`(`BRIGHT_AUG=1`) · `--bright-gain-range 0.8 1.2` · `--bright-ramp-prob 0.5` · `--bright-ramp-amp 0.15`. 전부 기본 off/위 값. epoch당 1회 로그: gain·ramp 샘플 통계 + **|ΔL_in − ΔL_tgt| 평균(DC 오프셋 크기)**.

### 3.3 결정 (변경 시 사용자 승인)
① **입력은 증강 ΔL, 타깃·가중은 깨끗한 ΔL**(denoising 형태). 타깃까지 증강이면 모델이 DC를 재현하도록 배워 불변성이 안 생긴다. guard-7 가중도 깨끗한 |ΔL|로 계산해야 정지 패치가 photometric 때문에 motion으로 오인되지 않는다. 구현 = `_recon_dL(m_channel, …)`을 `(m_in, m_tgt, …)`로 분리, off면 `m_in is m_tgt`.
② **caseA_prob 1.0 · indep_rot 0.1 = 제출본과 동일.** 제출본 S가 그대로 "증강 off" 팔(C0)이 되게 하기 위함. STEP 1 plain·스칼펠(caseA 0.25·indep_rot 0)은 설정이 달라 새 표에 넣지 않는다.

### 3.4 sanity 게이트 (본 학습 전 필수)
1. CPU smoke: `BRIGHT_AUG=0`에서 기존 ckpt forward loss가 수정 전과 일치(≤1e-6).
2. 1ep sanity(MAX_VIDEOS=1000, mig 또는 1 GPU): L_mB가 제출본 sanity(~0.02–0.04) 자리 · L_mA가 0으로 붕괴하지 않고 **양수 유지**(photometric→0을 배우는 신호) · DC 오프셋 로그가 motion |ΔL|와 같은 자릿수(≈0.05–0.1). 다르면 §2.4.
3. 본 학습 SUFFIX = `refine_comp_s_bright`. 제출 즉시 cluster_sessions 갱신.

---

## 4. 학습 셀 (전부 C1 설정 공유 = §2.1 + `BRIGHT_AUG=1`)

| 셀 | env 차이 | 답하는 질문 | 새 코드 | GPU·h |
|---|---|---|---|---|
| **C1 새 base** | `BRIGHT_AUG=1` | — | §3 | ~110 |
| **C2 거울 ablation** | `V15_COMP_MAE=0 V15_PIXEL_PRED=1 V15_MASKED_ANCHOR=1 V11_ROUTING_MODE=v_from_p` | M-recon이 필요한가 — routing은 두고 M-recon만 제거(3자 지적) | 없음¹ | ~110 |
| **C3 표준 cross-attn** | `V15_COMP_MAE=1 V11_ROUTING_MODE=v_from_m` | value-ownership routing이 기여하나 — Q=owner, K·V=helper, 파라미터 수 동일 | 없음² | ~110 |
| **C4 M-only ΔL MAE** | 별도 클래스 | P와의 co-recon이 단일 스트림 ΔL MAE보다 나은가 | **있음**³ | ~60 |

¹ C2 = MCP-MAE(`pixel_pred`) 경로. 제출본 plain(36652564)과의 차이는 `v_from_m→v_from_p` 하나. 잔여 차이(null helper: pixel_pred는 `M_enc(ΔL=0)` full-pass, CoMP는 learned null token — cluster_sessions 07-04) — **맞추지 말고** 각주로 공개.
² `RoutingInterpreterStep`이 M-recon 블록에도 `routing_mode`를 전달(`two_stream_v15.py:459`) → 플래그만으로 양쪽 모두 표준 cross-attn. `v_source`는 기본 `owner`(`v_from_m`+`helper`는 코드가 거부). CPU smoke로 `comp_mae + v_from_m` forward/backward 확인 후 제출.
³ C4 스펙: M 인코더(depth 6·384/6, C1의 M과 동일 파라미터 수 — 로그로 확인) + 3층 decoder + ΔL head. 입력·타깃 = §3.3 ①(증강 입력·깨끗한 타깃). 마스크 0.5·guard 7(floor 0.02)·Case A/B 동일. **P 인코더·routing 없음**(self-attn만). `_recon_dL`의 helper 없는 형제 함수로.

사전 예측(판정 아님): C2는 M motion이 plain처럼 붕괴 · C3는 probe 효율은 C1 근접, cross-leakage(M-identity·P-motion) 증가 · C4는 F2와 C1 사이.

비용: S 제출본 실측 13h42m ≈ 110 GPU·h ≈ 28만 원(H100 61,000원/GPU·일, VAT 별도). 4셀 ≈ 390 GPU·h ≈ 100만 원. **월 누적 ceil 청구**라 같은 달 안에 몰아 돌린다. part1 학습이라 full-data I/O 병목은 무관.

---

## 5. 평가 설계 (학습 0, probe만)

### 5.1 공통 규율
- **판독기 동일**: 전 팔 `AttentivePoolProbe`(learnable query pool + linear) 하나. 입력 토큰 [B, n, D]만 다름. mean-pool 병기 금지(제출본에서 중립 아님 확정).
- **probe 데이터 동일**: 팔 간 split·샘플 수·gap 완전 일치. 새 ckpt 평가 전 `eval_protocols.md` §0 체크리스트 7항목을 로그에 명시.
- **벤치**: CALVIN xfold(`SPLIT=training CROSS_FOLDER=1 MAX_EPISODES=200`, gap 30) · LIBERO spatial/object/goal(gap 20, `READOUT=attentive`). 정확한 값·스크립트 = `eval_protocols.md` §4 · `paper_artifacts/tables/step0_ood_efficiency/` **그대로 재사용**. EgoDex in-domain은 참고 열(천장 0.47).
- **시험 두 종류 — 표·그림에 라벨 필수**:
  - ⓘ **분포 동일**: probe 학습·시험이 같은 도메인 split(위 4벤치 표준 split·EgoDex). 그 "OOD"는 사전학습 기준일 뿐이고 raw ΔL엔 그 프레임이 적용조차 안 된다. **raw 유리·parity가 구조적 정상 → 참조 열, 판정 근거 아님.**
  - ⓢ **분포 상이**: probe 학습에 없는 조건 — nuisance 교란(§5.3-e)·suite 간 무재학습 전이(§5.3-d). **인코더가 이겨야 하는 유일한 자리 → 판정 근거.**
- **M 단독 열이 1차**, 헤드라인 P_t⊕X는 2차(§5.3-a).

### 5.2 팔
| 팔 | 토큰 | 답하는 질문 |
|---|---|---|
| **C0** 제출본 CoMP-S M(`attentive_m`) | 196×384 | 증강 off 기준 |
| **C1** 새 base M | 196×384 | 본 대상 |
| **F1 raw ΔL** | ΔL 16×16 패치 raw, 196×256 | 학습 없이 입력만으로 (리뷰어 바닥선) |
| **F1′** | F1을 고정 랜덤 투영으로 384 | 차원 공정성(라벨 sweep 반론 차단) |
| **F1-aug** | F1 + probe 학습 데이터에 §3.2 증강 동일 적용 | "raw probe는 증강을 못 봤다" 반박 차단. 예측: gain·ramp 격차 닫힘(저차원 → 선형 투영 제거), **shadow는 안 닫힘**(고차원·국소) |
| **F2 normalized ΔL** | F1에서 전역 평균 제거 + 패치별 (x−μ)/σ | 손 정규화의 몫 |
| **F3 random-init M** | C1 구조, 미학습, seed 3 | 구조만 있고 학습 없을 때 |
| **F4 DINOv2 + ΔL** | ΔL 3채널 복제 → DINOv2 ViT-B/14 패치 토큰 | 범용 인코더에 같은 입력(UnGc W1) |
| F5 R3M + ΔL (선택) | 같은 방식 | ICRA R3M^Δ 0.525 vs CoMP 0.576 교차 확인 |
| **F6 raw + 표현력 판독기 (천장)** | F1 → M 크기 transformer(depth 6·384/6) 타깃 라벨 100% scratch + pool + linear | 타깃 라벨로 도달 가능한 상한. C1이 가까우면 인코더가 그 일을 미리 한 것, 훨씬 높으면 **동결 인코더가 정보를 버린 것**(BC parity 단서). **판정 미사용, 참조 행** |
| C2·C3·C4 M | 196×384 | §4 |

P_t⊕X 형식: P_t 토큰(C1의 P, 384)과 X 토큰을 **토큰 축 concat** → 같은 pool. D≠384인 X는 probe 안 학습 선형(256→384, 768→384) — probe 용량에 포함됨을 표에 명시.

### 5.3 시험

**(a) 효율 사다리 — 헤드라인 표 형식.** 한 표, 사다리 순서, P_t는 전 행 동일(C1의 P). 각 단이 리뷰어 질문 하나.
| 단 | 행 | 답하는 질문 | 근거 |
|---|---|---|---|
| 0 | P_t⊕P_tk(두 프레임, 제출본 배포 readout) → P_t⊕raw ΔL | **입력**으로 ΔL을 주는 것의 몫. 제출본·ICRA 격차(0.38 vs 0.52–0.70)는 여기까지만 — **판정 아님** | 기존 + F1 |
| 1 | P_t⊕raw ΔL → P_t⊕CoMP M | ΔL을 **학습**하는 것이 raw보다 나은가 | F1 vs C1 |
| 2 | P_t⊕M_alone → P_t⊕CoMP M | **RGB와 연계**한 학습이 M 단독보다 나은가 (AI W5) | C4 vs C1 |
| 3 | P_t⊕M_cross-attn → P_t⊕CoMP M | 연계의 **형태**(value-ownership)가 기여하나 (ote3 W7) | C3 vs C1 |
각 단 Δ를 ⓘ/ⓢ별로. 사전 추정: 1+2 합산 ≈ 0.05(ICRA 근거). 1·2가 0이면 "입력이 전부" — 그것도 결론.

**(b) 라벨 효율 sweep.** probe 학습 라벨 100→20→5→2%(seed 3, 비율마다 재샘플). F1은 라벨 적을 때 무너지고 C1은 버텨야. F1′ 포함. 구현 `--label-frac`. F6 천장 대비 도달률 병기. **표상 품질 진단으로만 서술.**

**(c) 다중 타깃.** 같은 특징·같은 판독기, 타깃만 교체(`TARGET=` 확장, 기존 `action`/`identity` 불변).
| 타깃 | 라벨 | 기대 |
|---|---|---|
| EE 변위(기존) | LIBERO/CALVIN 상태 | F1 높음(거의 선형) |
| **물체 이동**(조작 대상 pose Δ) | LIBERO 데모 sim 상태 object pose — **필드 존재 먼저 확인**(§2.4) | F1은 그리퍼/물체 변화 구분 못 해 열위 기대 |
| **시간 방향**(t→t+k vs 역순) | 페어 순서 뒤집기, 자동 | ΔL 부호 반전이라 F1도 가능할 수 있음 — 어느 쪽이든 보고 |
| **motion attribution**(패치별: 정지/로봇/물체/photometric) | LIBERO 저장 상태 재생 → robosuite segmentation 렌더 → 패치 다수 클래스 + ΔL 임계; photometric 클래스는 shadow 주입으로 — **재렌더 가능성 먼저 확인**(§2.4) | **routing이 도울 메커니즘적 이유가 있는 첫 타깃**(P가 grouping·M이 값). raw·R3M+ΔL은 "변했다"만 보여 원리적 불가. **M 단독은 외형 없어 실패 예상** → 1차 비교 = **P_t⊕CoMP M vs P_t⊕raw ΔL**(routing의 몫). 패치별 타깃은 pool 없이 토큰별 linear head — 별도 표기 |
| **접촉 이벤트**(물체 이동 원인 = 그리퍼?) | 물체 pose Δ>임계 ∧ 그리퍼–물체 거리<임계, 또는 contact | 두 움직임의 관계 → 복잡. 조작 성공 판정 자체(ote3 W2 affordance) |

**(d) probe 무재학습 전이 (ⓢ).** LIBERO 한 suite에서 맞춘 probe를 다른 suite에 **재학습 없이** 적용, 6방향 평균(같은 카메라·로봇 = 좌표계 공유, 물체·배치만 다름). raw probe는 픽셀 배치에 묶여 무너질 것. 구현 `--probe-load <ckpt> --no-fit`. **CALVIN↔LIBERO 전이는 좌표계가 달라 넣지 않는다.**

**(e) nuisance 강건성 (ⓢ) — "ΔL≠물리 motion"의 실험적 답.**
```
1. 깨끗한 train split로 각 팔 probe fit → 고정·저장
2. eval split의 t+k 프레임(gain·ramp·shadow) 또는 양 프레임(noise)에 교란 주입, 강도 s 단계
3. 교란된 프레임 쌍에서 ΔL 재계산 → 인코더 → 고정 probe   ← 프레임을 건드리고 ΔL을 다시 계산. ΔL을 직접 건드리지 않음
4. ΔR²(s) 곡선, 전 팔 겹쳐 그림
```
| 교란 | 정의 | 학습 증강에 | 해석 |
|---|---|---|---|
| gain | ×g, g∈{0.9,1.1,1.2,1.3} | 있음 | by-construction 참고 열 |
| ramp | 임의 방향, a∈{0.1,0.2,0.3} | 있음(파라미터 랜덤) | in-distribution nuisance |
| **shadow** | 임의 위치 타원 내부 ×(1−b), b∈{0.2,0.4,0.6}, soft 경계 | **없음** | **핵심 시험** — 못 본 국소 photometric, 손 정규화·선형 probe로 못 지움 |
| noise | Gaussian σ∈{0.01,0.02,0.04} 양 프레임 독립 | 없음 | ΔL은 분산 2배 증폭 |
구현 `--perturb {gain,ramp,shadow,noise} --perturb-levels …`, probe 1회 fit 후 여러 s를 한 잡에서(`--eval-perturb-list`). **회전·translation 금지** — 독립 crop·회전 증강으로 학습 때 본 nuisance라 증거가 못 된다(제출본 App A에 회전만 적고 독립 crop은 미기재 — 재투고 시 공개).
**반대편 측정**: 같은 특징에서 교란 파라미터(g,a,b) 자체를 회귀(`TARGET=nuisance`). F1은 거의 완벽 복원. C1이 F1·F2·F1-aug보다 **낮게** 복원하면 선택적 폐기 = refinement의 직접 증거. 그림: x = nuisance 복원 R², y = shadow 하 motion R². 원하는 자리 = 좌상단.

---

## 6. 사전 등록 판정 기준 (측정 전 고정 · 사후 수정 금지)

**refinement 성립** = 아래 세 축 중 **둘 이상**에서 C1 M이 F1·F1-aug·F2·F3을 이기고, 그중 **(A) 또는 (C)가 반드시 포함**(둘 다 ⓢ — probe 학습 분포 밖의 증거 없이는 인코더의 환원 불가능한 이점이 없다. (B) 단독 불가). M 단독 열 · seed 3 · 95% CI 비겹침.

- **(A) 범용성** — probe 무재학습 전이(§5.3-d) 6방향 평균 우위. 다중 타깃(§5.3-c)은 보조: attribution(P_t⊕M vs P_t⊕raw)·물체 이동·접촉에서 이기면 강한 근거, EE 변위만이면 약한 근거. **ⓘ 4벤치는 (A)에 넣지 않는다.**
- **(B) 효율성** — 라벨 5%·2%에서 우위(F1′ 포함). 100% parity여도 성립. 진단 축.
- **(C) 강건성** — shadow에서 **F1-aug**의 R²가 절반으로 떨어지는 s*에서 C1 하락이 그 절반 미만 **그리고** nuisance 복원 R²가 F1·F2·F1-aug보다 유의하게 낮음. gain·ramp에서 F1-aug와 parity는 예상 결과.

**해석 규칙**
- ⓘ 시험(4벤치 표준 split·EgoDex) 100% 라벨에서 F1 parity 또는 F1 우위 = **예상**. 실패로 읽지 말 것.
- 한 축만 성립 → 그 축만 주장((C)만이면 "photometric 강건성"으로 축소).
- 셋 다 불성립 → **효율 주장 폐기**(F4가 C1과 parity면 특히). 남는 논문 = C2·C3·factorization 메커니즘.
- 증강 효과 = C1 vs C0를 각 축에서 별도 열. 같으면 증강은 불필요했던 것 — 그것도 결과.
- 사다리(§5.3-a) 1·2단이 0 → "입력이 전부". 그것도 결론.

**계획 중지 판단 (사용자 확정 09-25) — 사다리 1단을 ⓢ에서 본다.**
- **중지 신호** = ⓢ 시험(shadow s*·noise·§5.3-d 전이) 전부에서 **P_t⊕C1 M ≤ P_t⊕raw ΔL(F1·F1-aug)** 또는 **≤ P_t⊕random-init M(F3)**. M 단독 열도 동일. raw나 random-init에 지는 것만은 어떤 변명도 없다(인코더가 raw에 없는 것을 어떤 조건에서도 더하지 못함 / 학습이 해를 끼침).
- **중지 신호가 아닌 것**: ⓘ에서 raw와 parity(예상 결과) · F4(DINOv2+ΔL)에 열위(효율 프레임으로 변명 가능: 86M·인터넷 스케일 vs 32M·46k) · 라벨 sweep 열위(진단 축) · ⓘ 4벤치 열위.
- **사다리 0단**(P_t⊕P_tk vs P_t⊕raw ΔL)이 뒤집히면 과학적 신호가 아니라 **파이프라인 고장 신호**(ICRA 0.38 vs 0.52 격차가 사라질 이유 없음) — probe 배선 점검.
- **시점** = §7 4단계(C1 완료 + §5 전체 측정) 한 곳. §7 2단계(C0 vs F1)는 조기 경보일 뿐 중지 아님(C1이 기록 모델이고 이미 학습 중). §3.4 sanity는 기술적 중단으로 별개.
- 중지 신호 시 **C2–C4 제출 전에** 사용자 보고(§2.4). C2–C4는 factorization 논문에 여전히 필요할 수 있으므로 사용자가 결정.

---

## 7. 실행 순서·체크포인트

1. **§3 구현** → CPU smoke(off = 기존과 동일) → 1ep sanity(§3.4) → **C1 본 학습 제출**. cluster_sessions 갱신.
2. **C1 학습 중(≈14h)**: §5.2 F1·F1′·F1-aug·F2·F3·F4 + C0 probe 배선·측정 → §5.3-a(0단·1단은 C0로 선측정)·(b)·(c)(라벨 확보 확인 포함)·(d). 이 시점에 **F1 vs C0**가 나온다 — 첫 실질 정보. STATUS "열린 것"에 숫자.
3. **§5.3-e 배선**: C0·F1·F2로 곡선 먼저(파이프라인 검증).
4. **C1 완료** → §5 전체 측정 → §6 판정 기록. **★ 유일한 계획 중지 판단 시점**: ⓢ에서 P_t⊕M이 raw ΔL·random-init에 밀리면 멈추고 보고(§6 중지 판단).
5. **C2·C3**(플래그만) → **C4**(새 클래스) 제출, 같은 달 안에. 각 셀에 §5 동일 팔·시험.
6. 결과 표 = `paper_artifacts/tables/refinement_floor/`(CSV + 생성 스크립트).

---

## 8. 용어

- **C0** 제출본 CoMP-S(잡 36177296, ckpt `two_stream_v15b_step1_comp_mae_s/20260629_101634`) = 증강 off.
- **C1** 새 base = C0 + BRIGHT_AUG. 이후 연구의 기준 모델. **C2/C3/C4** §4.
- **F1–F6** 학습 없는 바닥선·참조 팔(§5.2).
- **ⓘ/ⓢ** probe 학습·시험 분포 동일/상이(§5.1). 판정은 ⓢ에서만.
- **refinement/정제** = 같은 판독기에서 raw ΔL보다 읽기 쉽고(라벨 적어도) photometric nuisance에 덜 흔들리며 다른 조건으로 옮겨가는 성질. 정보량이 아니다.
- **value-ownership routing** = `routing_mode=v_from_p`: Q·K는 helper, V·residual은 owner. **표준 cross-attn** = `v_from_m`: Q=owner, K·V=helper. 정의 = `src/models/common/blocks.py` MotionRoutingBlock docstring.
- **거울 ablation**(C2) = routing 유지·M-recon 제거. **스칼펠**(V_P) = V 소유만 뒤집은 다른 실험 — 이번엔 안 쓴다.
- **guard 7/8** = comp_mae_plan의 per-patch |ΔL| 가중(floor) / M 마스크 비율 < P.

---

## 9. 라운드 2 (2026-09-26 지시) — 학습 없음. probe 3건 + 진단 2건

### 9.0 왜
라운드 1이 남긴 질문 셋에 학습 전에 답한다. **§2 행동 규칙 전부 그대로 적용.** 학습(Case A 입력 노이즈 수정 등)은 라운드 2 결과를 본 뒤 사용자가 정한다 — dev가 먼저 학습을 제출하지 않는다.

- **Q1 정확히-0 의존이 진짜 원인인가.** 라운드 1: 학습된 M이 σ0.01 노이즈에 붕괴, raw·random-init은 불변. CALVIN 정지 ΔL의 73%가 정확히 0 → Case A(ΔL≡0→0)로 배운 sim 전용 단서 의심. 이게 맞으면 **같은 분포의 0.46–0.71도 그 단서에 기대 있을 수 있다.**
- **Q2 P가 어디까지 짐인가.** P_t⊕raw 전이 −10 vs raw 단독 −0.32. 그러나 (i) 전이 시험은 전 팔 음수(변별력 없음) (ii) raw 토큰을 **zero-pad**로 384에 맞췄음(계획 §5.2는 학습 선형 투영) → 노름 불균형 의심 (iii) CALVIN P_t⊕X probe best epoch 2–3 = 즉시 과적합(wd 0).
- **Q3 P 두 장은 어떤가.** 사다리 0단 P_t⊕P_tk가 ⓢ 시험에 없다. P는 z-score라 gain 불변이고 정확히-0에 의존하지 않으므로, **noise·shadow에서 P_t⊕P_tk가 버티면 취약성은 M에 국한**(배포 P-only는 무사)이고, 같이 무너지면 인코더 전체 문제.

### 9.1 작업 (전부 probe, V100)
| # | 작업 | 팔 | 시험 | 비고 |
|---|---|---|---|---|
| R2-1 | **P_t⊕P_tk 팔을 ⓢ 시험에 추가** | C1의 P (`attentive_concat_p_t_p_tk`), 참조로 C0의 P | LIBERO 전이 6방향 · CALVIN perturbation 전 종(gain·ramp·shadow·noise 전 강도) · CALVIN clean · 라벨 sweep | 기존 readout 경로, 새 코드 없음. 결과 표에 사다리 0단 행으로 |
| R2-2 | **P_t⊕raw 패딩 수정 재측정** | P_t⊕raw ΔL — raw 토큰 256→384 **probe 안 학습 선형 투영**(§5.2 명세) | 라운드 1과 동일 전 시험(ⓘ·전이·perturbation·라벨) | zero-pad 결과는 **삭제하지 말고** `pad=zero` 라벨로 보존, 새 행 `pad=linear`. 같은 방식으로 P_t⊕C1 M도 재측정 불필요(D=384) |
| R2-3 | **"현실적 sim" 조건 — 같은 분포 표 재측정** | C1 M · C0 M · raw · F1′ · F3 random-init · P_t⊕P_tk · P_t⊕C1 M · P_t⊕raw(linear) | CALVIN clean(ⓘ) + LIBERO suite 내(ⓘ) · **probe 학습·시험 양쪽 프레임에 Gaussian σ∈{0.01, 0.02} 독립 주입 후 ΔL 재계산** | 새 플래그 `--probe-noise-sigma`(기본 0), 픽셀 프레임에 적용 → 모든 팔 같은 파이프라인. seed 고정. 이것은 §5.3-e(시험만 교란)와 **다른 시험** — 학습·시험 분포가 같으니 ⓘ로 라벨 |
| R2-4 | **진단: 정확히-0 비율** | — | CALVIN·LIBERO 3suite·EgoDex part1(Case B 실제 쌍) 각 500쌍에서 `ΔL == 0` 픽셀 비율, 패치 단위 "전부 0" 비율 | GPU 불필요. EgoDex는 실제 영상이라 ~0%여야 함 — 그러면 "정확히 0"은 Case A에서만 본 단서라는 진단이 확정 |
| R2-5 | **진단(선택): probe 정칙화** | P_t⊕raw(linear) · P_t⊕C1 M | CALVIN clean + LIBERO 전이, `PROBE_WEIGHT_DECAY∈{0.01, 0.1}` | "P에 즉시 과적합" 가설 확인. 판정 아님, 참고 |

### 9.2 사전 해석 (결과 전 고정 — 판정 기준이 아니라 읽는 법)
- **R2-3**: 노이즈 양쪽 주입에서 C1 M이 raw의 **2배 이상 유지** → 같은 분포 우위는 정확히-0 산물이 아니고, 노이즈 취약성은 분포 이동 문제로 국한. C1 M이 **raw 수준으로 하락** → 라운드 1과 제출본의 probe 수치가 sim 단서의 산물 → 논문의 모든 probe 표가 현실적 조건 재측정 대상.
- **R2-1**: P_t⊕P_tk가 noise·shadow에서 버팀(raw 수준 이상) → 취약성은 M 국한, 배포 표상 무사. 같이 무너짐 → 인코더 전체.
- **R2-2**: linear 투영 후 P_t⊕raw 전이가 raw 단독에 근접 → −10은 패딩 인공물. 여전히 크게 나쁨 → "P가 판독기를 무너뜨림"이 실재(제출본 BC causal confusion과 같은 현상으로 서술).
- **R2-4**: EgoDex Case B 쌍의 정확히-0 비율 ≈ 0% 확인 시 → Case A 입력을 정확히 0으로 준 것이 sim 전용 단서 학습의 원인 [진단 확정] → 다음 학습 수정 후보 = Case A 입력에 센서 노이즈. **이 학습은 사용자 결정 후.**

### 9.3 순서·기록
R2-4(즉시, CPU) → R2-1·R2-2·R2-3 잡 일괄 제출(같은 날) → R2-5(여유 시). 결과 = `paper_artifacts/tables/refinement_floor/round2_*.{csv,txt}` + README에 라운드 2 절 추가. STATUS "열린 것"에 Q1–Q3 답을 한 줄씩. **학습 제출 없음.**
