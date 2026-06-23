# Parvo (code v15b) 재학습 진행 상태 (세션 핸드오프)

> **명명** (2026-06-11): 논문 핵심 모델 = **`Parvo`** (현재 구현 = code `v15b`, student-anchor). 직전 **v15**(teacher-anchor)는 버전명 유지. "catalyst" → **scaffold**로 용어 통일. 정규 출처 = [`CLAUDE.md`](../CLAUDE.md) "명명 · 2논문 구조". 코드 식별자(`MODEL=two-stream-v15b` 등)는 rename 보류 중이라 유지.
> **브랜치**: `main` (Parvo = 새 구조, code v15b). v15(논문 teacher-anchor)는 `paper-corl2026` 브랜치와 영구 분리.
> **최종 업데이트**: 2026-06-22 (§10 Parvo BC-T 결과, §11 no-M ablation 계획). 이 문서는 다른 세션이 작업을 이어받기 위한 현황 메모.
> **후속 (2026-06-23)**: Parvo/no-M 학습이 **part1 서브셋**(사고)이고 clean ckpt 부재(collapse) 확인 → **subset-matched small 재시작** 결정. 실행순서·gate·주의 = [`restart_plan.md`](restart_plan.md). §10 BC 78.5는 서브셋·붕괴 잔재라 trend용(절대·앵커 주장 불가).

## 1. 목표

원래 v15(teacher-anchor)는 V-JEPA P anchor=`teacher_p(frame_t).detach()`라 **P encoder가 motion(M) gradient를 전혀 못 받고 순수 MAE로 독립 학습**됨. `0fb74c8`에서 anchor를 student P encoder로 바꿔(표준 V-JEPA 복원) P·M 모두 motion routing gradient를 받게 함. `b41b177` = **v15b** (동일 아키텍처 + collapse 방지 레시피: ① recon-first hard-gate ② EMA 0.996 ⑤ lr scaling).

**검증 질문**: M→P gradient를 실제로 연결했을 때 **scaffold**가 작동해 VideoMAE(+0.47)/v15(+0.39, P_t⊕P_tk)를 넘는가? 못 넘으면 "multi-frame MAE concat이 강력한 단순 baseline"으로 정직하게 재서술.

관련: `paper-corl2026:docs/PAPER_CORL2026_PLAN.md` §2 (D1/D2 확정 — symmetric multi-frame MAE가 핵심, motion routing은 load-bearing 아님).

## 2. 이번 세션 코드 변경 (commit 포함됨)

- `scripts/cluster/pretrain.sbatch`: v15 분기가 `two-stream-v15b`도 매칭. `V15_GATE_EPOCHS`(default 0=no-op) + 모델별 EMA init default(v15b=0.996, v15=0.999) 노출. CKPT_DIR `${MODEL//-/_}`로 v15/v15b 분리.
- `scripts/cluster/sanity_v15.sbatch`: `MODEL` override + `V15_GATE_EPOCHS` + `SAVE_INTERVAL` env 노출 + 모델별 EMA/CKPT 분리. (v15 기존 동작 불변.)
- 코드(`two_stream_v15.py`, `scripts/pretrain.py`, `src/training/pretrain.py`)는 이미 b41b177/0fb74c8에 v15b dispatch·gate 헬퍼·student-anchor 반영됨 (이번 세션 변경 없음).

## 3. 실행 환경

- 클러스터 olaf, `MODEL=two-stream-v15b` → `scripts/pretrain.py` dispatch (env-agnostic).
- EgoDex frames: `/proj/external_group/mrg/datasets/egodex/frames/part1..part5`
- conda: `/proj/external_group/mrg/conda_envs/aavrl-train`
- 로컬 워크스테이션 런처도 존재: `scripts/local/pretrain.sh --model two-stream-v15b` (DataParallel, `/mnt/data`).

## 4. Sanity 결과

### Sanity #1 — JobID 35478663 ✅ (gate=0, 50vid×5ep)
- **구조 안정** (NaN/발산 없음), **P MAE 건강** (L_t 0.049→0.0186, std_p 0.06→0.45, cos_intra_p 0.996→0.66).
- ⚠️ **L_pred trivial collapse**: 0.044→0.0014@ep2, cos(pred,tgt) 0.955→0.998 → student-anchor catalyst 채널 신호 ≈ 0.
- ⚠️ std_m ep2-3 near-collapse(0.010)→ep5 0.118 회복. L_compose ~0.42 stuck (total의 87%).
- **단, gate=0이라 미성숙 P에서 V-JEPA 켜진 preview** — gate=10 본학습과 timing 다름 → catalyst 사망 미확정.
- ckpt 미저장(save-interval 999)이라 diagnose 불가 → sanity #2 재실행.

### Sanity #2 — JobID 35481545 ✅ (gate=3, 200vid×8ep) — gate가 collapse 방어
- gate 중(λ_pred=0) baseline cos(pred,tgt)≈0.77 = **과제 intrinsic trivial 아님**. gate 후 L_pred ~0.08 / cos ~0.94 plateau (sanity#1 0.002/0.998 collapse와 질적으로 다름). P 최종 건강(std_p 0.53, cos_intra_p 0.52). → **gate=10 본학습 정당화.**
- ckpt: `/proj/external_group/mrg/checkpoints/two_stream_v15b_sanity_gate3diag/20260610_181459/checkpoint_epoch0008.pt`

### Diagnose — JobID 35493291 ✅ (M=0 vs M-on, sanity#2 ep8 ckpt)
- baseline cos 0.9015, predictor 0.9279, M=0 0.9147 → **Δ(M routing 기여)=+0.0132**.
- trivial collapse 아님. M 기여 작으나 ep8 미성숙(M 5ep만) — teacher-anchor ep50은 +0.31였음. **maturity 신호로 해석, 본학습으로 확인.**
- gap별 baseline: gap0-9 0.92 → gap30 0.81 (큰 gap=motion 신호 많음; follow-up lever).

## 5. 본학습 — ⏸ 보류 중 (1차 캔슬, 리팩토링 후 재제출 예정)

**진행 경과**:
- 35493293 (1차): 8 GPU DDP 정상 기동, **63min/ep 실측**(50ep ~52h). 34m39s만에 **CANCELLED** — forward 최적화 적용 위해 조기 중단 (4.6 GPU·h).
- **forward 무손실 최적화 적용+검증 완료** (§6 참조). smoke sanity 35493303 통과.
- **⏸ 재제출 보류**: 사용자가 **추가 리팩토링을 다른 워크스테이션에서 진행** 후 재제출 예정. 리팩토링 내용 파악 대기 중.

### 🔧 forward 무손실 최적화 (2026-06-11, 적용됨)

**병목 진단**: 실행 중 GPU util 88-98% = **compute-bound** (데이터/3-frame 추출 아님). step당 full unmasked depth-12 P-encoder 6회 호출 중 2회 중복.

**수정** (`src/models/two_stream_v15.py`): forward에서 unique frame당 unmasked P 인코딩 1회만 계산해 `_vjepa_p_one_segment(anchor_repr_S=, target_repr_T=)`로 전달. student {t,t+n}, teacher {t+n,t+m} → full P forward 6→4. dropout 없어 deterministic + gradient 합산 동치라 **무손실**(smoke 검증: well-conditioned 항 sanity#1과 1-3% 일치). wall-clock ~15-20% 단축 예상.

### ▶ 다음 (리팩토링 파악 후)

1. 사용자의 워크스테이션 리팩토링 변경사항 파악 (diff 리뷰).
2. 통합 후 smoke sanity 재검증 (gate=0, 50vid, 3ep → loss 규모 sanity#1 대조).
3. 본학습 2차 재제출 (§6 명령). 첫 ep에서 **새 per-ep 시간 실측** (최적화 효과 확인).
4. **ep12-18 abort 모니터링**: cos(pred,tgt)>0.99 & L_pred<0.01(trivial collapse) 또는 train·eval 동반 발산 시 중단.
5. **mid-run diagnose** (ep20/30 ckpt): `diagnose_vjepa_p_trivial.py --ckpt <epXX>` → M 기여(+0.013→?) 성장 추적 = student-anchor 성공 핵심 지표.
6. **완주 후**: EgoDex P_t⊕P_tk probing vs 원래 v15(+0.39)/VideoMAE(+0.47).

### 미적용/보류 (필요 시 follow-up)
- **L_pred-only gate 분리**(M을 ep1부터 성숙): 미검증·M trivial collapse 위험 → shared gate 유지.
- **MAX_GAP 60**: RESEARCH_PLAN 검증된 개선이나 v15 비교 위해 30 유지.
- **lr 1e-4 / batch 64**: 안정성/속도 lever이나 comparability 위해 보류 (batch 키우면 update수·LR 변함).
- **추가 forward 최적화 후보**(다음 model ver): torch.compile, V-JEPA anchor를 masked context로(MAE 토큰 재사용), segment 3→2.

## 6. 본학습 2차 재제출 명령 (리팩토링 통합 후)

```bash
sbatch --partition=AIP_long --nodes=2 --ntasks-per-node=4 --gres=gpu:4 \
    --cpus-per-task=8 --time=5-00:00:00 --job-name=aavrl_v15b_main \
    --output=/proj/external_group/mrg/logs/pretrain_v15b_%j.out \
    --error=/proj/external_group/mrg/logs/pretrain_v15b_%j.err \
    --export=ALL,MODEL=two-stream-v15b,EPOCHS=50,BATCH_SIZE_PER_GPU=32,LR=2e-4,\
SPLITS=part1,part2,part3,part4,part5,V11_ROUTING_MODE=v_from_p,\
V15_GATE_EPOCHS=10,V15_EMA_INIT=0.996,\
V15_LAMBDA_PRED_WARMUP_START=0,V15_LAMBDA_PRED_WARMUP_EPOCHS=10,\
V15_LAMBDA_M_JEPA_WARMUP_START=0,V15_LAMBDA_M_JEPA_WARMUP_EPOCHS=10,\
V15_LAMBDA_COMPOSE_WARMUP_START=0,V15_LAMBDA_COMPOSE_WARMUP_EPOCHS=10,\
NUM_WORKERS=8 \
    scripts/cluster/pretrain.sbatch
```
- ckpt → `/proj/external_group/mrg/checkpoints/two_stream_v15b/`. 1차 63min/ep(미최적화) → 최적화 후 ~50min/ep 예상(첫 ep 실측 필요).

## 7. 결정된 설계 판단 (재론 불필요)

- **인코더(P/M) size 축소 ❌**: 원래 v15는 overfitting이 아니라 **P-CLS collapse + L_m_jepa 폭증 + late divergence(train·eval 동반 상승)**. M은 건강(cos_intra_m 0.27). +0.39 자산은 patch-level P_t⊕P_tk라 P 축소 시 자산 손상. capacity lever는 **predictor(p_motion_decoder)** 가 on-target (V-JEPA trivial 방지). 진짜 overfit 증거 나오면 weight decay/drop-path가 정공법.
- 아키텍처: embed_dim=768, P depth 12(~85M), M depth 6(~42M), decoder_depth_m=3, interpreter_depth=3.

## 8. 붕괴 진단 확정 + Run A/B 분리 (2026-06-15)

**붕괴 근인 재진단** (SSIM/LR/warmup 5연패 후): recon 노브는 전부 *증상*만 건드림. 근인 = **V-JEPA P 자기참조 collapse**. target = `teacher_p(frame_tk).detach()` = student P의 EMA 거울이라, student가 표현을 저분산/상수로 만들면 teacher(target)도 따라 무너져 loss→0. cos(pred,tgt)→0.993·P 균일 = 이 **constant 모드**. recon(L_t)은 약한 anti-collapse라 못 버팀.

**붕괴 종류 분리** (핵심):
- **constant 붕괴** (= 우리가 본 것): target 정규화로만 막힘. anchor 마스킹 무관.
- **identity 복사** (full anchor + 작은 gap): masked anchor가 막음. 단 이건 *본 붕괴와 다른 모드*.

**M 기여 측정 정직화**: 지금까지 "M no-op"의 실제 근거는 **옛 v15의 구조적 gradient=0**(코드 사실) 하나뿐. v15b ablation +0.013은 **inference 시 predictor 의존도**를 잴 뿐 *학습 중 P 표현 기여*가 아님 (옛 v15 ep50 ablation은 +0.31인데 encoder shaping은 구조적 0 → 이 metric이 엉뚱한 양 측정 증명). v15b의 M 표현기여 = **깨끗한 판정 없음**.

**Run A (지금) — 붕괴부터 끄기, 최소 변경**:
- 현재 구조(full anchor, pair_mode) 그대로 + **target 정규화**: ① target LayerNorm(V-JEPA 표준, scale collapse 방어) ② variance reg(VICReg식, std collapse 직접 금지 = 본 붕괴 정공법).
- 성공 기준: ep12+ 균일 붕괴 없음(std_p 유지) + L_t 복구 단조감소 + P_t⊕P_tk probing이 비붕괴값.
- 예상: 붕괴 멈춤·P 건강·복구 생존. **단 M은 여전히 no-op 가능**(full anchor identity → M 무시) — 정상. A는 scaffold 검증이 아니라 *깨끗한 비붕괴 baseline* 확보용.

**Run B (A 확인 후) — scaffold 검증**:
- A에 **masked anchor + completion(self-attn)→routing 순서** 추가. completion은 predictor(버려질 모듈)에 가두고 student encoder는 visible-only로 → 마스킹 이득을 encoder에 남김.
- 측정: **학습-시 ablation**(with-M vs no-M 두 run, P probing 비교)이 gold standard. inference ablation은 의존도라 부적합.
- 귀속: A=붕괴가 원인이었나, B=마스킹이 M을 살리나. 한 run에 섞지 말 것.

> A/B는 논문 두 경쟁가설과 대응: A만으로 P_t⊕P_tk 좋으면 "multi-frame MAE concat 강한 baseline", B에서 M 살아나야 좋아지면 scaffold 지지.

### ▶ 다음 단계 (모니터링·결정 흐름)

**구현 (2026-06-15, 커밋됨)**: `lambda_var`(VICReg variance reg, P enc 출력 per-dim std<1 hinge) + `target_ln`(I-JEPA target LayerNorm) 플래그. [two_stream_v15.py](../src/models/two_stream_v15.py) `_variance_loss`/`_vjepa_p_one_segment`/`_forward_pair`, [pretrain.py](../scripts/pretrain.py), sanity·pretrain sbatch env wiring. CPU smoke + sanity(35763563) 통과 — **std_p 0.58→1.0 궤적 역전 확인**.

**1. Run A 본학습 35764680 모니터링** (cap 20ep, ckpt 매 ep):
- **ep4~5 조기경보**: L_t 역행(옛 0.008→0.019) 없어야. 있으면 lambda_var 부족 의심.
- **ep8**: std_p 유지(↓0 아님), std_m 회복(옛 transient면 ep5+ 회복).
- **ep12 확정 판정**: cos(pred,tgt)<0.97 + std_p 유지 + viz(V100 mig, [feedback_viz_inference_min_spec]) GT급 → **통과**. cos→0.99 or 균일 viz → **붕괴, abort**.

**2. ep12 분기**:
- **통과** → ep16~20에서 멈춤(50 안 감). ep16/20 ckpt로 **EgoDex P_t⊕P_tk probing** → "비붕괴 baseline 수치"(vs v15 +0.39 / VideoMAE +0.47). 이게 "multi-frame MAE concat 강한 baseline" 가설 답.
- **붕괴** → lambda_var↑(1→4) 또는 target만으로 부족분 진단 후 재제출.

**3. Run B 준비** (A 통과 후 착수, scaffold 검증):
- masked anchor(student visible-only) + **completion(self-attn)→routing 순서 교정**(RoutingInterpreterStep 뒤집기, 토글 플래그로) + Run A의 target norm 유지.
- completion은 predictor(버려질 모듈)에 가두고 encoder는 visible-only → 마스킹 이득 encoder에 잔류.
- **측정 = 학습-시 ablation**: full Parvo vs M routing 제거 두 run → P probing 비교(= M의 *표현* 기여, gold standard). inference ablation(의존도)은 부적합 — [cluster_sessions](cluster_sessions.md) §35493291 참조.

### Run B 설계 원리 — P encoder의 task format 일관성 (핵심 렌즈, 2026-06-15)

> Run B를 "왜·어떻게" 설계하는지의 단일 출처. anti-collapse를 넘어선 *본질적* 근거.

**문제 인식**: 현재 P encoder student는 *일관되지 않은 두 종류의 task*를 동시에 받는다.
- MAE 복구: **masked**(visible-only) 입력 → 마스크 위치 추론 (interpreter_1 + recon_head)
- 모션 라우팅(현재): **full**(unmasked, `_encode_p_unmasked`) 입력 → 미래 표현 변환 (motion predictor)

→ 인코더 관점에서 입력 포맷이 다르다. MAE="부분에서 추론", 모션="전체에서 변환". 같은 가중치가 *두 다른 종류*의 능력으로 끌려가 — 양자택일 압력.

**통일 원리**: P encoder가 배우는 건 *복구 정보 자체*가 아니라 **"부분 관찰을 (interpreter/predictor가) 안 본 것으로 도출하게 하는 substrate"**다. 이 substrate는 공통 통화 — 공간 추론(복구)이든 시간 추론(모션)이든 같은 "부분→예측" 능력이 떠받친다. 모션도 **masked 입력**으로 통일하면:
- MAE: visible frame_t → masked frame_t 예측 (공간 target)
- 모션: visible frame_t → frame_tk 예측 (시간 target)
- 인코더에 *동일* 요구("부분을 예측-지원형으로 표현"), target/head만 다름 → **두 loss가 경쟁이 아니라 시너지.**

**masked anchor의 3중 명분** (이 일관성이 세 번째이자 가장 본질적):
1. anti-collapse: identity-복사 trivial 모드 차단 (full anchor + 작은 gap).
2. M load-bearing: predictor가 full frame_t로 못 풀게 해 모션을 필수화.
3. **task format 일관성**: 모션을 MAE와 *같은 종류*("부분→예측")로 만들어 인코더가 *단일 능력*을 학습 → 시너지.

**정직한 경계**: masked anchor는 *task type*(부분→예측)은 통일하나 *target 추상도*는 다름 — MAE=픽셀(저수준), 모션=latent(고수준). MAE↔JEPA tension은 *줄지만* 완전히 사라지진 않음. (후속 갈래: 모션 target도 masked 픽셀로 가면 추상도까지 통일 가능 — 검토 대상.) 또 우리가 *본* 파괴적 양자택일은 collapse dynamic이었고(variance reg로 차단됨), 붕괴 막은 뒤 full anchor는 "경쟁"보다 "no-op(무임승차)"에 가까움 → 본 논리는 부정형("full=경쟁")보다 **긍정형("masked=시너지")**으로 쓸 것.

**구현 함의**:
- V-JEPA P anchor를 `_encode_p_unmasked`(full) → `_student_p_encode_visible`(masked)로 교체. **MAE의 visible 인코딩을 *그대로 재사용*** (같은 mask·같은 forward) → 두 loss가 *같은 latent* 공유 + forward 절약.
- 모션 loss를 **visible 위치**에 둠 (각 visible 패치의 미래 예측).
- completion(self-attn)→routing 순서 (구멍 채운 뒤 모션 적용; 현재 RoutingInterpreterStep은 routing→interp 역순).
- completion은 predictor(버려질 모듈)에 가두고 encoder는 visible-only 유지 → bottleneck이 encoder에 박혀 마스킹 이득 잔류.

**loss 위치 비대칭 — 마스킹이 좋은 또 다른 이유** (제4 근거):
- MAE loss = **masked 위치** → visible latent *간접* 채점(interpreter 거쳐 "남을 복구하나").
- 모션 loss = **visible 위치** → visible latent *직접* 채점("제 미래를 예측하나").
- → 모션이 *배포 표현(visible patch)*을 **직접·덜 희석된** 신호로 co-shape. masking이 MAE/모션을 *같은 forward*로 묶어 인코더가 하나의 latent으로 둘 다 만족하게 함(full anchor는 masked/full 다른 forward = 분리됨).
- **단 양날**: directness는 *증폭기*. 모션 건강하면 강한 scaffold, degenerate하면 collapse 가속. directness 이득은 non-triviality(masking+붕괴방지)와 세트로만.

**검증 ablation (variance reg 필요 여부 — masking만으로 충분한가 가설)**:
- **B-1**: masked anchor + variance reg (안전, belt-and-suspenders).
- **B-2**: masked anchor **without** variance reg (가설: 마스킹이 상수해 basin을 자연 회피 — I-JEPA 선례).
- B-2 안 무너지면 → variance reg 폐기(덜 왜곡된 표현, std≥1 강제 없음). 무너지면 보험 유지.
- 주의: 마스킹은 상수 *최소점을 제거하진 못함*(teacher=EMA student라 상수도 loss=0) → basin of attraction 문제, 순수 경험적.
- **왜 B-2가 "보장 아님"인지** (붕괴 메커니즘 정밀화): 우리가 본 붕괴는 *per-image 평균*(이미지마다 제 평균색)이 아니라 **global constant**(모든 샘플이 같은 남보라, viz 확인) = 자기참조가 만든 단일 벡터. 만약 "full 접근 → 그 이미지 평균 계산 → 출력"이 driver였다면 masking이 평균 접근을 끊어 *확실히* 고쳤겠지만, 실제는 자기참조라 masking은 상수로 가는 *추론 장벽만 높일 뿐* 최소점은 잔존 → B-2는 유망하나 시험 필요. (masking이 "평균 읽기"를 막는 효과가 정확히 맞는 곳은 MAE 픽셀 쪽 = "평균색만 맞춰도 MSE 낮음".)
- Run A의 target norm(variance reg + target LayerNorm)은 B-1에 유지, B-2에서 variance reg만 제거(target LayerNorm은 scale 정합용이라 유지 검토).

## 9. 설계 방향 검토 — interpreter_1 흡수 + JEPA→pixel prediction 전환 (2026-06-19)

> Run B-2 self-pair 라우팅 viz(36115626)에서 출발한 설계 논의. **다음 모델 버전 후보**. 현재 Parvo BC-T(36053622~) 결과 본 뒤 채택 결정.

### 출발점 — viz가 드러낸 모듈 중복

self-pair 라우팅(M(x,x)≈0)을 픽셀로 그렸더니 단일프레임 복구와 ≈동일. **근인**: viz가 라우팅 latent을 `interpreter_1+recon_head`로 렌더링 → 복구 열과 같은 디코더. 학습 시엔 복구(interpreter_1→픽셀)와 라우팅(p_motion_decoder→latent)이 별 모듈이나, **완성(completion) 기능이 중복**(Run B `_vjepa_p_masked`는 이미 mask token 주입→p_motion_decoder로 완성 수행).

### 단계 1 — interpreter_1을 p_motion_decoder에 흡수

복구 전용 `interpreter_1` 모듈 제거. 복구 = `p_motion_decoder(P, M=null) + recon_head`, 라우팅 = `p_motion_decoder(P, M) + (head)`. **"M 없으면 복구, M 있으면 예측"이 한 메커니즘** = Run B task-format 일관성의 모듈 레벨 실현. 파라미터↓, 복구↔라우팅 결합. (단 복구 grad가 p_motion_decoder로 흐름 — 간섭 가능성은 실험.)

### 단계 2 — JEPA 제거, 전부 pixel reconstruction으로 통일 (핵심 제안)

세 경우를 **동일 포맷(픽셀 복구)**으로:
- `L_t`: `motion_decoder(P_t, no-M) + recon_head` → **frame_t 픽셀**
- `L_tk`: `motion_decoder(P_tk, no-M) + recon_head` → **frame_tk 픽셀**
- `L_pred`: `motion_decoder(P_t, M(t,tk)) + recon_head` → **frame_tk 픽셀** (motion-conditioned 미래 픽셀 예측)

→ **EMA teacher·latent target·JEPA 전면 제거.** 모델 정체성 = "V-JEPA scaffold" → **"motion-conditioned predictive (pixel) MAE"**.

### 왜 강력한가 — 붕괴 문제를 *원천 소거*

Run A/B 내내 싸운 붕괴 = **JEPA 자기참조**(target=EMA student → 상수해도 loss=0). 미래 **실제 픽셀**을 target으로 두면 target이 고정 실데이터 → **상수 붕괴 자체가 불가능.** 게다가 JEPA(latent)는 본 프로젝트에서 우위 미입증(scaffold 0, 오히려 VideoMAE 픽셀-MAE가 EgoDex +0.47 > v15 latent +0.39) → **JEPA 버려 잃을 게 적고, 붕괴 면역을 얻음.**

### scaffold는 살아남나 — 그렇다

복구는 no-M, **예측(L_pred)만 M 조건** → M이 P_t를 통해 gradient로 P를 shaping(scaffold 메커니즘 유지). 즉 latent→pixel로 target만 바꾼 것이지, M→P scaffold 가설은 그대로 검증 가능.

### M-stream — zero/small motion도 처리해야 한다 (사용자 원칙, 2026-06-19)

초안의 "복구는 M 끄기(no-M)" 해소는 **철회**. 사용자 지적: **M이 변화가 *매우 작은* 상태에서도 동작해야 한다. 작다고 안 되면 그건 "motion" 인코더가 아니라 "큰-motion 검출기"다.** magnocellular 비유(미세 temporal 변화까지 반응)와도 정합. → M을 *끄지 말고*, **전체 motion 스펙트럼(0→큰 변화)을 학습**시키는 게 정공법.

**정확한 메커니즘** (M(0)은 "고장"이 아님):
- M 채널 = ΔL. frame_t=frame_tk면 ΔL=0 → M 인코더가 **상수 null code** 출력(모든 샘플 동일). routing이 이 null을 **identity**로 학습 → P_t 그대로 복구. 즉 **"motion 없음 → null motion code → identity"** 가 *올바른 연속 거동*이지 퇴화가 아님.
- 그래서 frame_t/frame_tk 복구 = motion 스펙트럼의 **zero 끝점**. 정당한 학습 케이스.

**가장 깔끔한 실현 — gap으로 파라미터화**: recon vs prediction을 따로 두지 말고 **단일 task "P_a + M(a,b) → frame_b 예측"**, b를 gap=0(=복구)부터 큰 gap(=예측)까지 샘플. **motion 크기 = gap.** M이 전 스펙트럼을 *특수 케이스 없이 자연스럽게* 학습. (현 max_gap=30 분포에 gap=0 끝점 추가.)

**균형 — 샘플링 말고 loss 가중치로** (사용자, 2026-06-19): gap 분포는 우리가 정함. 통일 설계는 step당 **gap=0 2개(L_t, L_tk) + real-gap 1개(L_pred)** 가 *확정적으로* 추가되는 구조(2:1). → 샘플링 파이프라인 건드릴 필요 없이 **비율이 결정적이니 λ_recon에 가중치만 조정**(importance weighting = resampling 등가). 분석적으로 계산 가능.

**단 무엇을 균형 맞추나 — 정밀화**:
- gap=0 recon 항의 *주 역할 = 픽셀 grounding(P anti-collapse)* → 너무 줄이면 붕괴 면역(이 설계의 존재 이유) 약화. 강하게 유지해야.
- **M의 real-motion 학습은 L_pred에서 옴**. gap=0은 M에 **trivial gradient**(zero→null code)만 줌 → M의 motion feature는 recon 가중치에 *둔감*. 즉 "M zero-과노출" 우려는 **빈도가 시사하는 것보다 작음**(zero-motion이 M을 거의 안 가르침, 자기제한적).
- 따라서 λ_recon 가중은 실제론 **"grounding(recon) ↔ prediction(motion)" loss 크기 균형**(표준 multi-task)이지, "M을 zero에서 구출"이 아님. → 과튜닝 말 것. **M의 real routing 기여를 측정**(학습-시 ablation)해서 조정하는 게 정공법, 빈도만 보고 미리 줄이지 말 것.
- **구현 결정**: λ_t/λ_tk/λ_pred(또는 λ_recon) **변수화하되 init=1.0**. 빈도(2:1) 보고 *바로 줄이지 않음* → 1로 시작해 실험·측정 후 조정.

**배포 관련성**: 실로봇은 고프레임레이트라 연속 프레임 motion이 *작음*. M이 small motion 못 다루면 배포에서 깨짐 → small/zero motion 처리는 deployment-critical. 사용자 원칙이 실용적으로도 맞음.

### 잠재 문제 점검 — pixel 통일·EMA 제거 (2026-06-19, 본학습 전 필수 검토)

> "EMA/JEPA 버리고 전부 픽셀로 통일"의 문제 점검. 붕괴는 죽지만 **새 실패 모드**가 생김.

**~~P1 — M-leakage / trivial add shortcut~~ → 철회 (2026-06-19, 사용자 반박, 코드 확인)**
- 초안 주장: "M=ΔL을 입력으로 주면 `P_t + ΔL = frame_tk` 더하기 shortcut" → **틀림.**
- **근거(아키텍처)**: routing은 `v_from_p` cross-attention. `MotionRoutingBlock` ([common/blocks.py](../src/models/common/blocks.py)): **Q,K = M**(`qk_m`), **V = P**(`v_p`). 출력 = `softmax(Q_M K_M^T)·V_P` + residual P. → **ΔL은 attention 가중치로만 들어가고, 출력 *내용*은 전적으로 P의 appearance remix.** "ΔL 더하기"는 구조적으로 불가능.
- 더 나아가 attention에 ΔL을 쓰려 해도 "ΔL→대응(flow) 추정"은 non-trivial → M이 *실제 motion 추정*을 학습해야지 trivial readout 아님. **scaffold도 정상**(V=P라 P가 routing value로 shaping됨).
- **내가 틀린 이유**: routing을 generic "frame_t + M → frame_tk"(M 내용이 additive)로 모델링했으나, 실제는 M=attention-only, V=P. pixel/latent 무관하게 같은 구조라 "pixel이 leak 증폭"도 틀림.
- **잔여(=P3로 흡수)**: V-from-P remix는 *P_t에 있는 내용만* 재배치 가능 → frame_tk의 **새로 드러난 영역(disocclusion/신규 객체)은 생성 불가** → 그 영역 blur/오차. 단 이건 leak이 아니라 **warping 커버리지 한계 = blur 문제(P3)**. latent에도 있으나 pixel에서 더 *가시화*될 뿐.

**🟠 P2 — 복구(gap=0)는 반드시 masked여야 (full copy 무의미)**
- gap=0 + full P_t(unmasked) → frame_t 복구는 **trivial copy**(정보 다 있음) → grounding 효과 0. **masked MAE 필수**(masked 위치 추론). Run B masked anchor가 이미 처리하나, 통일 설계에서 명시 유지.

**🟡 P3 — pixel blur** (아래 정직한 리스크와 동일, 가장 잘 알려진 trade).

**🟡 P4 — 표현 추상도 하락 가능**: pixel 예측은 저수준 feature를 빚음(JEPA가 latent 간 이유). 단 VideoMAE 픽셀-MAE가 EgoDex +0.47이라 *경험적으론 OK 신호*. action-relevance는 본학습 probing/BC-T로 확인.

**✅ 안 생기는 것**: constant collapse(고정 실픽셀 target이라 불가) · EMA 불안정 · ~~M-leak(P1 철회, V-from-P)~~. → 실패 모드가 "붕괴" → **"blur(P3)"** 로 교체. P1 철회로 새 1순위 리스크 = **blur**(disocclusion 커버리지 한계 포함), P2(masked 필수)는 유효.

### 정직한 리스크

1. **픽셀 예측 blur**: MSE 미래 예측은 불확실성 평균화로 흐려짐(과거 SSIM 사투의 그 문제). → SSIM/perceptual 보강 필요. JEPA(붕괴) ↔ pixel(blur) trade.
2. **narrative pivot**: Paper 2가 "V-JEPA scaffold"→"motion-conditioned predictive MAE"로 바뀜. 단 기존 story가 미입증·붕괴 중이라 정직한 재정렬에 가까움.
3. **미검증 가설**: 붕괴 회피·유용한 P 학습은 본학습으로 확인 필요.

### 설계 철학 — information bottleneck for generalization (근본 동기, 2026-06-19)

> 이 재설계(및 two-stream 분리 전체)의 *왜*. scaffold-인과보다 방어 가능한 정규 프레이밍 후보.

**동기**: 최근 로봇 world-model = 미래 예측 방향. 이에 공감하나, **RGB(P)에 미래 예측 정보를 전부 담으면 학습 dynamics 의존성이 급등** → in-domain 성능보다 **새 환경 적응력**을 우선하는 입장에서 역효과. → **의도적으로 P에서 미래-예측 정보 일부를 제거**(데이터 경향성↓), 변화 정보는 **M stream(가볍게)**으로 받아 routing의 "복구지능"이 **변화 × 이미지 요소 상호작용**을 일부 사전학습. = **정보 병목으로 일반화를 산다.**

**이미 있는 경험적 공명 — readout↔control dissociation**:
- VideoMAE(한 stream에 dynamics 다 담음) = in-domain readout 최강(+0.47)인데 **control 꼴찌**(LIBERO 0.22) = 과적합 전형.
- two-stream(병목/분리) = readout 양보, **control 압도**(0.63) = 일반화.
- → philosophy가 *예측하는 패턴이 데이터로 이미 나타남*.

**V-from-P가 구조적 실현**: M=어디(attention Q/K), P=무엇(content V) → appearance/change 명시적 factorize. ΔL은 출력 내용에 안 들어가고 routing만 함 → 분리가 아키텍처에 박힘.

**🔑 긴장 ② (본학습 전 필수 명시) — 병목 이상 vs scaffold 주장**:
- 병목: "P는 dynamics 인코딩 안 함"(과적합 방지). scaffold: "M이 P shaping"(예측 grad가 P로) → P가 dynamics 인코딩. **반대로 당김.**
- 현 설계는 P=routing의 V라 예측 grad 받음 → 병목 *강제 안 됨*(희망).
- **해소 가설**: M이 P를 "**warpable**(일반 변형가능) appearance"로 빚는 것이지 "**memorized dynamics**(특정 동역학)"가 아니라면 양립. P=일반 factorizable substrate, dynamics는 M+routing에. **검증 = P가 full-2-frame 인코더보다 OOD 전이 우수한가.**

**caveat ①**: ΔL은 motion 저수준 proxy, 조명·대비 의존 → M도 도메인 의존 *일부* 나름("0" 아니라 "낮음", full appearance보다 훨씬 저차원).

**가치 증명 ③**: "정보 많이 주면 좋다"는 자명 → 분리 이점은 **OOD/control에서 two-stream > 과거+현재 풀이미지**로만 성립. = 지금 Parvo BC-T + OOD probing이 그 테스트.

**논문 프레임 함의**: scaffold-인과(증명 난이도 高) 대신 **"information bottleneck → readout 양보·control/OOD 적응력 획득"**(dissociation이 증거)이 더 정직·강함.

### 판단

붕괴를 원천 소거하면서 task-format 통일까지 달성하는 **유망한 단순화**. M 우려도 깔끔히 해소됨. 채택 시 다음 본학습의 핵심 변경. **현 Parvo BC-T 결과 확인 후 결정** — BC-T가 약하면 이 재설계가 더 시급, 강하면 현 구조 유지하며 점진 도입 검토. **재설계는 necessary-not-sufficient**: 붕괴 블로커는 없애나 scaffold/병목 이점은 별도 검증(with-M vs no-M 학습-시 ablation + OOD 전이)이 판정.

### 구현 결정 (2026-06-22) — Full 재설계 채택 + SiamMAE 동시 준비

**채택 확정**(사용자): §9 **Full 재설계**(JEPA/EMA 제거·전부 pixel recon·interpreter_1 흡수·gap-parameterized 통일 task) 구현. SiamMAE baseline(2-size)과 **나란히 비교**하기 위함.

**통일 메커니즘** — 단일 pixel-predict 함수를 3가지로:
```
predict(P_a_visible, mask, M_routing) = recon_head( p_motion_decoder(build_full_seq(P_a, mask), M_routing) )
  L_t  : M=M(t,t)≈null   → frame_t  pixels  (masked)   # gap=0, self-pair routing
  L_tk : M=M(tk,tk)≈null → frame_tk pixels (masked)   # gap=0
  L_pred: M=M(t,tk) real → frame_tk pixels (masked)   # real gap, motion-conditioned
λ_t/λ_tk/λ_pred init=1.0 변수화. teacher_p/teacher_m/EMA/interpreter_1 제거. P2(masked 필수) 준수.
```

**Sizing**(사용자 2026-06-22): "사이즈" = **P encoder가 SiamMAE와 동급**(P=downstream representation stream, SiamMAE 단일 stream과 1:1). **M encoder는 더 경량**(shallow; routing dim 호환 위해 embed_dim 공유, depth↓).
- `small`: P=ViT-S(384/12/6) ≈ SiamMAE-small / M=경량(depth↓)
- `base` : P=ViT-B(768/12/12) ≈ SiamMAE-base(=VideoMAE-ours parity) / M=경량

**비교 매트릭스**: {SiamMAE, §9} × {ViT-S, ViT-B} (P 기준 동일 스케일) → cross-model 공정 비교.

## 10. Parvo BC-T LIBERO rollout 결과 (2026-06-21)

§9 판단이 기다리던 **Parvo BC-T 결과** = §9 가치증명 ③의 control 테스트. BC-T는 클러스터에서 학습(`parvo-ptptk` 어댑터, Run B-2 continuation ep30 encoder), ckpt 9개(3 suite × 3 seed) 로컬 전송 후 closed-loop rollout(50 trials × 3 seed = seed당 500 ep). 어댑터 = no-Sobel P=RGB 3ch, `patch_mean_concat_p_t_p_tk` 동치 ([parvo_pt_ptk.py](../src/encoders/adapters/parvo_pt_ptk.py)). 결과 → [paper_artifacts/libero_rollout/](../paper_artifacts/libero_rollout/) (summary/per_task/episodes).

**LIBERO SR** (mean over 3 seed × 500 ep):

| Encoder | goal | object | spatial | avg |
|---|---|---|---|---|
| **parvo-ptptk (runB2cont ep30)** | 0.827 | 0.885 | 0.644 | **0.785** |
| two-stream-v15-ptptk (ep50) | 0.837 | 0.864 | 0.630 | 0.777 |
| v15-ptptk vfromm (ep32) | 0.817 | 0.876 | 0.645 | 0.779 |
| siglip | 0.855 | 0.907 | 0.802 | 0.855 |
| vc1 | 0.857 | 0.875 | 0.732 | 0.821 |
| dinov2 | 0.838 | 0.880 | 0.715 | 0.811 |
| videomae-ours | 0.424 | 0.239 | 0.215 | 0.293 |
| two-stream-v11 | 0.264 | 0.030 | 0.061 | 0.118 |

**읽기**:
- **Parvo ≈ v15-ptptk** (0.785 vs 0.777, goal −0.01 / object +0.02 / spatial +0.01). **no-Sobel(P=RGB 3ch) 단순화 + Run B-2 masked anchor가 v15 BC SR을 무손실로 유지** — 붕괴 방어 설계가 control 성능을 깨지 않음 확인.
- 단 **frozen 단일프레임 baseline(siglip/vc1/dinov2)에 여전히 못 미침**, 특히 **spatial**(0.644 vs 0.715~0.802) 격차 큼. goal/object는 vc1/dinov2와 경합권.
- videomae-ours·v11은 BC-T 매우 약함 (참고).

**§9 결정 함의**: BC-T가 "강함"(baseline 추월)도 "약함"(v15 대비 퇴행)도 아닌 **현상 유지** = v15-ptptk 수준 동등. → §9 재설계는 *즉시 시급하진 않으나* 붕괴 블로커 제거·task-format 통일 명분은 유효. 판정은 여전히 **OOD 전이 + with-M/no-M 학습-시 ablation**에 달림 (BC SR 단독으론 scaffold/병목 이점 미판정). spatial 격차가 two-stream 공통 약점인지(v15-ptptk도 0.630) 별도 진단 가치.

## 11. no-M ablation 계획 — M 기여 격리 (2026-06-22)

> §10의 "판정은 with-M/no-M ablation에 달림"을 구체화. 결론: **M 기여를 측정하는 단 하나의 깨끗한 실험 = 같은 코드에서 motion routing만 끈 인코더 학습 후 동일 downstream eval.** 이번 세션은 계획·주의사항만 기록(구현/실행은 별도 dev session).

### 11.1 핵심 발견 — LIBERO downstream은 이미 P-only (M 미사용)

- BC-T 어댑터 [`parvo_pt_ptk.py`](../src/encoders/adapters/parvo_pt_ptk.py)는 `_encode_p_unmasked`로 **P 인코더만** 2회 호출(P_t, P_tk) → patch-mean → concat. docstring 그대로 **"M encoder / motion-routing 미사용"**.
- ∴ **"LIBERO를 M 없이 수행"은 새 실험이 아니라 현 상태.** 현재 모든 LIBERO 수치(§10)는 전부 P-only 결과. downstream에서 뺄 M이 없음.
- M이 LIBERO에 영향을 줄 수 있는 **유일 경로 = 사전학습 때 motion-routing gradient가 P 가중치를 빚는 것.** 학습 종료 후 M은 LIBERO에서 버려지므로 M의 효과는 전부 "P 가중치에 frozen-in".

### 11.2 따라서 toggle은 downstream이 아니라 pretraining

- **인코더 A** = Parvo, motion routing 연결(M→P gradient on) = 현재.
- **인코더 B (no-M)** = 동일 Parvo에서 **motion routing(+ M→P 예측 경로 `L_pred`)만 차단**. 나머지(P 크기·recon loss `L_t`/`L_tk`·masked anchor·데이터·epoch·params) **전부 동일**.
- 두 인코더에 **동일 frozen P-only LIBERO BC** → **ΔSR(A−B) = M이 P를 빚어 얻은 기여.**
- ⚠️ **confound 가드 (핵심)**: 이전 v15→Parvo 비교는 routing 외에 Sobel 제거·레시피·epoch가 동시에 바뀐 **confounded** 비교였음. 여기선 **routing 하나만** toggle해야 격리 성립. **구현 시 확인**: routing을 꺼도 `L_compose`/`L_m_jepa` 등 다른 손실이 P 인코더로 back-prop되지 않는지 — 흐른다면 그것도 함께 차단해야 "pure MAE 대조"가 성립.

### 11.2.1 구현 완료 (2026-06-22) — `--v15-no-motion` 플래그 (효율형)

forward 추적 결과 **M→P gradient의 유일한 경로 = `L_pred`** (`_vjepa_p_masked`: `m_local`(M routing) + `p_t_visible`(P) → `p_motion_decoder` → teacher_tk 예측). 나머지 손실은 P를 안 빚음:
- `L_t`/`L_tk`: P MAE 단독(`_mae_one_frame`, M 미사용) — encoder B의 *유일* 학습 신호.
- `L_m_jepa`: M encoder/decoder만 (P 안 거침). **Run B-2는 이미 `λ_m_jepa=0`** (M = conditioning oracle).
- `L_compose`: pair_mode에 없음. `L_var`: P patch에만, Run B-2는 0.

**최소 toggle = `λ_pred=0`이면 결과상 동치**(P가 L_t+L_tk만 받음)이나, dead branch(M encoder ×2, teacher_p full depth-12 forward 등)가 매 step ×0으로 계산돼 **~절반 낭비**. → 효율형으로 **`--v15-no-motion` 플래그 구현**([two_stream_v15.py](../src/models/two_stream_v15.py) `_forward_pair` early-return + M/routing 모듈 동결 + `update_teacher` no-op, [pretrain.py](../scripts/pretrain.py) `--v15-no-motion`, [pretrain.sbatch](../scripts/cluster/pretrain.sbatch) `V15_NO_MOTION=1`).

**smoke 검증 (CPU, batch 2)**: `loss = loss_t + loss_tk` (loss_pred=loss_m_jepa=0), **192 trainable 텐서 전부 grad 수신·unused 0 → DDP-safe**(`find_unused_parameters=False` hang 없음). trainable = P MAE 경로만(blocks_p/patch_embed_p/.../interpreter_1/recon_head), frozen = M stream 전체 + p_motion_decoder. encoder B에서 M·p_motion_decoder는 미학습이나 downstream 어댑터(`parvo_pt_ptk`)는 P encoder만 써서 무해.

> 참고(효율, no-M 무관): full Parvo 경로(향후 §9 재학습)에 `patch_embed_m`가 `_encode_m_unmasked`/`_encode_m_masked`에서 같은 `m_chan`에 2회 적용되는 사소한 중복 있음. conv라 저비용 + drive-by 금지로 미수정 — 재학습 시 patch embed 1회 공유로 개선 가능.

> ⚠️ **CLS 붕괴 함정 (재발, [feedback_no_cls_collapse_judgement])**: "encoder A 붕괴함 vs B 안 붕괴함"을 confound로 든 적 있으나 **철회**. 근거였던 std_p≈0.008·cos_intra_p→1.0은 **CLS metric**(이 모델 학습압력 0). **patch는 healthy**(probing R²=0.30, recon 작동) → BC-T(patch-mean)가 보는 표현은 안 붕괴. patch 레벨에선 A·B 차이 = "routing gradient가 P patch를 추가로 빚었나"뿐 → ablation은 깨끗. (단 `cos(pred,tgt)→1.0`은 CLS 아닌 predictor trivial 신호 = routing gradient가 *약했을* 가능성 → B≈A 예측 근거. 이건 발견이지 confound 아님.)

### 11.3 no-M ≈ two-frame image MAE = Paper1 vs Paper2 비교

- routing을 끄면 P에 남는 학습 신호는 `L_t`/`L_tk_recon`뿐 → **두 프레임 각각 masked reconstruction = 사실상 (two-frame) image MAE.**
- ∴ no-M ablation = **Paper 1(image MAE) vs Paper 2(motion routing)** 을 한 코드베이스에서 깨끗하게 수행하는 것. 이건 새 허들이 아니라 [`Projects/Action-Agnostic Paper/README.md`](Vault) 의 성패 기준 *"Paper 1을 넘어야 성립"* / `2. Experiments.md §4` *"대조군 = standalone image MAE(P단독), 못 넘으면 구조적 연결 불필요 → 가설 기각"* 을 실제로 측정.
- ⚠️ **off-the-shelf image MAE / VideoMAE-ours와 비교 금지** — 아키텍처·mask·params·데이터 confound. 반드시 **같은 Parvo 코드 routing-off** 버전이라야 함.

### 11.4 이 실험이 잡는 축 / 못 잡는 축

- **잡힘 (in-domain method 정당화)**: "motion routing이 LIBERO control 표현에 load-bearing이냐."
  - routing-P **>** MAE-P (noise 넘게): M이 P를 유용하게 빚었다는 **첫 positive 증거** → Paper 2 독립 성립.
  - routing-P **≈** MAE-P: 구조가 LIBERO에 0 기여 → method 정당성 상실, Paper 1 null로 수렴.
  - routing-P **<** MAE-P: routing이 해로움(강한 음성).
- **못 잡음 (thesis 심장 = OOD 일반화)**: LIBERO는 OOD 시험대 아님(sim manip, 대체로 in-domain). §일반화 프레임(factorization-as-regularization, in-dist 양보↔OOD 적응력)은 LIBERO로 판정 불가.
  - LIBERO null이어도 thesis 즉사 아님 — 입증 부담이 **전부 OOD로 이동**(CALVIN ABC→D, cross-domain probing, view-sensitivity, SIMPLER).
  - LIBERO 우위여도 OOD 증거 **별도 필요**.

### 11.5 효율적 실행 — 인코더 하나로 두 축

- 비싼 건 인코더 B 학습뿐. 한 번 학습해두면 **LIBERO BC + OOD/cross-domain probing을 동일 인코더에 둘 다** 돌려 method 축(LIBERO)과 thesis 축(OOD)을 같이 답함.
- 권장 계획 단위: **"no-M 인코더 B 학습 → LIBERO BC + OOD eval 한 묶음."**

### 11.6 주장 규율 (결과 해석 시 — §5.1 Vault와 동기)

- **"광범위 벤치 우수" ≠ "motion 학습 증거."** breadth는 "쓸모 있는 표현"만 강화. motion 인과는 ablation(11.2) + motion-selective 우위(축별 분해, 예: CALVIN continuous-motion pos-dim)로만 licensing.
- confounded 비교로 "M 좋다"도 "M 무용"도 단정 금지 (v15 오귀속 철회의 대칭 오류 방지). 현재 정직한 문장 = **"M 기여는 현 증거상 작아 보이며 load-bearing 근거 없음, 미입증"** → 11.2가 확정.
