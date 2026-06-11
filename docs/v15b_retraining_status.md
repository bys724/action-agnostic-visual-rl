# v15b 재학습 진행 상태 (세션 핸드오프)

> **브랜치**: `main` (student-anchor = 새 구조). 논문 v15는 `paper-corl2026` 브랜치(teacher-anchor)와 영구 분리.
> **최종 업데이트**: 2026-06-10. 이 문서는 다른 세션이 작업을 이어받기 위한 현황 메모.

## 1. 목표

원래 v15(teacher-anchor)는 V-JEPA P anchor=`teacher_p(frame_t).detach()`라 **P encoder가 motion(M) gradient를 전혀 못 받고 순수 MAE로 독립 학습**됨. `0fb74c8`에서 anchor를 student P encoder로 바꿔(표준 V-JEPA 복원) P·M 모두 motion routing gradient를 받게 함. `b41b177` = **v15b** (동일 아키텍처 + collapse 방지 레시피: ① recon-first hard-gate ② EMA 0.996 ⑤ lr scaling).

**검증 질문**: M→P gradient를 실제로 연결했을 때 catalyst가 작동해 VideoMAE(+0.47)/원래 v15(+0.39, P_t⊕P_tk)를 넘는가? 못 넘으면 "multi-frame MAE concat이 강력한 단순 baseline"으로 정직하게 재서술.

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
