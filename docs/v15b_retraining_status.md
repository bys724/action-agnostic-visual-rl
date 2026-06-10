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

### Sanity #2 — JobID 35481545 🔄 RUNNING (gate=3, 200vid×8ep, SAVE_INTERVAL=8)
- **목적**: L_pred trivial이 (a) P 미성숙 탓인지 (b) 과제 자체 trivial인지 분리. gate=3로 P가 3ep MAE 성숙 후 V-JEPA 켜지는 본학습 timing 모사. ckpt 저장 → diagnose.
- ckpt 경로 예상: `/proj/external_group/mrg/checkpoints/two_stream_v15b_sanity_gate3diag/<timestamp>/checkpoint_epoch0008.pt`
- 로그: `/proj/external_group/mrg/logs/sanity_v15_35481545.{out,err}`

## 5. ▶ 다음 세션 할 일 (순서)

1. **sanity #2 결과 확인**: `grep -E "\[v14\]|schedule\] λ_pred" /proj/external_group/mrg/logs/sanity_v15_35481545.out`
   - ep4+(gate 해제 후)에서 L_pred·cos(pred,tgt) 추이 확인. 여전히 trivial(cos>0.99, L_pred<0.005)인지?
2. **diagnose 실행** (ckpt 대상): `scripts/eval/diagnose_vjepa_p_trivial.py` — **baseline cos(teacher_t, teacher_tk)** + **M=0 vs M-on ablation**.
   - baseline cos ~0.99 → 프레임이 원래 너무 비슷 = 과제 intrinsic trivial → **predictor 좁히기 / gap 늘리기** (인코더 size 아님).
   - M-on이 M=0보다 예측 개선 → catalyst 신호 있음 → 본학습 진행.
3. **분기 결정 후**:
   - **catalyst 신호 O** → 본학습 제출 (아래 §6).
   - **catalyst 신호 X (intrinsic trivial)** → 본학습 전에 triviality 완화 파라미터 조정 (predictor bottleneck 축소 / MAX_GAP↑ / EMA 0.999로 되돌림 검토). EMA 0.996은 target이 student를 바짝 따라와 L_pred를 *더* trivial하게 만드는 긴장 있음.

## 6. 본학습 명령 (catalyst 확인 후 제출)

```bash
sbatch --partition=AIP_long --nodes=2 --ntasks-per-node=4 --gres=gpu:4 \
    --time=2-12:00:00 \
    --export=ALL,MODEL=two-stream-v15b,EPOCHS=50,BATCH_SIZE_PER_GPU=32,\
V15_GATE_EPOCHS=10,V15_EMA_INIT=0.996,\
V15_LAMBDA_PRED_WARMUP_START=0,V15_LAMBDA_M_JEPA_WARMUP_START=0,V15_LAMBDA_COMPOSE_WARMUP_START=0,\
NUM_WORKERS=8 \
    scripts/cluster/pretrain.sbatch
```
- ckpt → `/proj/external_group/mrg/checkpoints/two_stream_v15b/`
- 예상 ~43-52h (원래 v15 343~420 GPU·h 수준). `--time` partition max 권장.
- **abort 기준**: ep12-15(V-JEPA 켜진 직후, P 성숙)에서 cos(pred,tgt)>0.99 & L_pred<0.01이면 catalyst 사망 → 중단 후 재서술.

## 7. 결정된 설계 판단 (재론 불필요)

- **인코더(P/M) size 축소 ❌**: 원래 v15는 overfitting이 아니라 **P-CLS collapse + L_m_jepa 폭증 + late divergence(train·eval 동반 상승)**. M은 건강(cos_intra_m 0.27). +0.39 자산은 patch-level P_t⊕P_tk라 P 축소 시 자산 손상. capacity lever는 **predictor(p_motion_decoder)** 가 on-target (V-JEPA trivial 방지). 진짜 overfit 증거 나오면 weight decay/drop-path가 정공법.
- 아키텍처: embed_dim=768, P depth 12(~85M), M depth 6(~42M), decoder_depth_m=3, interpreter_depth=3.
