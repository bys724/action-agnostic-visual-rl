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

## 5. 본학습 — JobID 35493293 🔄 RUNNING (2026-06-11 시작)

- 8 GPU(AIP_long 2×4) DDP 정상 기동(world_size=8). global batch 256, lr 2e-4, gate=10, EMA 0.996, nw=8, MAX_GAP=30(원래 v15 유지), shared gate.
- ckpt → `/proj/external_group/mrg/checkpoints/two_stream_v15b/<timestamp>/` (save_interval=4 → ep4/8/12/...).
- 로그: `/proj/external_group/mrg/logs/pretrain_v15b_35493293.{out,err}`

### ▶ 다음 세션 할 일

1. **ep12-18 abort 모니터링** (V-JEPA gate 해제 ep11 직후, P 성숙):
   - `grep "\[v14\]" /proj/external_group/mrg/logs/pretrain_v15b_35493293.out | sed -n '11,18p'`
   - **abort 조건**: cos(pred,tgt)>0.99 & L_pred<0.01 (trivial collapse) → 중단. 또는 train·eval 동반 발산(원래 v15 ep45-50 선례).
2. **ETA 재검산**: 첫 1-2 ep elapsed로 `잔여ep × per_ep + wall > TIME_LIMIT` 확인 (cluster_sessions --time 가이드).
3. **mid-run diagnose** (ep20+, ep30+ ckpt): `diagnose_vjepa_p_trivial.py --ckpt <epXX> --tag v15b_epXX` → **M 기여(+0.013→?)가 자라는지** 추적. 이게 student-anchor 성공 여부의 핵심 지표.
4. **완주 후**: EgoDex P_t⊕P_tk probing → 원래 v15(+0.39)/VideoMAE(+0.47) 대비. 넘으면 M→P motion 기여 성립, 못 넘으면 "multi-frame MAE concat baseline" 재서술.

### 미적용/보류 (필요 시 follow-up)
- **L_pred-only gate 분리**(M을 ep1부터 성숙): 미검증·M trivial collapse 위험 → 이번엔 shared gate 유지. M 기여가 끝까지 약하면 다음 arm에서 시도.
- **MAX_GAP 60**: RESEARCH_PLAN 검증된 개선(eval -16%, probing +0.054) + diagnose의 "gap 작아 너무 쉬움". 이번엔 v15 비교 위해 30 유지.
- **lr 1e-4**(linear scaling): v15 divergence 대비 안정. 이번엔 2e-4(v15 동일) 유지.

## 7. 결정된 설계 판단 (재론 불필요)

- **인코더(P/M) size 축소 ❌**: 원래 v15는 overfitting이 아니라 **P-CLS collapse + L_m_jepa 폭증 + late divergence(train·eval 동반 상승)**. M은 건강(cos_intra_m 0.27). +0.39 자산은 patch-level P_t⊕P_tk라 P 축소 시 자산 손상. capacity lever는 **predictor(p_motion_decoder)** 가 on-target (V-JEPA trivial 방지). 진짜 overfit 증거 나오면 weight decay/drop-path가 정공법.
- 아키텍처: embed_dim=768, P depth 12(~85M), M depth 6(~42M), decoder_depth_m=3, interpreter_depth=3.
