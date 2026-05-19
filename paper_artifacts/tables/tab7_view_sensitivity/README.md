# Tab 7 — LIBERO View-Sensitivity Sub-analysis (paper §5 ¶6)

**핵심 metric**: Δ R² = R²(agentview + eye_in_hand combined) − R²(agentview only)
→ "wrist view 추가 시 representation이 얼마나 더 많은 action 정보를 인코딩하는가"

## Source

- 잡 ID: 34619063~069 (av_only, 7잡) + 34619135~143 (av+eih, 9잡) — `docs/cluster_sessions.md`
- v15ep50 object/goal av_only는 기존 `paper_artifacts/libero_action_probing/two-stream-v11_libero_{object,goal}_20260515_161112_v15ep50/` 재활용
- raw cells: [summary.csv](summary.csv) (72 rows)
- Δ table: [delta.csv](delta.csv) (36 rows)

## 결과 요약

### Encoder Δ 평균 (3 suite × 4 gap = 12 cells 평균)

| Rank | Encoder | Δ avg | 격차 |
|------|---------|-------|------|
| 1 | **vc1** | **+0.221** | — |
| 2 | **v15** (P encoder only) | **+0.211** | vc1과 −0.010 |
| 3 | siglip | +0.114 | v15과 −0.097 |

### Per-suite Δ (v15 vs baseline 상세)

| Encoder | spatial | object | goal | overall |
|---------|---------|--------|------|---------|
| v15     | +0.263  | +0.132 | **+0.238** ★ | +0.211 |
| siglip  | +0.164  | +0.062 | +0.115 | +0.114 |
| vc1     | **+0.276** ★ | **+0.155** ★ | +0.231 | +0.221 |

★ = suite 1위

### Gap별 Δ 평균 (3 enc × 3 suite = 9 cells 평균)

| gap | Δ avg | LIBERO 20Hz 환산 |
|-----|-------|-----|
| 1   | +0.277 | 0.05s (가장 큰 gain) |
| 13  | +0.148 | 0.65s |
| 20  | +0.134 | 1.00s |
| 40  | +0.169 | 2.00s |

## 해석 (비판적)

**Paper §5 ¶6 가설 결과**:
- 가설 1 (v15 Δ 최대) — **부분적 부합**. siglip은 명확히 추월(+0.097), 그러나 vc1과 사실상 동률(−0.010). 
- "Motion-routing이 unique advantage" 강한 claim **불가**
- 대신 가능한 framing 두 가지:
  1. "**action-relevant encoders (v15-P, VC-1) integrate wrist view more effectively than vision-language SSL (SigLIP)**" — Δ 0.21~0.22 vs 0.11
  2. "v15 excels at **goal task** specifically (Δ +0.238 vs siglip +0.115), where long-horizon trajectories benefit most from motion-rich wrist context"

**Suite별 비대칭**:
- spatial/object → vc1이 1위 (VC-1의 robotics-pretrained pixel feature가 wrist view에 적합)
- goal → v15이 1위 (long-horizon에서 action-agnostic motion pretrain의 이점)

**Gap=1에서 Δ 가장 큼 (+0.277)**:
- gap=1 av_only는 거의 변화 없는 두 frame → av 단독에서 action 정보 부족
- wrist view가 close-up motion context로 큰 신호 추가
- gap≥13에선 av 자체가 충분한 motion 정보 → wrist 추가 효과 절반 수준

## 약점 + 후속 조치

- **vc1과 동률**: paper main에서 "motion-routing 우위" 단정 표현 회피. "**comparable to robotics-specialized VC-1, both outperform vision-language SSL**" 같은 fair framing 권장
- **DINOv2/VideoMAE-ours 미포함**: 5/18 사용자 결정으로 BC SR 상위 2 baseline만. 필요 시 후속 추가
- 결과는 §5 ¶6 본문 1-2 문장 + Tab 7 appendix 채움 (NeurIPS supplementary)

## 사용된 raw 데이터 위치

| Condition | Encoder | Suite | 폴더 |
|-----------|---------|-------|------|
| av_only | v15 ep50 | spatial | `libero_action_probing/two-stream-v11_libero_spatial_20260519_085921_v15ep50_av/` |
| av_only | v15 ep50 | object | `libero_action_probing/two-stream-v11_libero_object_20260515_161112_v15ep50/` (기존) |
| av_only | v15 ep50 | goal | `libero_action_probing/two-stream-v11_libero_goal_20260515_161112_v15ep50/` (기존) |
| av_only | siglip | × 3 suite | `libero_action_probing/siglip_libero_*_2026051[0-9]_*_av/` |
| av_only | vc1 | × 3 suite | `libero_action_probing/vc1_libero_*_2026051[0-9]_*_av/` |
| av+eih | (3 enc × 3 suite) | × 9 | `libero_action_probing/*_2026051[0-9]_*_both/` |
