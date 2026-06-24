# Tab 3 — Architecture Ablation

**Paper**: 2 (AAAI, Action-Agnostic)

**Paper position**: §5.1 / §5.2 (motion-routing source + BC adapter ablation)
**Status**: 🔴 not started — C1 결과 + v15 BC round2 대기

## Spec (Vault [[7. Outline § Tab 3]])

| Variant | EgoDex P_t+P_tk | LIBERO probing | BC SR (ptptk) | BC SR (mp) |
|---|---|---|---|---|
| **v15 V from P (ours)** | +0.390 | +0.584 | 83.7/63.0/(obj) | 43.3/.../... |
| v15 V from M (C1) | TBD | TBD | TBD | TBD |
| v11 (reference) | +0.010 | +0.041 | 26.4/6.1/3.0 | n/a |

## Source data 의존성

| Cell | 어디서 옴 |
|------|----------|
| v15 V-from-P probing | [`../tab2_probing/`](../tab2_probing/) — TODO |
| v15 V-from-P BC | [`../../fig3_bc_main/summary.csv`](../../fig3_bc_main/summary.csv) — TODO (round2) |
| v15 V-from-M (C1) | [`../../fig6_motion_routing_ablation/`](../../fig6_motion_routing_ablation/) — C1 학습 대기 |
| v11 reference | tab2/fig3 round1 |

## TODO

본 표는 다른 폴더의 데이터를 **참조 집계**. 의존성 모두 충족 후 생성.

- [ ] tab2_probing v15 12-mode 완료
- [ ] fig3_bc_main round2 v15-ptptk + v15-mp 완료
- [ ] fig6_motion_routing_ablation C1 학습 + probing + BC 완료
- [ ] **집계 스크립트** — 위 source CSV에서 6 cell 추출하는 짧은 Python 스크립트

## Notes

본 표는 paper §5의 **결정적 ablation table**. C1 결과가 paper main claim의 정당화 근거.
