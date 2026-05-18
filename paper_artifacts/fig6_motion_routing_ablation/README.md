# Fig 6 — Motion-Routing Source Ablation (V from P vs V from M)

**Paper position**: §5.2 Analysis (★ main architectural ablation)
**Status**: 🔴 not started — **C1 학습 대기** (cluster_sessions 34464714/34464715)

## Spec (Vault [[7. Outline § Fig 6]])

**2-panel**:

- **Panel A — Schematic**: V from P (ours) vs V from M (standard cross-attn) side-by-side
  - 강조: output space (P vs M) 차이, motion routing block 직후 representation
- **Panel B — Bar chart**: probing R² + LIBERO BC SR head-to-head
  - 2 encoder × 4 metric (EgoDex / LIBERO probing / LIBERO BC ptptk / LIBERO BC mp)

## Argument

"Motion semantic을 따라 P value를 routing"의 architectural choice 정당화. Same training budget / same data / 단일 hyperparameter 차이 (`V11_ROUTING_MODE`) — 깔끔한 ablation.

## Current artifacts

없음. C1 본 학습 진행 중 (34464715, AIP_long 2×4 H100 50ep, 약 43h 예상, 2026-05-15 시작).

## TODO

C1 학습 완료 후:
- [ ] **v15-vfromm ckpt에 12-mode EgoDex probing** — fair pair (v15 ep50 vs v15-vfromm ep50)
- [ ] **v15-vfromm ckpt에 LIBERO 12-cell probing**
- [ ] **v15-vfromm BC-T 학습** (ptptk + mp 어댑터 모두) — 3 suite × 3 seed × 2 adapter = 18 BC runs
- [ ] **v15-vfromm BC rollout** — 로컬 LIBERO closed-loop SR
- [ ] **schematic 스크립트** — TikZ block diagram 2-encoder side-by-side
- [ ] **bar chart 스크립트** — 2 encoder × 4 metric grouped bar

## 학습 cfg (cluster_sessions 참조)

```
sbatch ... --export=ALL,V11_ROUTING_MODE=v_from_m,CHECKPOINT_SUFFIX=vfromm,...
scripts/cluster/pretrain.sbatch
```

ckpt → `/proj/external_group/mrg/checkpoints/two_stream_v15_vfromm/`

## Notes

본 ablation은 paper §5.2 **main ablation** — fig6 미완성이면 paper main claim 약화. C1 timeline이 paper deadline에 가장 critical.
