# Fig 4 — Reconstruction Quality Symmetry (v11 vs v15)

**Paper**: 2 (AAAI, Action-Agnostic)

**Paper position**: §5 Analysis (design intent realization)
**Status**: 🔴 not started — ckpt forward 필요

## Spec (Vault [[7. Outline § Fig 4]])

**2×2 grid**:

|         | frame_t recon | frame_tk recon |
|---------|---------------|----------------|
| v11     | direct        | motion-routed (asymmetric path) → **quality dispersion** |
| v15     | shared path   | shared path (symmetric) → **quality symmetry** |

- 2-3 EgoDex sample × 2 frames each
- 각 panel 아래 MSE annotation
- 강조: v11은 frame_tk가 noisy / patch grid artifact, v15는 두 frame 유사

## Argument

§3.3 space-level paradigm separation의 **direct visual evidence** — v11의 비대칭 reconstruction path가 design intent 미실현 증거, v15의 대칭 path가 catalyst 발현의 architectural 조건.

## Current artifacts

figure PNG 없음 (not started). source 메트릭만 존재:
- `v11_ep44_vs_v15_ep50_20260515_162817/summary.json` — v11 vs v15 recon 비교 run 요약 (구 `recon_quality/`에서 통합)

## TODO

- [ ] **샘플 선정** — EgoDex test split에서 2-3개 (manipulation phase 다양성)
- [ ] **v11 ckpt forward** — `checkpoints/two_stream_v11/20260426_014333/checkpoint_epoch00044.pt` (champion)
  - `scripts/eval/visualize_attn_v11.py` 의 nomask reconstruction 분기 활용 (frame_t, frame_tk 각각 MSE 보고)
- [ ] **v15 ckpt forward** — `/mnt/data/checkpoints/two_stream_v15/.../latest.pt` (ep50) 또는 `checkpoint_epoch00032.pt` (probing champion)
  - `scripts/eval/visualize_v15_no_mask.py` 활용
- [ ] **2×2 grid 합성 스크립트** — matplotlib subplot 또는 PIL collage
- [ ] **MSE 정량 보고** — per-sample, per-frame MSE 표 (figure caption 보조)

## Source

- v11 reconstruction viz: [`scripts/eval/visualize_attn_v11.py`](../../scripts/eval/visualize_attn_v11.py) `--no-mask --recon-only` 옵션
- v15 reconstruction viz: [`scripts/eval/visualize_v15_no_mask.py`](../../scripts/eval/visualize_v15_no_mask.py)
- EgoDex test split: `/mnt/data/egodex_frames/test/` (로컬) / `/proj/external_group/mrg/datasets/egodex_frames/test/` (클러스터)
