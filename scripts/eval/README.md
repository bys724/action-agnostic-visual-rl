# scripts/eval/ — Evaluation Scripts

평가 / 분석 / 시각화 스크립트 인덱스. 자세한 사용법은 파일 내부 docstring 참조.
모델 명명·구조는 [docs/FILE_INDEX.md](../../docs/FILE_INDEX.md) · [docs/REFACTOR_PLAN.md](../../docs/REFACTOR_PLAN.md).

## Action Probing (frozen encoder → linear/MLP probe → action target)

| Script | Status | 용도 | 의존 |
|--------|--------|------|------|
| [`probe_action.py`](probe_action.py) | active | EgoDex within-domain probing. v15-family mode 지원 | base 함수 (다른 probe가 import) |
| [`probe_action_v11.py`](probe_action_v11.py) | **active 공유 util** | two-stream P/M 인코더 추출·`load_v11_model`·`CLS_MODES`. v15 ckpt 로드(strict=False). v15_loader·viz·libero probe 공유. ⚠️'v11' 이름은 deferred rename | imports `probe_action` |
| [`probe_action_droid.py`](probe_action_droid.py) | active | DROID cross-domain probing | imports `probe_action` |
| [`probe_action_libero.py`](probe_action_libero.py) | active | LIBERO Action Probing (gap-matched DROID protocol) | imports `probe_action_v11` |
| [`probe_action_calvin.py`](probe_action_calvin.py) | active | CALVIN segment-based + cross-folder OOD probing | imports `probe_action_v11` |

## Value Alignment / BC fine-tune

| Script | Status | 용도 |
|--------|--------|------|
| [`value_alignment.py`](value_alignment.py) | active | LIBERO trajectory ρ (Phase 2.5, negative result) — imports `probe_action_v11` |
| [`finetune_libero_bct.py`](finetune_libero_bct.py) | **active (BC main)** | LIBERO 공식 BC-Transformer + 우리 어댑터(parvo-ptptk 등). encoder 비교 main table |

## Visualization / 진단

| Script | Status | 용도 |
|--------|--------|------|
| [`visualize_v15_no_mask.py`](visualize_v15_no_mask.py) | active | two-stream(v15-family) nomask reconstruction 시각화 |
| [`diagnose_v15_collapse.py`](diagnose_v15_collapse.py) | active | P-stream collapse 진단 |
| [`diagnose_ssim_scale.py`](diagnose_ssim_scale.py), [`diagnose_vjepa_p_trivial.py`](diagnose_vjepa_p_trivial.py) | active | SSIM scale·V-JEPA trivial 진단 |
| [`visualize_videomae_recon.py`](visualize_videomae_recon.py) | active | VideoMAE recon 시각화 |
| `scripts/viz/pca_overlay.py`, `grad_cam_arrow.py` | active | PCA·Grad-CAM viz (`probe_action_v11` util 공유) |
| [`analyze_delta_l.py`](analyze_delta_l.py) | legacy | v7-big σ 결정용 \|ΔL\| 분포 분석 (사용 안 함) |

## 클러스터 sbatch 매핑

| Script | Cluster sbatch |
|--------|---------------|
| `finetune_libero_bct.py` | [`finetune_libero_bct.sbatch`](../cluster/finetune_libero_bct.sbatch) |
| `probe_action.py` | [`probe_action.sbatch`](../cluster/probe_action.sbatch) |
| `probe_action_droid.py` / `_libero.py` / `_calvin.py` | [`probe_action_droid`](../cluster/probe_action_droid.sbatch) / [`_libero`](../cluster/probe_action_libero.sbatch) / [`_calvin`](../cluster/probe_action_calvin.sbatch) `.sbatch` |
| `value_alignment.py` | [`value_alignment.sbatch`](../cluster/value_alignment.sbatch) |

## 의존 그래프 (active)

```
probe_action ← probe_action_v11(공유 util) ← probe_action_libero
                                          ← probe_action_calvin
                                          ← value_alignment
             ← probe_action_droid

probe_action_v11(공유 util) → src/cortexbench/v15_loader.py (Paper-1 P-only)
                            → scripts/viz/{pca_overlay, grad_cam_arrow}.py
```

> **2026-06-23 리팩토링**: v4-10/v11 attention viz(`visualize_attn_*`), `probe_action_droid_v11`, `recon_quality_v11_vs_v15`, 구 어댑터 삭제 (git history). `probe_action_v11.py`는 활성 공유 util이라 keep. 상세 [docs/REFACTOR_PLAN.md](../../docs/REFACTOR_PLAN.md).
