# scripts/eval/ — Evaluation Scripts

평가 / 분석 / 시각화 스크립트 인덱스. **active** vs **legacy** 구분 + 의존 관계
명시. 매 스크립트의 자세한 사용법은 파일 내부 docstring 참조.

> ⚠️ Refactor 정책: legacy 표시된 스크립트는 paper 마감까지 유지 (Phase 1.5
> 결과 재생성 필요할 수 있음). 마감 후 `_archive/`로 이동 검토.

## Action Probing (frozen encoder → linear/MLP probe → action target)

| Script | Status | 용도 | 의존 |
|--------|--------|------|------|
| [`probe_action.py`](probe_action.py) | **active** | EgoDex within-domain probing (within-EgoDex action target). v6/v10 ckpt | base 함수 (다른 probe가 import) |
| [`probe_action_v11.py`](probe_action_v11.py) | **active** | EgoDex probing for v11. 12-mode 지원 (A/B/D/D' + concat) | imports `probe_action` |
| [`probe_action_droid.py`](probe_action_droid.py) | **active** | DROID cross-domain probing. v6/v10 | imports `probe_action` |
| [`probe_action_droid_v11.py`](probe_action_droid_v11.py) | **active** | DROID cross-domain probing for v11 | imports `probe_action_v11`, `probe_action_droid` |
| [`probe_action_libero.py`](probe_action_libero.py) | **active** | LIBERO Action Probing (Phase 2 보강, gap-matched DROID protocol) | imports `probe_action_v11` |

## Value Alignment (Phase 2.5, frozen encoder → trajectory ρ)

| Script | Status | 용도 |
|--------|--------|------|
| [`value_alignment.py`](value_alignment.py) | **active** | LIBERO trajectory frame-wise embedding → Spearman ρ(t, cos(e_t, e_T)) |

## BC fine-tune (encoder → policy → action)

| Script | Status | 용도 |
|--------|--------|------|
| [`finetune_libero_bct.py`](finetune_libero_bct.py) | **active (Phase 3 main)** | LIBERO 공식 BC-Transformer + 우리 어댑터. 5 encoder 비교 main table |

## Visualization

| Script | Status | 용도 |
|--------|--------|------|
| [`visualize_attn_compare.py`](visualize_attn_compare.py) | **active** | Two-Stream attention + prediction cross-domain 비교 (v6/v10/v11). high-change sample 자동 선별 |
| [`visualize_attn_v11.py`](visualize_attn_v11.py) | **active** | v11 전용 attention + reconstruction. 4 sample × 8 column grid |
| [`visualize_attn_rotation.py`](visualize_attn_rotation.py) | **active** | Rotation diagnostic — content-driven (equivariant) vs position-prior (invariant) 검증 |
| [`visualize_inference.py`](visualize_inference.py) | **active** | Two-Stream 추론 결과 시각화 (v4/v6/v10) |
| [`visualize_sample_detail.py`](visualize_sample_detail.py) | **active** | 단일 샘플 상세 시각화 (M/P channel + attention) |

## 일회성 분석

| Script | Status | 용도 |
|--------|--------|------|
| [`analyze_delta_l.py`](analyze_delta_l.py) | **legacy (Phase 1.5)** | v7-big Gaussian weighting σ 결정용 \|ΔL\| 분포 분석. v8 폐기 후 사용 안 함 |

## 클러스터 sbatch 매핑

| Script | Cluster sbatch |
|--------|---------------|
| `finetune_libero_bct.py` | [`scripts/cluster/finetune_libero_bct.sbatch`](../cluster/finetune_libero_bct.sbatch) |
| `probe_action.py` | [`scripts/cluster/probe_action.sbatch`](../cluster/probe_action.sbatch) |
| `probe_action_droid.py` | [`scripts/cluster/probe_action_droid.sbatch`](../cluster/probe_action_droid.sbatch) |
| `probe_action_libero.py` | [`scripts/cluster/probe_action_libero.sbatch`](../cluster/probe_action_libero.sbatch) |
| `value_alignment.py` | [`scripts/cluster/value_alignment.sbatch`](../cluster/value_alignment.sbatch) |
| `visualize_attn_*` | [`scripts/cluster/viz_two_stream.sbatch`](../cluster/viz_two_stream.sbatch) |

## 의존 그래프 (active scripts)

```
probe_action ← probe_action_v11 ← probe_action_libero
                              ← probe_action_droid_v11
                              ← value_alignment
              ← probe_action_droid

visualize_attn_compare ← visualize_attn_v11
                       ← visualize_attn_rotation

probe_action_v11 ← src/encoders/adapters/two_stream_v11.py (어댑터)
```
