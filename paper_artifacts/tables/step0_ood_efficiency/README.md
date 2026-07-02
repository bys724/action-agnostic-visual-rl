# STEP 0 — OOD Action-Probing Efficiency (paper material)

**Claim (3b, surviving)**: A representation pretrained on **narrow, unlabeled, first-person
home-video (EgoDex)** at **~32M params** matches or edges **internet-scale ViT-B frozen
encoders** and approaches a **larger same-data video baseline** on robot action probing —
supporting a *data/parameter-efficiency* reading of action-agnostic pretraining.

> **Slope/dissociation (3a) was dropped** — see "Why not slope" below. This efficiency claim
> uses **absolute** OOD probing R² and is not affected by that confound.

Regenerate (numbers read from source artifacts, no hand-transcription):
`python3 scripts/eval/build_step0_efficiency_table.py` → `efficiency.csv` + this table.

## Table — position R² (translational dims 0–2), higher = better

LIBERO는 3 suite(spatial / object / goal, 전부 `agentview_rgb`, gap20)로 확장. CALVIN은
cross-folder OOD. CoMP-MAE-S·VideoMAE-ours만 mean/attn 둘 다 산출(그 외는 canonical readout).

| Encoder | Params | Pretrain data | CALVIN (xfold) | LIBERO-spatial | LIBERO-object | LIBERO-goal |
|---|---:|---|---:|---:|---:|---:|
| **CoMP-MAE-S  P_t⊕M  (attn)** | **~32M** | EgoDex part1 subset (~46k, unlabeled) | **+0.487** | **+0.814** | **+0.851** | **+0.751** |
| CoMP-MAE-S  P_t⊕M  (mean) | ~32M | EgoDex part1 subset | +0.405 | +0.809 | +0.838 | +0.708 |
| CoMP-MAE-S  P_t⊕P_tk (attn) | ~22M | EgoDex part1 subset | +0.175 | +0.766 | +0.829 | +0.709 |
| CoMP-MAE-S  P_t⊕P_tk (mean) | ~22M | EgoDex part1 subset | +0.257 | +0.741 | +0.811 | +0.702 |
| VideoMAE-ours (attn) *matched data* | 86M | EgoDex full (~314k) | +0.610 | +0.879 | +0.903 | +0.830 |
| VideoMAE-ours (mean) *matched data* | 86M | EgoDex full | +0.529 | +0.853 | +0.867 | +0.791 |
| VC-1 (frozen) | 86M | Ego4D+ (internet) | +0.536 | +0.891 | +0.905 | +0.836 |
| DINOv2 (frozen) | 86M | LVD-142M (internet) | +0.223 | +0.766 | +0.677 | +0.489 |
| SigLIP (frozen) | 86M | WebLI (internet) | −0.314 | +0.787 | +0.749 | +0.547 |

## How to read

- **vs internet-scale frozen (ViT-B, 86M)**: CoMP-MAE-S `P_t⊕M` **beats DINOv2 and SigLIP**
  clearly on CALVIN (+0.49 vs +0.22 / −0.31) and the margin **widens on object/goal**:
  object +0.85 vs +0.68 / +0.75, **goal +0.75 vs +0.49 / +0.55**. On the goal suite the
  appearance-only internet models *collapse* while CoMP-MAE-S holds — a ~32M home-video model
  outscoring 86M internet-scale by +0.20–0.26. (Spatial is the closest suite: +0.81 vs
  +0.77 / +0.79.) It approaches VC-1 (the one motion/ego-centric internet baseline) across all
  suites. → narrow home-video at ~32M competes with internet-scale on action-relevant probing.
- **vs matched-data VideoMAE (86M)**: CoMP-MAE-S approaches it at **~2.7× fewer params** on
  every suite (LIBERO spatial 0.81 vs 0.85–0.88 / object 0.85 vs 0.87–0.90 / goal 0.75 vs
  0.79–0.83; CALVIN 0.49 vs 0.53–0.61). Framing = **"small but close"**, not "scale it up to
  win" (ViT-B variant did *worse*, per project note).
- **M stream is load-bearing**: `P_t⊕M` ≫ `P_t⊕P_tk` (appearance-only) on both benchmarks —
  the efficiency comes from the motion stream, consistent with the action-agnostic thesis.

## Methodology / parity (recurring-incident guard: eval_protocols §0)

- **Target**: 7-DoF pose-derived; **position = dims 0–2** (translational Δ). Rotation/gripper
  excluded (noisier, not the efficiency-relevant channel).
- **CALVIN**: cross-folder OOD (train on `training/`, eval on `validation/`), gap30 (~1 s),
  MAX_EPISODES=200. **Parity anchor: n_eval = 32,183 pairs — identical across all 10 encoders.**
- **LIBERO**: `libero_{spatial,object,goal}`, `agentview_rgb`, gap20 (~1 s). **Per-suite parity
  anchors: n_eval = spatial 9,690 / object 12,710 / goal 11,100 — identical across all encoders
  within each suite** (build script asserts this). ⚠️ Never average across suites (earlier draft
  did → wrong); report each suite as its own column.
- **VideoMAE readout**: reported values use the **VLA token path** (same forward convention as
  CoMP-MAE, mean/attentive self-consistent), *not* the BC-T adapter. Canonical adapter
  reference (CALVIN gap30) = +0.553 for cross-checking.
- **Probe**: frozen encoder → linear (mean) / AttentivePoolProbe (attn), 20 ep. `mean` =
  patch-mean concat; `attn` = per-stream learnable-query pool.

## Why not slope (3a dropped)

In-domain EgoDex action probing **ceilings at ~0.47** across **all 300 probe runs** (best =
VideoMAE +0.470); CoMP-MAE `P_t⊕M` in-domain = +0.095 (rightHand, dim-matched) / +0.099 (18d).
OOD position (end-effector) is intrinsically **easier** (ceiling ~0.85). So *every* encoder's
R² **rises** going in-domain→OOD, and `slope = in − OOD` is dominated by this
**target-difficulty asymmetry**, not generalization. `slope_comp − slope_VideoMAE = −0.247`
(p_t_m) largely reflects CoMP-MAE starting lower on the hard in-domain and converging toward the
shared easier-OOD ceiling — **regression-to-ceiling, not motion-driven robustness**. Dim-matching
(rightHand 3D ≈ 18d aggregate) and 2-domain replication did not remove it. → slope is not a
trustworthy claim; **mechanism selectivity (STEP 0.5 ③, corrupt-in-place ΔR²) is the clean test**.

## Caveats

- **SigLIP −0.31 on CALVIN**: language-aligned contrastive model, weak at fine geometric
  position — not a fair "vision" baseline for translational decoding. DINOv2 also appearance-
  pretrained. CoMP-MAE beating them ⇒ beating *appearance-only* pretraining, consistent with
  "action needs motion/geometric info."
- **LIBERO-spatial is in-suite** (easier OOD than CALVIN cross-folder); margins there compress.
- `P_t⊕M` uses the M stream at inference (not the P-only deployment). Efficiency claim is about
  the probed representation, not the scaffold-then-drop deployment.

## Provenance

- Generator: [`scripts/eval/build_step0_efficiency_table.py`](../../../scripts/eval/build_step0_efficiency_table.py)
- CoMP-MAE-S ckpt: `two_stream_v15b_step1_comp_mae_s/20260629_101634/latest.pt`
- Jobs (2026-07-01): CALVIN/LIBERO-spatial step0 matrix `36224980–991` (+ `36225602` vmae-attn CALVIN rerun).
- Jobs (2026-07-02): LIBERO-object/goal step0 matrix `36300651–662` (+ `36303903` vmae-attn object rerun, mem120G OOM fix).
- Baselines: CALVIN `*_training_20260526_213639_gapsweep`; LIBERO (all 3 suites) `tables/tab2_probing/libero_all_gaps_summary.csv`.
- Session log: `docs/cluster_sessions.md` (2026-07-01 STEP 0).
