#!/usr/bin/env python3
"""STEP 0 — OOD action-probing efficiency 표 생성 (paper_artifacts, 재현 가능).

restart_plan §3b: slope(3a)는 dataset 난이도 confound로 폐기. 살아남은 주장 =
**efficiency 절대값** — CoMP-MAE-S(~32M P+M, 좁은 unlabeled home-video)가
internet-scale ViT-B frozen(DINOv2/SigLIP/VC-1) 및 same-data VideoMAE(86M)와
로봇 action probing에서 얼마나 경합하나.

수치는 전부 소스 산출물(summary.json / all_gaps.csv)에서 직접 읽음 — 손 전사 없음.
parity: CALVIN cross-folder gap30 n_eval=32183 / LIBERO_spatial gap20 n_eval=9690
(전 인코더 동일, agentview_rgb). 실행: python3 scripts/eval/build_step0_efficiency_table.py
"""
import csv
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAL_DIR = REPO / "paper_artifacts" / "calvin_action_probing"
LIB_DIR = REPO / "paper_artifacts" / "libero_action_probing"
LIB_BASELINE_CSV = REPO / "paper_artifacts" / "tables" / "tab2_probing" / "libero_all_gaps_summary.csv"
OUT_DIR = REPO / "paper_artifacts" / "tables" / "step0_ood_efficiency"
POS = (0, 1, 2)  # 7-DoF: dim0-2 = position Δ (translational)

# (display, params_M, pretrain, calvin step0 dir glob | None, libero step0 dir glob | None,
#  calvin baseline run | None, libero baseline suite-row enc | None)
# step0 = 이번 세션 신규(comp/vmae-vla) / baseline = 기존 canonical(213639 gapsweep / summary CSV)
ENCODERS = [
    # ours (CoMP-MAE-S) — P+M(p_t_m) 및 P-only(p_t_p_tk), mean/attentive
    ("CoMP-MAE-S  P_t⊕M  (mean)",  32.3, "EgoDex part1 subset (~46k, unlabeled)",
     "parvo_training_*step0_mean_ptm", "parvo_libero_spatial_*step0_mean_ptm", None, None),
    ("CoMP-MAE-S  P_t⊕M  (attn)",  32.3, "EgoDex part1 subset (~46k, unlabeled)",
     "parvo_training_*step0_attn_ptm", "parvo_libero_spatial_*step0_attn_ptm", None, None),
    ("CoMP-MAE-S  P_t⊕P_tk (mean)", 21.6, "EgoDex part1 subset (~46k, unlabeled)",
     "parvo_training_*step0_mean_ptptk", "parvo_libero_spatial_*step0_mean_ptptk", None, None),
    ("CoMP-MAE-S  P_t⊕P_tk (attn)", 21.6, "EgoDex part1 subset (~46k, unlabeled)",
     "parvo_training_*step0_attn_ptptk", "parvo_libero_spatial_*step0_attn_ptptk", None, None),
    # matched-data baseline (VideoMAE-ours) — VLA self-consistent (comp와 동일 forward 규약)
    ("VideoMAE-ours (mean, vla)",  86.0, "EgoDex full (~314k)",
     "videomae-ours_training_*step0_vla_mean", "videomae-ours_libero_spatial_*step0_vla_mean", None, None),
    ("VideoMAE-ours (attn, vla)",  86.0, "EgoDex full (~314k)",
     "videomae-ours_training_*step0_vla_attn", "videomae-ours_libero_spatial_*step0_vla_attn", None, None),
    # internet-scale frozen (unmatched reference, 전부 ViT-B ~86M)
    ("VC-1 (frozen)",     86.0, "Ego4D+ (internet)",   None, None,
     "vc1_training_20260526_213639_gapsweep", "vc1"),
    ("DINOv2 (frozen)",   86.0, "LVD-142M (internet)", None, None,
     "dinov2_training_20260526_213639_gapsweep", "dinov2"),
    ("SigLIP (frozen)",   86.0, "WebLI (internet)",    None, None,
     "siglip_training_20260526_213639_gapsweep", "siglip"),
]

CAL_GAP, LIB_GAP = 30, 20
CAL_NEVAL, LIB_NEVAL = 32183, 9690  # parity 앵커 (틀리면 오source)


def _pos_from_summary(path):
    d = json.load(open(path))
    per = d.get("r2_per_dim") or [d.get(f"r2_dim{i}") for i in range(7)]
    n_eval = d.get("n_eval_pairs")
    return sum(per[i] for i in POS) / len(POS), n_eval


def calvin_step0(glob_pat):
    fs = glob.glob(str(CAL_DIR / glob_pat / f"gap{CAL_GAP}" / "summary.json"))
    return _pos_from_summary(sorted(fs)[-1]) if fs else (None, None)


def calvin_baseline(run):
    csvp = CAL_DIR / run / "all_gaps.csv"
    if not csvp.exists():
        return None, None
    for row in csv.DictReader(open(csvp)):
        if int(float(row["gap"])) == CAL_GAP:
            return sum(float(row[f"r2_dim{d}"]) for d in POS) / len(POS), None  # baseline CSV엔 n_eval 없음
    return None, None


def libero_step0(glob_pat):
    fs = glob.glob(str(LIB_DIR / glob_pat / f"gap{LIB_GAP}" / "summary.json"))
    return _pos_from_summary(sorted(fs)[-1]) if fs else (None, None)


def libero_baseline(enc):
    for row in csv.DictReader(open(LIB_BASELINE_CSV)):
        if (row["encoder"] == enc and int(float(row["gap"])) == LIB_GAP
                and row["task_suite"] == "libero_spatial" and row["view"] == "agentview_rgb"):
            pos = sum(float(row[f"r2_dim{d}"]) for d in POS) / len(POS)
            return pos, int(row["n_eval_pairs"])
    return None, None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    warn = []
    for disp, params, data, cal_s, lib_s, cal_b, lib_b in ENCODERS:
        cal, cal_n = calvin_step0(cal_s) if cal_s else calvin_baseline(cal_b)
        lib, lib_n = libero_step0(lib_s) if lib_s else libero_baseline(lib_b)
        # parity 검증
        if cal_n is not None and cal_n != CAL_NEVAL:
            warn.append(f"CALVIN n_eval mismatch {disp}: {cal_n}≠{CAL_NEVAL}")
        if lib_n is not None and lib_n != LIB_NEVAL:
            warn.append(f"LIBERO n_eval mismatch {disp}: {lib_n}≠{LIB_NEVAL}")
        rows.append((disp, params, data, cal, lib))

    csv_path = OUT_DIR / "efficiency.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "params_M", "pretrain_data",
                    "calvin_pos_r2_gap30_xfold", "libero_spatial_pos_r2_gap20"])
        for disp, params, data, cal, lib in rows:
            w.writerow([disp, params, data,
                        f"{cal:.4f}" if cal is not None else "",
                        f"{lib:.4f}" if lib is not None else ""])
    print(f"wrote {csv_path}")

    # 콘솔 미리보기
    print(f"\n{'encoder':30s} {'params':>7s} {'CALVIN':>8s} {'LIBERO_sp':>10s}")
    for disp, params, data, cal, lib in rows:
        print(f"{disp:30s} {params:6.1f}M {cal:+8.4f} {lib:+10.4f}")
    print(f"\nparity 앵커: CALVIN n_eval={CAL_NEVAL} / LIBERO_spatial n_eval={LIB_NEVAL}")
    if warn:
        print("\n⚠️ PARITY 경고:")
        for w_ in warn:
            print("  " + w_)
    else:
        print("✅ parity 경고 없음 (검증된 n_eval 인코더 전부 앵커 일치)")


if __name__ == "__main__":
    main()
