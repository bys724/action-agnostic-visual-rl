#!/usr/bin/env python3
"""STEP 0 — dissociation slope 집계 (restart_plan.md §3/§5, 게이트 아님).

기존 probing 산출물만 집계 (학습 없음). 비교군의 도메인 취약성(dissociation
headroom)을 사전 확인하기 위한 도구.

slope = (in-domain EgoDex R²) − (OOD CALVIN pos R²)
  → "집(EgoDex)에서 낯선 곳(로봇)으로 갈 때 행동 디코딩이 얼마나 떨어지나".

⚠️ slope 절대값은 해석 불가: in-domain은 EgoDex 18-dim hand-pose(gap10),
   OOD는 CALVIN 3-dim position Δ(gap30)로 action-space·gap이 달라 confound됨.
   **의미 있는 신호 = `slope_ours − slope_VideoMAE`** (동일 target이라 target-space
   confound 상쇄) → restart_plan: ours slope가 monolithic보다 작으면 factorization 이득.

대상 (restart_plan §3):
  - slope 산출 = EgoDex-trained 모델만 (VideoMAE-ours; ours-S는 STEP 1 후 append).
  - frozen baseline(DINOv2/SigLIP/VC-1) = EgoDex 학습 없음 → in-domain 앵커 부재
    → OOD 절대값만 (dissociation floor 참조). slope = N/A.

데이터 소스 (parity 단일출처 = docs/eval_protocols.md §1·§4):
  - EgoDex in-domain : data/probing_results/probe_<enc>_patch_mean_concat_p_t_p_tk_gap10_test_*.json (key 'r2', 18-dim hand pose)
  - CALVIN OOD(진짜) : paper_artifacts/calvin_action_probing/<enc>_training_20260526_213639_gapsweep/all_gaps.csv
                       (§4 cross-folder: train=training/ eval=validation/, gap=30, pos=mean(r2_dim0..2)).
                       ⚠️ `_validation_*_seg` 런은 within-validation(in-distribution)이라 OOD 아님 — 쓰지 말 것.
  - LIBERO(보조)     : paper_artifacts/tables/tab2_probing/libero_all_gaps_summary.csv (gap=20, pos).
                       ⚠️ within-suite probe(cross-folder 아님) → CALVIN보다 약한 OOD. slope엔 미사용, 참고만.

사용: python3 scripts/eval/aggregate_dissociation_slope.py
"""
import csv
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "data" / "probing_results"
CALVIN_DIR = REPO / "paper_artifacts" / "calvin_action_probing"
LIBERO_CSV = REPO / "paper_artifacts" / "tables" / "tab2_probing" / "libero_all_gaps_summary.csv"

CALVIN_XFOLDER_RUN = "_training_20260526_213639_gapsweep"  # §4 cross-folder OOD 정규 런 (gap30, 4인코더 일치 검증)
CALVIN_GAP = 30
LIBERO_GAP = 20
POS_DIMS = (0, 1, 2)  # CALVIN/LIBERO 7-DoF: dim0-2=position Δ, dim3-5=rotvec, dim6=gripper

# (display, kind, egodex_json_glob, calvin_enc, libero_enc)
#   kind: 'egodex_trained' = in-domain 앵커 있음 → slope 산출 / 'frozen' = OOD 참조만
ENCODERS = [
    ("VideoMAE-ours", "egodex_trained",
     "probe_videomae_patch_mean_concat_p_t_p_tk_gap10_test_*.json", "videomae-ours", "videomae-ours"),
    ("DINOv2", "frozen", None, "dinov2", "dinov2"),
    ("SigLIP", "frozen", None, "siglip", "siglip"),
    ("VC-1",   "frozen", None, "vc1",    "vc1"),
    # MS-JEPA(=구 Parvo, code v15b) — ours, 붕괴 caveat·CALVIN 미측정 → slope N/A, 참고용.
    # ⚠️ probe_parvo_* = 디스크상 파일명(코드 식별자, rename deferred). 모델 기능명 = MS-JEPA.
    ("MS-JEPA (ours, 붕괴 caveat)", "egodex_trained",
     "probe_parvo_*patch_mean_concat_p_t_p_tk*test*.json", None, None),
]

# §4 baseline 표(gap=30 pos avg) — 읽기 정합성 sanity 대조용 (틀리면 경로/파싱 오류)
CALVIN_EXPECTED_POS = {"videomae-ours": 0.553, "vc1": 0.536, "dinov2": 0.223, "siglip": -0.314}


def read_egodex_r2(json_glob):
    """EgoDex in-domain R² (canonical: cls_mode=patch_mean_concat_p_t_p_tk, gap10, test)."""
    if json_glob is None:
        return None
    files = sorted(glob.glob(str(PROBE_DIR / json_glob)))
    if not files:
        return None
    with open(files[-1]) as f:  # 최신
        return json.load(f).get("r2")


def _pos_from_all_gaps(csv_path, gap):
    """all_gaps.csv에서 해당 gap 행의 pos R²(mean dim0-2) 반환."""
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if int(float(row["gap"])) == gap:
                return sum(float(row[f"r2_dim{d}"]) for d in POS_DIMS) / len(POS_DIMS)
    return None


def read_calvin_pos(calvin_enc):
    """CALVIN cross-folder OOD pos R² (§4 정규 런, gap=30)."""
    if calvin_enc is None:
        return None
    csv_path = CALVIN_DIR / f"{calvin_enc}{CALVIN_XFOLDER_RUN}" / "all_gaps.csv"
    if not csv_path.exists():
        return None
    return _pos_from_all_gaps(csv_path, CALVIN_GAP)


def read_libero_pos(libero_enc):
    """LIBERO OOD pos R² (gap=20, suites/views 평균). 보조 지표."""
    if libero_enc is None or not LIBERO_CSV.exists():
        return None
    vals = []
    with open(LIBERO_CSV) as f:
        for row in csv.DictReader(f):
            if row["encoder"] == libero_enc and int(float(row["gap"])) == LIBERO_GAP:
                vals.append(sum(float(row[f"r2_dim{d}"]) for d in POS_DIMS) / len(POS_DIMS))
    return sum(vals) / len(vals) if vals else None


def fmt(x):
    return f"{x:+.3f}" if isinstance(x, float) else "  —  "


def main():
    rows = []
    for disp, kind, ego_glob, calvin_enc, libero_enc in ENCODERS:
        ego = read_egodex_r2(ego_glob)
        calvin = read_calvin_pos(calvin_enc)
        libero = read_libero_pos(libero_enc)
        slope = (ego - calvin) if (kind == "egodex_trained" and ego is not None and calvin is not None) else None
        rows.append(dict(disp=disp, kind=kind, ego=ego, calvin=calvin, libero=libero, slope=slope))

        # sanity: 읽은 CALVIN pos가 §4 표와 일치하는지 (parity 회귀 가드)
        exp = CALVIN_EXPECTED_POS.get(calvin_enc)
        if exp is not None and calvin is not None and abs(calvin - exp) > 0.02:
            print(f"  ⚠️ CALVIN parity 경고 {calvin_enc}: 읽음 {calvin:+.3f} ≠ eval_protocols §4 {exp:+.3f}")

    print("\n=== STEP 0 — dissociation slope (in-domain EgoDex − OOD CALVIN pos) ===")
    print("    slope 절대값 ✗해석 / 신호 = slope_ours − slope_VideoMAE (target confound 상쇄)\n")
    hdr = f"{'encoder':<26} {'EgoDex(in)':>11} {'CALVIN-OOD':>11} {'LIBERO(참고)':>13} {'slope':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        note = "" if r["kind"] == "egodex_trained" else "  (frozen: OOD floor만)"
        print(f"{r['disp']:<26} {fmt(r['ego']):>11} {fmt(r['calvin']):>11} {fmt(r['libero']):>13} {fmt(r['slope']):>9}{note}")

    ref = next((r for r in rows if r["disp"] == "VideoMAE-ours"), None)
    print("\n해석:")
    if ref and ref["slope"] is not None:
        print(f"  • VideoMAE-ours slope = {ref['slope']:+.3f} = 비교 기준(beat-target).")
    print("  • frozen baseline = OOD 절대 R²(dissociation floor) 참조 — slope 개념 부재(EgoDex 미학습).")
    print("  • ours-S(MCP-MAE-S 등)는 STEP 1 학습+probing 후 ENCODERS에 append → slope_ours < slope_VideoMAE 확인.")
    print(f"  • CALVIN=gap{CALVIN_GAP} cross-folder(진짜 OOD) / LIBERO=gap{LIBERO_GAP} within-suite(참고), pos=dim0-2(gripper 제외, §4 aggregate는 gripper-dominated).")


if __name__ == "__main__":
    main()
