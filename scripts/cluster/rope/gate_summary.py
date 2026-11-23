"""RoPE 건강 관문 집계 (rope_encoder_plan.md §3-1 문턱 · 대조값 = 손 위치만 0.579 / 0.314, refinement_floor README).
motion = 위치 R² 차원 0–2 평균 · Δ = (특징 ⊕ 손 위치) − 손 위치만 · identity = 10 과제 정확도.
argv: <SUFFIX 접두 예 rope_R0_ep2> [<비교용 접두 예 fdn>]"""
import csv, glob, sys

BASE = "paper_artifacts/libero_action_probing"
POS_MOTION, POS_ID = 0.579, 0.314


def read(tag, cell):
    d = sorted(glob.glob(f"{BASE}/parvo_libero_object_*_{tag}_{cell}"))
    if not d:
        return None
    row = next(csv.DictReader(open(f"{d[-1]}/all_gaps.csv")))
    if "accuracy" in row:
        return float(row["accuracy"])
    return sum(float(row[f"r2_dim{i}"]) for i in range(3)) / 3


for tag in sys.argv[1:]:
    v = {c: read(tag, c) for c in ("m_act", "m_act_pos", "m_id", "m_id_pos", "pt_act", "pt_act_pos", "pt_id", "pt_id_pos")}
    if any(x is None for x in v.values()):
        print(tag, "missing cells:", [c for c, x in v.items() if x is None]); continue
    m_d, m_idr, pt_d = v["m_act_pos"] - POS_MOTION, v["m_id_pos"] - POS_ID, v["pt_act_pos"] - POS_MOTION
    gates = {"P identity >= 0.9": v["pt_id"] >= 0.9, "M motion Δ >= 0.160": m_d >= 0.160, "M identity residual <= 0.291": m_idr <= 0.291}
    print(f"== {tag}: M motion {v['m_act']:.3f}/Δ{m_d:+.3f} · M identity {v['m_id']:.3f}/잔여{m_idr:+.3f} · "
          f"P_t motion {v['pt_act']:.3f}/Δ{pt_d:+.3f} · P_t identity {v['pt_id']:.3f}")
    print("   gates:", {k: ("PASS" if ok else "FAIL") for k, ok in gates.items()})
