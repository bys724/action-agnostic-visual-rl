"""RoPE §7-2 사전 점검 ①② 집계 (docs/rope_encoder_plan.md §7-2 "dev 조작화 11-25") — 사후 보고 행 · §3 판정 불변.

① 해상도 보간: R0 칸 pc_ri_R0_m_act_pos_s{seed} 의 res(외삽 · §3 재현 확인) · resi(보간) 낙폭. APE 판(A0 · C1-DN)은
   보간 = §3 res 값(같은 코드 경로). 읽기: 버팀 = 두 해상도 모두 R0 보간 낙폭 ≤ A0 평균 × 2 · 무너짐 = 어느 쪽이든 > 1.0.
② 판독기 측 위치 임베딩: pc_pe_{m}_{cell}_s{seed} 를 원판 rj_ 와 나란히. 읽기: 격차가 좁아짐 = "R0 가 A0 보다 비중첩 하락"인
   OOD 벤치 수가 원판보다 줄어듦.
출력 = paper_artifacts/rope_judge/precheck.{md,json}. 칸이 빠지면 '미완'.
"""
import json

import numpy as np

from judge_summary import CAL, LIB, MODELS, OUT, POS_ID, POS_MOTION, SEEDS, _summary, fmt, lt, pos_r2, seeds_of

BENCH = ("calvin", "spatial", "object", "goal")


def get(prefix, m, name, s, base=LIB, gap=20):
    return _summary(base, f"{prefix}_{m}_{name}_s{s}", gap)


def vals(prefix, m, name, f, base=LIB, gap=20):
    return seeds_of(lambda s: (lambda c: None if c is None else f(c))(get(prefix, m, name, s, base, gap)))


def drop(kind, r):
    return lambda c: None if not c.get("perturb") else pos_r2(c) - float(np.mean(c["perturb"][kind][str(r)]["r2_per_dim"][:3]))


def main():
    J = {}
    L = ["# RoPE §7-2 사전 점검 ①② (사후 보고 행 · §3 판정 불변)\n", "값 = seed 3 평균 [최솟값, 최댓값]\n", "## ① 해상도 보간\n",
         "| 자 | R0 외삽(§3) | R0 외삽(재측정) | R0 보간 | A0 (= 보간) | C1-DN (= 보간) |", "|---|---|---|---|---|---|"]
    one = {}
    for r in (288, 320):
        row = {"R0 외삽 §3": vals("rj", "R0", "m_act_pos", drop("res", r)),
               "R0 외삽 재": vals("pc_ri", "R0", "m_act_pos", drop("res", r)),
               "R0 보간": vals("pc_ri", "R0", "m_act_pos", drop("resi", r)),
               "A0": vals("rj", "A0", "m_act_pos", drop("res", r)),
               "C1-DN": vals("rj", "fdn", "m_act_pos", drop("res", r))}
        one[r] = row
        L.append(f"| 해상도 {r} 낙폭 | " + " | ".join(fmt(v) for v in row.values()) + " |")
    ri = [one[r]["R0 보간"] for r in (288, 320)]
    if None in ri or None in [one[r]["A0"] for r in (288, 320)]:
        J["① 읽기"] = "미완"
    else:
        hold = all(np.mean(one[r]["R0 보간"]) <= 2 * np.mean(one[r]["A0"]) for r in (288, 320))
        fall = any(np.mean(one[r]["R0 보간"]) > 1.0 for r in (288, 320))
        J["① 읽기"] = "버팀" if hold else ("무너짐 → 구현 점검 뒤 본학습(제출 보류)" if fall else "중간 → 보고 · Vault")
    rep = [None if one[r]["R0 외삽 §3"] is None or one[r]["R0 외삽 재"] is None
           else bool(np.allclose(one[r]["R0 외삽 §3"], one[r]["R0 외삽 재"], atol=1e-6)) for r in (288, 320)]
    J["① §3 외삽 값 재현"] = rep

    L += ["\n## ② 판독기 측 위치 임베딩\n", "| 자 | " + " | ".join(f"{m} 원판 | {m} PE" for m in MODELS) + " |",
          "|---|" + "---|" * (2 * len(MODELS))]
    gate = {"M motion Δ": ("m_act_pos", lambda c: pos_r2(c) - POS_MOTION), "M identity 잔여": ("m_id_pos", lambda c: c["accuracy"] - POS_ID),
            "P identity": ("pt_id", lambda c: c["accuracy"]), "P_t motion Δ": ("pt_act_pos", lambda c: pos_r2(c) - POS_MOTION)}
    T = {}
    for k, (name, f) in gate.items():
        T[k] = {m: (vals("rj", m, name, f), vals("pc_pe", m, name, f)) for m in MODELS}
    for b in BENCH:
        base, gap = (CAL, 30) if b == "calvin" else (LIB, 20)
        T[f"OOD {b} attn"] = {m: (vals("rj", m, f"ood_{b}_attn", pos_r2, base, gap), vals("pc_pe", m, f"ood_{b}_attn", pos_r2, base, gap))
                              for m in MODELS}
    for k, d in T.items():
        L.append(f"| {k} | " + " | ".join(f"{fmt(d[m][0])} | {fmt(d[m][1])}" for m in MODELS) + " |")
    gaps = []
    for b in BENCH:
        (r_o, r_p), (a_o, a_p) = T[f"OOD {b} attn"]["R0"], T[f"OOD {b} attn"]["A0"]
        gaps.append({"bench": b, "원판 R0−A0": None if None in (r_o, a_o) else float(np.mean(r_o) - np.mean(a_o)),
                     "PE R0−A0": None if None in (r_p, a_p) else float(np.mean(r_p) - np.mean(a_p)),
                     "원판 비중첩 하락": lt(r_o, a_o), "PE 비중첩 하락": lt(r_p, a_p)})
    L += ["\n| OOD 벤치 | 원판 R0−A0 | PE R0−A0 | 원판 비중첩 하락 | PE 비중첩 하락 |", "|---|---|---|---|---|"]
    L += [f"| {g['bench']} | {g['원판 R0−A0']} | {g['PE R0−A0']} | {g['원판 비중첩 하락']} | {g['PE 비중첩 하락']} |" for g in gaps]
    if any(g["PE 비중첩 하락"] is None or g["원판 비중첩 하락"] is None for g in gaps):
        J["② 읽기"] = "미완"
    else:
        n_o, n_p = sum(g["원판 비중첩 하락"] for g in gaps), sum(g["PE 비중첩 하락"] for g in gaps)
        J["② 비중첩 하락 벤치 수 (원판 → PE)"] = (n_o, n_p)
        J["② 읽기"] = ("격차 좁아짐 → R-full OOD 자 = PE 판(주) + 원판(보고)" if n_p < n_o
                      else "격차 그대로 → OOD 하락은 실질(R-full 기대 문장 (b) 쪽)")
    L += ["\n## 읽기 (결과 전 고정 규칙 적용)\n"] + [f"- {k}: {v}" for k, v in J.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "precheck.md").write_text("\n".join(L) + "\n")
    json.dump({"resolution": {str(r): v for r, v in one.items()}, "probe_pe": T, "ood_gaps": gaps, "reading": J},
              open(OUT / "precheck.json", "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
