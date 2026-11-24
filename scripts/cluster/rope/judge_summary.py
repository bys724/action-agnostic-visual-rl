"""RoPE 판정 단계 집계 (docs/rope_encoder_plan.md §3 규칙 · "dev 조작화") — judge_chain.sh 산출물을 읽어 표 + 판정.

모델 = fdn(C1-DN) · R0(최종 R) · A0(최종 A). seed = 판독기 seed 0·1·2. CI = seed 값의 [최솟값, 최댓값](비중첩 = 범위가 안 겹침).
대조값(손 위치만) = 움직임 0.579 · 정체성 0.314 (gate_summary.py 와 같음 · 모델 무관 상수).
출력 = paper_artifacts/rope_judge/summary.{md,json}. 칸이 빠지면 그 자는 '미완'으로 적고 판정하지 않는다.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LIB, CAL = ROOT / "paper_artifacts/libero_action_probing", ROOT / "paper_artifacts/calvin_action_probing"
OUT = ROOT / "paper_artifacts/rope_judge"
MODELS, SEEDS = ("fdn", "R0", "A0"), (0, 1, 2)
POS_MOTION, POS_ID = 0.579, 0.314


def _summary(base, suffix, gap):
    d = sorted(glob.glob(str(base / f"parvo_*_{suffix}")))
    if not d:
        return None
    f = Path(d[-1]) / f"gap{gap}" / "summary.json"
    return json.load(open(f)) if f.exists() else None


def pos_r2(s):
    return float(np.mean(s["r2_per_dim"][:3]))


def cell(m, name, seed, base=LIB, gap=20):
    return _summary(base, f"rj_{m}_{name}_s{seed}", gap)


def seeds_of(fn):
    v = [fn(s) for s in SEEDS]
    return None if any(x is None for x in v) else v


def rng(v):
    return None if v is None else (min(v), max(v))


def fmt(v):
    return "미완" if v is None else f"{np.mean(v):+.3f} [{min(v):+.3f}, {max(v):+.3f}]"


def lt(a, b):   # a 가 b 보다 작다 · 범위 비중첩
    return None if a is None or b is None else max(a) < min(b)


def main():
    T, J, L = {}, {}, ["# RoPE 판정 단계 집계 (rope_encoder_plan.md §3)\n", "값 = seed 3 평균 [최솟값, 최댓값]\n"]
    for m in MODELS:
        def g(name, f):
            return seeds_of(lambda s: (lambda c: None if c is None else f(c))(cell(m, name, s)))
        t = {
            "M motion Δ": g("m_act_pos", lambda c: pos_r2(c) - POS_MOTION),
            "M identity 잔여": g("m_id_pos", lambda c: c["accuracy"] - POS_ID),
            "P identity": g("pt_id", lambda c: c["accuracy"]),
            "P_t motion Δ": g("pt_act_pos", lambda c: pos_r2(c) - POS_MOTION),
        }
        for k in (8, 16, 32):   # 평행이동 낙폭 = 깨끗한 R² − 이동 R² (같은 판독기)
            t[f"shift {k} 낙폭"] = g("m_act_pos", lambda c, k=k: None if not c.get("perturb") else
                                   pos_r2(c) - float(np.mean(c["perturb"]["shift"][str(k)]["r2_per_dim"][:3])))
        for r in (288, 320):   # 해상도 낙폭 = 깨끗한 224 R² − 같은 판독기로 r 해상도 평가한 R²
            t[f"해상도 {r} 낙폭"] = g("m_act_pos", lambda c, r=r: None if not c.get("perturb") else
                                   pos_r2(c) - float(np.mean(c["perturb"]["res"][str(r)]["r2_per_dim"][:3])))
        for b in ("calvin", "spatial", "object", "goal"):
            for ro in ("attn", "mean"):
                base, gap = (CAL, 30) if b == "calvin" else (LIB, 20)
                t[f"OOD {b} {ro}"] = seeds_of(lambda s: (lambda c: None if c is None else pos_r2(c))(
                    cell(m, f"ood_{b}_{ro}", s, base, gap)))
        p = OUT / f"position_{m}.json"
        if p.exists():
            q = json.load(open(p))
            t["등변성 P"], t["등변성 M"] = (q["equivariance"][s] for s in ("P", "M"))
            t["위치 판독 top-1"] = [r["top1"] for r in q["position_readout"]["per_seed"]]
        a4 = [json.load(open(f))["age_mean"]["beyond"] for f in sorted(glob.glob(str(OUT / f"alpha4a_{m}/*.json")))]
        t["4-a 위치 너머(나이 평균)"] = a4 if len(a4) == 3 else None
        T[m] = t

    keys = [k for k in T["fdn"] if not k.startswith("등변성")]
    L.append("| 자 | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
    for k in keys:
        L.append(f"| {k} | " + " | ".join(fmt(T[m].get(k)) for m in MODELS) + " |")
    for s in ("P", "M"):
        L.append(f"| 등변성 {s} (옮겨 간 비율 · 부트스트랩) | " + " | ".join(
            "미완" if f"등변성 {s}" not in T[m] else
            f"{T[m][f'등변성 {s}']['frac_moved']:.3f} [{T[m][f'등변성 {s}']['frac_ci'][0]:.3f}, {T[m][f'등변성 {s}']['frac_ci'][1]:.3f}]"
            for m in MODELS) + " |")

    R, A, C = T["R0"], T["A0"], T["fdn"]
    mean = lambda v: None if v is None else float(np.mean(v))
    # 3-1 건강 관문 (R · seed 평균)
    gate = None if any(R[k] is None for k in ("P identity", "M motion Δ", "M identity 잔여")) else (
        mean(R["P identity"]) >= 0.9 and mean(R["M motion Δ"]) >= 0.160 and mean(R["M identity 잔여"]) <= 0.291)
    J["3-1 건강 관문(R)"] = gate
    # ① 위치 의존성: 평행이동 낙폭 R < A (k 셋 중 ≥ 2 비중첩)
    sh = [lt(R[f"shift {k} 낙폭"], A[f"shift {k} 낙폭"]) for k in (8, 16, 32)]
    J["① 평행이동 낙폭 R < A (k 별)"] = sh
    J["① 성립"] = None if None in sh else sum(sh) >= 2
    eq = []
    for s in ("P", "M"):
        if f"등변성 {s}" in R and f"등변성 {s}" in A:
            eq.append(R[f"등변성 {s}"]["frac_ci"][0] > A[f"등변성 {s}"]["frac_ci"][1])
    J["보조 등변성 R > A (P, M)"] = eq or None
    # ② 범용: OOD attn R ≥ A 전 벤치(비중첩 하락 없음) ∧ ≥ 2 벤치 비중첩 상승
    ood = [(lt(R[f"OOD {b} attn"], A[f"OOD {b} attn"]), lt(A[f"OOD {b} attn"], R[f"OOD {b} attn"]))
           for b in ("calvin", "spatial", "object", "goal")]
    J["② OOD (R 하락, R 상승) 벤치별"] = ood
    ood_ok = None if any(None in x for x in ood) else (not any(d for d, _ in ood) and sum(u for _, u in ood) >= 2)
    res = [lt(R[f"해상도 {r} 낙폭"], A[f"해상도 {r} 낙폭"]) for r in (288, 320)]
    J["② 해상도 낙폭 R < A (288, 320)"] = res
    res_ok = None if None in res else all(res)
    guard = None
    if all(R[k] is not None and C[k] is not None for k in ("P identity", "M motion Δ", "M identity 잔여")):
        guard = (mean(R["P identity"]) >= 0.95 and mean(R["M motion Δ"]) >= mean(C["M motion Δ"]) - 0.05
                 and mean(R["M identity 잔여"]) <= mean(C["M identity 잔여"]) + 0.05)
    J["② 같은 분포 가드(R vs C1-DN)"] = guard
    J["② 성립"] = None if None in (ood_ok, res_ok, guard) else (ood_ok and res_ok and guard)
    one, two = J["① 성립"], J["② 성립"]
    if gate is False:
        concl = "공학 단계 출구 재측정 미달 — 판정 열지 않음(보고)"
    elif None in (gate, one, two):
        concl = "미완 칸 있음 — 결론 보류"
    else:
        concl = {(True, True): "① ∧ ② 성립 → 위치 의존성을 낮춘 RoPE 인코더가 범용 판독 · 해상도 외삽에서 C1-DN 을 넘는다",
                 (True, False): "① 성립 · ② 불성립 → 위치 의존성은 낮아졌으나 범용 기능 이득은 없다"
                                + ("(가드 위반 = 위치 정보를 잃은 비용)" if guard is False else ""),
                 (False, True): "① 불성립 · ② 성립 → 이득은 있으나 위치 의존성 기전으로 설명되지 않는다",
                 (False, False): "둘 다 불성립 → 닫음"}[(one, two)]
    J["결론(§3-4)"] = concl
    L.append("\n## 판정\n")
    L += [f"- {k}: {v}" for k, v in J.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    json.dump({"table": {m: {k: v for k, v in t.items()} for m, t in T.items()}, "judgment": J},
              open(OUT / "summary.json", "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
