"""T1 θ 점검 집계 (docs/rope_encoder_plan.md §7-2 ③ "dev 조작화 11-25 (2)") — 처음부터 4 ep 건강 관문 · 공학 단계 수치.

판 = T1th100(θ 100) · T1th10000(θ 10000). 칸 = t1_{run}_{cell}_s{seed} · 판독기 seed 0·1·2 · CI = [최솟값, 최댓값].
채택 규칙(결과 전 고정): θ 100 채택 = 두 판 모두 관문 셋 통과 ∧ 세 값(M motion Δ ↑ · M identity 잔여 ↓ · P identity ↑) 중
≥ 1 에서 θ 100 이 비중첩으로 더 좋고 어느 값에서도 비중첩으로 더 나쁘지 않음. θ 100 만 관문 미달이면 10000 · 그 외 = 10000 유지.
출력 = paper_artifacts/rope_judge/theta_t1.{md,json}.
"""
import json

import numpy as np

from judge_summary import LIB, OUT, POS_ID, POS_MOTION, SEEDS, _summary, fmt, pos_r2, seeds_of

RUNS = {"T1th100": 100, "T1th10000": 10000}
VALS = {"M motion Δ": ("m_act_pos", lambda c: pos_r2(c) - POS_MOTION, +1),
        "M identity 잔여": ("m_id_pos", lambda c: c["accuracy"] - POS_ID, -1),
        "P identity": ("pt_id", lambda c: c["accuracy"], +1),
        "P_t motion Δ": ("pt_act_pos", lambda c: pos_r2(c) - POS_MOTION, 0)}   # 0 = 보고만


def main():
    T = {r: {k: seeds_of(lambda s, n=n, f=f: (lambda c: None if c is None else f(c))(_summary(LIB, f"t1_{r}_{n}_s{s}", 20)))
             for k, (n, f, _) in VALS.items()} for r in RUNS}
    L = ["# T1 θ 점검 — 처음부터 4 ep 건강 관문 (공학 단계 수치 · 판독기 seed 3)\n", "| 자 | θ 100 | θ 10000 |", "|---|---|---|"]
    L += [f"| {k} | {fmt(T['T1th100'][k])} | {fmt(T['T1th10000'][k])} |" for k in VALS]
    J = {}
    if any(T[r][k] is None for r in RUNS for k in VALS):
        J["채택"] = "미완"
    else:
        m = lambda v: float(np.mean(v))
        gate = {r: m(T[r]["P identity"]) >= 0.9 and m(T[r]["M motion Δ"]) >= 0.160 and m(T[r]["M identity 잔여"]) <= 0.291 for r in RUNS}
        better, worse = [], []
        for k, (_, _, d) in VALS.items():
            if d == 0:
                continue
            a, b = T["T1th100"][k], T["T1th10000"][k]   # a = θ 100
            up, down = min(a) > max(b), max(a) < min(b)   # a 가 비중첩으로 큼 / 작음
            if (up and d > 0) or (down and d < 0):
                better.append(k)
            elif (down and d > 0) or (up and d < 0):
                worse.append(k)
        J["관문 통과"] = gate
        J["θ 100 이 비중첩으로 더 좋음"], J["θ 100 이 비중첩으로 더 나쁨"] = better, worse
        J["채택"] = 100 if (all(gate.values()) and better and not worse) else 10000
    L += ["\n## 채택 (결과 전 고정 규칙)\n"] + [f"- {k}: {v}" for k, v in J.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "theta_t1.md").write_text("\n".join(L) + "\n")
    json.dump({"table": T, "decision": J}, open(OUT / "theta_t1.json", "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
