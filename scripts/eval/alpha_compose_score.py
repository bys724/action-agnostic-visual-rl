"""alpha_plan §4-g 합성 복원 시험 ② 채점 — r7/{linear,tf1} npz (학습 0).

① ★ 재적합 재현: r7 value_star 예측 vs 저장된 r2(선형) · r6/base(tf1) — 폴드별 MAE 차 ≤ 0.1 ∧ 상관 ≥ 0.99.
② 등록률_s = (V̂(t_inj+s) − V̂(t_inj−1) + (s+1)) / (regrasp_end − t_inj) (alpha_latency 와 같은 식) · s ∈ {2, 5, 10}
   팔: ★(재적합) · P̂ 합성(나이 1 5 10 20) · 보고용 P+M · 앵커 P만(r2/r6 저장값 · 같은 나이).
읽기(결과 전 고정 · §4-g dev 조작화): 주 = 선형 · s = 10 · P̂ 나이 4 개 균등 평균 / ★ (같은 seed) →
   seed 비 최솟값 ≥ 0.80 = 합성 성립 · 최댓값 ≤ 0.50 = 합성 깨짐 · 그 외 = 사이.

python scripts/eval/alpha_compose_score.py  → 표 출력 + paper_artifacts/alpha_value/r7/compose.json
"""
import json
from pathlib import Path
import numpy as np

A = Path("/proj/home/mrg/bys724/action-agnostic-visual-rl/paper_artifacts/alpha_value")
IDX = json.load(open("/proj/external_group/mrg/sfa_export/alpha_v1/index.json"))["episodes"]
EV = {e["src"]: e["event"] for e in IDX if e["source"] == "recover" and e["labeled"]}
AGES, S = (1, 5, 10, 20), (2, 5, 10)
SAVED = {"linear": "r2", "tf1": "r6/base"}


def npz(root, arm, f, s):
    return np.load(A / root / f"value_{arm}_f{f}_s{s}_preds.npz")


def reg_mae(ds, age):
    """5 폴드 npz → (s 별 등록률 에피소드 평균, 전체 MAE)."""
    rows, err = [], []
    for d in ds:
        key = f"t_age{age}" if f"t_age{age}" in d.files else "t_age1"
        t, yh = d[key], d[key.replace("t_", "yhat_")]
        err.append(np.abs(yh - d["y"][t]))
        V = np.full(int(t.max()) + 12, np.nan); V[t] = yh
        for j, src in enumerate(d["ep_src"]):
            if src in EV:
                e = EV[src]; g0 = int(d["ep_start"][j]) + e["t_inj"] - 1
                r = np.array([V[g0 + 1 + s] - V[g0] + (s + 1) for s in S])
                if not np.isnan(r).any():
                    rows.append(r / (e["regrasp_end"] - e["t_inj"]))
    return np.mean(rows, 0), float(np.concatenate(err).mean())


out = {}
for ro, saved in SAVED.items():
    rep = []
    for s in (1, 2, 3):
        for f in range(5):
            a, b = npz(f"r7/{ro}", "star", f, s), npz(saved, "star", f, s)
            y = a["y"][a["t_age1"]]
            rep.append((abs(np.abs(a["yhat_age1"] - y).mean() - np.abs(b["yhat_age1"] - y).mean()),
                        np.corrcoef(a["yhat_age1"], b["yhat_age1"])[0, 1]))
    rep = np.array(rep)
    res = {"repro": {"max_mae_diff": float(rep[:, 0].max()), "min_corr": float(rep[:, 1].min()),
                     "pass": bool((rep[:, 0] <= 0.1).all() and (rep[:, 1] >= 0.99).all())}, "seeds": {}}
    for s in (1, 2, 3):
        F5 = range(5)
        star_r, star_m = reg_mae([npz(f"r7/{ro}", "star", f, s) for f in F5], 1)
        row = {"star": {"reg": star_r.tolist(), "mae": star_m}}
        for arm, root in (("compose", f"r7/{ro}"), ("comp", saved), ("none", saved)):
            ds = [npz(root, arm, f, s) for f in F5]
            per = {a: reg_mae(ds, a) for a in AGES}
            row[arm] = {"reg_by_age": {a: v[0].tolist() for a, v in per.items()},
                        "reg": np.mean([v[0] for v in per.values()], 0).tolist(),
                        "mae": float(np.mean([v[1] for v in per.values()]))}
        row["ratio_s10"] = row["compose"]["reg"][-1] / star_r[-1]
        res["seeds"][s] = row
    r = [res["seeds"][s]["ratio_s10"] for s in (1, 2, 3)]
    res["ratio_min"], res["ratio_max"] = min(r), max(r)
    res["reading"] = ("composition holds" if min(r) >= 0.8 else "composition broken" if max(r) <= 0.5 else "in between")
    out[ro] = res
    print(f"\n[{ro}] ★ 재현: MAE 차 최대 {rep[:, 0].max():.3f} · 상관 최소 {rep[:, 1].min():.4f} · {'PASS' if res['repro']['pass'] else 'FAIL'}")
    for arm in ("star", "compose", "comp", "none"):
        regs = np.array([res["seeds"][s][arm]["reg"] for s in (1, 2, 3)]); maes = [res["seeds"][s][arm]["mae"] for s in (1, 2, 3)]
        print(f"  {arm:8s} 등록률 s=2/5/10 " + " ".join(f"{m:+.3f}[{lo:+.3f},{hi:+.3f}]" for m, lo, hi in
              zip(regs.mean(0), regs.min(0), regs.max(0))) + f" · MAE {np.mean(maes):.2f} [{min(maes):.2f}, {max(maes):.2f}]")
    for a in AGES:
        v = np.array([res["seeds"][s]["compose"]["reg_by_age"][a] for s in (1, 2, 3)]).mean(0)
        print(f"    P̂ 나이 {a:2d} 등록률 s=2/5/10 " + " ".join(f"{x:+.3f}" for x in v))
    print(f"  P̂/★ (s=10) seed 비 {[round(x, 3) for x in r]} → {res['reading']}")
json.dump(out, open(A / "r7" / "compose.json", "w"), indent=1)
