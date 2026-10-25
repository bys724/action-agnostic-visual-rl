"""alpha_plan §4-d 실시간 자 (10-25 Vault (3)) — 기존 판독 npz 재채점 · 학습 0.

지연 곡선: 등록률_s = (V̂(t_inj+s) − V̂(t_inj−1) + (s+1)) / (regrasp_end − t_inj), s = 1..10
  (s = 10 이 §4 J1 의 등록률과 같은 식). 팔별 배포 평균 = 배포 나이 집합 균등(alpha_j1 과 같은 규약).
감지 시간: 문턱 θ_(a,s) = 같은 팔·나이·s 의 무교란 창(시연 에피소드) 상승분 V̂(t+s) − V̂(t−1) + (s+1) 의 상위 5 % 분위.
  사건의 상승분이 θ 를 처음 넘는 s = s_det · 넘지 못하면 미감지(∞). 배포 나이 집합의 (사건 × 나이) 표본을 모아 중앙값 · 미감지 비율.
읽기(결과 전 고정 · §4-d): s* = ★ 등록률_s 가 ★ 등록률_10 의 절반에 처음 도달하는 s.

python scripts/eval/alpha_latency.py [tf1]  → 표 출력 + paper_artifacts/alpha_value/r5/latency.json
  (tf1 = §4-f 비선형 판독기 npz · r6/{base,p2,p9,m192x2} → r6/latency.json)
"""
import json
import sys
from pathlib import Path
import numpy as np

A = Path("/proj/home/mrg/bys724/action-agnostic-visual-rl/paper_artifacts/alpha_value")
IDX = json.load(open("/proj/external_group/mrg/sfa_export/alpha_v1/index.json"))["episodes"]
EV = {e["src"]: e["event"] for e in IDX if e["source"] == "recover" and e["labeled"]}
DEMO = {e["src"] for e in IDX if e["source"] == "demo"}
S = range(1, 11)
# 팔 = (출력 폴더, 파일 팔 이름, 배포 나이, 스텝당 GF)
ARMS = {
    "star (P every step)":        ("r2", "star", [1], 9.20),
    "P+M 4.58 (period 20)":       ("r2", "comp", range(1, 21), 5.04),
    "P+M 0.43 (period 20)":       ("r4/m192x2", "comp", range(1, 21), 0.89),
    "period-2 P":                 ("r3", "p2", (1, 2), 4.60),
    "period-9 P":                 ("r4/competitor_p9", "p9", range(1, 10), 1.02),
    "anchor P + raw dL":          ("r2", "raw", range(1, 21), None),
    "(ref) anchor P only 11..20": ("r2", "none", range(11, 21), None),
}


TF1 = sys.argv[1:2] == ["tf1"]
OUT = "r6" if TF1 else "r5"
if TF1:  # §4-f: 같은 팔 · 같은 규약 · 판독기만 다른 npz
    ROOT6 = {"r2": "r6/base", "r3": "r6/p2", "r4/competitor_p9": "r6/p9", "r4/m192x2": "r6/m192x2"}
    ARMS = {k: (ROOT6[v[0]],) + v[1:] for k, v in ARMS.items()}


def load(root, arm):
    """seed -> list of npz (5 폴드)."""
    out = {}
    for f in sorted((A / root).glob(f"value_{arm}_f*_s*_preds.npz")):
        if "smoke" in f.name:
            continue
        out.setdefault(int(f.name.split("_s")[-1].split("_")[0]), []).append(np.load(f))
    return out


def per_seed(npzs, ages):
    """한 seed 의 5 폴드 → (등록률_s 나이 평균 [10], 감지 표본 s_det 리스트)."""
    reg = []        # 나이별 (s 별 에피소드 평균)
    det = []
    for a in ages:
        rows, clean = [], {s: [] for s in S}
        ev_rise = []
        for d in npzs:
            key = f"t_age{a}" if f"t_age{a}" in d.files else "t_age1"   # star = 나이 무관
            t, yh = d[key], d[key.replace("t_", "yhat_")]
            V = np.full(int(t.max()) + 12, np.nan); V[t] = yh               # 전역 프레임 → V̂ (평가 안 된 프레임 = nan)
            for j, src in enumerate(d["ep_src"]):
                st = int(d["ep_start"][j])
                if src in EV:
                    e = EV[src]; g0 = st + e["t_inj"] - 1                    # 창 시작 = t_inj − 1
                    rise = np.array([V[g0 + 1 + s] - V[g0] + (s + 1) for s in S])  # V̂(t_inj+s) − V̂(t_inj−1) + (s+1)
                    if not np.isnan(rise).any():
                        rows.append(rise / (e["regrasp_end"] - e["t_inj"]))
                        ev_rise.append(rise)
            # 무교란 창: 시연 에피소드의 모든 t (t−1 · t+1..t+10 이 평가 프레임 · 에피소드 경계 안)
            n_ep = np.diff(np.append(d["ep_start"], len(d["y"])))
            for j, src in enumerate(d["ep_src"]):
                if src not in DEMO:
                    continue
                st, T = int(d["ep_start"][j]), int(n_ep[j])
                tt = np.arange(st + 1, st + T - 10)
                for s in S:
                    r = V[tt + s] - V[tt - 1] + (s + 1)
                    clean[s].append(r[~np.isnan(r)])
        clean = {s: np.concatenate(v) for s, v in clean.items()}
        reg.append(np.mean(rows, 0))
        th = np.array([np.quantile(clean[s], 0.95) for s in S])
        for r in ev_rise:
            hit = np.nonzero(r > th)[0]
            det.append(int(hit[0]) + 1 if len(hit) else np.inf)
    return np.mean(reg, 0), det


res = {}
for name, (root, arm, ages, gf) in ARMS.items():
    seeds = load(root, arm)
    assert len(seeds) == 3 and all(len(v) == 5 for v in seeds.values()), (name, {k: len(v) for k, v in seeds.items()})
    curves, dets = {}, {}
    for sd, npzs in sorted(seeds.items()):
        curves[sd], dets[sd] = per_seed(npzs, list(ages))
    C = np.stack(list(curves.values()))
    med = [float(np.median(v)) for v in dets.values()]
    und = [float(np.mean(np.isinf(v))) for v in dets.values()]
    res[name] = dict(gf=gf, reg_mean=C.mean(0).tolist(), reg_lo=C.min(0).tolist(), reg_hi=C.max(0).tolist(),
                     sdet_median=med, undetected=und, n_samples=[len(v) for v in dets.values()])

star = np.array(res["star (P every step)"]["reg_mean"])
sstar = next((s for s, v in zip(S, star) if v >= 0.5 * star[-1]), None)
res["s_star"] = sstar
print("등록률_s (seed 평균 [범위]) · s = 1..10")
for name in ARMS:
    r = res[name]
    print(f"{name:28s} " + " ".join(f"{m:.3f}" for m in r["reg_mean"]))
    print(f"{'':28s} lo " + " ".join(f"{m:.3f}" for m in r["reg_lo"]) + " | hi " + " ".join(f"{m:.3f}" for m in r["reg_hi"]))
print("\n감지 시간 (seed 별 중앙값 s_det · 미감지 비율)")
for name in ARMS:
    r = res[name]
    print(f"{name:28s} median {r['sdet_median']} | undetected {[round(u, 3) for u in r['undetected']]} | n {r['n_samples']}")
print(f"\ns* = {sstar} (★ 등록률_10 = {star[-1]:.3f}, 절반 = {0.5 * star[-1]:.3f})")
if sstar is not None and sstar <= 5:
    i = sstar - 1
    for a, b in (("P+M 4.58 (period 20)", "period-2 P"), ("P+M 0.43 (period 20)", "period-9 P")):
        lo_a, hi_b = res[a]["reg_lo"][i], res[b]["reg_hi"][i]
        ge = lo_a > hi_b
        print(f"  s*={sstar}: {a} {res[a]['reg_mean'][i]:.3f} [{lo_a:.3f}, {res[a]['reg_hi'][i]:.3f}] vs {b} "
              f"{res[b]['reg_mean'][i]:.3f} [{res[b]['reg_lo'][i]:.3f}, {hi_b:.3f}] → P+M ≥ (disjoint): {ge}")
(A / OUT).mkdir(exist_ok=True)
json.dump(res, open(A / OUT / "latency.json", "w"), indent=1)
