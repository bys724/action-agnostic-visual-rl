"""J1·J2 사건 창 점프 등록률 (alpha_plan §4 · 10-19 규약: 같은 나이 a, 판정 a = 11..20).
--deploy <root…>: §4-b 비용 대등 대조 (10-20) — 배포 평균(P+M 나이 1..20 균등 vs 주기-2 P 나이 1·2) 등록률 · MAE."""
import json, sys
from pathlib import Path
import numpy as np
DEPLOY = sys.argv[1:2] == ["--deploy"]
roots = [Path(p) for p in (sys.argv[2:] if DEPLOY else sys.argv[1:2])] or \
    [Path("/proj/home/mrg/bys724/action-agnostic-visual-rl/paper_artifacts/alpha_value")]  # 2차(나이 입력) = .../alpha_value/r2
ev = {e["src"]: e["event"] for e in json.load(open("/proj/external_group/mrg/sfa_export/alpha_v1/index.json"))["episodes"]
      if e["source"] == "recover" and e["labeled"]}
R = {}  # (arm, seed) -> {src: rate[20]}
E = {}  # (arm, seed) -> (나이별 절대오차 합, 개수) · 배포 MAE 용
for f in sorted(p for r in roots for p in r.glob("value_*_preds.npz")):
    tag = f.name[6:-10]
    if "smoke" in tag: continue
    arm, fold, seed = tag.rsplit("_", 2)
    d = np.load(f); st, src = d["ep_start"], d["ep_src"]
    acc = R.setdefault((arm, int(seed[1:])), {})
    es, ec = E.setdefault((arm, int(seed[1:])), (np.zeros(20), np.zeros(20)))
    for a in range(1, 21):
        # star 는 나이 무관(t_age1 하나) · 주기-2 P 는 나이 1·2 만 저장 → 없는 나이는 비움
        key = f"t_age{a}" if f"t_age{a}" in d.files else ("t_age1" if arm.endswith("star") else None)
        if key is None: continue
        t, yh = d[key], d[key.replace("t_", "yhat_")]
        es[a - 1] += np.abs(yh - d["y"][t]).sum(); ec[a - 1] += len(t)
        lut = dict(zip(t.tolist(), yh.tolist()))
        for j, s in enumerate(src):
            if s not in ev: continue
            e = ev[s]; g0, g1 = st[j] + e["t_inj"] - 1, st[j] + e["t_inj"] + 10
            if g0 in lut and g1 in lut:
                acc.setdefault(s, np.full(20, np.nan))[a - 1] = (lut[g1] - lut[g0] + 11) / (e["regrasp_end"] - e["t_inj"])
curve = {k: np.nanmean(np.stack(list(v.values())), 0) for k, v in R.items()}  # (arm, seed) -> 나이별 에피소드 평균

if DEPLOY:
    # 배포 평균: 비용이 배포 전체에 걸리므로 나이 균등 평균 (alpha_plan §4-b · 결과 전 고정)
    AGES = {"comp": range(1, 21), "p2": (1, 2), "none": (1, 2), "star": range(1, 21)}  # none = r2 혼합 학습 재채점(참고)
    out = {}
    for arm, ages in AGES.items():
        idx = [a - 1 for a in ages]
        for (b, s), c in sorted(curve.items()):
            if b != arm: continue
            es, ec = E[(b, s)]
            out.setdefault(arm, {})[s] = dict(reg=float(np.mean(c[idx])), mae=float(np.mean(es[idx] / ec[idx])))
    for arm, v in out.items():
        r = [x["reg"] for x in v.values()]; m = [x["mae"] for x in v.values()]
        print(f"{arm:5s} seeds {sorted(v)} | 배포 등록률 {np.mean(r):.4f} [{min(r):.4f}, {max(r):.4f}] | "
              f"배포 MAE {np.mean(m):.2f} [{min(m):.2f}, {max(m):.2f}]")
    if {"comp", "p2"} <= set(out):
        rg = lambda arm, k: [x[k] for x in out[arm].values()]
        print("§4-b 주 (등록률) comp > p2, seed ranges disjoint:", "PASS" if min(rg("comp", "reg")) > max(rg("p2", "reg")) else "FAIL")
        print("§4-b 부 (MAE, 보고) comp < p2, seed ranges disjoint:", min(rg("p2", "mae")) > max(rg("comp", "mae")))
    json.dump(out, open(roots[-1] / "j1_deploy.json", "w"), indent=1)
    sys.exit()

out = {}
for (arm, seed), v in sorted(R.items()):
    M = np.stack(list(v.values()))  # (n_ep, 20)
    out.setdefault(arm, {})[seed] = dict(n_ep=len(v), judge=float(np.nanmean(np.nanmean(M[:, 10:], 0))),
                                         ref_1_10=float(np.nanmean(np.nanmean(M[:, :10], 0))),
                                         n_nan_judge=int(np.isnan(M[:, 10:]).sum()))
for arm, v in out.items():
    j = [x["judge"] for x in v.values()]; r = [x["ref_1_10"] for x in v.values()]
    print(f"{arm:9s} n_ep {[x['n_ep'] for x in v.values()]} nan {[x['n_nan_judge'] for x in v.values()]} | a11-20 {np.mean(j):.4f} [{min(j):.4f}, {max(j):.4f}] | ref a1-10 {np.mean(r):.4f} [{min(r):.4f}, {max(r):.4f}]")
rng = lambda arm: [x["judge"] for x in out[arm].values()]
for pre in ("", "dino_"):  # dino_ = J4 (DINOv2 블록 안 같은 규칙 · 10-20)
    if {pre + "comp", pre + "none", pre + "raw"} <= set(out):
        print(f"{pre}J1 comp > none, seed ranges disjoint:", "PASS" if min(rng(pre + "comp")) > max(rng(pre + "none")) else "FAIL")
        print(f"{pre}J2-event comp > raw, seed ranges disjoint:", "PASS" if min(rng(pre + "comp")) > max(rng(pre + "raw")) else "FAIL")
json.dump(out, open(roots[0] / "j1_event_window.json", "w"), indent=1)
