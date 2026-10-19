"""J1·J2 사건 창 점프 등록률 (alpha_plan §4 · 10-19 규약: 같은 나이 a, 판정 a = 11..20)."""
import json, sys
from pathlib import Path
import numpy as np
root = Path("/proj/home/mrg/bys724/action-agnostic-visual-rl/paper_artifacts/alpha_value")
ev = {e["src"]: e["event"] for e in json.load(open("/proj/external_group/mrg/sfa_export/alpha_v1/index.json"))["episodes"]
      if e["source"] == "recover" and e["labeled"]}
R = {}  # (arm, seed) -> {src: rate[20]}
for f in sorted(root.glob("value_*_preds.npz")):
    tag = f.name[6:-10]
    if "smoke" in tag: continue
    arm, fold, seed = tag.rsplit("_", 2)
    d = np.load(f); st, src = d["ep_start"], d["ep_src"]
    acc = R.setdefault((arm, int(seed[1:])), {})
    for a in range(1, 21):
        key = f"t_age{a}" if f"t_age{a}" in d.files else "t_age1"
        lut = dict(zip(d[key].tolist(), d[key.replace("t_", "yhat_")].tolist()))
        for j, s in enumerate(src):
            if s not in ev: continue
            e = ev[s]; g0, g1 = st[j] + e["t_inj"] - 1, st[j] + e["t_inj"] + 10
            if g0 in lut and g1 in lut:
                acc.setdefault(s, np.full(20, np.nan))[a - 1] = (lut[g1] - lut[g0] + 11) / (e["regrasp_end"] - e["t_inj"])
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
print("J1 comp > none, seed ranges disjoint:", "PASS" if min(rng("comp")) > max(rng("none")) else "FAIL")
print("J2-event comp > raw, seed ranges disjoint:", "PASS" if min(rng("comp")) > max(rng("raw")) else "FAIL")
json.dump(out, open(root / "j1_event_window.json", "w"), indent=1)
