# 라운드 2 (refinement_floor_plan §9) 결과 요약 — 팔×시험별 pos R²(dim 0-2 평균), probe seed 평균 ± 95% CI(t, n=3).
# 사용: python paper_artifacts/tables/refinement_floor/agg_round2.py   (aavrl-train env python)
import glob, json, re
from collections import defaultdict
import numpy as np

pos = lambda m: float(np.mean(m["r2_per_dim"][:3]))
T95 = {1: float("nan"), 2: 12.706, 3: 4.303}
cells = defaultdict(list)   # (test, arm, metric) → [값 per seed]
for d in glob.glob("paper_artifacts/*_action_probing/*_refine2_*"):
    test, arm, seed = re.search(r"refine2_([a-z0-9]+)_(\w+?)_s(\d+)$", d).groups()
    for f in glob.glob(d + "/gap*/summary.json"):
        j = json.load(open(f))
        cells[(test, arm, "clean")].append(pos(j))
        for k, lv in (j.get("perturb") or {}).items():
            for l, v in lv.items():
                cells[(test, arm, f"{k}{l}")].append(pos(v))
        for fr, v in (j.get("label_frac") or {}).items():
            cells[(test, arm, f"label{fr}")].append(pos(v))
    for f in glob.glob(d + "/transfer_gap*.json"):
        j = json.load(open(f))
        cells[(test, arm, "insuite")].append(j["insuite_pos_r2_mean"])
        cells[(test, arm, "transfer6")].append(j["transfer_pos_r2_mean"])
for (test, arm, met), v in sorted(cells.items()):
    n = len(v); m = np.mean(v); h = T95[n] * np.std(v, ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"{test:<9}{arm:<11}{met:<12} {m:+.3f} ±{h:.3f}  n={n}")
