# E0 (docs/claim_spine_v2.md §3) 결과 요약 — 팔×시험별 pos R²(dim 0-2 평균), probe seed 평균 ± 95% CI(t, n=3).
# 사용: python paper_artifacts/tables/refinement_floor/agg_e0.py   (agg_round2.py와 같은 규약)
import glob, json, re
from collections import defaultdict
import numpy as np

pos = lambda m: float(np.mean(m["r2_per_dim"][:3]))
T95 = {2: 12.706, 3: 4.303}
cells = defaultdict(list)   # (arm, metric) → [값 per seed]
for d in glob.glob("paper_artifacts/*_action_probing/*_e0_*"):
    test, arm, seed = re.search(r"_e0_([a-z]+)_(\w+?)_s(\d+)$", d).groups()
    for f in glob.glob(d + "/gap30/summary.json"):
        cells[(arm, "calvin")].append(pos(json.load(open(f))))
    for f in glob.glob(d + "/transfer_gap*.json"):
        j = json.load(open(f))
        cells[(arm, "lib_insuite")].append(j["insuite_pos_r2_mean"])
        cells[(arm, "lib_transfer6")].append(j["transfer_pos_r2_mean"])
for (arm, met), v in sorted(cells.items()):
    n = len(v); m = np.mean(v); h = T95[n] * np.std(v, ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"{arm:<15}{met:<14} {m:+.3f} ±{h:.3f}  n={n}")
