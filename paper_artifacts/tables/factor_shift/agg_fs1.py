"""분리 축 1단계 집계 (docs/factor_shift_plan.md §2).

팔별 Δshift = R²(ABC→D) − R²(D→D), R² = EE 위치 Δ dims 0–2 평균. probe seed {42,1,2}별로
같은 seed끼리 차를 낸 뒤 평균 ± 95% CI (mean ± 4.303·sd/√3, 라운드 1 규약).
"""
import glob
import json
import re
from collections import defaultdict

import numpy as np

ROOT = "paper_artifacts/calvin_action_probing"
ARMS = ["c0_m", "c0_ptptk", "plain_ptptk", "c0_ptm", "raw", "vmae"]

r2 = defaultdict(dict)  # (arm, gap) -> {(scenes, seed): r2}
for f in glob.glob(f"{ROOT}/*_fs1_*/gap*/summary.json"):
    m = re.search(r"_fs1_(\w+?)_(ABC|D)_s(\d+)/gap(\d+)/", f)
    arm, sc, seed, gap = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    r2[(arm, gap)][(sc, seed)] = float(np.mean(json.load(open(f))["r2_per_dim"][:3]))


def ci(x):
    x = np.asarray(x)
    return x.mean(), (4.303 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


for gap in (30, 10, 20, 45):
    print(f"\n== gap {gap}{'  (주 판정)' if gap == 30 else ''} ==")
    print(f"{'arm':<12} {'n':>2} {'ABC→D':>14} {'D→D':>14} {'Δshift':>16}")
    for arm in ARMS:
        d = r2.get((arm, gap), {})
        seeds = sorted({s for (sc, s) in d if ("ABC", s) in d and ("D", s) in d})
        if not seeds:
            continue
        a = [d[("ABC", s)] for s in seeds]
        b = [d[("D", s)] for s in seeds]
        sh = [x - y for x, y in zip(a, b)]
        (ma, ca), (mb, cb), (ms, cs) = ci(a), ci(b), ci(sh)
        print(f"{arm:<12} {len(seeds):>2} {ma:+.3f}±{ca:.3f} {mb:+.3f}±{cb:.3f} {ms:+.3f}±{cs:.3f}")
