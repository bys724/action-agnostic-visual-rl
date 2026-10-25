"""지연 곡선 그림 (alpha_plan §4-d) — <dir>/latency.json → <dir>/latency_curve.png · 인자 없으면 r5(선형) · `r6` = §4-f 비선형."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).parent.parent / (sys.argv[1] if len(sys.argv) > 1 else "r5")
R = json.load(open(D / "latency.json"))
S = np.arange(1, 11)
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
ROLE = {"pm": "#2a78d6", "per": "#eb6834", "raw": "#1baf7a", "star": MUTED}   # reference palette slots 1-3
PANELS = [("Current M (4.58 GF) vs P every 2 steps", "P+M 4.58 (period 20)", "period-2 P", "P+M (5.04 GF/step)", "P every 2 steps (4.60)"),
          ("Small M (0.43 GF) vs P every 9 steps", "P+M 0.43 (period 20)", "period-9 P", "P+M (0.89 GF/step)", "P every 9 steps (1.02)")]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURF)
for ax, (title, pm, per, pm_lab, per_lab) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    ends = []
    for key, role, lab, ls in (("star (P every step)", "star", "P every step (9.20)", "--"), (pm, "pm", pm_lab, "-"),
                               (per, "per", per_lab, "-"), ("anchor P + raw dL", "raw", "anchor P + raw ΔL", "-")):
        r = R[key]
        ax.fill_between(S, r["reg_lo"], r["reg_hi"], color=ROLE[role], alpha=0.18, lw=0)
        ax.plot(S, r["reg_mean"], ls, color=ROLE[role], lw=2, marker="o", ms=4, label=lab)
        ends.append((r["reg_mean"][-1], lab))
    ys = []   # 끝 라벨 겹침 방지: 아래에서부터 최소 간격 0.03
    for y, lab in sorted(ends):
        ys.append(max(y, ys[-1] + 0.03) if ys else y)
        ax.annotate(lab, (10, y), xytext=(10.25, ys[-1]), textcoords="data", va="center", fontsize=8.5, color=INK)
    ax.axvline(R["s_star"], color=MUTED, lw=1, ls=":")
    ax.text(R["s_star"] + 0.1, -0.135, f"s* = {R['s_star']}", color=MUTED, fontsize=8.5)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_xticks(S); ax.set_xlim(0.7, 13.6); ax.set_xlabel("steps after the injected event (s)")
axes[0].set_ylabel("event registration_s (deployment average)")
axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), frameon=False, fontsize=8)
fig.suptitle(f"alpha §4-d — how fast each arm registers the event ({'nonlinear tf1' if D.name == 'r6' else 'linear'} readout · seed 3 × 5 folds · band = seed range)", fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(D / "latency_curve.png", dpi=150, facecolor=SURF)
