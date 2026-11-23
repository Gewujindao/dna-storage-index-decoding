"""Summary figure for the README: index misidentification rate, baseline vs. design.

Run: python make_figures.py   (writes summary.png next to this script)
"""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1f4e79"
GREY = "#9aa5b1"
AMBER = "#d98c1f"
INK = "#222222"
MUTED = "#5f6b76"
AXIS = "#444444"
GRID = "#e8ebee"

FS_TITLE, FS_MAIN, FS_SMALL = 13, 10, 8.5

# Misidentification rate (hard decision), raw values as reported.
# Dataset 1, held-out half, 2,000 reads, M = 10,000
D1_HELDOUT_BASE = "0.2645"  # results/frame_error_rate_dataset1_heldout.txt, line 1 (per-symbol i.i.d. + temperature)
D1_HELDOUT_JOINT_IID = "0.1650"  # see README (joint likelihood, i.i.d. channel)
D1_HELDOUT_DESIGN = "0.1375"  # results/frame_error_rate_dataset1_heldout.txt, line 2 (design)
# Dataset 1, fresh clusters never used before, 29,323 reads
D1_FRESH_BASE = "0.260"  # see README (26.0%)
D1_FRESH_DESIGN = "0.142"  # see README (14.2%)
# Dataset 3, held-out half, 2,000 reads, M = 30,589
D3_HELDOUT_BASE = "0.322"  # see README
D3_HELDOUT_DESIGN = "0.171"  # see README


def pct(s):
    return float((Decimal(s) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FS_SMALL,
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.125)

groups = [
    ("Dataset 1, held-out\n2,000 reads", [("Per-symbol baseline", D1_HELDOUT_BASE, GREY),
                                          ("Joint, i.i.d. channel", D1_HELDOUT_JOINT_IID, AMBER),
                                          ("This design", D1_HELDOUT_DESIGN, NAVY)]),
    ("Dataset 1, fresh reads\n29,323 reads", [("Per-symbol baseline", D1_FRESH_BASE, GREY),
                                              ("This design", D1_FRESH_DESIGN, NAVY)]),
    ("Dataset 3, held-out\n2,000 reads", [("Per-symbol baseline", D3_HELDOUT_BASE, GREY),
                                          ("This design", D3_HELDOUT_DESIGN, NAVY)]),
]

width = 0.26
gap = 0.04
for gi, (label, bars) in enumerate(groups):
    n = len(bars)
    span = n * width + (n - 1) * gap
    x0 = gi - span / 2 + width / 2
    for bi, (name, raw, color) in enumerate(bars):
        x = x0 + bi * (width + gap)
        v = pct(raw)
        ax.bar(x, v, width=width, color=color, edgecolor="none", zorder=2)
        ax.text(x, v + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=FS_MAIN, color=INK)
        if gi == 0:
            ax.text(x, 1.0, name, rotation=90, ha="center", va="bottom", fontsize=FS_SMALL,
                    color="white" if color == NAVY else INK)

ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=FS_MAIN, color=INK)
ax.tick_params(axis="x", length=0, pad=5)
ax.set_xlim(-0.55, len(groups) - 0.45)
ax.set_ylim(0, 36)
ax.set_yticks(range(0, 36, 10))
ax.set_yticklabels([f"{t}%" for t in range(0, 36, 10)])
ax.tick_params(axis="y", labelsize=FS_SMALL, colors=AXIS)
ax.set_ylabel("Misidentification rate", fontsize=FS_MAIN, color=INK)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)

fig.text(0.015, 0.955, "Joint index posterior roughly halves index misidentification on real reads",
         ha="left", va="top", fontsize=FS_TITLE, fontweight="bold", color=INK)
fig.text(0.015, 0.89, "Hard-decision error vs. the per-symbol baseline (with temperature); "
         "each set evaluated once", ha="left", va="top", fontsize=FS_MAIN, color=MUTED)

fig.savefig(Path(__file__).with_name("summary.png"), dpi=200, facecolor="white")
