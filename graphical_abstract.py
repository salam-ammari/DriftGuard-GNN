"""Graphical abstract: 2656 x 1062 px (500:200 ratio) at 300 dpi.
Only elements that appear in the paper; no performance numbers."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

W_IN, H_IN, DPI = 2656 / 300, 1062 / 300, 300
INK, MUTED = "#1f2a36", "#5b6773"
BLUE, LBLUE, ORANGE, GREEN, PURPLE, SAND = ("#0072B2", "#DCE9F4", "#D55E00",
                                            "#2E8B57", "#7a3e9d", "#F3EBC8")

fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 500)
ax.set_ylim(0, 200)
ax.axis("off")
fig.patch.set_facecolor("white")


def rbox(x, y, w, h, fc, ec=INK, lw=1.0, ls="-", r=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(x1, y1, x2, y2, c=INK, lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=lw, color=c, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def text(x, y, s, size=8, c=INK, w="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=c, fontweight=w, ha=ha, va=va, **kw)


# ---------------- title band
ax.add_patch(Rectangle((0, 178), 500, 22, fc=INK, ec="none"))
text(250, 189, "DriftGuard-GNN: drift-adaptive, uncertainty-aware graph learning for "
     "non-technical loss detection", 10.5, "white", "bold")

# ---------------- panel 1: the problem
rbox(6, 14, 128, 158, "#F7F9FB", "#9aa5b1")
text(70, 163, "Latin American distribution grid", 8.2, INK, "bold")
rng = np.random.default_rng(3)
feeders = [(40, 128, "#9fbfd9"), (100, 128, "#9fbfd9"), (40, 78, "#f2c49b"), (100, 78, "#b9d8b0")]
for fx, fy, col in feeders:
    ax.add_patch(Rectangle((fx - 5, fy - 4), 10, 8, fc="#4a5563", ec="none"))
    for k in range(8):
        ang = 2 * np.pi * k / 8 + 0.3
        cx, cy = fx + 17 * np.cos(ang), fy + 13 * np.sin(ang)
        ax.plot([fx, cx], [fy, cy], color="#9aa5b1", lw=0.6, zorder=1)
        fraud = (fx, fy, k) in [(100, 128, 2), (40, 78, 5), (100, 78, 1)]
        ax.add_patch(Circle((cx, cy), 2.6, fc=ORANGE if fraud else col, ec=INK, lw=0.4, zorder=2))
text(70, 51, "feeder communities\n(consumers + transformer)", 6.6, MUTED)
for i, (lab, col) in enumerate((("legitimate collective drift", "#f2c49b"),
                                ("individual fraud", ORANGE))):
    ax.add_patch(Circle((22, 37 - 9 * i), 2.6, fc=col, ec=INK, lw=0.4))
    text(28, 37 - 9 * i, lab, 6.6, INK, ha="left")
text(70, 20, "scarce, biased inspection labels", 6.6, MUTED, style="italic")

arrow(136, 93, 150, 93, lw=2.0)

# ---------------- panel 2: the closed loop
rbox(152, 14, 228, 158, "#FBFCFE", BLUE, lw=1.3)
text(266, 163, "DriftGuard-GNN closed loop", 8.6, BLUE, "bold")
steps = [
    (162, 116, "1  Graph context", "feeder-relative descriptors\n+ load-shape similarity graph"),
    (272, 116, "2  Self-supervised GNN", "two-layer R-SAGE encoder;\nmasked reconstruction"),
    (272, 58, "3  Per-feeder drift monitor", "level + shape channels;\nlocalized alarms"),
    (162, 58, "4  Targeted re-inspection", "labels from flagged feeders;\ncalibrated retraining"),
]
for x, y, h1, h2 in steps:
    rbox(x, y, 100, 38, LBLUE, BLUE, lw=0.9)
    text(x + 50, y + 27, h1, 7.4, INK, "bold")
    text(x + 50, y + 12, h2, 6.4, MUTED)
arrow(262, 135, 272, 135, BLUE)
arrow(322, 116, 322, 96, BLUE)
arrow(272, 77, 262, 77, BLUE)
arrow(212, 96, 212, 116, BLUE)
text(266, 28, "alarms show where to spend re-inspections; random audits reduce label bias",
     6.4, MUTED, style="italic")

arrow(382, 93, 396, 93, lw=2.0)

# ---------------- panel 3: the decision
rbox(398, 14, 96, 158, "#FBFAF4", "#b59b3a", lw=1.2)
text(446, 163, "Utility decision", 8.2, INK, "bold")
rows = [("calibrated probability", BLUE), ("epistemic uncertainty", PURPLE),
        ("recoverable energy", GREEN), ("inspection cost", ORANGE)]
for i, (lab, c) in enumerate(rows):
    y = 140 - 15 * i
    ax.add_patch(Rectangle((406, y - 4), 8, 8, fc=c, ec="none"))
    text(418, y, lab, 6.6, INK, ha="left")
rbox(406, 30, 80, 44, SAND, "#b59b3a", lw=0.9)
text(446, 63, "Budgeted top-k list", 7.4, INK, "bold")
for j in range(3):
    ax.add_patch(Rectangle((416, 52 - 8 * j), 60, 5, fc="#e3d39b" if j else "#c9a93a", ec="none"))
text(446, 22, "abstain when uncertain:\nhuman review, not accusation", 6.2, MUTED)

fig.savefig("../ga/graphical_abstract.png", dpi=DPI)
print("saved")
