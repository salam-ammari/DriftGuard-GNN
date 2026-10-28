"""Regenerate every figure of the revised manuscript from results/."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
RES, FIG = "../results", "../Figures"
SEEDS = [11, 23, 37, 51, 73]
# colour-blind-safe categorical palette (Okabe-Ito based)
COL = {"DriftGuard-GNN": "#0072B2", "GraphCtx-MLP": "#56B4E9", "LR": "#E69F00",
       "MLP": "#999999", "RF": "#CC79A7", "R-SAGE-sup": "#009E73"}
STRAT = [("static", "Static", "#999999"), ("finetune", "Global fine-tune", "#E69F00"),
         ("local", "Local heads", "#CC79A7"), ("scheduled", "Scheduled retrain", "#009E73"),
         ("uniform", "Untargeted retrain", "#56B4E9"),
         ("driftguard", "DriftGuard-GNN", "#0072B2")]


def load(scen, var="topology"):
    return [json.load(open(f"{RES}/sim_{scen}_{var}_{s}.json")) for s in SEEDS]


# ------------------------------------------------------------- Fig. 1
def fig_architecture():
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    ax.axis("off")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.01, 1.0)

    def box(x, y, w, h, txt, fc, ls="-", fs=6.0, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                    fc=fc, ec="#333333", lw=0.7, ls=ls))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arr(x1, y1, x2, y2, c="#333333", ls="-", txt=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color=c, ls=ls))
        if txt:
            ax.text((x1 + x2) / 2 + 0.012, (y1 + y2) / 2, txt, fontsize=5.4,
                    color=c, va="center", ha="left")

    inp, enc, mon, dec = "#E8EEF4", "#CFE0EE", "#F7E1CF", "#DCEBD3"
    box(0.00, 0.885, 0.30, 0.10, "Smart-meter\nconsumption", inp)
    box(0.34, 0.885, 0.30, 0.10, "Feeder topology\n(BDGD; optional)", inp, ls="--")
    box(0.68, 0.885, 0.32, 0.10, "Inspection labels\n+ random audits", inp)
    box(0.00, 0.735, 0.64, 0.105, "Context descriptors $x_v$: 12 window\nstatistics, also relative to the feeder\n"
        "(or similarity-neighbour) median", enc)
    box(0.00, 0.555, 0.60, 0.135, "Two-layer R-SAGE GNN encoder\n(physical + similarity relations);\n"
        "SSL: masked reconstruction\n+ neighbourhood consistency", enc)
    box(0.66, 0.555, 0.34, 0.135, "$r_v=[x_v, z_v, e_v, \\delta_v]$\nDeep ensemble +\ncross-fitted Platt\n"
        "$\\rightarrow \\hat p_v,\\ u_v$", enc)
    box(0.00, 0.37, 0.46, 0.13, "Per-feeder drift monitor:\nlevel (median YoY) and\nshape (embedding) channels", mon)
    box(0.54, 0.37, 0.46, 0.13, "Alarm: targeted\nre-inspection of flagged\nfeeders, then retraining", mon)
    box(0.00, 0.185, 0.46, 0.13, "Uncertainty-adjusted\nconfidence $c_v$;\nabstain below coverage", dec)
    box(0.54, 0.185, 0.46, 0.13, "Inspection priority\n$S_v=\\hat p_v(1-0.3u_v)$\n$\\times(\\hat E_v+1)^{0.7}/C_v$", dec)
    box(0.20, 0.015, 0.60, 0.11, "Budgeted, confidence-qualified\ninspection list (top-$k$)", "#F3EBC8", bold=True)
    arr(0.15, 0.885, 0.15, 0.84)
    arr(0.49, 0.885, 0.49, 0.84, ls="--")
    arr(0.84, 0.885, 0.84, 0.69)
    arr(0.30, 0.735, 0.30, 0.69)
    arr(0.60, 0.62, 0.66, 0.62)
    arr(0.23, 0.555, 0.23, 0.50)
    arr(0.46, 0.435, 0.54, 0.435)
    arr(0.77, 0.50, 0.77, 0.555, c="#8a5a2b", txt="retrained head")
    arr(0.23, 0.555, 0.23, 0.50)
    arr(0.80, 0.37, 0.80, 0.315)
    arr(0.23, 0.37, 0.23, 0.315)
    arr(0.46, 0.25, 0.54, 0.25)
    arr(0.64, 0.185, 0.60, 0.125)
    fig.savefig(f"{FIG}/architecture.png")
    plt.close(fig)


# ------------------------------------------------------------- Fig. 2
def fig_detection_drift():
    S = json.load(open(f"{RES}/summary.json"))
    comb = load("combined")
    fig = plt.figure(figsize=(7.16, 2.75))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1.15, 0.05, 0.22, 1], wspace=0.42,
                          left=0.07, right=0.99, top=0.88, bottom=0.30)
    # (a) label efficiency
    ax = fig.add_subplot(gs[0])
    fr = ["0.01", "0.05", "0.1", "0.2"]
    x = [1, 5, 10, 20]
    for m in ("DriftGuard-GNN", "GraphCtx-MLP", "LR", "MLP"):
        mu = np.array([S["detection"][f][m]["prauc"][0] for f in fr])
        sd = np.array([S["detection"][f][m]["prauc"][1] for f in fr])
        ax.plot(x, mu, "-o", ms=3, lw=1.3, color=COL[m], label=m)
        ax.fill_between(x, mu - sd, mu + sd, color=COL[m], alpha=0.10, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}%" for v in x])
    ax.minorticks_off()
    ax.set_xlabel("labeled fraction")
    ax.set_ylabel("PR-AUC")
    ax.set_title("(a) Label efficiency", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
              handlelength=1.4, columnspacing=0.8, fontsize=6.4)
    # (b) drift heat map, seed 11
    ax = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])
    r = comb[0]
    wk = sorted(int(k) for k in r["ztrace"])
    Z = np.array([[max(r["ztrace"][str(w)][str(f)]) for w in wk] for f in range(24)])
    im = ax.imshow(np.clip(Z, 0, 20), aspect="auto", cmap="YlOrRd",
                   extent=[wk[0] - 1, wk[-1] + 1, 23.5, -0.5], vmin=0, vmax=20)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("drift score $z_{f,t}$ (clipped)", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.3)
    for w, c in ((124, "#0072B2"), (128, "#009E73"), (136, "#7a3e9d")):
        ax.axvline(w, color=c, lw=1.1, ls="--")
    for f, w in r["drift"]["alarm_week"].items():
        ax.plot(w, int(f), marker="*", color="k", ms=6, mec="w", mew=0.4, ls="none")
    for yv in (5.5, 11.5, 17.5):
        ax.axhline(yv, color="w", lw=0.9)
    ax.set_yticks([2.5, 8.5, 14.5, 20.5])
    ax.set_yticklabels(["R0", "R1", "R2", "R3"], fontsize=6.8)
    ax.set_xlabel("week")
    ax.grid(False)
    ax.set_title("(b) Drift score, seed 11", loc="left")
    h = [Line2D([], [], color=c, ls="--", lw=1.1, label=l) for c, l in
         (("#0072B2", "covariate onset (R2)"), ("#009E73", "gradual onset (R3: f18-20)"),
          ("#7a3e9d", "concept onset (R1)"))] + \
        [Line2D([], [], marker="*", color="k", ls="none", ms=6, label="alarm")]
    ax.legend(handles=h, frameon=False, loc="upper center", bbox_to_anchor=(0.55, -0.27),
              ncol=2, fontsize=6.0, handlelength=1.6, columnspacing=0.8)
    # (c) prequential curve on drifted feeders
    ax = fig.add_subplot(gs[4])
    wks = comb[0]["curve_weeks"]
    for s, lab, c in STRAT:
        if s == "local":
            continue
        cur = np.array([r["adapt"][s]["curve_drifted"] for r in comb], dtype=float)
        ax.plot(wks, np.nanmean(cur, 0), lw=1.5 if s == "driftguard" else 1.0,
                color=c, label=lab)
    for w, c in ((124, "#0072B2"), (128, "#009E73"), (136, "#7a3e9d")):
        ax.axvline(w, color=c, lw=0.8, ls=":")
    ax.set_xlabel("week")
    ax.set_ylabel("PR-AUC, drifted feeders")
    ax.set_title("(c) Online, drifted feeders", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.27), ncol=2,
              fontsize=6.0, handlelength=1.4, columnspacing=0.8)
    fig.savefig(f"{FIG}/detection_drift.png")
    plt.close(fig)


# ------------------------------------------------------------- Fig. 3
def fig_adapt_calib_rank():
    S = json.load(open(f"{RES}/summary.json"))
    comb = load("combined")
    fig = plt.figure(figsize=(7.16, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.42,
                          left=0.07, right=0.99, top=0.89, bottom=0.30)
    # (a) per-seed outcomes for ALL strategies
    ax = fig.add_subplot(gs[0])
    rng = np.random.default_rng(0)
    for j, (s, lab, c) in enumerate(STRAT):
        for k, (grp, mk) in enumerate((("stable", "o"), ("drifted", "s"))):
            v = np.array(S["adapt"][s]["per_seed_" + grp], dtype=float)
            xx = j + (-0.17 if k == 0 else 0.17) + rng.uniform(-0.05, 0.05, v.size)
            ax.scatter(xx, v, s=11, marker=mk, facecolor=c if k else "white",
                       edgecolor=c, lw=0.8, zorder=3)
            ax.hlines(np.mean(v), j + (-0.30 if k == 0 else 0.04),
                      j + (-0.04 if k == 0 else 0.30), color="k", lw=1.0, zorder=4)
    ax.set_xticks(range(len(STRAT)))
    ax.set_xticklabels(["Static", "Global fine-tune", "Local heads", "Scheduled retrain",
                        "Untargeted retrain", "DriftGuard-GNN"], fontsize=6.0,
                       rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylabel("PR-AUC (weeks 138-156)")
    ax.set_title("(a) Adaptation strategies, all seeds", loc="left")
    h = [Line2D([], [], marker="o", ls="none", mfc="white", mec="k", ms=4, label="stable feeders"),
         Line2D([], [], marker="s", ls="none", mfc="k", mec="k", ms=4, label="drifted feeders"),
         Line2D([], [], color="k", lw=1.0, label="mean of 5 seeds")]
    ax.legend(handles=h, frameon=False, loc="upper center", ncol=3, fontsize=6.3,
              bbox_to_anchor=(0.5, -0.33), handlelength=1.2, columnspacing=0.8)
    # (b) reliability
    ax = fig.add_subplot(gs[1])
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    for key, c, lab, mk in (("reliab_uncal", "#E69F00", "uncalibrated", "s"),
                            ("reliab", "#0072B2", "cross-fitted Platt", "o")):
        pts = np.array(comb[0]["calib"][key])
        ax.plot(pts[:, 0], pts[:, 1], "-" + mk, ms=3, lw=1.1, color=c, label=lab)
    e0, e1 = S["calib"]["ece_uncal"][0], S["calib"]["ece_cal"][0]
    ax.text(0.04, 0.96, f"mean ECE\n{e0:.3f} $\\rightarrow$ {e1:.3f}", fontsize=6.4,
            transform=ax.transAxes, va="top", ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title("(b) Calibration, seed 11", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.3,
              ncol=1)
    # (c) recovered energy
    ax = fig.add_subplot(gs[2])
    ks = ["50", "100", "200", "400"]
    xx = np.arange(len(ks))
    for off, key, c, lab in ((-0.19, "prob", "#999999", "probability only"),
                             (0.19, "util", "#0072B2", "priority $S_v$")):
        mu = [S["ranking"][key][k][0] for k in ks]
        sd = [S["ranking"][key][k][1] for k in ks]
        ax.bar(xx + off, mu, 0.36, yerr=sd, color=c, capsize=2,
               error_kw=dict(lw=0.6), label=lab)
    ax.set_xticks(xx)
    ax.set_xticklabels(ks)
    ax.set_xlabel("inspection budget $k$")
    ax.set_ylabel("stolen energy recovered")
    ax.set_title("(c) Budgeted inspection", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.3,
              ncol=1)
    fig.savefig(f"{FIG}/adapt_calib_rank.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    fig_architecture()
    if "arch" not in sys.argv:
        fig_detection_drift()
        fig_adapt_calib_rank()
    print("figures written")
