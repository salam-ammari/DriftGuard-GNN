
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 8.5, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C_DG, C_B1, C_B2, C_B3 = "#1f5f8b", "#c0392b", "#7f8c8d", "#e67e22"
FIGD = "/home/claude/driftguard/Figures"
R = json.load(open("/home/claude/driftguard/results/results.json"))


def agg(path):
    vals = []
    for r in R:
        v = r
        for p in path:
            v = v[p]
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------- fig 1
def fig_architecture():
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.012",
                                    fc=fc, ec="#333333", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=6.6)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color="#333333"))

    box(0.02, 0.86, 0.44, 0.11, "Smart-meter records\n+ feeder topology", "#dde8f0")
    box(0.54, 0.86, 0.44, 0.11, "Sparse, biased\ninspection labels", "#dde8f0")
    box(0.02, 0.68, 0.96, 0.11,
        "Relational contextualization (feeder-median message passing)\n"
        "+ kNN load-shape similarity graph", "#cfe0ee")
    box(0.02, 0.50, 0.46, 0.11,
        "SSL encoder: masked recon.\n+ neighborhood consistency", "#bcd4e6")
    box(0.52, 0.50, 0.46, 0.11,
        "Deep-ensemble head\n+ Platt calibration", "#bcd4e6")
    box(0.02, 0.32, 0.46, 0.11,
        "Two-channel localized\ndrift monitor (per feeder)", "#f5d9c8")
    box(0.52, 0.32, 0.46, 0.11,
        "Guarded selective\nadaptation (flagged only)", "#f5d9c8")
    box(0.02, 0.14, 0.46, 0.11,
        "Epistemic / aleatoric\nuncertainty + abstention", "#dcead2")
    box(0.52, 0.14, 0.46, 0.11,
        "Budget-aware inspection\nranking $S_i$", "#dcead2")
    box(0.25, 0.00, 0.50, 0.09, "Prioritized, confidence-\nqualified inspections",
        "#efe6c8")
    arrow(0.24, 0.86, 0.30, 0.795)
    arrow(0.76, 0.86, 0.70, 0.795)
    arrow(0.25, 0.68, 0.25, 0.615)
    arrow(0.75, 0.68, 0.75, 0.615)
    arrow(0.48, 0.555, 0.52, 0.555)
    arrow(0.25, 0.50, 0.25, 0.435)
    arrow(0.48, 0.375, 0.52, 0.375)
    arrow(0.75, 0.50, 0.75, 0.435)
    arrow(0.25, 0.32, 0.25, 0.255)
    arrow(0.75, 0.32, 0.75, 0.255)
    arrow(0.48, 0.195, 0.52, 0.195)
    arrow(0.62, 0.14, 0.55, 0.095)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 1.0)
    fig.savefig(f"{FIGD}/architecture.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_detection():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    # (a) label-efficiency sweep
    ax = axes[0]
    fr = [1, 5, 10, 20]
    for m, c in [("DriftGuard", C_DG), ("MLP", C_B1), ("RF", C_B2)]:
        mu = [agg(["sweep", s, m])[0] for s in ["0.01", "0.05", "0.1", "0.2"]]
        sd = [agg(["sweep", s, m])[1] for s in ["0.01", "0.05", "0.1", "0.2"]]
        lbl = "DriftGuard-GNN" if m == "DriftGuard" else m
        ax.plot(fr, mu, "-o", ms=3.5, lw=1.3, color=c, label=lbl)
        ax.fill_between(fr, np.array(mu) - sd, np.array(mu) + sd,
                        color=c, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(fr)
    ax.set_xticklabels([f"{f}%" for f in fr])
    ax.set_xlabel("Labeled fraction of consumers")
    ax.set_ylabel("PR-AUC")
    ax.set_title("(a) Label efficiency", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    # (b) main comparison bars at 1% and 5%
    ax = axes[1]
    models = ["LR", "RF", "GB", "MLP", "GraphCtx-sup", "DriftGuard"]
    labels = ["LR", "RF", "GB", "MLP", "GCtx-sup", "DriftGuard"]
    x = np.arange(len(models))
    for off, key, col, lab in [(-0.19, "main_1", C_DG, "1% labels"),
                               (0.19, "main", "#7fb2d6", "5% labels")]:
        mu = [agg([key, m, "prauc"])[0] for m in models]
        sd = [agg([key, m, "prauc"])[1] for m in models]
        ax.bar(x + off, mu, 0.36, yerr=sd, color=col, capsize=2,
               error_kw=dict(lw=0.7), label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_ylabel("PR-AUC")
    ax.set_title("(b) Chronological test, post-drift period", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(f"{FIGD}/detection.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_drift():
    r = R[0]
    curve = r["drift"]["curve"]
    wks = sorted(int(k) for k in curve)
    Z = np.array([[curve[str(w)][str(f)] for w in wks] for f in range(24)])
    det = {int(k): v for k, v in r["drift"]["detect_week"].items()}
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    im = ax.imshow(np.clip(Z, 0, 8), aspect="auto", cmap="YlOrRd",
                   extent=[wks[0] - 1, wks[-1] + 1, 23.5, -0.5])
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("standardized drift score $z_{f,t}$", fontsize=7.5)
    ax.axvline(124, color=C_DG, lw=1.2, ls="--")
    ax.axvline(128, color="#2c7a4b", lw=1.2, ls="--")
    ax.text(124, -1.2, "abrupt onset (R2)", color=C_DG, fontsize=7, ha="center")
    ax.text(139, -1.2, "gradual onset (R3, partial)", color="#2c7a4b",
            fontsize=7, ha="center")
    for f, w in det.items():
        ax.plot(w, f, marker="*", color="k", ms=7, mec="w", mew=0.4)
    for y in (5.5, 11.5, 17.5):
        ax.axhline(y, color="w", lw=0.8)
    ax.set_yticks([2.5, 8.5, 14.5, 20.5])
    ax.set_yticklabels(["Region 0", "Region 1", "Region 2", "Region 3"],
                       fontsize=7.5)
    ax.set_xlabel("week")
    ax.grid(False)
    fig.savefig(f"{FIGD}/drift_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_adapt_unc():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    # (a) adaptation strategies
    ax = axes[0]
    strat = ["static", "finetune", "retrain", "selective"]
    names = ["Static", "Fine-tune", "Retrain", "Selective\n(ours)"]
    x = np.arange(4)
    for off, key, col, lab in [(-0.19, "stable", C_B2, "stable feeders"),
                               (0.19, "drifted", C_DG, "drifted feeders")]:
        mu = [agg(["adapt", s, key])[0] for s in strat]
        sd = [agg(["adapt", s, key])[1] for s in strat]
        ax.bar(x + off, mu, 0.36, yerr=sd, color=col, capsize=2,
               error_kw=dict(lw=0.7), label=lab)
    # per-seed stable outcomes of fine-tuning (forgetting risk)
    ys = [r["adapt"]["finetune"]["stable"] for r in R]
    ax.scatter([1 - 0.19] * len(ys), ys, s=8, color="k", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("PR-AUC")
    ax.set_title("(a) Adaptation after drift", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    # (b) reliability diagram
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for key, c, lab in [("reliab_uncal", C_B1, "uncalibrated"),
                        ("reliab", C_DG, "calibrated")]:
        pts = np.array(R[0]["uncertainty"][key])
        ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3, lw=1.1, color=c, label=lab)
    e0, e1 = agg(["uncertainty", "ece_uncal"]), agg(["uncertainty", "ece_cal"])
    ax.text(0.03, 0.9, f"ECE {e0[0]:.3f} → {e1[0]:.3f}", fontsize=7)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title("(b) Calibration", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    # (c) risk-coverage
    ax = axes[2]
    rc = np.array([r["uncertainty"]["risk_cov"] for r in R]).mean(0)
    rc0 = np.array([r["uncertainty"]["risk_cov_naive"] for r in R]).mean(0)
    ax.plot(rc0[:, 0], rc0[:, 1] * 100, "-s", ms=3, lw=1.1, color=C_B1,
            label="max-prob confidence")
    ax.plot(rc[:, 0], rc[:, 1] * 100, "-o", ms=3, lw=1.1, color=C_DG,
            label="uncertainty-adjusted")
    ax.set_xlabel("coverage (fraction not abstained)")
    ax.set_ylabel("selective risk (% errors)")
    ax.set_title("(c) Abstention", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{FIGD}/adapt_uncertainty.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig_ranking():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ks = ["50", "100", "200", "400"]
    x = np.arange(len(ks))
    for off, key, col, lab in [(-0.19, "prob_only", C_B2, "probability-only"),
                               (0.19, "utility", C_DG, "utility-aware (ours)")]:
        mu = [agg(["ranking", key, k])[0] for k in ks]
        sd = [agg(["ranking", key, k])[1] for k in ks]
        ax.bar(x + off, mu, 0.36, yerr=sd, color=col, capsize=2,
               error_kw=dict(lw=0.7), label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks], fontsize=7.5)
    ax.set_ylabel("recovered energy @ k")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(f"{FIGD}/ranking.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_detection()
    fig_drift()
    fig_adapt_unc()
    fig_ranking()
    print("figures written")
