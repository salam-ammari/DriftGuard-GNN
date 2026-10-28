"""Video abstract: 1920x1080 slides + neural TTS narration (Piper) + captions.

usage: python make_video.py "<MANUSCRIPT ID>" <piper_model.onnx>
Writes ../video/DriftGuard-GNN_video_abstract.mp4, the .srt captions, and
the narration script.  The first frame shows the title and the manuscript ID.
"""

import json
import os
import subprocess
import sys
import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyBboxPatch

MSID = sys.argv[1] if len(sys.argv) > 1 else "[Manuscript ID]"
MODEL = sys.argv[2]
OUT = "../video"
TMP = os.path.join(OUT, "build")
os.makedirs(TMP, exist_ok=True)
S = json.load(open("../results/summary.json"))
TITLE = ("DriftGuard-GNN: Self-Supervised, Drift-Adaptive, and Uncertainty-Aware\n"
         "Graph Learning for Non-Technical Loss Detection in Latin American Smart Grids")
INK, MUTED, BLUE, BAND = "#1f2a36", "#5b6773", "#0072B2", "#1f2a36"

d1 = S["detection"]["0.01"]
d5 = S["detection"]["0.05"]
sc = S["scenarios"]
ad = S["adapt"]
sg = S["sgcc"]


def f3(x):
    return f"{x:.3f}".replace("0.", "0 point ", 1)


def slide(fn, title, draw, footer=True):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920); ax.set_ylim(0, 1080); ax.axis("off")
    if title:
        ax.add_patch(Rectangle((0, 960), 1920, 120, fc=BAND, ec="none"))
        ax.text(70, 1020, title, fontsize=34, color="white", fontweight="bold", va="center")
    if footer:
        ax.text(70, 32, "DriftGuard-GNN  |  IEEE Latin America Transactions  |  " + MSID,
                fontsize=15, color=MUTED, va="center")
    draw(fig, ax)
    fig.savefig(fn, dpi=100)
    plt.close(fig)


def bullets(ax, items, x=90, y=860, dy=105, size=27):
    for i, t in enumerate(items):
        ax.text(x, y - i * dy, "•", fontsize=size, color=BLUE, va="top")
        ax.text(x + 45, y - i * dy, t, fontsize=size, color=INK, va="top", linespacing=1.3)


def image(fig, path, rect):
    a = fig.add_axes(rect)
    a.imshow(mpimg.imread(path))
    a.axis("off")


slides = []

# 1 ------------------------------------------------------------- title card
def s1(fig, ax):
    ax.add_patch(Rectangle((0, 0), 1920, 1080, fc="#F4F7FA", ec="none"))
    ax.add_patch(Rectangle((0, 760), 1920, 320, fc=BAND, ec="none"))
    ax.text(960, 920, TITLE, fontsize=31, color="white", fontweight="bold",
            ha="center", va="center", linespacing=1.35)
    ax.text(960, 640, "Manuscript ID: " + MSID, fontsize=40, color=BLUE,
            fontweight="bold", ha="center")
    ax.text(960, 520, "Salam Al-E'mari, Yousef Sanjalawe, Muder Almiani", fontsize=30,
            color=INK, ha="center")
    ax.text(960, 455, "University of Petra (Jordan)  ·  University of Jordan  ·  "
            "Gulf University for Science and Technology (Kuwait)", fontsize=21,
            color=MUTED, ha="center")
    ax.text(960, 330, "Video abstract  ·  IEEE Latin America Transactions", fontsize=26,
            color=INK, ha="center", style="italic")
slides.append((("" if MSID.startswith("[") else "Video abstract for manuscript " +
                MSID.replace("-", " ") + ". ") +
               "Submitted to I triple E Latin America Transactions. "
               "Drift Guard G N N: self-supervised, drift-adaptive, and uncertainty-aware "
               "graph learning for non-technical loss detection in Latin American smart grids.",
               None, s1, False))

# 2 ------------------------------------------------------------- problem
def s2(fig, ax):
    bullets(ax, [
        "Non-technical losses: meter tampering, illegal connections, theft",
        "In Brazil, 14.6% of all low-voltage losses in 2022 (ANEEL)",
        "Inspection labels are scarce and biased toward selected consumers",
        "Consumption drifts with tariffs, seasons, and the economy",
        "Offenders adapt to enforcement",
        "Utilities need a short, reliable inspection list, not just a score"])
slides.append(("Non-technical losses, such as meter tampering, illegal connections and theft, "
               "are a major burden for Latin American distributors. In Brazil they were "
               "fourteen point six percent of all low-voltage losses in twenty twenty-two. "
               "Detectors face four conditions together: few and biased inspection labels, "
               "consumption that drifts with tariffs and the economy, offenders who adapt, "
               "and the need for a short, reliable inspection list.",
               "The problem", s2, True))

# 3 ------------------------------------------------------------- idea
def s3(fig, ax):
    rng = np.random.default_rng(4)
    t = np.arange(104)
    a = fig.add_axes([0.07, 0.17, 0.55, 0.66])
    for i in range(14):
        base = 40 * rng.lognormal(0, 0.15) * (1 + 0.1 * np.cos(2 * np.pi * (t - 6) / 52))
        y = base * rng.normal(1, 0.05, t.size)
        y[60:] *= 0.86
        a.plot(t, y, color="#9fbfd9", lw=1.2)
    y = 42 * (1 + 0.1 * np.cos(2 * np.pi * (t - 6) / 52)) * rng.normal(1, 0.05, t.size)
    y[60:] *= 0.86
    y[40:] *= 0.55
    a.plot(t, y, color="#D55E00", lw=2.6)
    a.axvline(60, color=INK, ls="--", lw=1.2)
    a.text(61, 63, "tariff change:\nwhole feeder drops", fontsize=17, color=INK, va="top")
    a.text(42, 11, "one consumer drops alone: suspicious", fontsize=17, color="#D55E00")
    a.set_ylim(7, 68)
    a.set_xlabel("week", fontsize=18); a.set_ylabel("kWh per week", fontsize=18)
    a.tick_params(labelsize=14)
    a.spines[["top", "right"]].set_visible(False)
    ax.text(1250, 800, "Key observation", fontsize=30, color=BLUE, fontweight="bold")
    ax.text(1250, 720, "Legitimate drift is mostly\ncollective; fraud is individual.\n\n"
            "Describe each consumer relative\nto its feeder community, then\nencode it with a graph network.",
            fontsize=25, color=INK, va="top", linespacing=1.4)
slides.append(("The key observation is that legitimate drift tends to be collective, while fraud "
               "is individual. When a tariff changes, the whole feeder drops together. When one "
               "consumer drops alone, that is suspicious. So Drift Guard describes every "
               "consumer relative to its feeder community.", "Key idea", s3, True))

# 4 ------------------------------------------------------------- framework
def s4(fig, ax):
    image(fig, "../Figures/architecture.png", [0.03, 0.08, 0.46, 0.8])
    bullets(ax, ["Community-relative descriptors",
                 "Two-layer relational GNN encoder\n(R-SAGE), self-supervised",
                 "Calibrated deep ensemble:\nprobability + uncertainty",
                 "Per-feeder drift monitor",
                 "Alarm → targeted re-inspection\n→ retraining",
                 "Inspection priority under a budget"], x=1000, y=860, dy=125, size=24)
slides.append(("The framework is a closed loop. A two-layer relational graph neural network, "
               "pre-trained without labels by masked reconstruction, encodes each consumer. "
               "A calibrated deep ensemble gives a fraud probability and an uncertainty. "
               "A drift monitor watches every feeder; an alarm triggers re-inspection of that "
               "feeder and retraining. Finally, an inspection priority combines probability, "
               "uncertainty, recoverable energy and cost under a fixed budget. Feeder topology "
               "is optional; Brazilian distributors already register it in the regulator's "
               "geographic database.", "DriftGuard-GNN", s4, True))

# 5 ------------------------------------------------------------- detection
def s5(fig, ax):
    image(fig, "../Figures/detection_drift.png", [0.03, 0.30, 0.94, 0.58])
    ax.text(90, 230, f"1% labels: PR-AUC {d1['DriftGuard-GNN']['prauc'][0]:.3f} "
            f"vs. best baseline {max(v['prauc'][0] for k, v in d1.items() if k != 'DriftGuard-GNN'):.3f}",
            fontsize=26, color=INK)
    cov, gr = sc["combined"]["covariate"]["delay"][0], sc["combined"]["gradual"]["delay"][0]
    ax.text(90, 160, f"Every abruptly and gradually drifting feeder flagged: "
            f"{cov:.1f} and {gr:.1f} weeks after onset", fontsize=26, color=INK)
    ax.text(90, 90, "Concept drift that leaves consumption unchanged is not visible to "
            "unsupervised monitoring", fontsize=24, color="#D55E00")
best_b = max(v["prauc"][0] for k, v in d1.items() if k != "DriftGuard-GNN")
slides.append((f"On a three-year simulated network anchored to Brazilian consumption statistics, "
               f"with only one percent of consumers labelled, Drift Guard reached a precision "
               f"recall A U C of {f3(d1['DriftGuard-GNN']['prauc'][0])}, against "
               f"{f3(best_b)} for the best of six baselines. The monitor flagged every "
               f"abruptly and gradually drifting feeder. Concept drift that leaves consumption "
               f"unchanged was not detected, which we report as a limitation.",
               "Simulation results", s5, True))

# 6 ------------------------------------------------------------- real data
def s6(fig, ax):
    a = fig.add_axes([0.14, 0.15, 0.49, 0.68])
    models = [m for m in ("LR", "RF", "GB", "RF-raw", "MLP", "R-SAGE-sup", "GraphCtx-MLP",
                          "DriftGuard-GNN") if m in sg["0.05"]]
    mu = [sg["0.05"][m]["prauc"][0] for m in models]
    sd = [sg["0.05"][m]["prauc"][1] for m in models]
    a.barh(range(len(models)), mu, xerr=sd, color=["#0072B2" if m == "DriftGuard-GNN"
                                                    else "#9aa5b1" for m in models])
    a.set_yticks(range(len(models))); a.set_yticklabels(models, fontsize=16)
    a.set_ylim(-1.3, len(models) - 0.4)
    a.axvline(sg["meta"]["theft_rate"], color="#D55E00", ls="--")
    a.text(sg["meta"]["theft_rate"] + 0.003, -0.9, "theft rate", color="#D55E00",
           fontsize=15)
    a.set_xlabel("PR-AUC, held-out customers, 5% labels", fontsize=17)
    a.tick_params(labelsize=14)
    a.spines[["top", "right"]].set_visible(False)
    ax.text(1290, 800, "SGCC real data", fontsize=30, color=BLUE, fontweight="bold")
    ax.text(1290, 720, "42,372 labelled customers\nno feeder topology\n"
            "out-of-time, customer-disjoint test\n\n" + SG_SUMMARY, fontsize=23, color=INK,
            va="top", linespacing=1.45)
slides.append((None, "Validation on real data", s6, True))

# 7 ------------------------------------------------------------- adaptation & decisions
def s7(fig, ax):
    image(fig, "../Figures/adapt_calib_rank.png", [0.03, 0.30, 0.94, 0.58])
    ax.text(90, 230, f"Targeted re-inspection + retraining: drifted-feeder PR-AUC "
            f"{ad['static']['drifted'][0]:.3f} → {ad['driftguard']['drifted'][0]:.3f}",
            fontsize=25, color=INK)
    ax.text(90, 160, f"Calibration error {S['calib']['ece_uncal'][0]:.3f} → "
            f"{S['calib']['ece_cal'][0]:.3f}; uncertain cases go to human review",
            fontsize=25, color=INK)
    ax.text(90, 90, "Value-weighted priority helps most when the inspection budget is tight",
            fontsize=25, color=INK)
slides.append((f"After drift alarms, targeted re-inspection and retraining raised precision "
               f"recall A U C on the drifted feeders from {f3(ad['static']['drifted'][0])} to "
               f"{f3(ad['driftguard']['drifted'][0])}. Calibration reduced the expected "
               f"calibration error from {f3(S['calib']['ece_uncal'][0])} to "
               f"{f3(S['calib']['ece_cal'][0])}, so probabilities can be trusted, and uncertain "
               f"cases are sent to human review instead of enforcement.",
               "Adaptation and decisions", s7, True))

# 8 ------------------------------------------------------------- merits / limits
def s8(fig, ax):
    ax.text(90, 880, "For utilities", fontsize=30, color=BLUE, fontweight="bold")
    bullets(ax, ["Works with very few inspection labels",
                 "Says which feeders changed and where\nto spend re-inspections",
                 "Calibrated probabilities with abstention",
                 "Runs on a standard CPU in seconds"], x=90, y=800, dy=120, size=24)
    ax.text(1000, 880, "Limitations", fontsize=30, color="#D55E00", fontweight="bold")
    bullets(ax, ["Main evaluation on a data-anchored\nsimulator; utility data needed",
                 "Concept drift needs label-based\nmonitoring",
                 "Gradual drift takes about ten weeks\nto detect"], x=1000, y=800, dy=150,
            size=24)
    ax.text(960, 130, "Code, seeds and data processing are released with the paper.",
            fontsize=24, color=MUTED, ha="center", style="italic")
slides.append(("For a utility, the framework works with very few inspections, says which "
               "feeders changed, gives calibrated probabilities, and runs on a standard computer. "
               "Its main limitations are that the full loop was evaluated on a data-anchored "
               "simulator, that concept drift needs label-based monitoring, and that gradual "
               "drift takes about ten weeks to detect. Thank you for watching.",
               "Merits and limitations", s8, True))


def tts(text, path):
    subprocess.run([sys.executable, "-m", "piper", "-m", MODEL, "-f", path],
                   input=text.encode(), check=True, capture_output=True)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def srt_time(s):
    h, r = divmod(s, 3600); m, r = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(r):02d},{int((r % 1) * 1000):03d}"


if __name__ == "__main__":
    global SG_SUMMARY
    SG_SUMMARY = sys.argv[3] if len(sys.argv) > 3 else ""
    SG_NARR = sys.argv[4] if len(sys.argv) > 4 else ""
    segs, t0, script = [], 0.0, []
    srt = []
    for i, (narr, title, fn, foot) in enumerate(slides):
        if narr is None:
            narr = SG_NARR
        png = os.path.join(TMP, f"s{i:02d}.png")
        slide(png, title, fn, foot)
        wav = os.path.join(TMP, f"s{i:02d}.wav")
        dur = tts(narr, wav)
        hold = dur + 1.2
        mp4 = os.path.join(TMP, f"s{i:02d}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", png,
                        "-i", wav, "-af", "apad", "-t", f"{hold:.2f}", "-c:v", "libx264",
                        "-preset", "veryfast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                        "-r", "25", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", mp4],
                       check=True)
        segs.append(mp4)
        srt.append(f"{i + 1}\n{srt_time(t0)} --> {srt_time(t0 + dur)}\n{narr}\n")
        script.append(f"[Slide {i + 1}: {title or 'Title card'}]\n{narr}\n")
        t0 += hold
    with open(os.path.join(TMP, "list.txt"), "w") as f:
        for s in segs:
            f.write(f"file '{os.path.abspath(s)}'\n")
    out = os.path.join(OUT, "DriftGuard-GNN_video_abstract.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                    os.path.join(TMP, "list.txt"), "-c", "copy", "-movflags", "+faststart", out],
                   check=True)
    open(os.path.join(OUT, "DriftGuard-GNN_video_abstract.srt"), "w").write("\n".join(srt))
    open(os.path.join(OUT, "narration_script.txt"), "w").write("\n".join(script))
    print("video", out, f"{t0:.1f}s")
