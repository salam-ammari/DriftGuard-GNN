"""Aggregate all result files into results/summary.json and print the
numbers quoted in the manuscript (mean +- std over the five test seeds)."""

import glob
import json
import numpy as np

RES = "../results"
SEEDS = [11, 23, 37, 51, 73]


def load(scen, var="topology"):
    out = []
    for s in SEEDS:
        try:
            out.append(json.load(open(f"{RES}/sim_{scen}_{var}_{s}.json")))
        except FileNotFoundError:
            pass
    return out


def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)


def fmt(t, d=3):
    return "n/a" if t[0] is None else f"{t[0]:.{d}f}+-{t[1]:.{d}f}"


S = {}
comb = load("combined")
models = list(comb[0]["detection"]["0.01"].keys())
S["detection"] = {f: {m: {k: ms([r["detection"][f][m][k] for r in comb])
                          for k in ("prauc", "rocauc", "r200")} for m in models}
                  for f in ("0.01", "0.05", "0.1", "0.2")}
print("== SIM detection (combined, topology), n_eval:",
      [r["n_eval"] for r in comb])
for f in ("0.01", "0.05", "0.1", "0.2"):
    print(f, {m: fmt(S["detection"][f][m]["prauc"]) for m in models})
for f in ("0.01", "0.05"):
    print(f, "ROC", {m: fmt(S["detection"][f][m]["rocauc"]) for m in models})
    print(f, "R@200", {m: fmt(S["detection"][f][m]["r200"]) for m in models})

sim_ = load("combined", "similarity")
S["detection_similarity"] = {f: {m: ms([r["detection"][f][m]["prauc"] for r in sim_])
                                 for m in models} for f in ("0.01", "0.05")}
print("== topology-free variant (n=%d)" % len(sim_))
for f in ("0.01", "0.05"):
    print(f, {m: fmt(S["detection_similarity"][f][m]) for m in models})

strategies = list(comb[0]["adapt"].keys())
S["adapt"] = {}
print("== adaptation (combined), post-drift window mean PR-AUC")
for s in strategies:
    d = {"stable": ms([r["adapt"][s]["stable"] for r in comb]),
         "drifted": ms([r["adapt"][s]["drifted"] for r in comb]),
         "per_seed_stable": [r["adapt"][s]["stable"] for r in comb],
         "per_seed_drifted": [r["adapt"][s]["drifted"] for r in comb]}
    for k in ("covariate", "gradual", "concept"):
        d[k] = ms([r["adapt"][s]["per_type"].get(k) for r in comb])
    S["adapt"][s] = d
    print(f"{s:10s} stable {fmt(d['stable'])} drifted {fmt(d['drifted'])} "
          f"cov {fmt(d['covariate'])} grad {fmt(d['gradual'])} con {fmt(d['concept'])}")
    print("           per-seed stable", np.round(d["per_seed_stable"], 3),
          "drifted", np.round(d["per_seed_drifted"], 3))

print("== scenarios")
S["scenarios"] = {}
for sc in ("none", "covariate", "gradual", "concept", "combined", "individual"):
    R = load(sc)
    if not R:
        continue
    d = {"n": len(R), "false_alarms": ms([r["drift"]["false_alarms"] for r in R])}
    for k in ("covariate", "gradual", "concept"):
        if k in R[0]["drift"]:
            d[k] = {"det_frac": ms([r["drift"][k]["detected"] / r["drift"][k]["of"] for r in R]),
                    "delay": ms([r["drift"][k]["delay_w"] for r in R])}
    for s in ("static", "scheduled", "uniform", "driftguard"):
        d[s] = {"stable": ms([r["adapt"][s]["stable"] for r in R]),
                "drifted": ms([r["adapt"][s]["drifted"] for r in R])}
    d["det_dg_1"] = ms([r["detection"]["0.01"]["DriftGuard-GNN"]["prauc"] for r in R])
    d["det_dg_5"] = ms([r["detection"]["0.05"]["DriftGuard-GNN"]["prauc"] for r in R])
    d["det_lr_5"] = ms([r["detection"]["0.05"]["LR"]["prauc"] for r in R])
    d["det_gctx_5"] = ms([r["detection"]["0.05"]["GraphCtx-MLP"]["prauc"] for r in R])
    d["det_mlp_5"] = ms([r["detection"]["0.05"]["MLP"]["prauc"] for r in R])
    d["loc_f1"] = ms([r["drift"]["loc_f1"] for r in R])
    d["events"] = ms([r["timing"]["alarm_events"] for r in R])
    S["scenarios"][sc] = d
    print(sc, json.dumps({k: (v if not isinstance(v, tuple) else fmt(v, 2))
                          for k, v in d.items()}, default=str))

print("== calibration / decisions (combined)")
S["calib"] = {k: ms([r["calib"][k] for r in comb]) for k in ("ece_uncal", "ece_cal", "brier")}
rc = np.array([r["calib"]["risk_cov"] for r in comb]).mean(0)
rc0 = np.array([r["calib"]["risk_cov_naive"] for r in comb]).mean(0)
S["calib"]["risk_cov"] = rc.tolist()
S["calib"]["risk_cov_naive"] = rc0.tolist()
print({k: fmt(v) for k, v in S["calib"].items() if isinstance(v, tuple)})
print("risk-cov (cov, adj, naive):", [(round(a, 2), round(b * 100, 2), round(c * 100, 2))
                                      for (a, b), (_, c) in zip(rc, rc0)])
S["ranking"] = {k: {kk: ms([r["ranking"][k][kk] for r in comb]) for kk in ("50", "100", "200", "400")}
                for k in ("prob", "util", "hit_prob", "hit_util")}
for k, v in S["ranking"].items():
    print(k, {kk: fmt(vv) for kk, vv in v.items()})
S["decision"] = {k: ms([r["decision"][k] for r in comb]) for k in ("spearman_Ehat", "medrelerr_Ehat")}
print("decision", {k: fmt(v) for k, v in S["decision"].items()})

print("== timing (combined, 3,600 consumers)")
keys = [k for k in comb[0]["timing"] if comb[0]["timing"][k] is not None]
S["timing"] = {k: ms([r["timing"][k] for r in comb if r["timing"][k] is not None]) for k in keys}
print({k: fmt(v, 3) for k, v in S["timing"].items()})

try:
    G = json.load(open(f"{RES}/sgcc.json"))
    S["sgcc"] = {"meta": {k: G[k] for k in ("N", "weeks", "theft_rate", "missing_week_frac")},
                 "timing": G["timing"]}
    ms_ = [m for m in G["runs"][0]["0.05"] if isinstance(G["runs"][0]["0.05"][m], dict)]
    for f in ("0.01", "0.05"):
        S["sgcc"][f] = {m: {k: ms([r[f][m][k] for r in G["runs"]])
                            for k in ("prauc", "rocauc", "p100", "p500")} for m in ms_}
        S["sgcc"][f]["n_pos"] = ms([r[f]["n_pos"] for r in G["runs"]])
        print("== SGCC", f, "n_pos", fmt(S["sgcc"][f]["n_pos"], 1))
        for m in ms_:
            print(f"   {m:15s}", {k: fmt(v) for k, v in S["sgcc"][f][m].items()})
    print("SGCC timing", G["timing"])
except FileNotFoundError:
    print("no sgcc yet")

json.dump(S, open(f"{RES}/summary.json", "w"), indent=1)
