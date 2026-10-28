"""Real-data validation of the detection stage on the public SGCC
electricity-theft dataset (42,372 labelled customers, daily kWh,
2014-01-01 to 2016-10-31; Zheng et al., IEEE TII 2018).

No feeder topology exists in SGCC, so DriftGuard-GNN runs in its
topology-free configuration: context and message passing use only the
load-shape similarity graph built from 2014 data.

Protocol: customer-disjoint (70/30 stratified) AND out-of-time: models
are trained on features of the window ending at week 104 (Dec 2015) and
evaluated on held-out customers at the window ending at week 147
(Oct 2016).  Labelled budgets of 1% and 5% of all customers are drawn
from training customers with a bias toward high consumption.

usage: python exp_sgcc.py <path/to/data.csv> <seed> [<seed> ...]
"""

import json
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from features import window_features, knn_shape
from gnn import RGNNEncoder, RGNNClassifier, knn_adjacency
from driftguard import EnsembleHead, representation, prauc, rocauc, topk

T1, WIN = 104, 24


def load_weekly(path):
    df = pd.read_csv(path)
    y = df["FLAG"].to_numpy().astype(int)
    D = df.drop(columns=["CONS_NO", "FLAG"])
    dates = pd.to_datetime(D.columns, format="%Y/%m/%d")
    D = D.iloc[:, np.argsort(dates.values)].to_numpy(dtype=float)
    n_w = D.shape[1] // 7
    D = D[:, :n_w * 7].reshape(len(y), n_w, 7)
    cnt = np.isfinite(D).sum(2)
    W = np.where(cnt >= 4, np.nanmean(D, axis=2) * 7, np.nan)
    W = np.where(np.isfinite(W), np.clip(W, 0, None), np.nan)
    miss = np.isnan(W).mean(1)
    Mw = np.isnan(D).mean(2)                     # weekly missing fraction
    # per-customer linear interpolation over weeks, edges held constant
    idx = np.arange(n_w)
    for i in np.where(np.isnan(W).any(1))[0]:
        ok = np.isfinite(W[i])
        W[i] = np.interp(idx, idx[ok], W[i, ok]) if ok.sum() >= 2 else 0.0
    return W, y, miss, Mw


def run(W, y, seed, nb8, nb30, A_sim, X, XI, timing):
    rng = np.random.default_rng(seed)
    N = len(y)
    # customer-disjoint stratified split
    test = np.zeros(N, bool)
    for c in (0, 1):
        ids = np.where(y == c)[0]
        test[rng.choice(ids, int(0.3 * ids.size), replace=False)] = True
    train = ~test
    bias = np.log1p(W[:, T1 - WIN:T1].mean(1))
    p = (0.5 + (bias - bias.min()) / np.ptp(bias)) * train
    while True:      # protocol: >= 2 confirmed cases in the smallest budget
        order = rng.choice(N, size=int(0.05 * N), replace=False, p=p / p.sum())
        if y[order[:int(round(0.01 * N))]].sum() >= 2:
            break
    rels = [A_sim]

    t0 = time.time()
    enc = RGNNEncoder(X[T1].shape[1], 32, 16, 1, rng)
    enc.pretrain(X[T1], rels, A_sim)
    timing.setdefault("ssl_pretrain_s", []).append(time.time() - t0)
    t0 = time.time()
    R = {t: representation(enc, X[t], rels, A_sim)[0] for t in X}
    timing.setdefault("gnn_infer_s", []).append((time.time() - t0) / len(X))
    scR = StandardScaler().fit(R[T1])
    R = {t: scR.transform(R[t]) for t in R}
    T2 = max(X)
    yT = y[test]
    res = {}
    raw = W / (W[:, :T1].mean(1, keepdims=True) + 1e-6)          # RF-raw
    for f in (0.01, 0.05):
        m = np.zeros(N, bool)
        m[order[:int(round(f * N))]] = True
        ym = y[m]
        P = {}
        for name, clf, Z in (
                ("LR", LogisticRegression(max_iter=3000, class_weight="balanced"), XI),
                ("RF", RandomForestClassifier(300, class_weight="balanced",
                                              n_jobs=2, random_state=seed), XI),
                ("GB", GradientBoostingClassifier(random_state=seed), XI)):
            clf.fit(Z[T1][m], ym)
            P[name] = clf.predict_proba(Z[T2])[:, 1]
        rf = RandomForestClassifier(300, class_weight="balanced", n_jobs=2,
                                    random_state=seed)
        rf.fit(raw[m, T1 - 52:T1], ym)
        P["RF-raw"] = rf.predict_proba(raw[:, T2 - 52:T2])[:, 1]
        h = EnsembleHead(XI[T1].shape[1], rng)
        h.fit(XI[T1][m], ym, calibrate=False)
        P["MLP"] = h.predict(XI[T2], calibrated=False)[0]
        g = RGNNClassifier(XI[T1].shape[1], 32, 1, rng)
        g.fit(XI[T1], rels, np.where(m)[0], ym)
        P["R-SAGE-sup"] = g.predict(XI[T2], rels)
        h = EnsembleHead(X[T1].shape[1], rng)
        h.fit(X[T1][m], ym, calibrate=False)
        P["GraphCtx-MLP"] = h.predict(X[T2], calibrated=False)[0]
        t0 = time.time()
        h = EnsembleHead(R[T1].shape[1], rng)
        h.fit(R[T1][m], ym)
        timing.setdefault("head_fit_s_%g" % f, []).append(time.time() - t0)
        P["DriftGuard-GNN"] = h.predict(R[T2])[0]
        res[str(f)] = {k: {"prauc": prauc(yT, v[test]), "rocauc": rocauc(yT, v[test]),
                           "p100": float(yT[topk(v[test], 100)].mean()),
                           "p500": float(yT[topk(v[test], 500)].mean())}
                       for k, v in P.items()}
        res[str(f)]["n_labels"] = int(m.sum())
        res[str(f)]["n_pos"] = int(ym.sum())
    return res


if __name__ == "__main__":
    path = sys.argv[1]
    seeds = list(map(int, sys.argv[2:]))
    timing = {}
    t0 = time.time()
    W, y, miss, Mw = load_weekly(path)
    timing["load_s"] = time.time() - t0
    N, T = W.shape
    print("customers", N, "weeks", T, "theft rate", y.mean(),
          "missing-week frac", miss.mean(), flush=True)
    t0 = time.time()
    nb30 = knn_shape(W, 52, 30)                  # 2014 data only
    nb8 = knn_shape(W, 52, 8)
    A_sim = knn_adjacency(nb8, N)
    timing["graph_s"] = time.time() - t0
    t0 = time.time()
    # window-limited missing-reading fraction appended for ALL models
    mfrac = {t: Mw[:, t - WIN:t].mean(1, keepdims=True) for t in (T1, T)}
    F = {t: np.column_stack([window_features(W, t, ("knn", nb30)), mfrac[t]])
         for t in (T1, T)}
    FI = {t: np.column_stack([window_features(W, t, None), mfrac[t]])
          for t in (T1, T)}
    timing["features_s"] = (time.time() - t0) / 2
    sc = StandardScaler().fit(F[T1])
    X = {t: sc.transform(F[t]) for t in F}
    scI = StandardScaler().fit(FI[T1])
    XI = {t: scI.transform(FI[t]) for t in FI}
    out = {"N": N, "weeks": T, "theft_rate": float(y.mean()),
           "missing_week_frac": float(miss.mean()), "runs": []}
    for s in seeds:
        out["runs"].append(run(W, y, s, nb8, nb30, A_sim, X, XI, timing))
        print("seed", s, {k: round(v["prauc"], 3) for k, v in out["runs"][-1]["0.05"].items()
                          if isinstance(v, dict)}, flush=True)
    out["timing"] = {k: (float(np.mean(v)) if isinstance(v, list) else v)
                     for k, v in timing.items()}
    with open("../results/sgcc.json", "w") as f:
        json.dump(out, f)
    print("timing", out["timing"])
