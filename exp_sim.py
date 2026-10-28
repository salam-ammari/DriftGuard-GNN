"""Simulation experiments for the revised DriftGuard-GNN paper.

usage: python exp_sim.py <scenario> <variant> <seed> [<seed> ...]
  scenario in {none, covariate, gradual, concept, combined, individual}
  variant  in {topology, similarity}
Writes results/sim_<scenario>_<variant>_<seed>.json
"""

import copy
import json
import sys
import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import sim
from sim import SCENARIOS, TRAIN_END, VAL_END, T_WEEKS, N_FEEDERS, \
    FEEDERS_PER_REGION, WIN
from features import window_features, knn_shape, knn_shape_within
from gnn import RGNNEncoder, RGNNClassifier, knn_adjacency, group_adjacency
from driftguard import (EnsembleHead, representation, group_stats,
                        DriftMonitor, audit_z, ece, reliability, prauc,
                        rocauc, topk, n_params_head)

CHECKPOINTS = list(range(118, T_WEEKS + 1, 2))   # 20 biweekly checkpoints
CAL_CHECKS = list(range(80, VAL_END + 1, 3))     # drift-free calibration
THR, PERSIST = 5.0, 2          # chosen on drift-free validation seeds 101-103
AUDIT_RATE, AUDIT_THR = 0.005, 3.0  # random audits / checkpoint (validation)
BATCH_FRAC = 0.10                                # re-inspection batch
BUDGETS = (0.01, 0.05, 0.10, 0.20)
EVAL_WINDOW = [t for t in CHECKPOINTS if t >= 138]


def run(seed, scen_name, variant="topology"):
    T0 = time.time()
    tm = {}
    rng = np.random.default_rng(seed)
    C = sim.gen_corpus(rng, SCENARIOS[scen_name])
    N = len(C.y)
    region_of_feeder = np.arange(N_FEEDERS) // FEEDERS_PER_REGION

    # ------------------------------------------------ graph and context
    t0 = time.time()
    if variant == "topology":
        ctx = ("group", C.feeder)
        A_sim = knn_adjacency(knn_shape_within(C.load, TRAIN_END, C.feeder, 8), N)
        A_phy = group_adjacency(C.feeder, rng, 16)
        rels = [A_phy, A_sim]
        mon_groups = C.feeder
    else:                                        # no topology available
        nb = knn_shape(C.load, TRAIN_END, 30)
        ctx = ("knn", nb)
        A_sim = knn_adjacency(knn_shape(C.load, TRAIN_END, 8), N)
        rels = [A_sim]
        mon_groups = C.region * FEEDERS_PER_REGION  # billing areas only
    tm["graph_s"] = time.time() - t0

    weeks = sorted(set([TRAIN_END, VAL_END] + CAL_CHECKS + CHECKPOINTS))
    t0 = time.time()
    Fc = {t: window_features(C.load, t, ctx) for t in weeks}
    tm["features_per_ckpt_s"] = (time.time() - t0) / len(weeks)
    Fi = {t: window_features(C.load, t, None) for t in (TRAIN_END, T_WEEKS)}
    sc = StandardScaler().fit(Fc[TRAIN_END])
    X = {t: sc.transform(Fc[t]) for t in weeks}
    scI = StandardScaler().fit(Fi[TRAIN_END])
    XI = {t: scI.transform(Fi[t]) for t in Fi}

    # ------------------------------------------- self-supervised GNN
    t0 = time.time()
    enc = RGNNEncoder(X[TRAIN_END].shape[1], 32, 16, len(rels), rng)
    enc.pretrain(X[TRAIN_END], rels, A_sim)
    tm["ssl_pretrain_s"] = time.time() - t0
    t0 = time.time()
    REP = {t: representation(enc, X[t], rels, A_sim) for t in weeks}
    tm["gnn_infer_per_ckpt_s"] = (time.time() - t0) / len(weeks)
    scR = StandardScaler().fit(REP[TRAIN_END][0])
    R = {t: scR.transform(REP[t][0]) for t in weeks}

    # ----------------------------------------------- labels (nested)
    bias = np.log1p(C.load[:, TRAIN_END - WIN:TRAIN_END].mean(1))
    p = 0.5 + (bias - bias.min()) / np.ptp(bias)
    while True:      # protocol: >= 2 confirmed cases in the smallest budget
        order = rng.choice(N, size=int(0.20 * N), replace=False, p=p / p.sum())
        if C.y[order[:int(round(BUDGETS[0] * N))]].sum() >= 2:
            break
    lab = {f: np.zeros(N, bool) for f in BUDGETS}
    for f in BUDGETS:
        lab[f][order[:int(round(f * N))]] = True
    ever = lab[0.20].copy()                      # updated by audits/batches

    out = {"seed": seed, "scenario": scen_name, "variant": variant,
           "truth": {k: v for k, v in C.truth.items() if k != "individual_ids"}}

    # --------------------------------- static detection (main + sweep)
    def fit_models(f):
        m = lab[f]
        ym = C.y[m]
        P = {}
        for name, clf in (("LR", LogisticRegression(max_iter=3000,
                                                    class_weight="balanced")),
                          ("RF", RandomForestClassifier(300, class_weight="balanced",
                                                        random_state=seed)),
                          ("GB", GradientBoostingClassifier(random_state=seed))):
            clf.fit(XI[TRAIN_END][m], ym)
            P[name] = clf.predict_proba(XI[T_WEEKS])[:, 1]
        h = EnsembleHead(XI[TRAIN_END].shape[1], rng)
        h.fit(XI[TRAIN_END][m], ym, calibrate=False)
        P["MLP"] = h.predict(XI[T_WEEKS], calibrated=False)[0]
        g = RGNNClassifier(XI[TRAIN_END].shape[1], 32, len(rels), rng)
        g.fit(XI[TRAIN_END], rels, np.where(m)[0], ym)
        P["R-SAGE-sup"] = g.predict(XI[T_WEEKS], rels)
        h = EnsembleHead(X[TRAIN_END].shape[1], rng)
        h.fit(X[TRAIN_END][m], ym, calibrate=False)
        P["GraphCtx-MLP"] = h.predict(X[T_WEEKS], calibrated=False)[0]
        t0 = time.time()
        h = EnsembleHead(R[TRAIN_END].shape[1], rng)
        h.fit(R[TRAIN_END][m], ym, calibrate=True)
        fit_s = time.time() - t0
        P["DriftGuard-GNN"] = h.predict(R[T_WEEKS])[0]
        return P, h, fit_s

    det = {}
    heads = {}
    for f in BUDGETS:
        P, h, fit_s = fit_models(f)
        det[str(f)] = P
        heads[f] = h
        if f == 0.05:
            tm["head_fit_s"] = fit_s
    head0 = heads[0.05]

    # ---------------------------------------- drift monitor calibration
    mon = DriftMonitor([group_stats(X[t], REP[t][1], REP[t][2], mon_groups)
                        for t in CAL_CHECKS])
    ref_lvl = {g: 0 for g in mon.g}

    # ------------------------------------------------------ online loop
    # Every strategy receives the same random-audit labels.  On an alarm,
    # DriftGuard re-inspects the flagged groups (targeted batch); the
    # "uniform" control re-inspects the same number of consumers drawn at
    # random from all groups, isolating the value of localization.
    strategies = ["static", "finetune", "local", "scheduled", "uniform",
                  "driftguard"]
    H = {"static": head0, "finetune": copy.deepcopy(head0),
         "scheduled": head0, "uniform": head0, "driftguard": head0}
    local = {}                                   # group -> local head
    pool_t = {int(i): TRAIN_END for i in np.where(lab[0.05])[0]}
    pool_u = dict(pool_t)                        # uniform-control pool
    pool_s = dict(pool_t)                        # scheduled: audits only
    persist = {g: 0 for g in ref_lvl}
    flagged, alarm_week, alarm_src = set(), {}, {}
    aud_O = np.zeros(4); aud_E = np.zeros(4); aud_V = np.zeros(4)
    aud_persist = np.zeros(4, int)
    curves = {s: [] for s in strategies}
    ztrace, atrace = {}, {}
    t_mon, t_loc, t_ret, n_events = [], [], [], 0

    truth_fe = set()
    for k in ("covariate", "gradual", "concept"):
        truth_fe |= set(C.truth.get(k, []))
    fe_drift = np.isin(C.feeder, sorted(truth_fe))

    def grp_mask(g):
        return mon_groups == g

    def predict(s, t):
        if s == "local":
            p, u = head0.predict(R[t])
            for g, h in local.items():
                m = grp_mask(g)
                pl, ul = h.predict(R[t][m])
                p[m] = 0.5 * (p[m] + pl)
                u[m] = 0.5 * (u[m] + ul)
            return p, u
        return H[s].predict(R[t])

    def retrain(pool):
        ids = np.array(sorted(pool))
        Zp = np.vstack([R[pool[i]][i] for i in ids])
        h = EnsembleHead(Zp.shape[1], rng)
        h.fit(Zp, C.y[ids])
        return h

    for t in CHECKPOINTS:
        P_now = {s: predict(s, t) for s in strategies}
        for s in strategies:
            curves[s].append((t, P_now[s][0]))
        t0 = time.time()
        zz = mon.score(group_stats(X[t], REP[t][1], REP[t][2], mon_groups))
        z = {g: max(v) for g, v in zz.items()}
        ztrace[t] = {int(g): v for g, v in zz.items()}
        new = []
        for g, v in z.items():
            persist[g] = persist[g] + 1 if v > THR else 0
            if persist[g] >= PERSIST and g not in flagged:
                new.append(g); alarm_src[int(g)] = "unsup"
        p_stat = P_now["static"][0]
        for r in range(4):
            cand = np.where((C.region == r) & ~ever)[0]
            a = rng.choice(cand, size=max(int(AUDIT_RATE * N / 4), 1), replace=False)
            ever[a] = True
            for i in a:
                pool_t[int(i)] = t; pool_u[int(i)] = t; pool_s[int(i)] = t
            aud_O[r] += C.y[a].sum(); aud_E[r] += p_stat[a].sum()
            aud_V[r] += (p_stat[a] * (1 - p_stat[a])).sum()
            za = audit_z(aud_O[r], aud_E[r], aud_V[r])
            atrace.setdefault(t, {})[r] = float(za)
            aud_persist[r] = aud_persist[r] + 1 if za > AUDIT_THR else 0
            if aud_persist[r] >= PERSIST:
                for g in np.unique(mon_groups[C.region == r]):
                    if g not in flagged and g not in new:
                        new.append(g); alarm_src[int(g)] = "audit"
        t_mon.append(time.time() - t0)
        if (CHECKPOINTS.index(t) + 1) % 4 == 0:          # every 8 weeks
            H["scheduled"] = retrain({i: v for i, v in pool_s.items()})
        if not new:
            continue
        n_events += 1
        batch = []
        for g in new:
            flagged.add(g); alarm_week[int(g)] = t
            cand = np.where(grp_mask(g) & ~ever)[0]
            b = rng.choice(cand, size=max(int(BATCH_FRAC * grp_mask(g).sum()), 3),
                           replace=False)
            ever[b] = True
            batch.extend(b.tolist())
            for i in b:
                pool_t[int(i)] = t
        batch = np.array(batch)
        cand = np.where(~ever)[0]                         # uniform control
        bu = rng.choice(cand, size=batch.size, replace=False)
        ever[bu] = True
        for i in bu:
            pool_u[int(i)] = t
        H["finetune"].finetune(R[t][batch], C.y[batch])
        t0 = time.time()
        H["driftguard"] = retrain(pool_t)
        t_ret.append(time.time() - t0)
        H["uniform"] = retrain(pool_u)
        t0 = time.time()
        for g in new:
            ids = np.array([i for i in pool_t if mon_groups[i] == g])
            Zs = np.vstack([R[pool_t[i]][i] for i in ids])
            ys = C.y[ids]
            h = copy.deepcopy(head0)
            if ys.sum() > 0:
                h.finetune(Zs, ys)
                if ys.sum() >= 2:
                    h.cross_fit_platt(Zs, ys, epochs=250)
            local[g] = h
        t_loc.append(time.time() - t0)

    eval_m = ~ever
    out["n_eval"] = int(eval_m.sum())
    out["n_labels_used"] = int(ever.sum())

    # -------------------------------------------- detection summary
    yE = C.y[eval_m]
    out["detection"] = {
        f: {name: {"prauc": prauc(yE, P[eval_m]), "rocauc": rocauc(yE, P[eval_m]),
                   "r200": float(C.y[eval_m][topk(P[eval_m], 200)].sum() / max(yE.sum(), 1))}
            for name, P in det[f].items()} for f in det}

    # ----------------------------------------- adaptation summary
    stable = eval_m & ~fe_drift
    drifted = eval_m & fe_drift
    adapt = {}
    for s in strategies:
        st, dr = [], []
        for t, p in curves[s]:
            if t in EVAL_WINDOW:
                st.append(prauc(C.y[stable], p[stable]))
                if drifted.sum():
                    dr.append(prauc(C.y[drifted], p[drifted]))
        adapt[s] = {"stable": float(np.mean(st)),
                    "drifted": float(np.mean(dr)) if dr else None}
        per_type = {}
        for k in ("covariate", "gradual", "concept"):
            if k in C.truth:
                mk = eval_m & np.isin(C.feeder, C.truth[k])
                per_type[k] = float(np.mean([prauc(C.y[mk], p[mk])
                                             for t, p in curves[s] if t in EVAL_WINDOW]))
        adapt[s]["per_type"] = per_type
        adapt[s]["curve_drifted"] = [prauc(C.y[drifted], p[drifted]) if drifted.sum() else None
                                     for t, p in curves[s]]
    out["adapt"] = adapt
    out["curve_weeks"] = CHECKPOINTS

    # ---------------------------------------------- drift localization
    loc = {}
    onset = {"covariate": sim.COV_WEEK, "gradual": sim.GRAD_WEEK,
             "concept": sim.CON_WEEK}
    mon_truth = {}
    for k in ("covariate", "gradual", "concept"):
        if k not in C.truth:
            continue
        tg = set(C.truth[k]) if variant == "topology" else \
            {sim.COV_REGION * 6} if k == "covariate" else \
            {GRAD_RG * 6 for GRAD_RG in [3]} if k == "gradual" else {sim.CON_REGION * 6}
        mon_truth[k] = tg
        det_g = tg & flagged
        loc[k] = {"detected": len(det_g), "of": len(tg),
                  "delay_w": float(np.mean([alarm_week[g] - onset[k] for g in det_g]))
                  if det_g else None,
                  "sources": sorted({alarm_src[g] for g in det_g})}
    all_truth = set().union(*mon_truth.values()) if mon_truth else set()
    fa = len(flagged - all_truth)
    tp = len(flagged & all_truth)
    fn = len(all_truth - flagged)
    loc["false_alarms"] = fa
    loc["loc_f1"] = 2 * tp / max(2 * tp + fa + fn, 1) if all_truth else None
    loc["alarm_week"] = {str(k): v for k, v in alarm_week.items()}
    loc["alarm_src"] = {str(k): v for k, v in alarm_src.items()}
    out["drift"] = loc
    out["ztrace"] = {str(k): {str(g): v for g, v in d.items()} for k, d in ztrace.items()}
    out["atrace"] = {str(k): {str(r): v for r, v in d.items()} for k, d in atrace.items()}

    # ------------------------------------ calibration and decisions (t=156)
    p_sel, u_sel = predict("driftguard", T_WEEKS)
    pu, _ = H["driftguard"].predict(R[T_WEEKS], calibrated=False)
    yv, pv, uv = C.y[eval_m], p_sel[eval_m], u_sel[eval_m]
    unorm = (uv - uv.min()) / (np.ptp(uv) + 1e-12)
    err = ((pv >= 0.5).astype(int) != yv).astype(float)
    conf = np.maximum(pv, 1 - pv) - 0.35 * unorm
    conf0 = np.maximum(pv, 1 - pv)
    rc, rc0 = [], []
    for c in np.linspace(0.5, 1.0, 11):
        k = int(c * len(pv))
        rc.append([float(c), float(err[np.argsort(-conf)[:k]].mean())])
        rc0.append([float(c), float(err[np.argsort(-conf0)[:k]].mean())])
    out["calib"] = {"ece_uncal": ece(yv, pu[eval_m]), "ece_cal": ece(yv, pv),
                    "brier": float(((pv - yv) ** 2).mean()),
                    "reliab": reliability(yv, pv),
                    "reliab_uncal": reliability(yv, pu[eval_m]),
                    "risk_cov": rc, "risk_cov_naive": rc0}

    # ranking: expected recoverable energy proxy and inspection cost
    base = C.load[:, T_WEEKS - 52 - WIN:T_WEEKS - 52].mean(1)
    ctx_yoy = np.empty(N)
    for g in np.unique(C.feeder if variant == "topology" else C.region):
        m = (C.feeder if variant == "topology" else C.region) == g
        ctx_yoy[m] = np.median(Fc[T_WEEKS][m, 0])
    E_hat = np.maximum(base * ctx_yoy - C.load[:, T_WEEKS - WIN:].mean(1), 0) \
        * (T_WEEKS - VAL_END)
    cost = 1.0 + 0.5 * (C.region == 3)
    u_all = (u_sel - u_sel[eval_m].min()) / (np.ptp(u_sel[eval_m]) + 1e-12)
    S = p_sel * (1 - 0.3 * u_all) * (E_hat + 1) ** 0.7 / cost
    st_E = C.stolen[eval_m]
    rk = {"prob": {}, "util": {}, "hit_prob": {}, "hit_util": {}}
    for k in (50, 100, 200, 400):
        for nm, sc_ in (("prob", p_sel[eval_m]), ("util", S[eval_m])):
            top = topk(sc_, k)
            rk[nm][str(k)] = float(st_E[top].sum() / st_E.sum())
            rk["hit_" + nm][str(k)] = float(yv[top].mean())
    fr = (C.y == 1) & eval_m
    rho = spearmanr(E_hat[fr], C.stolen[fr]).correlation
    rel_err = np.median(np.abs(E_hat[fr] - C.stolen[fr]) / (C.stolen[fr] + 1e-9))
    out["ranking"] = rk
    out["decision"] = {"spearman_Ehat": float(rho), "medrelerr_Ehat": float(rel_err)}

    # ------------------------------------------------------------ cost
    tm["monitor_per_ckpt_s"] = float(np.mean(t_mon))
    tm["local_per_event_s"] = float(np.mean(t_loc)) if t_loc else None
    tm["retrain_per_event_s"] = float(np.mean(t_ret)) if t_ret else None
    tm["alarm_events"] = n_events
    tm["pool_final"] = len(pool_t)
    tm["n_params_encoder"] = int(enc.n_params())
    tm["n_params_head"] = int(n_params_head(head0))
    tm["total_run_s"] = time.time() - T0
    out["timing"] = tm
    return out


if __name__ == "__main__":
    scen, variant = sys.argv[1], sys.argv[2]
    for s in map(int, sys.argv[3:]):
        res = run(s, scen, variant)
        with open(f"../results/sim_{scen}_{variant}_{s}.json", "w") as f:
            json.dump(res, f)
        print(scen, variant, s, f"{res['timing']['total_run_s']:.1f}s", flush=True)
