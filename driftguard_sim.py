
import copy
import json
import time
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler

from nn import Net, sigmoid

RNG_SEEDS = [11, 23, 37, 51, 73]


N_REGIONS = 4
FEEDERS_PER_REGION = 6
CUST_PER_FEEDER = 150
N_FEEDERS = N_REGIONS * FEEDERS_PER_REGION            # 24
N_CUST = N_FEEDERS * CUST_PER_FEEDER                  # 3600
T_WEEKS = 156
TRAIN_END, VAL_END = 104, 116                         # chronological split
DRIFT_WEEK_ABRUPT = 124                               # region-2 tariff shock
DRIFT_WEEK_GRADUAL = 132                              # partial economic drift
GRADUAL_FEEDERS = (18, 19, 20)                        # only half of region 3
FRAUD_RATE = 0.08
WIN = 24                                              # feature window (weeks)


@dataclass
class Corpus:
    load: np.ndarray          # reported consumption (N, T)
    true: np.ndarray          # counterfactual honest consumption (N, T)
    y: np.ndarray             # fraud indicator (N,)
    stolen: np.ndarray        # stolen energy inside test window (N,)
    feeder: np.ndarray        # feeder id (N,)
    region: np.ndarray        # region id (N,)
    fraud_type: np.ndarray    # 0 = honest, 1..4 fraud mechanisms


def gen_corpus(rng: np.random.Generator) -> Corpus:
    t = np.arange(T_WEEKS)
    region = np.repeat(np.arange(N_REGIONS), FEEDERS_PER_REGION * CUST_PER_FEEDER)
    feeder = np.repeat(np.arange(N_FEEDERS), CUST_PER_FEEDER)

    base_mu = np.array([120., 95., 140., 80.])[region]        # weekly kWh
    seas_amp = np.array([0.25, 0.40, 0.15, 0.35])[region]
    seas_phase = np.array([0.0, 1.3, 2.4, 4.0])[region]
    cust_scale = rng.lognormal(0.0, 0.35, N_CUST)
    noise_sd = 0.08 + 0.05 * rng.random(N_CUST)

    seas = 1.0 + seas_amp[:, None] * np.sin(2 * np.pi * t[None, :] / 52.0
                                            + seas_phase[:, None])
    trend = 1.0 + 0.0015 * t[None, :] * rng.normal(1.0, 0.5, (N_CUST, 1))
    true = (base_mu * cust_scale)[:, None] * seas * trend
    true *= rng.normal(1.0, noise_sd[:, None], (N_CUST, T_WEEKS))
    true = np.clip(true, 2.0, None)

    # ---- injected concept drift ---------------------------------------
    # Abrupt: tariff restructuring in region 2 -> level drop plus weekly
    # shape change (demand response) from DRIFT_WEEK_ABRUPT.
    m_ab = (region == 2)
    true[m_ab, DRIFT_WEEK_ABRUPT:] *= 0.78
    shape = 1.0 + 0.20 * np.sin(2 * np.pi * t / 13.0)
    true[m_ab, DRIFT_WEEK_ABRUPT:] *= shape[DRIFT_WEEK_ABRUPT:]
    # Gradual, spatially local: economic contraction affecting only three
    # feeders of region 3 from DRIFT_WEEK_GRADUAL.
    m_gr = np.isin(feeder, GRADUAL_FEEDERS)
    ramp = np.ones(T_WEEKS)
    gwks = np.arange(DRIFT_WEEK_GRADUAL, T_WEEKS)
    ramp[gwks] = 1.0 - 0.022 * (gwks - DRIFT_WEEK_GRADUAL)
    true[m_gr] *= ramp[None, :]

    # ---- fraud injection ----------------------------------------------
    y = (rng.random(N_CUST) < FRAUD_RATE).astype(int)
    fraud_type = np.zeros(N_CUST, dtype=int)
    load = true.copy()
    idx = np.where(y == 1)[0]
    fraud_type[idx] = rng.integers(1, 5, idx.size)
    onset = rng.integers(30, 120, N_CUST)
    for i in idx:
        s = onset[i]
        ft = fraud_type[i]
        if ft == 1:                                   # persistent under-report
            k = rng.uniform(0.35, 0.70)
            load[i, s:] = true[i, s:] * k
        elif ft == 2:                                 # intermittent bypass
            wk = rng.random(T_WEEKS - s) < 0.35
            load[i, s:][wk] = true[i, s:][wk] * rng.uniform(0.05, 0.20)
        elif ft == 3:                                 # progressive ramp-down
            L = T_WEEKS - s
            load[i, s:] = true[i, s:] * np.linspace(1.0, 0.30, L)
        else:                                         # cap / meter saturation
            cap = np.quantile(true[i, :s], 0.35)
            load[i, s:] = np.minimum(true[i, s:], cap)
    # Fraudster adaptation: after the enforcement/tariff events, active
    # fraud inside drifted feeders becomes subtler (adaptive adversary),
    # i.e. genuine P(y|x) concept drift, not just covariate shift.
    for i in idx:
        if feeder[i] >= 18 or feeder[i] < 12:
            continue                                  # only region 2 campaign
        k_new = rng.uniform(0.55, 0.75)               # subtler post-campaign skim
        load[i, DRIFT_WEEK_ABRUPT:] = true[i, DRIFT_WEEK_ABRUPT:] * k_new
    stolen = (true - load)[:, VAL_END:].sum(axis=1)
    return Corpus(load, true, y, stolen, feeder, region, fraud_type)


# ----------------------------------------------------------------------
# 2. Features (independent vs graph-contextualized) and similarity graph
# ----------------------------------------------------------------------

def window_features(load, feeder, t_end, graph=True, win=WIN, base_end=36):
    """Descriptors of the trailing `win` weeks ending at t_end.

    With graph=True every consumer is contextualized against its feeder
    median profile (one round of median-aggregator relational message
    passing over the physical relation); with graph=False consumers are
    treated independently against population medians, mirroring the
    prevailing CNN/LSTM literature.  The self-baseline is the same
    season one year earlier (leakage-safe, seasonality-corrected)."""
    X = load[:, t_end - win:t_end]
    if t_end - 52 - win >= 0:
        B = load[:, t_end - 52 - win:t_end - 52]
    else:
        B = load[:, :base_end]
    mu, sd = X.mean(1), X.std(1) + 1e-8
    bmu = B.mean(1) + 1e-8
    dec = mu / bmu
    fmu = np.zeros_like(mu)
    fdec = np.zeros_like(mu)
    prof = np.zeros(X.shape)
    if graph:
        for f in np.unique(feeder):
            m = feeder == f
            fmu[m] = np.median(mu[m])
            fdec[m] = np.median(dec[m])
            prof[m] = np.median(X[m], axis=0)
    else:
        fmu[:] = np.median(mu)
        fdec[:] = np.median(dec)
        prof[:] = np.median(X, axis=0)
    rel_level = mu / (fmu + 1e-8)
    rel_decay = dec / (fdec + 1e-8)                   # key NTL signature
    x1, x2 = X[:, :-1], X[:, 1:]
    ac1 = ((x1 - x1.mean(1, keepdims=True)) *
           (x2 - x2.mean(1, keepdims=True))).mean(1) / (x1.std(1) * x2.std(1) + 1e-8)
    slope = np.polyfit(np.arange(win), X.T, 1)[0] / (bmu / win)
    lowq = np.quantile(X, 0.1, axis=1) / bmu
    zero_frac = (X < 0.15 * bmu[:, None]).mean(1)
    half = win // 2
    half_decay = X[:, half:].mean(1) / (X[:, :half].mean(1) + 1e-8)
    cv = sd / (mu + 1e-8)
    diffE = (np.diff(X, axis=1) ** 2).mean(1) / (sd ** 2)
    xc = X - X.mean(1, keepdims=True)
    pc = prof - prof.mean(1, keepdims=True)
    corr_f = (xc * pc).mean(1) / (X.std(1) * prof.std(1) + 1e-8)
    return np.column_stack([
        dec, rel_decay, rel_level, cv, ac1, slope, lowq, zero_frac,
        half_decay, diffE, corr_f, np.log(mu + 1.0),
    ])


def build_graph(load, feeder, t_end, k=8):
    """Similarity relation: kNN over normalized load shapes computed
    from data before t_end only (leakage-safe), restricted to the same
    feeder block for tractability."""
    X = load[:, max(0, t_end - WIN):t_end]
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    sim_nbrs = np.zeros((len(feeder), k), dtype=int)
    for f in np.unique(feeder):
        m = np.where(feeder == f)[0]
        S = Z[m] @ Z[m].T / Z.shape[1]
        np.fill_diagonal(S, -np.inf)
        sim_nbrs[m] = m[np.argsort(-S, axis=1)[:, :k]]
    return sim_nbrs


# ----------------------------------------------------------------------
# 3. DriftGuard-GNN components (neural core in nn.py)
# ----------------------------------------------------------------------

class Encoder:
    """Masked-reconstruction + neighbourhood-consistency SSL encoder."""

    def __init__(self, d_in, d_hid, d_emb, rng):
        self.enc = Net([d_in, d_hid, d_emb], rng)
        self.dec = Net([d_emb, d_hid, d_in], rng)
        self.rng = rng

    def pretrain(self, X, sim_nbrs, epochs=320, lam_nbr=0.1, mask_p=0.3):
        n = X.shape[0]
        for ep in range(1, epochs + 1):
            idx = self.rng.choice(n, size=512, replace=False)
            xb = X[idx]
            mask = (self.rng.random(xb.shape) < mask_p).astype(float)
            xin = xb * (1.0 - mask)
            z_nb = self.enc.forward(X[sim_nbrs[idx]].mean(1)).copy()
            z = self.enc.forward(xin)
            xr = self.dec.forward(z)
            d_rec = 2.0 * (xr - xb) * mask / xb.shape[0]
            dz = self.dec.backward(d_rec)
            dz = dz + lam_nbr * 2.0 * (z - z_nb) / xb.shape[0]
            self.enc.backward(dz)
            self.enc.step(1e-3, ep)
            self.dec.step(1e-3, ep)

    def embed(self, X):
        return self.enc.forward(X)

    def recon_err(self, X):
        return ((self.dec.forward(self.enc.forward(X)) - X) ** 2).mean(1)


class EnsembleHead:
    """Deep-ensemble fraud head with Platt calibration."""

    def __init__(self, d_in, rng, n_members=5):
        self.rngs = [np.random.default_rng(rng.integers(10 ** 9))
                     for _ in range(n_members)]
        self.members = [Net([d_in, 32, 1], r) for r in self.rngs]
        self.a, self.b = 1.0, 0.0

    def fit(self, Z, y, epochs=500, pos_w=None):
        if pos_w is None:
            pos_w = (len(y) - y.sum()) / max(y.sum(), 1)
        w = np.where(y == 1, pos_w, 1.0)
        for m, r in zip(self.members, self.rngs):
            keep = r.random(len(y)) < 0.9              # light bagging
            Zb, yb, wb = Z[keep], y[keep], w[keep]
            for ep in range(1, epochs + 1):
                p = sigmoid(m.forward(Zb)[:, 0])
                d = ((p - yb) * wb / len(yb))[:, None]
                m.backward(d)
                m.step(3e-3, ep)

    def logits(self, Z):
        return np.stack([m.forward(Z)[:, 0] for m in self.members])

    def predict(self, Z):
        P = sigmoid(self.logits(Z))
        p = P.mean(0)
        epi = P.std(0)
        ale = -(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9))
        return p, epi, ale / np.log(2)

    def calibrate(self, Z, y):
        """Platt scaling (slope + intercept) on mean logits."""
        L = self.logits(Z).mean(0)
        a, b = 1.0, 0.0
        for _ in range(400):
            p = sigmoid(a * L + b)
            a -= 0.5 * ((p - y) * L).mean()
            b -= 0.5 * (p - y).mean()
        self.a, self.b = a, b

    def predict_cal(self, Z):
        P = sigmoid(self.a * self.logits(Z) + self.b)
        p = P.mean(0)
        epi = P.std(0)
        ale = -(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9))
        return p, epi, ale / np.log(2)


def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def reliability_points(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 5:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return xs, ys


def feeder_drift_scores(Z_ref, Z_now, rec_ref, rec_now, feeder):
    """Localized drift score per feeder combining embedding mean shift
    and reconstruction-error increase, both scaled by reference spread."""
    scores = {}
    for f in np.unique(feeder):
        m = feeder == f
        a, b = Z_ref[m], Z_now[m]
        d_emb = np.linalg.norm(a.mean(0) - b.mean(0)) / (a.std() + 1e-8)
        d_rec = (rec_now[m].mean() - rec_ref[m].mean()) / (rec_ref[m].std() + 1e-8)
        scores[int(f)] = float(0.6 * d_emb + 0.4 * max(d_rec, 0.0))
    return scores


# ----------------------------------------------------------------------
# 4. Experiment driver
# ----------------------------------------------------------------------

def label_mask(y, frac, rng, bias_scores=None):
    """Inspection labels cover only `frac` of consumers, biased toward
    higher consumption (historical inspection-targeting bias)."""
    n = len(y)
    k = int(frac * n)
    p = np.ones(n)
    if bias_scores is not None:
        p = 0.5 + (bias_scores - bias_scores.min()) / np.ptp(bias_scores)
    p /= p.sum()
    ins = rng.choice(n, size=k, replace=False, p=p)
    m = np.zeros(n, dtype=bool)
    m[ins] = True
    return m


def eval_detect(y, p):
    yh = (p >= 0.5).astype(int)
    return dict(prauc=float(average_precision_score(y, p)),
                rocauc=float(roc_auc_score(y, p)),
                f1=float(f1_score(y, yh)),
                brier=float(brier_score_loss(y, p)))


def recall_at_k(y, score, k):
    top = np.argsort(-score)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def energy_at_k(stolen, score, k):
    top = np.argsort(-score)[:k]
    return float(stolen[top].sum() / max(stolen.sum(), 1e-9))


def run_seed(seed, label_fracs=(0.01, 0.05, 0.10, 0.20)):
    rng = np.random.default_rng(seed)
    C = gen_corpus(rng)
    out = {"seed": seed}
    drifted_true = set(range(12, 18)) | set(GRADUAL_FEEDERS)

    # ----- chronological snapshots (graph vs independent features) -----
    fG_tr = window_features(C.load, C.feeder, TRAIN_END, graph=True)
    fG_va = window_features(C.load, C.feeder, VAL_END, graph=True)
    fG_te = window_features(C.load, C.feeder, T_WEEKS, graph=True)
    fN_tr = window_features(C.load, C.feeder, TRAIN_END, graph=False)
    fN_te = window_features(C.load, C.feeder, T_WEEKS, graph=False)
    nbS = build_graph(C.load, C.feeder, TRAIN_END)
    scG = StandardScaler().fit(fG_tr)
    XG_tr, XG_va, XG_te = (scG.transform(fG_tr), scG.transform(fG_va),
                           scG.transform(fG_te))
    scN = StandardScaler().fit(fN_tr)
    XN_tr, XN_te = scN.transform(fN_tr), scN.transform(fN_te)

    lm = {f: label_mask(C.y, f, rng, bias_scores=fG_tr[:, 11])
          for f in label_fracs}

    # ----- SSL encoder pretrained on unlabeled graph features ----------
    t0 = time.time()
    enc = Encoder(XG_tr.shape[1], 64, 16, rng)
    enc.pretrain(XG_tr, nbS)
    t_pre = time.time() - t0

    def dg_repr(Xs):
        z = enc.embed(Xs)
        re = enc.recon_err(Xs)[:, None]
        dev = np.linalg.norm(z - z[nbS].mean(1), axis=1, keepdims=True)
        return np.column_stack([Xs, z, re, dev])   # residual augmentation

    scZ = StandardScaler().fit(dg_repr(XG_tr))
    Ztr = scZ.transform(dg_repr(XG_tr))
    Zva = scZ.transform(dg_repr(XG_va))
    Zte = scZ.transform(dg_repr(XG_te))

    # ----- main comparison at 1% and 5% label budgets ------------------
    m5 = lm[0.05]
    y5 = C.y[m5]
    t_fit = 0.0
    for frac, key in ((0.01, "main_1"), (0.05, "main")):
        mB = lm[frac]
        yB = C.y[mB]
        res_main = {}
        for name, clf in {
            "LR": LogisticRegression(max_iter=3000, class_weight="balanced"),
            "RF": RandomForestClassifier(300, class_weight="balanced",
                                         random_state=seed),
            "GB": GradientBoostingClassifier(random_state=seed),
        }.items():
            clf.fit(XN_tr[mB], yB)
            p = clf.predict_proba(XN_te)[:, 1]
            res_main[name] = eval_detect(C.y, p)
            res_main[name]["r200"] = recall_at_k(C.y, p, 200)
        head_mlp = EnsembleHead(XN_tr.shape[1], rng)
        head_mlp.fit(XN_tr[mB], yB)
        p, _, _ = head_mlp.predict(XN_te)
        res_main["MLP"] = eval_detect(C.y, p)
        res_main["MLP"]["r200"] = recall_at_k(C.y, p, 200)
        head_g = EnsembleHead(XG_tr.shape[1], rng)
        head_g.fit(XG_tr[mB], yB)
        p, _, _ = head_g.predict(XG_te)
        res_main["GraphCtx-sup"] = eval_detect(C.y, p)
        res_main["GraphCtx-sup"]["r200"] = recall_at_k(C.y, p, 200)
        t0 = time.time()
        hd = EnsembleHead(Ztr.shape[1], rng)
        hd.fit(Ztr[mB], yB)
        hd.calibrate(Zva[lm[0.10]], C.y[lm[0.10]])
        t_fit = time.time() - t0
        p_b, _, _ = hd.predict_cal(Zte)
        res_main["DriftGuard"] = eval_detect(C.y, p_b)
        res_main["DriftGuard"]["r200"] = recall_at_k(C.y, p_b, 200)
        out[key] = res_main
        if frac == 0.05:
            head_dg, p_dg = hd, p_b
    res_main = out["main"]
    out["timing"] = {"pretrain_s": t_pre, "fit_s": t_fit}

    # ----- label-efficiency sweep --------------------------------------
    sweep = {}
    for f in label_fracs:
        m = lm[f]
        h = EnsembleHead(Ztr.shape[1], rng, n_members=3)
        h.fit(Ztr[m], C.y[m], epochs=400)
        pz, _, _ = h.predict(Zte)
        g = EnsembleHead(XN_tr.shape[1], rng, n_members=3)
        g.fit(XN_tr[m], C.y[m], epochs=400)
        pg, _, _ = g.predict(XN_te)
        rf = RandomForestClassifier(300, class_weight="balanced",
                                    random_state=seed)
        rf.fit(XN_tr[m], C.y[m])
        pr = rf.predict_proba(XN_te)[:, 1]
        sweep[str(f)] = {"DriftGuard": float(average_precision_score(C.y, pz)),
                         "MLP": float(average_precision_score(C.y, pg)),
                         "RF": float(average_precision_score(C.y, pr))}
    out["sweep"] = sweep

    # ----- localized drift monitoring (biweekly, two channels) ---------
    # Channel L (level): shift of the feeder-mean year-over-year
    #   consumption ratio (seasonality-corrected by construction).
    # Channel S (shape): SSL-embedding mean shift plus reconstruction-
    #   error increase (captures profile/shape changes).
    # Both are standardized per feeder against a drift-free calibration
    # year, and an alarm requires persistence over consecutive checks.
    rec_ref = enc.recon_err(XG_tr)
    Z_ref = enc.embed(XG_tr)
    ref_dec = {f: XG_tr[C.feeder == f, 0].mean() for f in range(N_FEEDERS)}
    cal_months = list(range(68, 117, 4))          # one seasonal cycle

    def channels_at(wk):
        fw = window_features(C.load, C.feeder, wk, graph=True)
        Xw = scG.transform(fw)
        Zw, rw = enc.embed(Xw), enc.recon_err(Xw)
        S = feeder_drift_scores(Z_ref, Zw, rec_ref, rw, C.feeder)
        L = {f: abs(Xw[C.feeder == f, 0].mean() - ref_dec[f])
             for f in range(N_FEEDERS)}
        return L, S

    calL = {f: [] for f in range(N_FEEDERS)}
    calS = {f: [] for f in range(N_FEEDERS)}
    for wk in cal_months:
        L, S = channels_at(wk)
        for f in range(N_FEEDERS):
            calL[f].append(L[f])
            calS[f].append(S[f])
    muL = {f: np.mean(v) for f, v in calL.items()}
    sdL = {f: np.std(v) + 1e-3 for f, v in calL.items()}
    muS = {f: np.mean(v) for f, v in calS.items()}
    sdS = {f: np.std(v) + 1e-3 for f, v in calS.items()}

    checkpoints = list(range(VAL_END + 2, T_WEEKS + 1, 2))
    THR_L, THR_S, PERSIST = 2.0, 3.0, 2
    drift_curve, detect_week = {}, {}
    persist = {f: 0 for f in range(N_FEEDERS)}
    for wk in checkpoints:
        L, S = channels_at(wk)
        zz = {}
        for f in range(N_FEEDERS):
            zL = (L[f] - muL[f]) / sdL[f]
            zS = (S[f] - muS[f]) / sdS[f]
            zz[f] = float(max(zL, zS * THR_L / THR_S))   # common 2.0 scale
        drift_curve[wk] = zz
        for f, v in zz.items():
            if v > THR_L:
                persist[f] += 1
                if persist[f] >= PERSIST and f not in detect_week:
                    detect_week[f] = wk
            else:
                persist[f] = 0
    flagged = set(detect_week.keys())
    tp = len(flagged & drifted_true)
    fp = len(flagged - drifted_true)
    fn = len(drifted_true - flagged)
    loc_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    del_ab = [detect_week[f] - DRIFT_WEEK_ABRUPT
              for f in flagged & drifted_true if f < 18]
    del_gr = [detect_week[f] - DRIFT_WEEK_GRADUAL
              for f in flagged & drifted_true if f >= 18]
    # global baseline: pooled combined score, same calibration protocol
    pooled_cal = []
    for wk in cal_months:
        L, S = channels_at(wk)
        pooled_cal.append(np.mean([max(L[f], S[f] * 0.5)
                                   for f in range(N_FEEDERS)]))
    pm, ps = np.mean(pooled_cal), np.std(pooled_cal) + 1e-3
    g_detect, gp = None, 0
    for wk in checkpoints:
        L, S = channels_at(wk)
        pooled = np.mean([max(L[f], S[f] * 0.5) for f in range(N_FEEDERS)])
        if (pooled - pm) / ps > THR_L:
            gp += 1
            if gp >= PERSIST and g_detect is None:
                g_detect = wk
        else:
            gp = 0
    out["drift"] = {
        "curve": {str(k): {str(f): float(v) for f, v in d.items()}
                  for k, d in drift_curve.items()},
        "detect_week": {str(k): int(v) for k, v in detect_week.items()},
        "loc_f1": float(loc_f1),
        "delay_abrupt_w": float(np.mean(del_ab)) if del_ab else None,
        "delay_gradual_w": float(np.mean(del_gr)) if del_gr else None,
        "false_alarms": int(fp), "missed": int(fn),
        "global_detect_week": g_detect,
        "drifted_feeders": sorted(drifted_true),
    }

    # ----- selective adaptation vs alternatives ------------------------
    m_dr = np.isin(C.feeder, sorted(drifted_true))
    m_stable = ~m_dr
    # the drift alarm triggers a targeted re-inspection batch covering
    # 10% of consumers on flagged feeders
    m_flag_all = np.isin(C.feeder, sorted(flagged))
    cand = np.where(m_flag_all)[0]
    take = rng.choice(cand, size=max(int(0.10 * cand.size), 1), replace=False)
    m_new = np.zeros(N_CUST, dtype=bool)
    m_new[take] = True

    def prf(mask_eval, p):
        return float(average_precision_score(C.y[mask_eval], p[mask_eval]))

    p_static, _, _ = head_dg.predict_cal(Zte)
    pre_stable = prf(m_stable, p_static)
    pre_drift = prf(m_dr, p_static)

    # (a) naive fine-tune on the post-drift batch only
    head_ft = copy.deepcopy(head_dg)
    head_ft.fit(Zte[m_new], C.y[m_new], epochs=250)
    p_ft, _, _ = head_ft.predict_cal(Zte)

    # (b) full retrain on old labels + new batch
    head_rt = EnsembleHead(Ztr.shape[1], rng)
    t0 = time.time()
    head_rt.fit(np.vstack([Ztr[m5], Zte[m_new]]),
                np.concatenate([y5, C.y[m_new]]))
    t_retrain = time.time() - t0
    head_rt.calibrate(Zva[lm[0.10]], C.y[lm[0.10]])
    p_rt, _, _ = head_rt.predict_cal(Zte)

    # (c) DriftGuard selective adaptation: clone of the global head is
    #     fine-tuned only on flagged-feeder data (new inspection batch +
    #     replayed historical labels from those feeders) and applied only
    #     to flagged feeders; stable feeders keep the frozen global head,
    #     so forgetting outside the drifted communities is structurally
    #     impossible.
    t0 = time.time()
    m_flag = m_flag_all
    m_replay = m5 & m_flag
    Zsel = np.vstack([Zte[m_new], Ztr[m_replay]])
    ysel = np.concatenate([C.y[m_new], C.y[m_replay]])
    head_sel = copy.deepcopy(head_dg)
    head_sel.fit(Zsel, ysel, epochs=300)
    head_sel.calibrate(Zsel, ysel)
    t_adapt = time.time() - t0
    p_loc, epi_loc, _ = head_sel.predict_cal(Zte)
    p_base, epi_base, _ = head_dg.predict_cal(Zte)
    # convex fusion of frozen global head and local adapted head on the
    # flagged feeders bounds the downside of a noisy adaptation batch
    p_ad = np.where(m_flag, 0.5 * (p_loc + p_base), p_base)
    epi_ad = np.where(m_flag, 0.5 * (epi_loc + epi_base), epi_base)

    out["adapt"] = {
        "static": {"stable": pre_stable, "drifted": pre_drift},
        "finetune": {"stable": prf(m_stable, p_ft),
                     "drifted": prf(m_dr, p_ft)},
        "retrain": {"stable": prf(m_stable, p_rt),
                    "drifted": prf(m_dr, p_rt), "time_s": t_retrain},
        "selective": {"stable": prf(m_stable, p_ad),
                      "drifted": prf(m_dr, p_ad), "time_s": t_adapt},
    }

    # ----- uncertainty, calibration, abstention ------------------------
    u = (epi_ad - epi_ad.min()) / np.ptp(epi_ad)
    conf = np.maximum(p_ad, 1 - p_ad) - 0.35 * u
    order = np.argsort(-conf)
    order0 = np.argsort(-np.maximum(p_ad, 1 - p_ad))
    yh = (p_ad >= 0.5).astype(int)
    err = (yh != C.y).astype(float)
    cov_grid = np.linspace(0.5, 1.0, 11)
    risk_cov, risk_cov0 = [], []
    for c in cov_grid:
        keep = order[: int(c * len(order))]
        risk_cov.append([float(c), float(err[keep].mean())])
        keep0 = order0[: int(c * len(order0))]
        risk_cov0.append([float(c), float(err[keep0].mean())])
    p_unc = sigmoid(head_dg.logits(Zte).mean(0))
    out["uncertainty"] = {
        "ece_uncal": float(ece(C.y, p_unc)),
        "ece_cal": float(ece(C.y, p_ad)),
        "risk_cov": risk_cov, "risk_cov_naive": risk_cov0,
        "reliab": [[float(a), float(b)] for a, b in
                   zip(*reliability_points(C.y, p_ad))],
        "reliab_uncal": [[float(a), float(b)] for a, b in
                         zip(*reliability_points(C.y, p_unc))],
    }

    # ----- utility-aware inspection ranking ----------------------------
    bmu = C.load[:, T_WEEKS - 52 - WIN:T_WEEKS - 52].mean(1)
    fdec = np.zeros(N_CUST)
    for f in range(N_FEEDERS):
        m = C.feeder == f
        fdec[m] = np.median(fG_te[m, 0])
    E_hat = np.maximum(bmu * fdec - C.load[:, T_WEEKS - WIN:].mean(1),
                       0.0) * (T_WEEKS - VAL_END)
    cost = 1.0 + 0.5 * (C.region == 3)                # rural premium
    S = p_ad * (1 - 0.3 * u) * (E_hat + 1) ** 0.7 / cost
    ks = [50, 100, 200, 400]
    out["ranking"] = {
        "prob_only": {str(k): energy_at_k(C.stolen, p_ad, k) for k in ks},
        "utility": {str(k): energy_at_k(C.stolen, S, k) for k in ks},
        "recall_prob": {str(k): recall_at_k(C.y, p_ad, k) for k in ks},
        "recall_util": {str(k): recall_at_k(C.y, S, k) for k in ks},
    }

    # ----- ablations ----------------------------------------------------
    abl = {"no_ssl": res_main["GraphCtx-sup"]["prauc"],
           "no_adapt_drifted": pre_drift,
           "full_drifted": out["adapt"]["selective"]["drifted"],
           "full": res_main["DriftGuard"]["prauc"],
           "ece_uncal": out["uncertainty"]["ece_uncal"],
           "ece_cal": out["uncertainty"]["ece_cal"],
           "energy200_prob": out["ranking"]["prob_only"]["200"],
           "energy200_util": out["ranking"]["utility"]["200"]}
    # w/o graph: same architecture on independent features
    enc2 = Encoder(XN_tr.shape[1], 64, 16, np.random.default_rng(seed + 1))
    enc2.pretrain(XN_tr, nbS, epochs=220)
    R2tr = np.column_stack([XN_tr, enc2.embed(XN_tr),
                            enc2.recon_err(XN_tr)[:, None]])
    R2te = np.column_stack([XN_te, enc2.embed(XN_te),
                            enc2.recon_err(XN_te)[:, None]])
    h2 = EnsembleHead(R2tr.shape[1], rng, n_members=3)
    h2.fit(R2tr[m5], y5, epochs=400)
    p2, _, _ = h2.predict(R2te)
    abl["no_graph"] = float(average_precision_score(C.y, p2))
    out["ablation"] = abl
    return out


if __name__ == "__main__":
    all_out = []
    for s in RNG_SEEDS:
        print(f"seed {s} ...", flush=True)
        t0 = time.time()
        all_out.append(run_seed(s))
        print(f"  done in {time.time() - t0:.1f}s")
    with open("/home/claude/driftguard/results/results.json", "w") as f:
        json.dump(all_out, f, indent=1)
    print("wrote results.json")
