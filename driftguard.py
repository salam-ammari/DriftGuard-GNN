"""DriftGuard-GNN components: calibrated ensemble head, representation
builder, drift-monitor channels, and evaluation helpers."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from nn import Net, sigmoid


# ---------------------------------------------------------------- head
class EnsembleHead:
    """Deep ensemble (M members, 2-layer ReLU nets) with cross-fitted
    Platt scaling: out-of-fold logits on the labelled budget itself are
    used to fit the slope/intercept, so no extra labels are consumed."""

    def __init__(self, d_in, rng, n_members=5, hidden=16):
        self.rng = rng
        self.d_in, self.M, self.hidden = d_in, n_members, hidden
        self.members = self._new_members()
        self.a, self.b = 1.0, 0.0

    def _new_members(self):
        return [Net([self.d_in, self.hidden, 1],
                    np.random.default_rng(self.rng.integers(1 << 30)))
                for _ in range(self.M)]

    @staticmethod
    def _train(members, Z, y, epochs, lr, rng):
        pos_w = (len(y) - y.sum()) / max(y.sum(), 1)
        w = np.where(y == 1, pos_w, 1.0)
        for m in members:
            keep = rng.random(len(y)) < 0.9
            if y[keep].sum() == 0:
                keep[:] = True
            Zb, yb, wb = Z[keep], y[keep], w[keep]
            for ep in range(1, epochs + 1):
                p = sigmoid(m.forward(Zb)[:, 0])
                m.backward(((p - yb) * wb / len(yb))[:, None])
                m.step(lr, ep)

    def fit(self, Z, y, epochs=200, lr=3e-3, calibrate=True, folds=3):
        self._train(self.members, Z, y, epochs, lr, self.rng)
        if calibrate:
            self.cross_fit_platt(Z, y, epochs, lr, folds)

    def cross_fit_platt(self, Z, y, epochs=200, lr=3e-3, folds=3):
        oof = np.zeros(len(y))
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        self.rng.shuffle(pos)
        self.rng.shuffle(neg)
        fold = np.zeros(len(y), dtype=int)
        fold[pos] = np.arange(pos.size) % folds
        fold[neg] = np.arange(neg.size) % folds
        for k in range(folds):
            tr, te = fold != k, fold == k
            if y[tr].sum() == 0 or te.sum() == 0:
                continue
            mem = [Net([self.d_in, self.hidden, 1],
                       np.random.default_rng(self.rng.integers(1 << 30)))
                   for _ in range(2)]
            self._train(mem, Z[tr], y[tr], epochs, lr, self.rng)
            oof[te] = np.mean([m.forward(Z[te])[:, 0] for m in mem], axis=0)
        self.fit_platt(oof, y)

    def fit_platt(self, L, y):
        a, b = 1.0, 0.0
        prior = (y.sum() + 1) / (len(y) + 2)            # Platt's smoothed targets
        t = np.where(y == 1, 1 - 1 / (y.sum() + 2), 1 / (len(y) - y.sum() + 2))
        for _ in range(800):
            p = sigmoid(a * L + b)
            a -= 0.3 * ((p - t) * L).mean()
            b -= 0.3 * (p - t).mean()
            a = max(a, 0.05)            # monotone: never invert the ranking
        self.a, self.b = a, b
        self.prior = prior

    def logits(self, Z):
        return np.stack([m.forward(Z)[:, 0] for m in self.members])

    def predict(self, Z, calibrated=True):
        L = self.logits(Z)
        P = sigmoid(self.a * L + self.b) if calibrated else sigmoid(L)
        return P.mean(0), P.std(0)

    def finetune(self, Z, y, epochs=150, lr=3e-3):
        self._train(self.members, Z, y, epochs, lr, self.rng)


def n_params_head(h):
    return sum(L.W.size + L.b.size for m in h.members for L in m.layers)


# ------------------------------------------------------ representation
def representation(enc, X, rels, A_sim):
    """r_v = [x_v, z_v, e_v, delta_v]  (residual augmentation)."""
    Z = enc.embed(X, rels)
    e = ((enc.dec.forward(Z) - X) ** 2).mean(1, keepdims=True)
    d = np.linalg.norm(Z - A_sim @ Z, axis=1, keepdims=True)
    return np.column_stack([X, Z, e, d]), Z, e[:, 0]


# ----------------------------------------------------- drift channels
def group_stats(Xstd, Z, e, groups):
    """Robust per-group statistics: median standardized YoY ratio (level),
    coordinate-wise median embedding (shape) and median reconstruction
    error.  Medians move under collective legitimate drift but are
    insensitive to the minority of individually decaying consumers."""
    out = {}
    for g in np.unique(groups):
        m = groups == g
        out[int(g)] = (float(np.median(Xstd[m, 0])), np.median(Z[m], 0),
                       float(np.median(e[m])))
    return out


class DriftMonitor:
    """Two-channel per-group monitor standardized against the empirical
    distribution of each statistic over a drift-free calibration period."""

    def __init__(self, calib_stats):
        self.g = sorted(calib_stats[0].keys())
        self.cal = {}
        for g in self.g:
            lv = np.array([c[g][0] for c in calib_stats])
            zv = np.array([c[g][1] for c in calib_stats])
            rv = np.array([c[g][2] for c in calib_stats])
            zc = zv.mean(0)
            dz = np.linalg.norm(zv - zc, axis=1)
            self.cal[g] = (lv.mean(), lv.std() + 1e-3, zc, dz.mean(),
                           dz.std() + 1e-3, rv.mean(), rv.std() + 1e-3)

    def score(self, stats):
        z = {}
        for g in self.g:
            lm, ls, zc, dm, ds, rm, rs = self.cal[g]
            lvl, zv, rv = stats[g]
            zL = abs(lvl - lm) / ls
            zS = 0.6 * (np.linalg.norm(zv - zc) - dm) / ds + 0.4 * max((rv - rm) / rs, 0)
            z[g] = (float(zL), float(zS))
        return z


def audit_z(obs, exp, var):
    return (obs - exp) / np.sqrt(var + 1e-6)


# ------------------------------------------------------------ metrics
def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 5:
            out.append([float(p[m].mean()), float(y[m].mean())])
    return out


def prauc(y, p):
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def rocauc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def topk(score, k):
    return np.argsort(-score)[:k]


def label_mask(n, frac, rng, bias, exclude=None):
    """Biased inspection sample: probability rises with consumption."""
    p = 0.5 + (bias - bias.min()) / (np.ptp(bias) + 1e-12)
    if exclude is not None:
        p = p * (~exclude)
    p = p / p.sum()
    k = max(int(round(frac * n)), 2)
    idx = rng.choice(n, size=k, replace=False, p=p)
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m
