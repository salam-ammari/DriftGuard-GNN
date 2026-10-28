"""Windowed consumption descriptors with three context modes.

context = ("group", ids)  -> contextualize against the median of the
                             consumer's physical group (feeder topology);
context = ("knn", nbrs)   -> contextualize against the median of its
                             load-shape similarity neighbours (no topology
                             needed; used for SGCC and the topology-free
                             variant);
context = None            -> independent consumers (population medians).
"""

import numpy as np

N_FEAT = 12
FEAT_NAMES = ["yoy", "rel_yoy", "rel_level", "cv", "ac1", "slope", "lowq",
              "zero_frac", "half_decay", "diffE", "corr_ctx", "log_mu"]


def _ctx_median(v, context):
    if context is None:
        return np.full_like(v, np.median(v), dtype=float)
    kind, obj = context
    if kind == "group":
        out = np.empty_like(v, dtype=float)
        for g in np.unique(obj):
            m = obj == g
            out[m] = np.median(v[m])
        return out
    return np.median(v[obj], axis=1)            # knn neighbour lists


def _ctx_profile(X, context):
    if context is None:
        return np.broadcast_to(np.median(X, axis=0), X.shape)
    kind, obj = context
    if kind == "group":
        P = np.empty_like(X)
        for g in np.unique(obj):
            m = obj == g
            P[m] = np.median(X[m], axis=0)
        return P
    P = np.empty_like(X)
    for s in range(0, X.shape[0], 4096):        # chunked for memory
        P[s:s + 4096] = np.median(X[obj[s:s + 4096]], axis=1)
    return P


def window_features(load, t_end, context, win=24, lag=52):
    X = load[:, t_end - win:t_end]
    B = load[:, t_end - lag - win:t_end - lag] if t_end - lag - win >= 0 \
        else load[:, :win]
    mu, sd = X.mean(1), X.std(1) + 1e-8
    bmu = B.mean(1) + 1e-8
    yoy = mu / bmu
    rel_level = mu / (_ctx_median(mu, context) + 1e-8)
    rel_yoy = yoy / (_ctx_median(yoy, context) + 1e-8)
    x1, x2 = X[:, :-1], X[:, 1:]
    ac1 = ((x1 - x1.mean(1, keepdims=True)) * (x2 - x2.mean(1, keepdims=True))
           ).mean(1) / (x1.std(1) * x2.std(1) + 1e-8)
    slope = np.polyfit(np.arange(win), X.T, 1)[0] / (bmu / win)
    lowq = np.quantile(X, 0.1, axis=1) / bmu
    zero_frac = (X < 0.15 * bmu[:, None]).mean(1)
    h = win // 2
    half_decay = X[:, h:].mean(1) / (X[:, :h].mean(1) + 1e-8)
    cv = sd / (mu + 1e-8)
    diffE = (np.diff(X, axis=1) ** 2).mean(1) / sd ** 2
    P = _ctx_profile(X, context)
    xc = X - X.mean(1, keepdims=True)
    pc = P - P.mean(1, keepdims=True)
    corr = (xc * pc).mean(1) / (X.std(1) * P.std(1) + 1e-8)
    F = np.column_stack([yoy, rel_yoy, rel_level, cv, ac1, slope, lowq,
                         zero_frac, half_decay, diffE, corr, np.log1p(mu)])
    return np.nan_to_num(F, nan=0.0, posinf=10.0, neginf=-10.0)


def knn_shape(load, t_end, k, win=52, chunk=2048):
    """kNN over z-normalized load shapes of the `win` weeks before t_end
    (leakage-safe: no data after t_end).  Cosine similarity, chunked."""
    X = load[:, max(0, t_end - win):t_end].astype(np.float32)
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-6
    n = Z.shape[0]
    nb = np.zeros((n, k), dtype=np.int64)
    for s in range(0, n, chunk):
        S = Z[s:s + chunk] @ Z.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        part = np.argpartition(-S, k, axis=1)[:, :k]
        nb[s:s + chunk] = part
    return nb


def knn_shape_within(load, t_end, groups, k, win=52):
    """kNN restricted to the same physical group (simulator)."""
    X = load[:, max(0, t_end - win):t_end]
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    nb = np.zeros((len(groups), k), dtype=np.int64)
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        S = Z[m] @ Z[m].T
        np.fill_diagonal(S, -np.inf)
        nb[m] = m[np.argsort(-S, axis=1)[:, :k]]
    return nb
