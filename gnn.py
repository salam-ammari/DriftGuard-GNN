"""Trainable relational message-passing layers (R-SAGE) for DriftGuard-GNN.

Each layer implements, for every relation r with row-normalized sparse
adjacency A_r,

    H' = act( H W_self + sum_r (A_r H) W_r + b )

which is the relational GraphSAGE update with mean aggregation.  Gradients
are propagated through the sparse aggregation (dH = dZ W_self^T +
sum_r A_r^T dZ W_r^T), so stacked layers train end-to-end.  Pure
NumPy/SciPy, full-batch.
"""

import numpy as np
import scipy.sparse as sp

from nn import Net, sigmoid


def row_normalize(A):
    A = sp.csr_matrix(A, dtype=np.float64)
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    return sp.diags(1.0 / d) @ A


def knn_adjacency(nbrs, n):
    rows = np.repeat(np.arange(n), nbrs.shape[1])
    return row_normalize(sp.csr_matrix((np.ones(rows.size), (rows, nbrs.ravel())),
                                       shape=(n, n)))


def group_adjacency(groups, rng, n_samp=16):
    """Physical relation: each node linked to n_samp sampled nodes of its
    own group (feeder).  Sampling keeps the operator O(N * n_samp)."""
    n = len(groups)
    rows, cols = [], []
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        for i in m:
            cand = m[m != i]
            pick = rng.choice(cand, size=min(n_samp, cand.size), replace=False)
            rows.extend([i] * pick.size)
            cols.extend(pick.tolist())
    return row_normalize(sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                       shape=(n, n)))


class Adam:
    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    def step(self, p, g, lr, t, wd=1e-4, clip=5.0):
        g = g + wd * p
        n = np.linalg.norm(g)
        if n > clip:
            g = g * (clip / n)
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        p -= lr * (self.m / (1 - 0.9 ** t)) / (np.sqrt(self.v / (1 - 0.999 ** t)) + 1e-8)


class RSAGELayer:
    def __init__(self, d_in, d_out, n_rel, rng, act="relu"):
        s = np.sqrt(2.0 / (d_in * (n_rel + 1)))
        self.W = [rng.normal(0, s, (d_in, d_out)) for _ in range(n_rel + 1)]
        self.b = np.zeros(d_out)
        self.act = act
        self.opt = [Adam(w.shape) for w in self.W] + [Adam(self.b.shape)]

    def forward(self, H, rels):
        self.H = H
        self.AH = [A @ H for A in rels]
        Z = H @ self.W[0] + self.b
        for AH, W in zip(self.AH, self.W[1:]):
            Z = Z + AH @ W
        self.Z = Z
        return np.maximum(Z, 0) if self.act == "relu" else Z

    def backward(self, dH, rels):
        dZ = dH * (self.Z > 0) if self.act == "relu" else dH
        self.gW = [self.H.T @ dZ] + [AH.T @ dZ for AH in self.AH]
        self.gb = dZ.sum(0)
        dIn = dZ @ self.W[0].T
        for A, W in zip(rels, self.W[1:]):
            dIn = dIn + A.T @ (dZ @ W.T)
        return dIn

    def step(self, lr, t):
        for W, g, o in zip(self.W, self.gW, self.opt):
            o.step(W, g, lr, t)
        self.opt[-1].step(self.b, self.gb, lr, t, wd=0.0)

    def n_params(self):
        return sum(w.size for w in self.W) + self.b.size


class RGNNEncoder:
    """Two-layer R-SAGE encoder pretrained with masked feature
    reconstruction plus neighborhood consistency (Eq. 1 of the paper)."""

    def __init__(self, d_in, d_hid, d_emb, n_rel, rng):
        self.l1 = RSAGELayer(d_in, d_hid, n_rel, rng, "relu")
        self.l2 = RSAGELayer(d_hid, d_emb, n_rel, rng, "linear")
        self.dec = Net([d_emb, d_hid, d_in], rng)
        self.rng = rng

    def encode(self, X, rels):
        return self.l2.forward(self.l1.forward(X, rels), rels)

    def pretrain(self, X, rels, A_cons, epochs=300, lam=0.1, mask_p=0.3,
                 lr=3e-3):
        n = X.shape[0]
        for ep in range(1, epochs + 1):
            M = (self.rng.random(X.shape) < mask_p).astype(float)
            Xin = X * (1.0 - M)
            Z = self.encode(Xin, rels)
            Zn = A_cons @ Z                      # stop-gradient target
            Xr = self.dec.forward(Z)
            dXr = 2.0 * (Xr - X) * M / n
            dZ = self.dec.backward(dXr) + lam * 2.0 * (Z - Zn) / n
            dH1 = self.l2.backward(dZ, rels)
            self.l1.backward(dH1, rels)
            self.l2.step(lr, ep)
            self.l1.step(lr, ep)
            self.dec.step(lr, ep)

    def embed(self, X, rels):
        return self.encode(X, rels)

    def recon_err(self, X, rels):
        Z = self.encode(X, rels)
        return ((self.dec.forward(Z) - X) ** 2).mean(1)

    def n_params(self):
        dec = sum(L.W.size + L.b.size for L in self.dec.layers)
        return self.l1.n_params() + self.l2.n_params() + dec


class RGNNClassifier:
    """End-to-end supervised two-layer R-SAGE node classifier (transductive,
    class-weighted BCE on labeled nodes).  Used as the supervised GNN
    baseline representing graph-based detectors from the literature."""

    def __init__(self, d_in, d_hid, n_rel, rng):
        self.l1 = RSAGELayer(d_in, d_hid, n_rel, rng, "relu")
        self.l2 = RSAGELayer(d_hid, 1, n_rel, rng, "linear")

    def fit(self, X, rels, lab_idx, y, epochs=400, lr=5e-3):
        pos_w = (len(y) - y.sum()) / max(y.sum(), 1)
        w = np.where(y == 1, pos_w, 1.0)
        n_l = len(lab_idx)
        for ep in range(1, epochs + 1):
            logit = self.l2.forward(self.l1.forward(X, rels), rels)[:, 0]
            p = sigmoid(logit[lab_idx])
            d = np.zeros((X.shape[0], 1))
            d[lab_idx, 0] = (p - y) * w / n_l
            dH = self.l2.backward(d, rels)
            self.l1.backward(dH, rels)
            self.l2.step(lr, ep)
            self.l1.step(lr, ep)

    def predict(self, X, rels):
        return sigmoid(self.l2.forward(self.l1.forward(X, rels), rels)[:, 0])


def grad_check(seed=0):
    """Finite-difference check of the sparse-aggregation backward pass."""
    rng = np.random.default_rng(seed)
    n, d = 30, 5
    X = rng.normal(size=(n, d))
    A = knn_adjacency(rng.integers(0, n, (n, 4)), n)
    L1 = RSAGELayer(d, 4, 1, rng, "relu")
    L2 = RSAGELayer(4, 3, 1, rng, "linear")
    T = rng.normal(size=(n, 3))

    def loss():
        return 0.5 * ((L2.forward(L1.forward(X, [A]), [A]) - T) ** 2).sum()

    loss()
    dH = L2.backward(L2.Z - T, [A])
    L1.backward(dH, [A])
    num, ana = [], []
    for W, g in ((L1.W[1], L1.gW[1]), (L1.W[0], L1.gW[0])):
        for _ in range(5):
            i, j = rng.integers(W.shape[0]), rng.integers(W.shape[1])
            old = W[i, j]
            W[i, j] = old + 1e-6; lp = loss()
            W[i, j] = old - 1e-6; lm = loss()
            W[i, j] = old
            num.append((lp - lm) / 2e-6)
            ana.append(g[i, j])
    return float(np.max(np.abs(np.array(num) - np.array(ana))))


if __name__ == "__main__":
    print("max |numeric - analytic| gradient:", grad_check())
