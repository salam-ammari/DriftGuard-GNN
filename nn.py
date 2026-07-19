"""Minimal, numerically careful numpy neural-network core used by
DriftGuard-GNN. Dense ReLU networks with Adam and gradient clipping."""

import numpy as np


class Dense:
    def __init__(self, d_in, d_out, rng, act="relu"):
        self.W = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, d_out))
        self.b = np.zeros(d_out)
        self.act = act
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        if self.act == "relu":
            self.h = np.maximum(self.z, 0)
        else:
            self.h = self.z
        return self.h

    def backward(self, dh):
        dz = dh * (self.z > 0) if self.act == "relu" else dh
        self.dW = self.x.T @ dz
        self.db = dz.sum(0)
        return dz @ self.W.T

    def step(self, lr, t, clip=5.0, wd=1e-4):
        self.dW = self.dW + wd * self.W
        for g in (self.dW, self.db):
            n = np.linalg.norm(g)
            if n > clip:
                g *= clip / n
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.mW = b1 * self.mW + (1 - b1) * self.dW
        self.vW = b2 * self.vW + (1 - b2) * self.dW ** 2
        self.mb = b1 * self.mb + (1 - b1) * self.db
        self.vb = b2 * self.vb + (1 - b2) * self.db ** 2
        self.W -= lr * (self.mW / (1 - b1 ** t)) / (np.sqrt(self.vW / (1 - b2 ** t)) + eps)
        self.b -= lr * (self.mb / (1 - b1 ** t)) / (np.sqrt(self.vb / (1 - b2 ** t)) + eps)


class Net:
    def __init__(self, dims, rng, last_act="linear"):
        self.layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            act = "relu" if i < len(dims) - 2 else last_act
            self.layers.append(Dense(a, b, rng, act))

    def forward(self, x):
        for L in self.layers:
            x = L.forward(x)
        return x

    def backward(self, dout):
        for L in reversed(self.layers):
            dout = L.backward(dout)
        return dout

    def step(self, lr, t):
        for L in self.layers:
            L.step(lr, t)

    def clone_arch(self, rng):
        dims = [self.layers[0].W.shape[0]] + [L.W.shape[1] for L in self.layers]
        return Net(dims, rng, self.layers[-1].act)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
