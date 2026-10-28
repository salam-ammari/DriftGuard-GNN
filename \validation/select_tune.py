import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '..')
import numpy as np, sim
from sim import SCENARIOS, TRAIN_END, T_WEEKS, WIN
from features import window_features, knn_shape_within
from gnn import RGNNEncoder, knn_adjacency, group_adjacency
from driftguard import EnsembleHead, representation, prauc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import driftguard
cfgs = {'h32_e400': (32, 400), 'h16_e400': (16, 400), 'h16_e200': (16, 200), 'h8_e300': (8, 300)}
res = {k: {0.01: [], 0.05: []} for k in list(cfgs) + ['LR_R', 'LR_X']}
for seed in (101, 102, 103):
    rng = np.random.default_rng(seed)
    C = sim.gen_corpus(rng, SCENARIOS['combined']); N = len(C.y)
    ctx = ('group', C.feeder)
    A_sim = knn_adjacency(knn_shape_within(C.load, TRAIN_END, C.feeder, 8), N)
    rels = [group_adjacency(C.feeder, rng, 16), A_sim]
    F = {t: window_features(C.load, t, ctx) for t in (TRAIN_END, T_WEEKS)}
    sc = StandardScaler().fit(F[TRAIN_END]); X = {t: sc.transform(F[t]) for t in F}
    enc = RGNNEncoder(12, 32, 16, 2, rng); enc.pretrain(X[TRAIN_END], rels, A_sim)
    Rr = {t: representation(enc, X[t], rels, A_sim)[0] for t in X}
    scR = StandardScaler().fit(Rr[TRAIN_END]); R = {t: scR.transform(Rr[t]) for t in Rr}
    bias = np.log1p(C.load[:, TRAIN_END - WIN:TRAIN_END].mean(1)); p = 0.5 + (bias - bias.min()) / np.ptp(bias)
    order = rng.choice(N, size=int(0.2 * N), replace=False, p=p / p.sum())
    ev = np.ones(N, bool); ev[order] = False
    for f in (0.01, 0.05):
        m = np.zeros(N, bool); m[order[:int(round(f * N))]] = True
        for k, (hid, ep) in cfgs.items():
            h = EnsembleHead(R[TRAIN_END].shape[1], np.random.default_rng(seed), hidden=hid)
            h.fit(R[TRAIN_END][m], C.y[m], epochs=ep)
            res[k][f].append(prauc(C.y[ev], h.predict(R[T_WEEKS])[0][ev]))
        for nm, Z in (('LR_R', R), ('LR_X', X)):
            lr = LogisticRegression(max_iter=3000, class_weight='balanced').fit(Z[TRAIN_END][m], C.y[m])
            res[nm][f].append(prauc(C.y[ev], lr.predict_proba(Z[T_WEEKS])[:, 1][ev]))
        print(seed, f, 'positives', C.y[m].sum(), flush=True)
for k, v in res.items():
    print(f'{k:10s}', {f: (np.round(x, 3).tolist(), round(np.mean(x), 3)) for f, x in v.items()})
