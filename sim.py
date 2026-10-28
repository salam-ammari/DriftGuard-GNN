"""LatAm-NTL-Sim v2: scenario-controlled smart-meter testbed.

Revision changes (reviewer H, comment 3):
  * residential load anchored to the published Brazilian mean residential
    consumption per consumer unit (BASE_MONTHLY_KWH, see paper Sec. V-A),
    with regional factors and a summer-peaking seasonal profile;
  * each drift mechanism can be switched on alone (scenario flags);
  * the adversary switch (concept drift) is decoupled from the tariff shock
    in both time and region;
  * an assumption-violating scenario adds INDIVIDUAL legitimate changes
    (e.g. rooftop PV, household-size change) among honest consumers, so
    "legitimate drift is collective" no longer holds by construction.
"""

from dataclasses import dataclass, field
import numpy as np

N_REGIONS = 4
FEEDERS_PER_REGION = 6
CUST_PER_FEEDER = 150
N_FEEDERS = N_REGIONS * FEEDERS_PER_REGION            # 24
N_CUST = N_FEEDERS * CUST_PER_FEEDER                  # 3600
T_WEEKS = 156
TRAIN_END, VAL_END = 104, 116
WIN = 24

# Anchors (set from EPE statistics; see paper).  Weekly = monthly*12/52.
BASE_MONTHLY_KWH = 179.1    # EPE Anuario 2025 (base 2024), national mean
# regional means span 136.2-220.3 kWh/month (EPE factsheet) -> factors
REGION_FACTOR = np.array([136.2, 183.8, 208.0, 220.3]) / 179.1
SEAS_AMP = np.array([0.08, 0.12, 0.15, 0.10])  # summer peak amplitude
FRAUD_RATE = 0.085                             # SGCC empirical prevalence

COV_REGION, COV_WEEK = 2, 124                  # abrupt tariff shock
GRAD_FEEDERS, GRAD_WEEK = (18, 19, 20), 128    # partial economic contraction
CON_REGION, CON_WEEK = 1, 136                  # adversary switch (decoupled)


@dataclass
class Scenario:
    covariate: bool = False
    gradual: bool = False
    concept: bool = False
    individual: bool = False
    shock: float = 0.12                        # tariff-shock level drop
    grad_rate: float = 0.012                   # weekly contraction
    indiv_frac: float = 0.06                   # honest consumers w/ own change
    name: str = "custom"


SCENARIOS = {
    "none": Scenario(name="none"),
    "covariate": Scenario(covariate=True, name="covariate"),
    "gradual": Scenario(gradual=True, name="gradual"),
    "concept": Scenario(concept=True, name="concept"),
    "combined": Scenario(covariate=True, gradual=True, concept=True,
                         name="combined"),
    "individual": Scenario(covariate=True, gradual=True, concept=True,
                           individual=True, name="individual"),
}


@dataclass
class Corpus:
    load: np.ndarray
    true: np.ndarray
    y: np.ndarray
    stolen: np.ndarray
    feeder: np.ndarray
    region: np.ndarray
    fraud_type: np.ndarray
    truth: dict = field(default_factory=dict)  # drifted feeders per type


def gen_corpus(rng, sc: Scenario, n_per_feeder=CUST_PER_FEEDER) -> Corpus:
    n_cust = N_FEEDERS * n_per_feeder
    t = np.arange(T_WEEKS)
    region = np.repeat(np.arange(N_REGIONS), FEEDERS_PER_REGION * n_per_feeder)
    feeder = np.repeat(np.arange(N_FEEDERS), n_per_feeder)

    base_w = BASE_MONTHLY_KWH * 12.0 / 52.0 * REGION_FACTOR[region]
    amp = SEAS_AMP[region]
    # summer (southern hemisphere) peak around week 6 (mid-February)
    seas = 1.0 + amp[:, None] * np.cos(2 * np.pi * (t[None, :] - 6) / 52.0)
    cust = rng.lognormal(-0.5 * 0.45 ** 2, 0.45, n_cust)       # mean-one
    trend = 1.0 + 0.0008 * t[None, :] * rng.normal(1.0, 0.6, (n_cust, 1))
    noise = 0.08 + 0.05 * rng.random(n_cust)
    true = (base_w * cust)[:, None] * seas * trend
    true *= rng.normal(1.0, noise[:, None], (n_cust, T_WEEKS))
    true = np.clip(true, 0.5, None)

    truth = {}
    if sc.covariate:
        m = region == COV_REGION
        true[m, COV_WEEK:] *= (1.0 - sc.shock)
        shape = 1.0 + 0.10 * np.sin(2 * np.pi * t / 13.0)    # demand response
        true[m, COV_WEEK:] *= shape[COV_WEEK:]
        truth["covariate"] = sorted(np.unique(feeder[m]).tolist())
    if sc.gradual:
        m = np.isin(feeder, GRAD_FEEDERS)
        ramp = np.ones(T_WEEKS)
        w = np.arange(GRAD_WEEK, T_WEEKS)
        ramp[w] = 1.0 - sc.grad_rate * (w - GRAD_WEEK)
        true[m] *= ramp[None, :]
        truth["gradual"] = list(GRAD_FEEDERS)

    y = (rng.random(n_cust) < FRAUD_RATE).astype(int)
    if sc.individual:
        hon = np.where(y == 0)[0]
        pick = rng.choice(hon, int(sc.indiv_frac * hon.size), replace=False)
        for i in pick:                                       # PV / household
            s = rng.integers(60, 150)
            true[i, s:] *= rng.uniform(0.5, 0.8)
        truth["individual_ids"] = pick.tolist()

    load = true.copy()
    ftype = np.zeros(n_cust, dtype=int)
    idx = np.where(y == 1)[0]
    ftype[idx] = rng.integers(1, 5, idx.size)
    onset = rng.integers(30, 120, n_cust)
    for i in idx:
        s = onset[i]
        if ftype[i] == 1:                                     # under-report
            load[i, s:] = true[i, s:] * rng.uniform(0.35, 0.70)
        elif ftype[i] == 2:                                   # bypass
            wk = rng.random(T_WEEKS - s) < 0.35
            load[i, s:][wk] = true[i, s:][wk] * rng.uniform(0.05, 0.20)
        elif ftype[i] == 3:                                   # ramp-down
            load[i, s:] = true[i, s:] * np.linspace(1.0, 0.30, T_WEEKS - s)
        else:                                                 # saturation
            cap = np.quantile(true[i, :s], 0.35)
            load[i, s:] = np.minimum(true[i, s:], cap)
    if sc.concept:                       # post-enforcement subtle skim
        m = idx[region[idx] == CON_REGION]
        for i in m:
            load[i, CON_WEEK:] = true[i, CON_WEEK:] * rng.uniform(0.60, 0.80)
        truth["concept"] = sorted(np.unique(feeder[region == CON_REGION]).tolist())
    stolen = (true - load)[:, VAL_END:].sum(axis=1)
    return Corpus(load, true, y, stolen, feeder, region, ftype, truth)
