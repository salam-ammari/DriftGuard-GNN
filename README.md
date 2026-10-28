# DriftGuard-GNN

**Research code and reproducibility artifacts** for *DriftGuard-GNN: Self-Supervised, Drift-Adaptive, and Uncertainty-Aware Graph Learning for Non-Technical Loss Detection in Latin American Smart Grids*.

DriftGuard-GNN studies non-technical loss (NTL) detection with scarce inspection labels, changing consumption behavior, and limited inspection budgets. Its components include relational graph learning, masked-feature self-supervision, group-level drift monitoring, selective retraining, ensemble calibration, and inspection ranking.

This repository now includes the **modular implementation, saved experimental results, validation traces, and external SGCC detection-validation outputs**.

## Implementation

| File | Purpose |
|---|---|
| `sim.py` | Synthetic smart-grid generator and drift scenarios |
| `features.py` | Consumption and community-context features |
| `gnn.py` | Relational GraphSAGE encoder, supervised baseline, gradient check |
| `driftguard.py` | Ensemble head, calibration, drift monitoring, metrics |
| `exp_sim.py` | Synthetic detection, drift, adaptation and ranking experiments |
| `exp_sgcc.py` | SGCC out-of-time, customer-disjoint detection validation |
| `aggregate.py` | Aggregate experiment outputs into `results/summary.json` |
| `make_figures.py`, `graphical_abstract.py`, `make_video.py` | Visualizations |
| `run_all.sh` | Batch execution over simulation scenarios and test seeds |
| `results/` | Per-seed JSONs, validation traces, aggregated results, SGCC outputs |
| `\\validation/` | Validation directory as currently named on GitHub |

**Repository setup warning:** `gnn.py` and `driftguard.py` import `nn.py`, but that file is missing from the current GitHub root. It exists in the accompanying source archive: upload it to the repository before running the implementation. GitHub currently names the validation folder `\\validation` (leading backslash), whereas the archive uses `validation/`. Normalize the directory name and check that `select_traces.py` and `select_tune.py` have been uploaded.

## Requirements

Python 3.10+; NumPy, SciPy, scikit-learn, pandas, and Matplotlib.

```bash
python -m pip install numpy scipy scikit-learn pandas matplotlib
```

## Reproduce the experiments

The code archive originally stored the Python files under `code/` and outputs under sibling `results/`. The published GitHub files currently sit at the repository root, but several scripts still write to `../results/` relative to the *working directory*. Either restore the archive's `code/` layout (recommended), or adjust the paths in the scripts. From a `code/` directory containing all the source files, including `nn.py`:

```bash
python gnn.py
bash run_all.sh
python aggregate.py
```

Run a specific configuration:

```bash
python exp_sim.py combined topology 11 23 37 51 73
```

Syntax: `python exp_sim.py <scenario> <variant> <seed> [<seed> ...]`.

Scenarios: `none`, `covariate`, `gradual`, `concept`, `combined`, `individual`. Variants: `topology`, `similarity`; the published topology-free experiment uses `combined similarity`. The batch script evaluates five test seeds **11, 23, 37, 51, 73**.

### SGCC external validation

The SGCC daily-consumption dataset is **not redistributed**. Obtain it separately and provide a CSV with `CONS_NO`, `FLAG`, and daily reading columns formatted `YYYY/MM/DD`.

From the same `code/` working directory:

```bash
python exp_sgcc.py /path/to/SGCC_data.csv 11 23 37 51 73
python aggregate.py
```

The SGCC procedure uses a customer-disjoint, stratified 70/30 split and later evaluation windows than the training features. Because feeder topology is unavailable, it builds a consumption-similarity graph. **SGCC validates the detection stage, not the full online drift-adaptation pipeline.**

### Validation and tuning

Validation-only seeds **101, 102, 103** are separate from the five test seeds. The provided validation scripts (`select_traces.py` and `select_tune.py` in the archive) support alarm-threshold and ensemble-head selection. The main implementation uses an alarm threshold of **5.0**, persistence **2** checkpoints, and a **16-hidden-unit, 200-epoch** ensemble head. Saved monitor traces are under `results/trace_*.json`.

## Results

The repository provides full per-seed and aggregated outputs:

- [Aggregated results](results/summary.json) — detection, drift, adaptation, uncertainty calibration, inspection ranking, timing, and SGCC evaluation.
- [Text summary](results/aggregate_all.txt) — printable experiment summaries.
- [SGCC results](results/sgcc.json) — external detection validation.
- [Simulation files](results/) — `sim_<scenario>_<variant>_<seed>.json` and `trace_*.json`.

Selected **DriftGuard-GNN** results from the saved five-seed summary (mean ± standard deviation):

| Dataset / setting | Label budget | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| Synthetic combined drift, topology graph | 5% | 0.569 ± 0.025 | 0.794 ± 0.033 |
| SGCC detection validation, similarity graph | 5% | 0.192 ± 0.014 | 0.668 ± 0.015 |

These figures come from **different datasets and protocols**; they should not be interpreted as directly comparable. See `summary.json` for baselines and other metrics. The results are archived research outputs and have not been independently rerun for this documentation update.

## Scope and limitations

The synthetic experimental testbed has **3,600 consumers, 24 feeders, four regions, and 156 weeks**. It is not a real Latin American utility dataset. Real-data validation is limited to the SGCC detection benchmark; no real-world end-to-end deployment or external drift-adaptation validation is claimed. Running the entire batch can require substantial compute resources.

## Citation

Please cite the accompanying manuscript: *DriftGuard-GNN: Self-Supervised, Drift-Adaptive, and Uncertainty-Aware Graph Learning for Non-Technical Loss Detection in Latin American Smart Grids*. Full bibliographic details and DOI will be added when available.

## Authors

Salam Al-E'mari, Yousef Sanjalawe, and Muder Almiani.

## License

Released under the [MIT License](LICENSE).
