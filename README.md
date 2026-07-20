# DriftGuard-GNN

Official research code for **“DriftGuard-GNN: Self-Supervised, Drift-Adaptive, and Uncertainty-Aware Graph Learning for Non-Technical Loss Detection in Latin American Smart Grids.”**

DriftGuard-GNN is a reproducible simulation framework for non-technical loss detection under label scarcity, regional concept drift, adaptive fraud, and limited inspection budgets. It combines feeder-aware graph contextualization, self-supervised learning, localized drift monitoring, selective adaptation, calibrated uncertainty, and utility-aware inspection ranking.

## Files

- `driftguard_sim.py` — generates the synthetic LatAm smart-grid testbed, trains the models and baselines, evaluates drift detection and adaptation, and saves all experimental results.
- `nn.py` — minimal NumPy implementation of the dense neural networks, Adam optimization, gradient clipping, and sigmoid function used by DriftGuard-GNN.

The `.pyc` files are automatically generated Python bytecode and do not need to be uploaded to GitHub.

## Requirements

- Python 3.10 or later (tested with Python 3.12)
- NumPy
- scikit-learn

Install the dependencies with:

```bash
pip install numpy scikit-learn
```

## Running the Experiment

Before running, replace the hard-coded output path at the end of `driftguard_sim.py`:

```python
with open("results.json", "w") as f:
```

Then execute:

```bash
python driftguard_sim.py
```

The script evaluates five fixed random seeds and writes the complete results to `results.json`. The output covers detection performance, label-efficiency experiments, localized drift monitoring, adaptation strategies, uncertainty calibration, abstention, inspection ranking, computational timing, and ablation studies.

## Reproducibility

The experiments use fixed seeds (`11, 23, 37, 51, 73`) and a chronological train–validation–test protocol. The synthetic testbed contains 3,600 consumers, 24 feeders, four regions, and 156 weeks of consumption data.

## Citation

If you use this code, please cite the DriftGuard-GNN paper. Full citation information will be added after publication.

## Authors

Salam Al-E'mari, Yousef Sanjalawe, and Muder Almiani.

## License

The original source code and documentation in this repository are released under the [MIT License](https://github.com/salam-ammari/DriftGuard-GNN/blob/main/LICENSE).
