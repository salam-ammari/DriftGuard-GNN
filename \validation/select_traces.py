"""Validation-seed selection (seeds 101-103 only; never the test seeds).
select_traces.py: collects monitor statistics with alarms disabled, used to
choose the alarm threshold (5) and persistence (2).
select_tune.py: compares ensemble-head sizes; 16 hidden units / 200 epochs chosen."""
import warnings; warnings.filterwarnings('ignore')
import sys, json; sys.path.insert(0, '..')
import exp_sim
exp_sim.THR = 99; exp_sim.AUDIT_THR = 99
scen = sys.argv[1]
for seed in map(int, sys.argv[2:]):
    r = exp_sim.run(seed, scen, 'topology')
    json.dump({'z': r['ztrace'], 'a': r['atrace'], 'truth': r['truth'],
               'det1': {k: v['prauc'] for k, v in r['detection']['0.01'].items()}},
              open(f'../../results/trace_{scen}_{seed}.json', 'w'))
    print(scen, seed, 'done', {k: round(v['prauc'], 3) for k, v in r['detection']['0.01'].items()}, flush=True)
