import sys
sys.path.insert(0, 'scripts')
import time
import os
import numpy as np
import polars as pl
import nutpie
import hierarchical as hr
import hierarchical_rust as rr

rows = []
d = (pl.read_parquet('data/processed/hierarchical_prediction.parquet')
     .filter(pl.col('speed') == 'Blitz', pl.col('time') < 1412121600)
     .sort('game_id', 'a_id').unique('game_id', keep='first', maintain_order=True))
players = np.unique(np.r_[d['a_id'].to_numpy(), d['b_id'].to_numpy()])
a, b = [np.searchsorted(players, d[c].to_numpy()) for c in ['a_id', 'b_id']]
X, Z = hr.design(d)
X, TX = hr.whiten(X)
Z, TZ = hr.whiten(Z)
ha, hb, y = [d[c].to_numpy() for c in ['h_a', 'h_b', 'y']]
for slope in [False, True]:
    original = nutpie.compile_pymc_model(hr.ordinal_model(X, Z, a, b, ha, hb, y, slope))
    theta = original.initial_point_func(9181)
    reference, gradient = original.logp_func(theta)
    for blocks in [0, 1, 4, 16]:
        f = original.logp_func if blocks == 0 else rr.compiled_model(X, Z, a, b, ha, hb, y, slope, blocks)._make_logp_func()
        value, grad = f(theta)
        np.testing.assert_allclose(value, reference, rtol=2e-12, atol=1e-6)
        np.testing.assert_allclose(grad, gradient, rtol=2e-9, atol=1e-7)
        durations = []
        for _ in range(20):
            tick = time.perf_counter()
            f(theta)
            durations.append(time.perf_counter()-tick)
        row = {'slope': slope, 'games': len(d), 'players': len(players), 'blocks': blocks,
               'rayon_threads': int(os.environ['RAYON_NUM_THREADS']),
               'median_seconds': np.median(durations), 'logp_error': abs(value-reference),
               'gradient_max_error': np.max(np.abs(grad-gradient))}
        rows.append(row)
        print(row, flush=True)
pl.DataFrame(rows).write_csv('outputs/tables/hierarchical_rust_benchmark.csv')
