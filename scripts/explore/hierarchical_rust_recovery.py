import sys
sys.path.insert(0, 'scripts')
import time
import json
from pathlib import Path
import numpy as np
import arviz as az
import nutpie
import hierarchical as hr
import hierarchical_rust as rr
import ordinal

rng = np.random.default_rng(572)
players, n = 80, 8000
u = rng.multivariate_normal([0, 0], [[.3**2, .3*.5*.25], [.3*.5*.25, .5**2]], players)
a = rng.integers(players, size=n)
b = (a+rng.integers(1, players, n)) % players
ha, hb = rng.binomial(1, .5, (2, n))-.5
X = np.column_stack([rng.choice([-1., 1.], n), rng.normal(size=n), ha-hb])
Z = np.column_stack([np.ones(n), rng.normal(0, .2, n)])
beta, wb = np.array([.1, .8, .2]), np.array([-1.5, .1])
p = ordinal.probs(X@beta+u[a, 0]-u[b, 0]+u[a, 1]*ha-u[b, 1]*hb, np.exp(Z@wb))
y = (rng.uniform(size=n)[:, None] > p.cumsum(1)).sum(1)
X, TX = hr.whiten(X)
Z, TZ = hr.whiten(Z)
model = rr.compiled_model(X, Z, a, b, ha, hb, y)
tick = time.monotonic()
fit = nutpie.sample(model, draws=1000, tune=1000, chains=4, cores=4, seed=572,
                    target_accept=.95, progress_bar=False, save_warmup=False)
summary = az.summary(fit, var_names=['beta', 'width_beta', 'sd', 'rho'], ci_prob=.95, round_to=8)
summary['truth'] = np.r_[np.linalg.solve(TX, beta), np.linalg.solve(TZ, wb), .3, .5, .25]
summary['standardized_error'] = (summary['mean']-summary['truth'])/summary['sd']
summary.to_csv('outputs/tables/hierarchical_rust_recovery.csv')
metadata = {'seconds': time.monotonic()-tick, 'divergences': int(fit['sample_stats']['diverging'].sum()),
            'rhat_max': float(summary['r_hat'].max()), 'ess_bulk_min': float(summary['ess_bulk'].min()),
            'standardized_error_max': float(summary['standardized_error'].abs().max())}
Path('outputs/tables/hierarchical_rust_recovery.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(summary, flush=True)
print(metadata, flush=True)
assert summary['standardized_error'].abs().max() < 4
assert summary['r_hat'].max() < 1.01 and summary['ess_bulk'].min() > 400
assert int(fit['sample_stats']['diverging'].sum()) == 0
