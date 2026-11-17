# likelihood 與模擬恢復檢查；binary 使用 Gee 原式，ordinal 驗證和局與雙方對稱。
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.insert(0, "scripts")
import time
import numpy as np
import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt
from scipy.special import expit
import hierarchical as hr
import ordinal

rng = np.random.default_rng(20261116)
eta, width, y = pt.dvector(), pt.dvector(), pt.lvector()
f = pytensor.function([eta, width, y], hr.ordinal_logp(eta, width, y))
e = rng.normal(0, 3, 1000)
w = np.exp(rng.normal(-2, 1, 1000))
yy = rng.integers(0, 3, 1000)
assert np.allclose(f(e, w, yy), np.asarray(ordinal.ll(e, w, yy)), atol=1e-12)
assert np.allclose(ordinal.probs(e, w), ordinal.probs(-e, w)[:, ::-1])

nplayers, n = 80, 16000
player = np.repeat(np.arange(nplayers), n//nplayers)
X = np.column_stack([rng.binomial(1, .5, n), rng.normal(0, 200, n)])
h = rng.binomial(1, .5, n)-.5
u = rng.multivariate_normal([0, 0], [[.3**2, .3*.5*.25], [.3*.5*.25, .5**2]], nplayers)
p = expit(X@np.array([.2, .004])+u[player, 0]+(.4+u[player, 1])*h)
y = rng.binomial(1, p)
model = hr.gee_model(X, h, player, y)
tick = time.monotonic()
with model:
    fit = pm.sample(draws=1000, tune=2000, chains=4, cores=4, random_seed=571,
                    nuts_sampler="nutpie", target_accept=.99, progressbar=False)
summary = az.summary(fit, var_names=["gamma", "mu_beta", "sd", "rho"], ci_prob=.95, round_to=8)
print(summary, flush=True)
summary.to_csv("outputs/tables/hierarchical_binary_check.csv")
posterior = fit["posterior"]
np.savez_compressed("data/processed/hierarchical_binary_check.npz",
                    **{name: posterior[name].values for name in ["gamma", "mu_beta", "sd", "rho"]})
truth = np.array([.2, .004, .4, .3, .5, .25])
summary["truth"] = truth
summary["standardized_error"] = (summary["mean"]-truth)/summary["sd"]
summary.to_csv("outputs/tables/hierarchical_binary_check.csv")
assert summary["standardized_error"].abs().max() < 4
assert summary["r_hat"].max() < 1.01 and summary["ess_bulk"].min() > 400
print("binary divergences", int(fit["sample_stats"]["diverging"].sum()), flush=True)
assert int(fit["sample_stats"]["diverging"].sum()) == 0
print("seconds", time.monotonic()-tick, flush=True)
print("likelihood and swap checks passed", flush=True)

# 不使用真實測試結果調整模型：先用已知參數的三類對局檢查 ordinal 延伸能恢復訊號。
n = 8000
a = rng.integers(0, nplayers, n)
b = (a+rng.integers(1, nplayers, n)) % nplayers
ha, hb = rng.binomial(1, .5, (2, n))-.5
X = np.column_stack([rng.choice([-1., 1.], n), rng.normal(0, 1, n), ha-hb])
Z = np.column_stack([np.ones(n), rng.normal(0, .2, n)])
beta, wb = np.array([.1, .8, .2]), np.array([-1.5, .1])
eta = X@beta+u[a, 0]-u[b, 0]+u[a, 1]*ha-u[b, 1]*hb
width = np.exp(Z@wb)
prob = ordinal.probs(eta, width)
y = (rng.uniform(size=n)[:, None] > prob.cumsum(1)).sum(1)
X, TX = hr.whiten(X)
Z, TZ = hr.whiten(Z)
model = hr.ordinal_model(X, Z, a, b, ha, hb, y)
tick = time.monotonic()
with model:
    fit = pm.sample(draws=1000, tune=1000, chains=4, cores=4, random_seed=572,
                    nuts_sampler="nutpie", target_accept=.95, progressbar=False)
summary = az.summary(fit, var_names=["beta", "width_beta", "sd", "rho"], ci_prob=.95, round_to=8)
truth = np.r_[np.linalg.solve(TX, beta), np.linalg.solve(TZ, wb), .3, .5, .25]
summary["truth"] = truth
summary["standardized_error"] = (summary["mean"]-truth)/summary["sd"]
summary.to_csv("outputs/tables/hierarchical_ordinal_check.csv")
print(summary, flush=True)
assert summary["standardized_error"].abs().max() < 4
assert summary["r_hat"].max() < 1.01 and summary["ess_bulk"].min() > 400
assert int(fit["sample_stats"]["diverging"].sum()) == 0
print("ordinal recovery seconds", time.monotonic()-tick, flush=True)
