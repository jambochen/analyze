import time

import arviz as az
import numpy as np
import nutpie
import polars as pl
import pymc as pm

b = (
    pl.read_parquet("data/processed/bad_weather.parquet")
    .filter(pl.col("y2_share").is_not_null())
    .with_columns((pl.col("y2_share") >= 1.0).cast(pl.Int8).alias("is_one"))
)
feat = ["arr_delay_min", "wx_origin_flight_category_ordinal", "wx_origin_visibility_m"]
b = b.drop_nulls(feat)

carriers = sorted(b["operating_carrier"].unique().to_list())
cidx = b["operating_carrier"].replace_strict({c: i for i, c in enumerate(carriers)}).to_numpy()
y = b["is_one"].to_numpy()
Z = b.select(feat).to_numpy().astype(float)
Z = (Z - Z.mean(0)) / Z.std(0)
print("n", len(y), "公司", len(carriers), flush=True)

K = 2
with pm.Model() as m:
    a = pm.Normal("a", 0.0, 2.0, shape=K, transform=pm.distributions.transforms.ordered,
                  initval=np.linspace(-6.0, 2.0, K), default_transform=None)
    beta = pm.Normal("beta", 0.0, 1.0, shape=(K, len(feat)))
    tau = pm.HalfNormal("tau", 1.0)
    u = pm.Normal("u", 0.0, tau, shape=(len(carriers), K - 1))
    eta = pm.math.concatenate([pm.math.zeros((len(carriers), 1)), u], axis=1)
    w = pm.Deterministic("w", pm.math.softmax(eta, axis=1))
    p = pm.math.sigmoid(a[None, :] + pm.math.dot(Z, beta.T))
    mix = (w[cidx] * p).sum(axis=1)
    pm.Bernoulli("obs", p=mix, observed=y)

DRAWS, TUNE, CHAINS = 1500, 2000, 4


def run_nutpie(backend):
    # compile 時間與抽樣時間分開記：JAX 的 jit 成本是一次性的，混在 wall time
    # 裡會讓短跑的模型看起來比實際差。
    t0 = time.perf_counter()
    cm = nutpie.compile_pymc_model(m, backend=backend, gradient_backend="jax" if backend == "jax" else None)
    t1 = time.perf_counter()
    idata = nutpie.sample(cm, draws=DRAWS, tune=TUNE, chains=CHAINS, target_accept=0.95,
                          seed=0, progress_bar=False)
    return idata, t1 - t0, time.perf_counter() - t1


# JAX 的裝置在 import 時就綁定，無法於同一行程內切換，故 JAX 那組的裝置由
# 外部的 JAX_PLATFORMS 決定，標籤取實際裝置而非寫死，環境變數沒生效時才看得出來。
import jax

dev = jax.devices()[0].platform
runs = [
    ("nutpie_numba_cpu", lambda: run_nutpie("numba")),
    (f"nutpie_jax_{dev}", lambda: run_nutpie("jax")),
]

rows = []
for label, fn in runs:
    print("=== " + label, flush=True)
    idata, compile_s, sample_s = fn()
    s = az.summary(idata, var_names=["a", "beta", "tau"])
    # ESS/s 用最差的參數，抽樣效率由瓶頸參數決定，不是平均值。
    ess_min = float(s["ess_bulk"].min())
    am = idata.posterior["a"].mean(("chain", "draw")).values
    rows.append({
        "backend": label, "compile_s": compile_s, "sample_s": sample_s,
        "ess_bulk_min": ess_min, "ess_per_s": ess_min / sample_s,
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
        "rhat_max": float(s["r_hat"].max()),
        "a0": float(am[0]), "a1": float(am[1]),
        "tau": float(idata.posterior["tau"].mean(("chain", "draw")).values),
    })
    print(rows[-1], flush=True)
    # 每跑完一個就落地，中途中斷也留得住已完成的部分。
    pl.DataFrame(rows).write_csv("outputs/tables/bench_nuts_backend.csv")

print(pl.DataFrame(rows))
