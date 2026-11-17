# 完整一個訓練段的短鏈，只估耗時與檢查執行，不將短鏈的結果用來判定預測優劣。
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.insert(0, "scripts")
import time
import json
import numpy as np
import polars as pl
import pymc as pm
import hierarchical as hr

speed = os.environ.get("HIERARCHICAL_SPEED", "Bullet")
d = (pl.read_parquet("data/processed/hierarchical_prediction.parquet")
     .filter(pl.col("speed") == speed, pl.col("time") < 1404172800)
     .sort("game_id", "a_id").unique("game_id", keep="first", maintain_order=True))
players = np.unique(np.r_[d["a_id"].to_numpy(), d["b_id"].to_numpy()])
a, b = [np.searchsorted(players, d[c].to_numpy()) for c in ["a_id", "b_id"]]
X, Z = hr.design(d)
X, TX = hr.whiten(X)
Z, TZ = hr.whiten(Z)
model = hr.ordinal_model(X, Z, a, b, d["h_a"].to_numpy(), d["h_b"].to_numpy(), d["y"].to_numpy())
print("sample", speed, d.height, len(players), flush=True)
tick = time.monotonic()
with model:
    fit = pm.sample(draws=50, tune=100, chains=2, cores=2, random_seed=613,
                    nuts_sampler="nutpie", target_accept=.9, progressbar=False)
stats = fit["sample_stats"]
out = {"speed": speed, "games": d.height, "players": len(players), "seconds": time.monotonic()-tick,
       "steps_mean": float(stats["n_steps"].mean()), "divergences": int(stats["diverging"].sum())}
print(out, flush=True)
with open("outputs/tables/hierarchical_benchmark.json", "w") as f:
    json.dump(out, f, indent=2)
