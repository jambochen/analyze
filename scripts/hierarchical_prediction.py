# 固定賽制、時間切分及模型；以環境變數指定工作，逐個執行以限制記憶體。
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time
import json
import hashlib
from pathlib import Path
import numpy as np
import polars as pl
import nutpie
import arviz as az
import hierarchical as hr
import hierarchical_rust as rr

speed = os.environ["HIERARCHICAL_SPEED"]
fold = os.environ["HIERARCHICAL_FOLD"]
kind = os.environ["HIERARCHICAL_KIND"]
assert kind in ["intercept", "slope"]
draws = int(os.environ.get("HIERARCHICAL_DRAWS", "1000"))
tune = int(os.environ.get("HIERARCHICAL_TUNE", "1000"))
cutoff = {"Jul-Sep": 1404172800, "Oct-Dec": 1412121600}[fold]
prefix = f"hierarchical_{speed}_{fold}_{kind}"
d = (pl.read_parquet("data/processed/hierarchical_prediction.parquet")
     .filter(pl.col("speed") == speed, pl.col("time") < cutoff))
expected = (pl.read_csv("outputs/tables/likelihood_selection_sample.csv")
            .filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("segment") == "train_refit").row(0, named=True))
assert d.height == expected["n"] and d["game_id"].n_unique() == expected["games"]
# 同一對局只取一個確定的視角，避免雙重 likelihood；沒有丟棄任何實際對局。
d = d.sort("game_id", "a_id").unique("game_id", keep="first", maintain_order=True)
players = np.unique(np.r_[d["a_id"].to_numpy(), d["b_id"].to_numpy()])
a, b = [np.searchsorted(players, d[c].to_numpy()) for c in ["a_id", "b_id"]]
X, Z = hr.design(d)
X, TX = hr.whiten(X)
Z, TZ = hr.whiten(Z)
model = rr.compiled_model(X, Z, a, b, d["h_a"].to_numpy(), d["h_b"].to_numpy(), d["y"].to_numpy(), kind == "slope")
print(prefix, "games", d.height, "players", len(players), flush=True)


def progress(chains):
    print({"model": prefix, "finished_draws": [c.finished_draws for c in chains],
           "total_draws": [c.total_draws for c in chains], "tuning": [c.tuning for c in chains]}, flush=True)


tick = time.monotonic()
fit = nutpie.sample(model, draws=draws, tune=tune, chains=4, cores=4, seed=9181,
                    target_accept=.95, progress_bar=False, save_warmup=False,
                    progress_callback=progress, progress_rate=30000)
names = ["beta", "width_beta", "sd"]+(["rho"] if kind == "slope" else [])
summary = az.summary(fit, var_names=names+["u"], ci_prob=.95, round_to=8)
summary.to_csv(f"outputs/tables/{prefix}_diagnostics.csv")
print(summary.loc[~summary.index.str.startswith("u[")], flush=True)
posterior = fit["posterior"]
values = {name: posterior[name].values for name in names+["u"]}
# 玩家係數存 float32 足以重建機率；全域係數保留 float64，變換與玩家索引一併保存。
values["u"] = values["u"].astype(np.float32)
np.savez_compressed(f"data/processed/{prefix}.npz", **values, players=players, TX=TX, TZ=TZ)
stats = fit["sample_stats"]
metadata = {"speed": speed, "fold": fold, "kind": kind, "draws_per_chain": draws, "tune": tune,
            "backend": "rust-logp-gradient+nutpie", "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
            "games": d.height, "players": len(players), "seconds": time.monotonic()-tick,
            "divergences": int(stats["diverging"].sum()),
            "diagnostic_columns": summary.columns.tolist(),
            "source_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                              ["scripts/hierarchical.py", "scripts/hierarchical_prediction.py", "scripts/hierarchical_rust.py",
                               "rust/likelihood/src/hierarchical.rs", "rust/likelihood/src/lib.rs", "rust/likelihood/Cargo.lock",
                               "docs/hierarchical-replication.md", "outputs/tables/likelihood_selection_models.json"]}}
Path(f"outputs/tables/{prefix}_fit.json").write_text(json.dumps(metadata, indent=2))
print(metadata, flush=True)
