# 與完整逐列預測比較，涵蓋兩種模型、新／舊玩家及同局雙方視角。
import sys
sys.path.insert(0, "scripts")
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import polars as pl
import hierarchical as hr
from hierarchical_predictive import chain_probabilities, game_rows

rows = []
for kind in ["slope", "intercept"]:
    prefix = f"hierarchical_Classical_Jul-Sep_{kind}"
    fit = np.load(f"data/processed/{prefix}.npz")
    saved = pl.read_parquet(f"data/processed/{prefix}_prediction.parquet")
    games = saved["game_id"].head(256).unique()
    expected = saved.filter(pl.col("game_id").is_in(games.implode()))
    d = pl.read_parquet("data/processed/hierarchical_prediction.parquet").join(expected.select("game_id", "a_id"), on=["game_id", "a_id"], how="semi", maintain_order="left")
    assert d.select("game_id", "a_id").equals(expected.select("game_id", "a_id"))
    players = fit["players"]
    ai, bi = [np.searchsorted(players, d[c].to_numpy()) for c in ["a_id", "b_id"]]
    ak, bk = [players[np.minimum(i, len(players)-1)] == d[c].to_numpy() for i, c in [(ai, "a_id"), (bi, "b_id")]]
    assert (ak & bk).any() and (~(ak & bk)).any()
    ai, bi = [np.minimum(i, len(players)-1) for i in [ai, bi]]
    X, Z = hr.design(d)
    X, Z = X@fit["TX"], Z@fit["TZ"]
    ha, hb = d["h_a"].to_numpy(), d["h_b"].to_numpy()
    first, inverse, reverse = game_rows(d, X, Z, ha, hb)
    assert reverse.any() and len(first) < len(d)
    samples = {k: fit[k] for k in ["beta", "width_beta", "u", "sd"]+(["rho"] if kind == "slope" else [])}
    worker = partial(chain_probabilities, samples=samples, X=X[first], Z=Z[first], ai=ai[first], bi=bi[first],
                     ak=ak[first], bk=bk[first], ha=ha[first], hb=hb[first], kind=kind)
    tick = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    p = np.stack([r[0] for r in results]).mean(0)[inverse]
    p[reverse] = p[reverse, ::-1]
    error = np.max(np.abs(p-expected.select("p_loss", "p_draw", "p_win").to_numpy()))
    assert error < 1e-12
    rows.append({"kind": kind, "rows": len(d), "games": len(first), "max_error": error, "seconds": time.monotonic()-tick})
pl.DataFrame(rows).write_csv("outputs/tables/hierarchical_predictive_check.csv")
print(pl.DataFrame(rows))
