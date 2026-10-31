# 直接效果模型（不含 h）的殘差，每個速度類別以全部有上一局的列各配適一次。
# 供 player_heterogeneity.py 讀入：重抽的行程池須在 jax 啟動之前 fork，所以配適獨立成一支腳本。

import os
import sys

# 先於 import jax 設定：float64 在這張 GPU 上很慢
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import polars as pl

from crossover_direct import SPEEDS, direct_columns, direct_resid, load

t = direct_columns(load()[0])
out = pl.concat([
    (ts := t.filter(pl.col("speed") == s)).select("game_id", "player", "speed").with_columns(r_dir=pl.Series(direct_resid(ts)))
    for s in SPEEDS
])
print(out.group_by("speed").agg(n=pl.len(), mean=pl.col("r_dir").mean(), sd=pl.col("r_dir").std()))
out.write_parquet("data/processed/arena_direct_resid.parquet")
