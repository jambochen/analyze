# 本次比較的資料快取；以相同開局時間彙總後再累積，避免同時開局資料互相洩漏結果。
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.insert(0, "scripts")
import time
import polars as pl
import temporal_models as tm

tick = time.monotonic()
t, folds, _ = tm.sample("Lichess")
players = pl.concat([t.select(player="a_id"), t.select(player="b_id")]).unique()
pg = (pl.scan_parquet("data/processed/player_games.parquet")
      .filter(pl.col("start") < pl.datetime(2015, 1, 1, time_zone="UTC"))
      .select("game_id", "player", "speed", "start", "result")
      .join(players.lazy(), on="player", how="semi").collect())
counts = (pg.group_by("player", "speed", "start")
          .agg(n=pl.len(), wins=(pl.col("result") == "win").sum())
          .sort("player", "speed", "start")
          .with_columns(past_n=pl.col("n").cum_sum().over("player", "speed")-pl.col("n"),
                        past_wins=pl.col("wins").cum_sum().over("player", "speed")-pl.col("wins"))
          .with_columns(past_win_rate=pl.when(pl.col("past_n") > 0)
                        .then(pl.col("past_wins") / pl.col("past_n")).otherwise(.5)))
history = (pg.join(counts.select("player", "speed", "start", "past_win_rate"), on=["player", "speed", "start"])
           .select("game_id", "player", "past_win_rate"))
t = (t.join(history.rename({"player": "a_id", "past_win_rate": "a_rate"}), on=["game_id", "a_id"])
     .join(history.rename({"player": "b_id", "past_win_rate": "b_rate"}), on=["game_id", "b_id"])
     .with_columns(h_a=(pl.col("prev_result") == "win").cast(pl.Float64)-pl.col("a_rate"),
                   h_b=pl.when(pl.col("opp_missing") == 0)
                   .then((pl.col("opp_prev_result") == "win").cast(pl.Float64)-pl.col("b_rate")).otherwise(0.))
     .sort("time", "game_id", "a_id"))
assert t.select(pl.struct("game_id", "a_id").n_unique()).item() == t.height
assert t.select(pl.col("h_a").is_null().sum()+pl.col("h_b").is_null().sum()).item() == 0
t.write_parquet("data/processed/hierarchical_prediction.parquet")
print(t.group_by("speed").agg(n=pl.len(), games=pl.col("game_id").n_unique(), players=pl.col("a_id").n_unique()))
print("seconds", time.monotonic()-tick, flush=True)
