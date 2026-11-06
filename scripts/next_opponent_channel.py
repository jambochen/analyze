# 下一局管道的機制：被分到強對手之後下一局的對手較弱，是否由「輸得快、較早空閒」說明。
# 規格見 docs/methods.md「剛被分到強對手的效果」的「下一局管道的機制」。只作描述：上一局的結果與局長
# 都發生在處理之後，不作因果的中介推論。變數以上一局所屬的批去平均，標準誤依玩家分群。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import polars as pl
import statsmodels.api as sm

from arena_batches import player_sample_rd
from crossover_direct import load

grp = ["player", "tournament_id"]
t, _, _ = load()
prev = player_sample_rd().sort(grp + ["start", "game_id"]).select("game_id", "player", prev_game=pl.col("game_id").shift(1).over(grp))
meta = pl.read_parquet("data/processed/game_meta.parquet", columns=["game_id", "created_ms", "last_move_ms"])
d = (
    t.filter(pl.col("speed").is_in(["Bullet", "Blitz"]), pl.col("layer_lag1") == "multi")
    .join(prev, on=["game_id", "player"])
    .join(meta.rename({"game_id": "prev_game"}), on="prev_game")
    .with_columns(
        log_dur=((pl.col("last_move_ms") - pl.col("created_ms")).clip(lower_bound=1000) / 1000).log(),
        prev_win=(pl.col("score_lag1") == 1).cast(pl.Float64),
        prev_draw=(pl.col("score_lag1") == 0.5).cast(pl.Float64),
        opp_elo_next=pl.col("opp_elo").cast(pl.Float64),
    )
)
print(d.group_by("speed").agg(n=pl.len(), players=pl.col("player").n_unique(), batches=pl.col("bid_lag1").n_unique()))

COLS = ["h_lag1", "score_lag1", "log_dur", "prev_win", "prev_draw", "opp_elo_next"]
rows = []
for spd in ["Bullet", "Blitz"]:
    s = d.filter(pl.col("speed") == spd).with_columns([(pl.col(c) - pl.col(c).mean().over("bid_lag1")).alias(c) for c in COLS])
    g = s["player"].to_numpy()

    def ols(y, xs, model):
        r = sm.OLS(s[y].to_numpy(), s.select(xs).to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": g})
        for j, x in enumerate(xs):
            rows.append({"speed": spd, "model": model, "y": y, "x": x, "est": r.params[j], "se": r.bse[j], "lo": r.conf_int()[j, 0], "hi": r.conf_int()[j, 1], "n": s.height})

    ols("score_lag1", ["h_lag1"], "step1_score")
    ols("log_dur", ["h_lag1"], "step1_duration")
    ols("opp_elo_next", ["h_lag1"], "step2_h_only")
    ols("opp_elo_next", ["h_lag1", "prev_win", "prev_draw", "log_dur"], "step2_with_mediators")

res = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(res)
res.write_csv("outputs/tables/next_opponent_channel.csv")
