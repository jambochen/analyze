# GAM 與 GLMM 的探索：匯出 state_adjusted_probs.py 同一份樣本與切分（上半年估計、下半年評估）給 R 的 mgcv。
# 樣本的建構照抄 state_adjusted_probs.py，之後若該腳本改動須同步。
import sys

sys.path.insert(0, "scripts")

import polars as pl

import novice
from arena_batches import player_sample_rd

UTC = "UTC"
grp = ["player", "tournament_id"]
base = (
    player_sample_rd()
    .sort("player", "tournament_id", "start", "game_id")
    .with_columns(
        prev_result=pl.col("result").shift(1).over(grp),
        dr=((pl.col("elo") - pl.col("elo").shift(1).over(grp)) / 100).cast(pl.Float64),
    )
)
opp = base.select("game_id", opponent="player", opp_prev_result="prev_result", opp_dr="dr")
t = (
    base.join(opp, on=["game_id", "opponent"], how="left")
    .filter(pl.col("prev_result").is_not_null())
    .with_columns(
        y=(pl.col("score") * 2).cast(pl.Int64) + 1,
        prev_draw=(pl.col("prev_result") == "draw").cast(pl.Float64),
        prev_win=(pl.col("prev_result") == "win").cast(pl.Float64),
        opp_missing=pl.col("opp_prev_result").is_null().cast(pl.Float64),
        opp_prev_draw=(pl.col("opp_prev_result") == "draw").fill_null(False).cast(pl.Float64),
        opp_prev_win=(pl.col("opp_prev_result") == "win").fill_null(False).cast(pl.Float64),
        opp_dr=pl.col("opp_dr").fill_null(0.0),
        novice_rd=novice.novice_rd("rd", "opp_rd").cast(pl.Float64),
        train=(pl.col("start") < pl.datetime(2014, 7, 1, time_zone=UTC)).cast(pl.Int64),
    )
    .with_columns(
        d_prev_draw=pl.col("prev_draw") - pl.col("opp_prev_draw"),
        d_prev_win=pl.col("prev_win") - pl.col("opp_prev_win"),
        d_dr=pl.col("dr") - pl.col("opp_dr"),
    )
)
cols = ["speed", "player", "opponent", "game_id", "train", "y", "eta", "w", "prev_draw", "prev_win", "dr", "d_prev_draw", "d_prev_win", "d_dr", "opp_missing", "rd", "opp_rd", "elo", "opp_elo", "games_in_tour", "novice_rd"]
for s in ["Bullet", "Blitz"]:
    d = t.filter(pl.col("speed") == s).select(cols)
    print(s, d.group_by("train").len().sort("train").to_dicts(), d["player"].n_unique())
    d.write_csv(f"data/processed/explore_gam_glmm_{s.lower()}.csv")
