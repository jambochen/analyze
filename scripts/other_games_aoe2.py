# AoE2 DE 1v1 ranked 的外部驗證：與 OGS 同一套方法（external.py），規格見 docs/plan.md「其他遊戲的外部驗證」。
# - 資料：aoe2-guide 的快照（data/raw/aoe2/SOURCE.md），2026 年 6 月 1 日至 7 月 3 日；6 月 24 日之前估計、之後評估
# - 系統的期望勝率：雙方 old_rating 代入 Elo 公式。AoE2 的實際更新規則未公開，這只是以表頭等級分換算的期望
# - 上一局：同一天（與上一場結束相隔 12 小時內）、且上一場的 new_rating 等於本場的 old_rating；快照有漏收的場次，
#   接不上時視為本日無上一局。兩者在本場開始前都已知，不用本局結果
# - 連勝連敗與停止規則只用一天內每一場都接得上的天
# - 新手：任一方局前的累計局數（career_wins 與 career_losses 含本局，兩者之和減一）未滿 30
# - 沒有和局；1v1 中不是恰一勝一負的場次排除。沒有 RD，主基準去掉 RD 收縮項

import datetime as dt
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import polars as pl

import novice
from external import evaluate, player_rows, stop_verdicts
from runs import runs_before

D = "data/raw/aoe2/2026-11-06/"
DAY_GAP = 12 * 3600
T_EVAL = int(dt.datetime(2026, 6, 24, tzinfo=dt.timezone.utc).timestamp())

m = pl.read_parquet(D + "matches.parquet").filter(pl.col("ladder") == "1v1").select("match_id", "start_time", "completion_time")
p = pl.read_parquet(D + "players.parquet").join(m, on="match_id")
p = p.filter((pl.col("won").cast(pl.Int32).sum().over("match_id") == 1) & (pl.len().over("match_id") == 2))
pg = (
    p.select(
        "match_id", "start_time", "completion_time", "team", "old_rating", "new_rating",
        player="profile_id", score=pl.col("won").cast(pl.Float64), games_before=pl.col("career_wins") + pl.col("career_losses") - 1,
    )
    .sort("player", "start_time", "match_id")
    .with_columns(gap=pl.col("start_time") - pl.col("completion_time").shift(1).over("player"), chain=pl.col("old_rating") == pl.col("new_rating").shift(1).over("player"))
    .with_columns(day=(pl.col("gap").is_null() | (pl.col("gap") > DAY_GAP)).cum_sum().over("player"))
    .with_columns(
        first_of_day=pl.col("day") != pl.col("day").shift(1).over("player").fill_null(-1),
        day_start=pl.col("start_time").first().over("player", "day"),
        result=pl.when(pl.col("score") == 1).then(pl.lit("win")).otherwise(pl.lit("loss")),
    )
    .with_columns(
        has_prev=~pl.col("first_of_day") & pl.col("chain").fill_null(False),
        day_complete=(pl.col("first_of_day") | pl.col("chain").fill_null(False)).all().over("player", "day"),
    )
    .with_columns(
        prev_win=pl.when("has_prev").then(pl.col("score").shift(1).over("player")).otherwise(0.0),
        missing=(~pl.col("has_prev")).cast(pl.Float64),
        dr=pl.when("has_prev").then((pl.col("old_rating") - pl.col("old_rating").shift(1).over("player")) / 100).otherwise(0.0),
        novice_side=pl.col("games_before") < novice.GAMES,
    )
)
pg = runs_before(pg, ["player", "day"])
a, b = pg.filter(pl.col("team") == 0), pg.filter(pl.col("team") == 1)
side = lambda x, s: x.select("match_id", *[pl.col(c).alias(f"{s}_{c}") for c in ["player", "old_rating", "score", "prev_win", "missing", "dr", "novice_side"]])  # noqa: E731
t = (
    side(a, "a").join(side(b, "b"), on="match_id").join(m, on="match_id")
    .with_columns(
        game_id="match_id", a_id="a_player", b_id="b_player", y="a_score", speed=pl.lit("1v1"),
        train=pl.col("start_time") < T_EVAL, const=pl.lit(1.0),
        xr=((pl.col("a_old_rating") + pl.col("b_old_rating")) / 2 - 1500) / 100,
        xd=((pl.col("a_old_rating") - pl.col("b_old_rating")) / 100).cast(pl.Float64),
        p_sys=1 / (1 + 10 ** (-(pl.col("a_old_rating") - pl.col("b_old_rating")) / 400)),
        d_prev_win=pl.col("a_prev_win") - pl.col("b_prev_win"),
        d_missing=pl.col("a_missing") - pl.col("b_missing"),
        d_dr=pl.col("a_dr") - pl.col("b_dr"),
        novice=pl.col("a_novice_side") | pl.col("b_novice_side"),
    )
    .with_columns(xr2=pl.col("xr") ** 2, xdxr=pl.col("xd") * pl.col("xr"))
)
sample = t.group_by("train").agg(
    n=pl.len(), players=pl.concat_list("a_id", "b_id").explode().n_unique(), novice=pl.col("novice").mean(), a_win=pl.col("y").mean(),
    has_prev=(1 - pl.col("a_missing")).mean(), rating_diff_sd=(pl.col("xd") * 100).std(),
).sort("train")
print(sample)
sample.write_csv("outputs/tables/other_games_aoe2_sample.csv")

BASE = ["const", "xr", "xr2", "xd", "xdxr"]
tables, te = evaluate(t, ["1v1"], BASE, lambda th, d: th[3] + th[4] * d["xr"])
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(tables["calibration"])
    print(tables["steps"])
    print(tables["prediction"].filter(pl.col("metric") == "log_loss"))
for k, v in tables.items():
    v.write_csv(f"outputs/tables/other_games_aoe2_{k}.csv")
ev = player_rows(te)
pgj = pg.select(game_id="match_id", player="player", day="day", day_complete="day_complete", loss_run="loss_run", win_run="win_run")
ev.join(pgj, on=["game_id", "player"]).write_parquet("data/processed/aoe2_eval_rows.parquet")

# 停止規則：評估段、整天都接得上的天，殘差取主基準
opp = pg.select("match_id", opponent="player", opp_novice="novice_side")
d = (
    pg.filter("day_complete")
    .join(opp, on="match_id").filter(pl.col("opponent") != pl.col("player"))
    .join(ev.select(match_id="game_id", player="player", e="e_base", speed="speed"), on=["match_id", "player"])
    .with_columns(resid=pl.col("score") - pl.col("e"), novice_rd=pl.col("novice_side") | pl.col("opp_novice"))
    .sort("player", "start_time", "match_id")
)
res, reset = stop_verdicts(d, pl.col("day_start") >= T_EVAL, ["1v1"], tables["coef"])
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(res.filter(pl.col("group") == "all"))
    print(reset)
res.write_csv("outputs/tables/other_games_aoe2_stop_rules.csv")
reset.write_csv("outputs/tables/other_games_aoe2_stop_rules_reset.csv")
